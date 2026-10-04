"""Evaluate frozen exploratory controls without fitting or changing original stages.

The clinical cohort has been inspected previously. These scores and intervals
are explicitly post hoc and cannot certify an untouched external evaluation.
Only hash-validated local control checkpoints are unpickled.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_evidence import (align_rows, dump_json, extension_dir, gene_metrics,
                              paired_group_bootstrap, rank_metrics, sha256,
                              subgroup_metrics)
from research_models import ALL_MODELS, predict_bundle


def checked_alignment(frame, ids, labels, key='row_id'):
    aligned = align_rows(frame, ids, key)
    if not np.array_equal(np.asarray(labels), aligned.LABEL_PATHOGENIC.to_numpy()):
        raise ValueError('Prediction labels differ from source cohort')
    return aligned


def aggregate_folds(scores, thresholds):
    matrix = np.asarray(scores, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 5 or cutoffs.shape != (5,):
        raise ValueError('Exactly five aligned frozen folds are required')
    if not np.isfinite(matrix).all() or not np.isfinite(cutoffs).all():
        raise ValueError('Nonfinite predictions or thresholds')
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError('Expected calibrated probabilities')
    return matrix.mean(axis=0), ((matrix >= cutoffs[:, None]).sum(axis=0) > 2).astype(np.int8)


def metrics(y, score, decisions):
    return {**rank_metrics(y, score),
            'brier': float(brier_score_loss(y, score)),
            'mcc': float(matthews_corrcoef(y, decisions))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/15_research_extensions')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    source = args.source_root
    extension = extension_dir(source, args.output_root)
    destination = extension / 'manuscript_analysis'
    destination.mkdir(exist_ok=True)
    protocol = json.loads((extension / 'protocol.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen control source changed: {name}')
    inputs = [source / '10_esm_features/internal_with_esm.parquet',
              source / '10_esm_features/clinvar_with_esm.parquet',
              source / '10_esm_features/clinvar_esm_embeddings.npy',
              source / '12_external_validation/external_predictions.npz',
              extension / 'protocol.json',
              source / '14_tuning/nested_inner_splits.json']
    hashes = {str(p.resolve()): sha256(p) for p in inputs}
    internal = pd.read_parquet(inputs[0])
    if hashes[str(inputs[0].resolve())] != protocol['input_sha256']:
        raise ValueError('Internal table differs from frozen protocol')
    # This legacy archive was produced locally and audited. Its full checksum
    # is recorded above; object fields are identities, not foreign downloads.
    prefix = 'clinvar_exact_variant_disjoint__'
    with np.load(inputs[3], allow_pickle=True) as archive:
        ids = archive[prefix + 'variant_ids'].astype(str)
        clinical_y = archive[prefix + 'y'].astype(np.int8)
        clinical_genes = archive[prefix + 'groups'].astype(str)
        original_clinical = {name: archive[prefix + name].copy() for name in
                             ['raw_esm_zero_shot', 'esm_conservation_logistic',
                              'gated_fusion', 'reliability_residual']}
    raw = pd.read_parquet(inputs[1])
    raw['embedding_index'] = np.arange(len(raw))
    raw['variant_id'] = raw.variant_id.astype(str).str.removeprefix('GRCh37:')
    clinical = checked_alignment(raw, ids, clinical_y, 'variant_id')
    if not np.array_equal(clinical.genename.fillna('').astype(str).str.strip().str.upper().to_numpy(), clinical_genes):
        raise ValueError('Clinical grouping differs from original evaluation')
    internal_variants = set(internal.variant_id.astype(str).str.removeprefix('GRCh37:'))
    if internal_variants.intersection(ids):
        raise ValueError('Exact training variant overlap in clinical cohort')
    all_embeddings = np.load(inputs[2], mmap_mode='r')
    clinical_embeddings = np.asarray(all_embeddings[clinical.embedding_index.to_numpy()], dtype=np.float32)
    internal_models, clinical_models, results = {}, {}, {'internal': {}, 'clinical': {}}
    internal_y = internal.LABEL_PATHOGENIC.to_numpy(np.int8)
    internal_ids = internal.row_id.astype(str).to_numpy()
    if hashes[str(inputs[5].resolve())] != protocol['split_plan_sha256']:
        raise ValueError('Frozen split plan checksum differs')
    split_plan = json.loads(inputs[5].read_text())
    fold_ids = np.zeros(len(internal), dtype=np.int8)
    for record in split_plan['folds']:
        indices = np.asarray(record['outer_validation']['indices'], dtype=int)
        if np.any(fold_ids[indices] != 0):
            raise ValueError('Repeated OOF partition assignment')
        fold_ids[indices] = int(record['outer_fold'])
    if np.any(fold_ids == 0):
        raise ValueError('Incomplete OOF partition coverage')
    all_prediction_arrays = {'clinical_ids': ids, 'clinical_y': clinical_y,
                             'clinical_genes': clinical_genes}
    for model in ALL_MODELS:
        print(f'Validating and scoring {model}', flush=True)
        path = extension / 'controls' / f'{model}_oof.npz'
        hashes[str(path.resolve())] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive['row_ids'].astype(str), internal_ids):
                raise ValueError(f'OOF identities differ for {model}')
            if not np.array_equal(archive['y'], internal_y):
                raise ValueError(f'OOF labels differ for {model}')
            score = archive['probabilities'].copy()
            decisions = archive['decisions'].copy()
        internal_models[model] = score
        results['internal'][model] = metrics(internal_y, score, decisions)
        results['internal'][model]['folds'] = {
            str(f): metrics(internal_y[fold_ids == f], score[fold_ids == f], decisions[fold_ids == f])
            for f in range(1, 6)}
        fold_scores, thresholds = [], []
        for fold in range(1, 6):
            folder = extension / 'controls' / model / f'fold_{fold}'
            record_path = folder / 'result.json'
            record = json.loads(record_path.read_text())
            if record['fingerprint'] != protocol['fingerprint'] or record['engineering_only']:
                raise ValueError('Checkpoint is not a full frozen-protocol result')
            checkpoint = folder / 'model.pkl'
            for filename, expected in record['artifacts'].items():
                if sha256(folder / filename) != expected:
                    raise ValueError('Control checkpoint checksum mismatch')
            hashes[str(record_path.resolve())] = sha256(record_path)
            hashes[str(checkpoint.resolve())] = record['artifacts']['model.pkl']
            with checkpoint.open('rb') as stream:
                bundle = pickle.load(stream)
            fold_scores.append(predict_bundle(bundle, clinical, clinical_embeddings))
            thresholds.append(float(bundle['threshold']))
            del bundle
        mean, votes = aggregate_folds(fold_scores, thresholds)
        clinical_models[model] = mean
        all_prediction_arrays[model + '__probabilities'] = mean
        all_prediction_arrays[model + '__decisions'] = votes
        all_prediction_arrays[model + '__fold_probabilities'] = np.stack(fold_scores)
        all_prediction_arrays[model + '__thresholds'] = np.asarray(thresholds)
        results['clinical'][model] = metrics(clinical_y, mean, votes)
    np.savez_compressed(destination / 'clinical_controls.npz', **all_prediction_arrays)
    internal_intervals = {}
    for model in ['same_input_lightgbm', 'same_input_mlp', 'fixed_quality_gate', 'unbounded_residual']:
        print(f'Paired homology-group interval {model} minus reliability_control', flush=True)
        internal_intervals[model + '_minus_reliability_control'] = paired_group_bootstrap(
            internal_y, internal_models['reliability_control'], internal_models[model],
            internal.split_group.astype(str).to_numpy(), iterations=args.bootstrap)
    clinical_intervals = {}
    for model in ['same_input_lightgbm', 'same_input_mlp']:
        for reference in ['reliability_control', 'gated_fusion']:
            print(f'Paired clinical interval {model} minus {reference}', flush=True)
            ref = clinical_models.get(reference, original_clinical.get(reference))
            clinical_intervals[model + '_minus_' + reference] = paired_group_bootstrap(
                clinical_y, ref, clinical_models[model], clinical_genes, iterations=args.bootstrap)
    all_clinical = {**original_clinical, **clinical_models}
    train_genes = set(internal.genename.fillna('').astype(str).str.strip().str.upper())
    conservation = ~clinical[['GERP++_RS__missing', 'phyloP100way_vertebrate__missing',
                             'phastCons100way_vertebrate__missing']].all(axis=1).to_numpy()
    strata = {'gene_seen_in_training': np.isin(clinical_genes, list(train_genes)),
              'structure_available': clinical.HAS_STRUCTURE.to_numpy(int),
              'conservation_available': conservation}
    pd.DataFrame(subgroup_metrics(clinical_y, all_clinical, strata)).to_csv(
        destination / 'clinical_control_subgroups.csv', index=False)
    macro = []
    for model, score in all_clinical.items():
        per_gene = gene_metrics(clinical_y, score, clinical_genes)
        valid = per_gene.auroc.notna()
        macro.append({'model': model, 'total_genes': len(per_gene),
                      'eligible_genes': int(valid.sum()),
                      'excluded_single_class_genes': int((~valid).sum()),
                      'macro_auroc': float(per_gene.loc[valid, 'auroc'].mean()),
                      'macro_auprc': float(per_gene.loc[valid, 'auprc'].mean())})
    pd.DataFrame(macro).to_csv(destination / 'clinical_control_gene_macro.csv', index=False)
    dump_json(destination / 'control_metrics.json', results)
    dump_json(destination / 'internal_paired_intervals.json', internal_intervals)
    dump_json(destination / 'clinical_paired_intervals.json', clinical_intervals)
    # Every source, including each trusted checkpoint, must remain unchanged.
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f'Source changed during analysis: {name}')
    dump_json(destination / 'analysis_manifest.json', {
        'role': 'exploratory_post_original_results_and_post_cohort_inspection',
        'untouched_external_evaluation': False, 'training_performed': False,
        'clinical_rows': len(clinical_y), 'clinical_genes': len(set(clinical_genes)),
        'exact_training_variant_overlap': 0, 'control_folds': 40,
        'clinical_aggregation': 'mean_of_five_fold_probabilities_strict_majority_of_frozen_fold_thresholds',
        'internal_interval_cluster': 'frozen_connected_gene_homology_group',
        'clinical_interval_cluster': 'gene_not_homology_disjoint',
        'intervals': 'paired_percentile_95_conditional_on_fixed_predictions_no_multiplicity_correction',
        'bootstrap_replicates': args.bootstrap,
        'source_sha256': hashes, 'analysis_code_sha256': sha256(Path(__file__)),
        'output_sha256': {p.name: sha256(p) for p in destination.iterdir()
                          if p.is_file() and p.name != 'analysis_manifest.json'}})
    print('Complete: frozen controls scored on the identical clinical cohort; no fitting.', flush=True)


if __name__ == '__main__':
    main()
