"""Lightweight post-run analyses for research questions 1 and 2; no fitting."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_evidence import (align_rows, attach_public_scores, dms_assay_metrics,
                              dump_json, extension_dir, gene_metrics,
                              paired_group_bootstrap, proteingym_aggregate,
                              rank_metrics, sha256, subgroup_metrics)

MODELS = ['raw_esm_zero_shot', 'esm_conservation_logistic', 'lightgbm',
          'concatenation', 'gated_fusion', 'reliability_residual']


def analyze_internal(source: Path, output: Path, iterations: int) -> None:
    path = source / '14_tuning/nested_tuning_oof.npz'
    keys = {
        'raw_esm_zero_shot': 'reference__raw_esm_zero_shot__probabilities',
        'esm_conservation_logistic': 'reference__esm_conservation_logistic__probabilities',
        'lightgbm': 'reference__lightgbm__probabilities',
        'concatenation': 'concatenation__probabilities',
        'gated_fusion': 'gated_fusion__probabilities',
        'reliability_residual': 'reliability_residual__probabilities',
    }
    with np.load(path, allow_pickle=True) as archive:
        y = archive['y'].astype(int)
        row_ids = archive['row_ids'].astype(str)
        folds = archive['fold_ids'].astype(int)
        models = {name: archive[key] for name, key in keys.items()}
    raw = pd.read_parquet(source / '10_esm_features/internal_with_esm.parquet',
                          columns=['row_id', 'LABEL_PATHOGENIC', 'genename', 'HAS_STRUCTURE',
                                   'GERP++_RS__missing', 'phyloP100way_vertebrate__missing',
                                   'phastCons100way_vertebrate__missing'])
    aligned = align_rows(raw, row_ids)
    if not np.array_equal(y, aligned.LABEL_PATHOGENIC.to_numpy(int)):
        raise ValueError('Internal OOF labels do not match source rows')
    genes = aligned.genename.fillna('unknown').astype(str).to_numpy()
    conservation = ~aligned[[
        'GERP++_RS__missing', 'phyloP100way_vertebrate__missing',
        'phastCons100way_vertebrate__missing']].all(axis=1).to_numpy()
    strata = {
        'outer_fold': folds,
        'structure_available': aligned.HAS_STRUCTURE.to_numpy(int),
        'conservation_available': conservation,
        'auxiliary_availability': np.where(aligned.HAS_STRUCTURE.to_numpy(int) == 1,
                                           np.where(conservation, 'both', 'structure_only'),
                                           np.where(conservation, 'conservation_only', 'neither')),
    }
    pd.DataFrame(subgroup_metrics(y, models, strata)).to_csv(output / 'internal_subgroups.csv', index=False)
    comparisons = {}
    for first in ['raw_esm_zero_shot', 'esm_conservation_logistic', 'lightgbm',
                  'concatenation', 'gated_fusion']:
        comparisons['reliability_minus_' + first] = paired_group_bootstrap(
            y, models[first], models['reliability_residual'], genes, iterations=iterations)
    dump_json(output / 'internal_paired_intervals.json', comparisons)
    rows = []
    for model, scores in models.items():
        rows.append({'model': model, 'scope': 'pooled_oof', **rank_metrics(y, scores)})
        for fold in sorted(set(folds)):
            keep = folds == fold
            rows.append({'model': model, 'scope': f'outer_fold_{fold}',
                         **rank_metrics(y[keep], scores[keep])})
    pd.DataFrame(rows).to_csv(output / 'internal_oof_metrics.csv', index=False)
    dump_json(output / 'internal_analysis.json', {
        'role': 'exploratory_post_result_analysis_of_primary_nested_oof',
        'rows': len(y), 'genes': int(pd.Series(genes).nunique()),
        'outer_folds': [int(value) for value in sorted(set(folds))],
        'paired_intervals': 'gene_clustered_conditional_on_frozen_oof_predictions',
        'training_uncertainty_included': False, 'multiplicity_adjusted': False,
        'bootstrap_iterations': iterations})
    print('Internal nested-OOF subgroup and paired analyses complete.', flush=True)


def analyze_clinical(source: Path, output: Path, iterations: int) -> None:
    path = source / '12_external_validation/external_predictions.npz'
    prefix = 'clinvar_exact_variant_disjoint__'
    with np.load(path, allow_pickle=True) as archive:
        y, groups = archive[prefix + 'y'], archive[prefix + 'groups'].astype(str)
        models = {name: archive[prefix + name] for name in MODELS}
        component = prefix + 'reliability_components__'
        codes = archive[component + 'availability_stratum_code'].astype(int)
        gate = archive[component + 'gate']
        variants = archive[prefix + 'variant_ids'].astype(str)
        raw = pd.read_parquet(source / '10_esm_features/clinvar_with_esm.parquet',
                              columns=['variant_id', 'LABEL_PATHOGENIC', 'PLDDT_SCORE',
                                       'HAS_STRUCTURE', 'CLINVAR_REVIEW_STATUS'])
        if not raw.variant_id.astype(str).str.startswith('GRCh37:').all():
            raise ValueError('Expected authenticated GRCh37 source identities')
        raw['variant_id'] = raw.variant_id.astype(str).str.removeprefix('GRCh37:')
        raw = raw.loc[raw.variant_id.astype(str).isin(variants)].copy()
        # Final primary consequences should be unique. Fail rather than picking
        # a possibly outcome-dependent annotation row if this contract changes.
        aligned = align_rows(raw, variants, 'variant_id')
        if not np.array_equal(y, aligned.LABEL_PATHOGENIC.to_numpy()):
            raise ValueError('External prediction labels do not match their rows')
        train_genes = set(pd.read_parquet(source / '10_esm_features/internal_with_esm.parquet',
                                         columns=['genename']).genename.astype(str))
        plddt = pd.to_numeric(aligned.PLDDT_SCORE, errors='coerce').to_numpy()
        structure = aligned.HAS_STRUCTURE.to_numpy() == 1
        quality = np.where(~structure | ~np.isfinite(plddt), 'unavailable',
                           np.where(plddt < 70, 'below70', np.where(plddt < 90, '70to90', '90plus')))
        strata = {'natural_reliability': codes,
                  'structure_quality': quality,
                  'gene_seen_in_training': np.isin(groups, list(train_genes)),
                  'review_status': aligned.CLINVAR_REVIEW_STATUS.fillna('unknown').astype(str)}
        pd.DataFrame(subgroup_metrics(y, models, strata)).to_csv(output / 'clinical_subgroups.csv', index=False)
        genes = []
        for model, scores in models.items():
            table = gene_metrics(y, scores, groups)
            table['model'] = model
            genes.append(table)
        by_gene = pd.concat(genes, ignore_index=True)
        by_gene.to_csv(output / 'clinical_per_gene.csv', index=False)
        macro = []
        for model, table in by_gene.groupby('model'):
            both = table.auroc.notna()
            macro.append({'model': model, 'total_genes': len(table),
                          'eligible_two_class_genes': int(both.sum()),
                          'excluded_single_class_genes': int((~both).sum()),
                          'macro_auroc': float(table.loc[both, 'auroc'].mean()),
                          'macro_auprc': float(table.loc[both, 'auprc'].mean())})
        pd.DataFrame(macro).to_csv(output / 'clinical_gene_macro.csv', index=False)
        comparisons = {}
        for first in ['esm_conservation_logistic', 'concatenation', 'gated_fusion']:
            comparisons['reliability_minus_' + first] = paired_group_bootstrap(
                y, models[first], models['reliability_residual'], groups, iterations=iterations)
        # Pair on each public predictor's coverage, avoiding the much smaller
        # intersection across every public tool. All variant/label joins checked.
        source_table = pd.read_parquet(source / '10_esm_features/clinvar_with_esm.parquet',
                                        columns=['variant_id', 'AlphaMissense_score', 'REVEL_score', 'MetaRNN_score'])
        source_table['variant_id'] = source_table.variant_id.astype(str).str.removeprefix('GRCh37:')
        source_table = source_table.loc[source_table.variant_id.astype(str).isin(variants)]
        source_table = align_rows(source_table, variants, 'variant_id')
        coverage_rows = []
        for name in ['AlphaMissense_score', 'REVEL_score', 'MetaRNN_score']:
            public = pd.to_numeric(source_table[name], errors='coerce').to_numpy(float)
            keep = np.isfinite(public)
            if not keep.any():
                continue
            for model in ['reliability_residual', 'gated_fusion']:
                key = model + '_minus_' + name
                comparisons[key] = paired_group_bootstrap(y[keep], public[keep], models[model][keep],
                                                          groups[keep], iterations=iterations)
                coverage_rows.append({'predictor': name, 'model': model, 'n': int(keep.sum()),
                                      'coverage': float(keep.mean()),
                                      'public_auprc': rank_metrics(y[keep], public[keep])['auprc'],
                                      'candidate_auprc': rank_metrics(y[keep], models[model][keep])['auprc']})
        pd.DataFrame(coverage_rows).to_csv(output / 'clinical_pairwise_coverage.csv', index=False)
        dump_json(output / 'clinical_paired_intervals.json', comparisons)
        dump_json(output / 'clinical_analysis.json', {
            'role': 'exploratory_post_result_analysis', 'n': len(y),
            'all_models_compared_on_identical_rows_within_each_stratum': True,
            'gene_macro_conditions_on_genes_with_both_classes': True,
            'subgroup_prevalence_is_not_controlled_by_stratification': True,
            'public_predictor_training_overlap_independence': 'not_established',
            'gate_mean': float(gate.mean()), 'gate_p05_p50_p95': np.quantile(gate, [.05,.5,.95]).tolist(),
            'paired_intervals': 'conditional_on_frozen_predictions_exploratory_not_multiplicity_adjusted',
            'bootstrap_iterations': iterations})
    print('Clinical subgroups, per-gene metrics and paired comparisons complete.', flush=True)


def analyze_dms(source: Path, output: Path, metadata_path: Path, public_scores: list[str]) -> None:
    columns = ['row_id', 'ASSAY_ID', 'DMS_SCORE', 'LABEL_PATHOGENIC', 'aa_ref', 'aa_pos', 'aa_alt']
    frame = pd.read_parquet(source / '10_esm_features/dms_with_esm.parquet', columns=columns)
    prefix = 'dms_exact_variant_disjoint__'
    with np.load(source / '12_external_validation/external_predictions.npz', allow_pickle=True) as archive:
        frame = align_rows(frame, archive[prefix + 'row_ids'])
        if not np.array_equal(frame.LABEL_PATHOGENIC.to_numpy(), archive[prefix + 'y']):
            raise ValueError('DMS labels do not match saved predictions')
        models = {name: archive[prefix + name] for name in MODELS}
    metadata = pd.read_csv(metadata_path)
    metrics = dms_assay_metrics(frame, models)
    metrics.to_csv(output / 'dms_per_assay.csv', index=False)
    summary = {'role': 'exploratory_post_result_analysis', 'rows': len(frame),
               'aggregation': 'mean_assays_within_UniProt_and_coarse_selection_type_then_category_mean_then_equal_category_mean',
               'reference': 'https://github.com/OATML-Markslab/ProteinGym',
               'cohort': 'retained_single_substitutions_with_valid_existing_extraction',
               'full_leaderboard_comparable': False,
               'spearman': proteingym_aggregate(metrics, metadata, 'spearman'),
               'auroc': proteingym_aggregate(metrics, metadata, 'auroc'),
               'public_comparators': {}}
    for spec in public_scores:
        # NAME|PATH|higher_is_functional (pipe also permits Windows drive colons).
        name, filename, direction = spec.split('|', 2)
        if not name.replace('_', '').isalnum() or name in models:
            raise ValueError('Public model name must be a new alphanumeric identifier')
        values = attach_public_scores(frame, Path(filename), direction=direction)
        keep = np.isfinite(values)
        if not keep.any():
            raise ValueError(f'No public scores matched {name}')
        paired = {k: v[keep] for k, v in models.items()}
        paired[name] = values[keep]
        paired_metrics = dms_assay_metrics(frame.loc[keep].reset_index(drop=True), paired)
        paired_metrics.to_csv(output / f'dms_public_{name}_paired_assays.csv', index=False)
        summary['public_comparators'][name] = {
            'source_sha256': sha256(Path(filename)), 'source_path': str(Path(filename).resolve()),
            'direction': direction, 'matched_rows': int(keep.sum()), 'coverage': float(keep.mean()),
            'same_rows_for_every_model': True,
            'spearman': proteingym_aggregate(paired_metrics, metadata, 'spearman'),
            'auroc': proteingym_aggregate(paired_metrics, metadata, 'auroc')}
    if not public_scores:
        summary['public_score_status'] = 'awaiting_per_variant_public_scores_not_fabricated'
    dump_json(output / 'dms_corrected_aggregation.json', summary)
    print('Protein/category-balanced DMS aggregation complete.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/15_research_extensions')
    parser.add_argument('--phase', choices=['internal','clinical','dms','all'], default='all')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'data/external/proteingym_metadata.csv')
    parser.add_argument('--public-score', action='append', default=[],
                        help='NAME|PATH_TO_CSV|higher_is_functional or higher_is_pathogenic')
    args = parser.parse_args()
    output = extension_dir(args.source_root, args.output_root) / 'saved_prediction_analysis'
    output.mkdir(parents=True, exist_ok=True)
    inputs = [args.source_root / '14_tuning/nested_tuning_oof.npz',
              args.source_root / '12_external_validation/external_predictions.npz',
              args.source_root / '10_esm_features/internal_with_esm.parquet',
              args.source_root / '10_esm_features/clinvar_with_esm.parquet',
              args.source_root / '10_esm_features/dms_with_esm.parquet', args.metadata]
    before = {str(p.resolve()): sha256(p) for p in inputs}
    if args.phase in ['internal','all']:
        analyze_internal(args.source_root, output, args.bootstrap)
    if args.phase in ['clinical','all']:
        analyze_clinical(args.source_root, output, args.bootstrap)
    if args.phase in ['dms','all']:
        analyze_dms(args.source_root, output, args.metadata, args.public_score)
    after = {str(p.resolve()): sha256(p) for p in inputs}
    if before != after:
        raise RuntimeError('Source artifacts changed during analysis')
    dump_json(output / f'{args.phase}_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': before,
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                        [Path(__file__), ROOT / 'src/research_evidence.py']},
        'original_artifacts_unchanged': True, 'training_run': False,
        'outputs': {p.name: sha256(p) for p in output.iterdir()
                    if p.is_file() and not p.name.endswith('_manifest.json')}})


if __name__ == '__main__':
    main()
