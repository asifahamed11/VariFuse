"""Export compact full-precision evidence; never modify fitted research artifacts."""
# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_evidence import dump_json, sha256


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    destination = ROOT / 'publication/evidence'
    destination.mkdir(parents=True, exist_ok=True)
    originals = ROOT / 'outputs'
    extensions = originals / '15_research_extensions'
    files = {
        'original_verified_evidence.json': ROOT / 'audit_reports/readiness_20261004/verified_evidence.json',
        'control_metrics.json': extensions / 'manuscript_analysis/control_metrics.json',
        'control_internal_intervals.json': extensions / 'manuscript_analysis/internal_paired_intervals.json',
        'control_clinical_intervals.json': extensions / 'manuscript_analysis/clinical_paired_intervals.json',
        'control_analysis_manifest.json': extensions / 'manuscript_analysis/analysis_manifest.json',
        'clinical_control_gene_macro.csv': extensions / 'manuscript_analysis/clinical_control_gene_macro.csv',
        'clinical_control_subgroups.csv': extensions / 'manuscript_analysis/clinical_control_subgroups.csv',
        'clinical_original_intervals.json': extensions / 'saved_prediction_analysis/clinical_paired_intervals.json',
        'clinical_original_subgroups.csv': extensions / 'saved_prediction_analysis/clinical_subgroups.csv',
        'dms_aggregation.json': extensions / 'saved_prediction_analysis/dms_corrected_aggregation.json',
        'functional_model_metrics.csv': extensions / 'functional_validation/functional_model_metrics.csv',
        'functional_cohort_manifest.json': extensions / 'functional_validation/cohort_manifest.json',
        'control_protocol.json': extensions / 'protocol.json',
        'control_summary_provenance.json': extensions / 'summary_repair/summary_provenance.json',
        'confirmation_results.json': originals / '14_tuning/confirmatory_repeated_cv_results.json',
        'homology_summary.json': originals / '08_prepare_esm/homology_summary.json',
    }
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files.values()}
    for name, source in files.items():
        shutil.copyfile(source, destination / name)
    selection_path = originals / '14_tuning/architecture_selection.json'
    selection = read(selection_path)
    compact_architectures = {}
    for name, value in selection['architectures'].items():
        compact = {k: value[k] for k in ['search', 'best_params_by_outer_fold',
                                        'production_selection', 'reproducibility']}
        compact['fold_results'] = []
        for fold in value['fold_results']:
            record = {k: v for k, v in fold.items() if k != 'outer_validation_rows'}
            record['outer_validation_row_count'] = len(fold['outer_validation_rows'])
            compact['fold_results'].append(record)
        compact_architectures[name] = compact
    dump_json(destination / 'original_selection_recipes.json', {
        'selection_protocol': selection['selection_protocol'],
        'feature_names_by_architecture': selection['feature_names_by_architecture'],
        'production_params_by_architecture': selection['production_params_by_architecture'],
        'architectures': compact_architectures,
        'row_identity_lists_included': False,
        'source_sha256': sha256(selection_path)})
    plan_path = originals / '14_tuning/nested_inner_splits.json'
    plan = read(plan_path)
    compact_plan = {key: value for key, value in plan.items() if key != 'folds'}
    compact_plan['folds'] = []
    for fold in plan['folds']:
        compact = {'outer_fold': fold['outer_fold'], 'final_training_seeds': fold['final_training_seeds']}
        for key in ['outer_train', 'outer_validation']:
            compact[key] = {k: v for k, v in fold[key].items() if k != 'indices'}
            compact[key]['n_rows'] = len(fold[key]['indices'])
        compact['final_partitions'] = {name: {**{k: v for k, v in record.items() if k != 'indices'},
                                               'n_rows': len(record['indices'])}
                                       for name, record in fold['final_partitions'].items()}
        compact_plan['folds'].append(compact)
    compact_plan['full_partition_source_sha256'] = sha256(plan_path)
    compact_plan['indices_included'] = False
    dump_json(destination / 'split_summary.json', compact_plan)
    external_path = originals / '12_external_validation/external_validation.json'
    external = read(external_path)
    compact_external = {'primary_evaluation_policy': external['primary_evaluation_policy'], 'sets': {}}
    for name, cohort in external['sets'].items():
        compact_external['sets'][name] = {key: cohort.get(key) for key in
            ['n', 'row_n', 'positives', 'prevalence', 'genes', 'assays', 'modality_coverage',
             'validation_scope', 'models', 'primary_endpoints', 'reliability_diagnostics']}
    dump_json(destination / 'original_external_summary.json', compact_external)
    for path in [selection_path, plan_path, external_path]:
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    dump_json(destination / 'evidence_manifest.json', {
        'date': '2026-10-04', 'role': 'compact_manuscript_numerical_evidence',
        'original_evidence_audit_is_historical': True,
        'new_control_metrics_and_status_supersede_old_pending_fields': True,
        'later_controls_untouched': False, 'raw_predictions_and_weights_included': False,
        'source_sha256': source_hashes,
        'output_sha256': {p.name: sha256(p) for p in destination.iterdir()
                          if p.is_file() and p.name != 'evidence_manifest.json'}})
    print('Exported compact evidence:', destination, flush=True)


if __name__ == '__main__':
    main()
