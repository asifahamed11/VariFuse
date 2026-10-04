"""Score the prepared structure-equipped DMS sensitivity cohort without fitting."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import common as C
from research_evidence import dms_assay_metrics, dump_json, extension_dir, proteingym_aggregate, sha256
from research_models import ALL_MODELS, predict_bundle

ORIGINAL_MODELS = ['raw_esm_zero_shot', 'esm_conservation_logistic', 'lightgbm',
                   'concatenation', 'gated_fusion', 'reliability_residual']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/15_research_extensions')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'data/external/proteingym_metadata.csv')
    args = parser.parse_args()
    root = extension_dir(args.source_root, args.output_root)
    folder = root / 'functional_validation'
    frame_path, embedding_path = folder / 'functional_with_structure.parquet', folder / 'functional_esm_embeddings.npy'
    inputs = [frame_path, embedding_path, args.metadata]
    before = {str(p.resolve()): sha256(p) for p in inputs}
    frame = pd.read_parquet(frame_path)
    embeddings = np.load(embedding_path, mmap_mode='r')
    if not len(frame) or embeddings.shape != (len(frame), C.ESM_DIM):
        raise ValueError('Prepared functional cohort is empty or misaligned')

    stage12 = importlib.import_module('12_external_validation')
    stage12.MODEL_DIR = args.source_root / '11_train_and_evaluate/models'
    internal = pd.read_parquet(args.source_root / '10_esm_features/internal_with_esm.parquet')
    pair = stage12.DatasetPair('functional_structure_sensitivity', frame, embeddings, 1.0)
    feature_names, tabular = stage12._validate_feature_schema(internal, {pair.name: pair})
    fold_probabilities = {name: [] for name in ORIGINAL_MODELS}
    fold_thresholds = {name: [] for name in ORIGINAL_MODELS}
    for fold in range(1, 6):
        predictions, thresholds = stage12._predict_fold(
            fold, pair, feature_names, tabular, ORIGINAL_MODELS)
        for name in ORIGINAL_MODELS:
            fold_probabilities[name].append(predictions[name])
            fold_thresholds[name].append(thresholds[name])
    original, _decisions, thresholds = stage12._aggregate_predictions(fold_probabilities, fold_thresholds)

    control_status = {}
    controls = {}
    for name in ALL_MODELS:
        paths = [root / 'controls' / name / f'fold_{fold}/model.pkl' for fold in range(1, 6)]
        if not all(path.exists() for path in paths):
            control_status[name] = {'status': 'not_scored',
                                    'missing_folds': [i + 1 for i, path in enumerate(paths) if not path.exists()]}
            continue
        values = []
        for path in paths:
            with path.open('rb') as stream:
                values.append(predict_bundle(pickle.load(stream), frame, embeddings))
        controls[name] = np.stack(values).mean(axis=0)
        control_status[name] = {'status': 'scored', 'folds': 5}
    models = {**original, **controls}
    assay_metrics = dms_assay_metrics(frame, models)
    assay_metrics.to_csv(folder / 'functional_model_metrics.csv', index=False)
    metadata = pd.read_csv(args.metadata)
    aggregation = {
        'spearman': proteingym_aggregate(assay_metrics, metadata, 'spearman'),
        'auroc': proteingym_aggregate(assay_metrics, metadata, 'auroc'),
    }
    prediction_table = frame[['row_id', 'ASSAY_ID', 'DMS_SCORE', 'LABEL_PATHOGENIC']].copy()
    for name, score in models.items():
        prediction_table[name] = score
    prediction_table.to_parquet(folder / 'functional_predictions.parquet', index=False)
    after = {str(p.resolve()): sha256(p) for p in inputs}
    if before != after:
        raise RuntimeError('Prepared cohort changed while it was being scored')
    dump_json(folder / 'functional_validation_results.json', {
        'scientific_role': 'exploratory_structure_transfer_sensitivity_analysis',
        'untouched': False, 'rows': len(frame), 'assays': int(frame.ASSAY_ID.nunique()),
        'genes': int(frame.genename.nunique()),
        'adequate_for_independent_validation_claim': False,
        'reason': 'only 19 availability-selected variants from one previously evaluated assay',
        'original_models_scored': ORIGINAL_MODELS, 'research_controls': control_status,
        'thresholds': thresholds, 'decisions_saved_in_memory_only': True,
        'aggregation': aggregation, 'source_hashes': before,
        'code_sha256': sha256(Path(__file__)),
        'outputs': {name: sha256(folder / name) for name in
                    ['functional_model_metrics.csv', 'functional_predictions.parquet']},
    })
    print(f'Scored {len(frame)} rows; result remains exploratory and underpowered.', flush=True)


if __name__ == '__main__':
    main()
