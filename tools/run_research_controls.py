"""Run one resumable same-input baseline/ablation fold, never the full pipeline.

Full-data jobs are explicit (--phase train). Defaults only prepare a locked
exploratory protocol. Existing ESM arrays and original split partitions are reused.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_evidence import digest, dump_json, extension_dir, sha256
from research_models import ALL_MODELS, BASELINE_MODELS, predict_bundle, research_factory
import common as C


def validate_partitions(plan: dict, frame: pd.DataFrame) -> None:
    """Check row identities plus train/stop/calibration/threshold/test separation."""
    n = len(frame)
    if n != plan['n_rows']:
        raise ValueError('Split plan row count differs')
    text = frame.row_id.astype(str).tolist()
    if len(set(text)) != n:
        raise ValueError('Nonunique input row IDs')
    groups = frame.split_group.astype(str).to_numpy()
    covered = np.zeros(n, dtype=int)
    for fold in plan['folds']:
        outer = np.asarray(fold['outer_validation']['indices'], dtype=int)
        if not len(outer) or len(set(outer)) != len(outer) or np.any(outer < 0) or np.any(outer >= n):
            raise ValueError('Invalid outer validation indices')
        covered[outer] += 1
        for part_number, partition_set in enumerate([dict(fold['final_partitions'], validation=fold['outer_validation']),
                              *[{k:v for k,v in record.items() if isinstance(v,dict) and 'indices' in v}
                                for record in fold['inner_folds']]]):
            previous_rows, previous_groups = set(), set()
            for record in partition_set.values():
                idx = np.asarray(record['indices'], dtype=int)
                if not len(idx) or len(set(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= n):
                    raise ValueError('Empty, repeated or out-of-range partition indices')
                if previous_rows.intersection(idx) or previous_groups.intersection(groups[idx]):
                    raise ValueError('Row/group leakage across fitting/evaluation partitions')
                previous_rows.update(idx)
                previous_groups.update(groups[idx])
                # Stage 14 uses SHA256 of newline-delimited ordered identities.
                import hashlib
                actual = hashlib.sha256(''.join(text[i] + '\n' for i in idx).encode()).hexdigest()
                if actual != record['row_ids_sha256']:
                    raise ValueError('Persisted partition row identity mismatch')
            if part_number == 0 and previous_rows != set(range(n)):
                raise ValueError('Final partitions do not cover the full row universe')
            if part_number > 0 and not previous_rows.issubset(set(fold['outer_train']['indices'])):
                raise ValueError('Inner partition crosses the outer test boundary')
        if set(fold['outer_train']['indices']) != set(range(n)) - set(outer):
            raise ValueError('Outer training complement mismatch')
    if not np.all(covered == 1):
        raise ValueError('Outer evaluation must cover every row exactly once')


def read_inputs(source: Path):
    table_path = source / '10_esm_features/internal_with_esm.parquet'
    embedding_path = source / '10_esm_features/internal_esm_embeddings.npy'
    plan = json.loads((source / '14_tuning/nested_inner_splits.json').read_text())
    if sha256(table_path) != plan['input_sha256'] or sha256(embedding_path) != plan['embedding_sha256']:
        raise ValueError('Frozen split plan does not match input table/ESM embeddings')
    frame = pd.read_parquet(table_path)
    embeddings = np.load(embedding_path, mmap_mode='r')
    if embeddings.shape != (len(frame), C.ESM_DIM):
        raise ValueError('ESM row count or dimension mismatch')
    validate_partitions(plan, frame)
    original = json.loads((source / '14_tuning/architecture_selection.json').read_text())
    names = original['feature_names_by_architecture']['reliability_residual']
    if len(names) != len(set(names)) or set(names) - set(frame):
        raise ValueError('Original reliability feature schema is unavailable')
    return frame, embeddings, plan, original, names


def protocol(source: Path, output: Path, frame, plan, original, names) -> dict:
    filenames = ['src/research_models.py', 'src/research_evidence.py',
                 'tools/run_research_controls.py', 'src/common.py', 'src/config.py', 'src/schema.py',
                 'src/gpu_runtime.py', 'src/14_tune_cross_attention.py']
    value = {
        'version': 2, 'role': 'exploratory_post_original_results',
        'untouched_external_evaluation': False, 'input_sha256': plan['input_sha256'],
        'embedding_sha256': plan['embedding_sha256'],
        'split_plan_sha256': sha256(source / '14_tuning/nested_inner_splits.json'),
        'selection_source_sha256': sha256(source / '14_tuning/architecture_selection.json'),
        'code_sha256': {p: sha256(ROOT/p) for p in filenames},
        'models': list(ALL_MODELS), 'shared_bio_features': names, 'esm_dimensions': C.ESM_DIM,
        'folds': [int(f['outer_fold']) for f in plan['folds']], 'rows': len(frame),
        'neural_parameter_policy': 'reuse_control_inner_selected_parameters_for_same_outer_fold_no_external_selection',
        'ablation_retuning': False, 'ablation_scope': 'matched_training_recipe_component_effect_not_best_possible_retuned_model',
        'baseline_selection': 'logistic_C_on_stop_partition;LightGBM_six_configurations_on_persisted_inner_folds',
        'neural_epochs': plan['final_epochs'], 'neural_patience': plan['final_patience'],
        'ensemble': plan['final_ensemble'], 'esm_extraction_repeated': False,
        'seed_policy': 'same_original_final_training_seeds_for_every_neural_control',
        'preprocessing': 'same_fold_fit_feature_mask;fit_only_imputation_and_scaling;raw_gate_scales_for_neural_models',
        'calibration': 'temperature_partition_only', 'threshold': 'threshold_partition_only',
        'comparison_policy': 'all_requested_folds_required_for_pooled_oof_no_best_model_cherry_picking',
        'runtime_packages': {name: importlib.metadata.version(name) for name in
                             ['torch','numpy','pandas','scikit-learn','lightgbm','scipy']},
    }
    value['fingerprint'] = digest(value)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        completed = list((output / 'controls').rglob('result.json')) if (output / 'controls').exists() else []
        if completed:
            raise ValueError('Research code/input/protocol changed: use a new output directory, never mix resumes')
        # A preparation-only protocol contains no fitted result. It may be
        # refreshed while engineering is still in progress.
    dump_json(path, value)
    return value


def final_indices(fold: dict) -> dict:
    return {k: np.asarray(v['indices'], dtype=int)
            for k,v in dict(fold['final_partitions'], validation=fold['outer_validation']).items()}


def fit_fold(frame, embeddings, fold, original, names, model_name, *, smoke=False,
             max_epochs=60, patience=10):
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    idx = final_indices(fold)
    y = frame[C.LABEL_COL].to_numpy(np.int8)
    bio = frame[names].to_numpy(np.float32)
    if smoke:
        # This never writes full-run checkpoints; each existing disjoint
        # partition is reduced to at most 32 rows/class for engineering only.
        idx = {key: np.concatenate([rows[y[rows] == c][:32] for c in [0,1]])
               for key,rows in idx.items()}
    for rows in idx.values():
        C.validate_binary_labels(y[rows])
    fit, stop, calibration, threshold, validation = (
        idx[k] for k in ['fit','early_stopping','temperature','threshold','validation'])
    mask = C.architecture_feature_mask(bio[fit], names, 'reliability_residual')
    kept = [n for n,k in zip(names, mask) if k]
    bio = bio[:, mask]
    parameters = next(f['best_parameters'] for f in original['architectures']['reliability_residual']['fold_results']
                      if f['outer_fold'] == fold['outer_fold']).copy()
    parameters.update({'N_ENSEMBLE': 1 if smoke else len(fold['final_training_seeds']),
                       'DEEP_MAX_EPOCHS': 1 if smoke else max_epochs,
                       'DEEP_PATIENCE': 1 if smoke else patience})
    if smoke:
        parameters['MIXUP_ALPHA'] = 0.0
    selected_parameters, selection = {}, {}
    if model_name in BASELINE_MODELS:
        # Identical raw input columns, with a standard fit-only scale for the
        # linear baseline and unchanged missing values for LightGBM.
        values = np.column_stack([bio, np.asarray(embeddings, dtype=np.float32)])
        if model_name == 'same_input_logistic':
            preprocessing = C.ArrayPreprocessor.fit(values[fit])
            xf, xs = preprocessing.transform(values[fit]), preprocessing.transform(values[stop])
            if smoke:
                model = LogisticRegression(C=.1, class_weight='balanced', max_iter=300).fit(xf, y[fit])
            else:
                model = C.fit_regularized_logistic(xf, xs, y[fit], y[stop], seed=fold['final_training_seeds'][0])
            calibrator = C.fit_direction_preserving_calibrator(
                y[calibration], model.decision_function(preprocessing.transform(values[calibration])))
            def predict(rows):
                return calibrator.predict_from_logits(model.decision_function(preprocessing.transform(values[rows])))
            selected_parameters = {'C': float(model.C)}
            selection = {'selection_uses_outer_test': False, 'selection_set': 'early_stopping',
                         'grid': [.01,.1,1.,10.] if not smoke else [.1]}
        else:
            stage14 = importlib.import_module('14_tune_cross_attention')
            # Use raw shared columns before the outer fit mask for inner HPO;
            # each inner call learns its own nonconstant mask on its fit rows.
            inner_values = np.column_stack([frame[names].to_numpy(np.float32), np.asarray(embeddings)])
            selected_parameters, selection = ({'num_leaves': 15, 'min_child_samples': 10}, {}) if smoke else (
                stage14._select_nested_lightgbm_parameters(inner_values, y, fold, 1701))
            tree_parameters = dict(C.LGBM_PARAMS)
            tree_parameters.update(selected_parameters)
            tree_parameters.update({'random_state': fold['final_training_seeds'][0],
                                    'scale_pos_weight': C.fold_class_weight(y[fit])})
            if smoke:
                tree_parameters.update(n_estimators=5, n_jobs=2)
            model = LGBMClassifier(**tree_parameters)
            model.fit(values[fit], y[fit], eval_set=[(values[stop], y[stop])], eval_metric='aucpr',
                      callbacks=[early_stopping(2 if smoke else C.LGBM_EARLY_STOP, verbose=False), log_evaluation(0)])
            preprocessing = None
            calibrator = C.fit_probability_calibrator(y[calibration], model.booster_.predict(values[calibration]))
            def predict(rows):
                return calibrator.predict(model.booster_.predict(values[rows]))
        threshold_value = C.select_threshold(y[threshold], predict(threshold))
        probs = predict(validation)
        bundle = {'model': model, 'preprocessors': preprocessing, 'calibrator': calibrator,
                  'model_type': model_name}
        parameter_count = None
    else:
        prep = C.fit_preprocessors(bio[fit], embeddings[fit], C.reliability_passthrough_indices(kept))
        with C.temporary_model_config(parameters), research_factory(model_name):
            model = C.train_deep_model(
                prep.transform_bio(bio[fit]), prep.transform_esm(embeddings[fit]), y[fit],
                prep.transform_bio(bio[stop]), prep.transform_esm(embeddings[stop]), y[stop],
                prep.transform_bio(bio[calibration]), prep.transform_esm(embeddings[calibration]), y[calibration],
                max_epochs=parameters['DEEP_MAX_EPOCHS'], patience=parameters['DEEP_PATIENCE'],
                architecture='research_extension', feature_names=kept,
                seeds=fold['final_training_seeds'][:parameters['N_ENSEMBLE']])
            threshold_value = C.select_threshold(y[threshold], C.predict(model,
                prep.transform_bio(bio[threshold]), prep.transform_esm(embeddings[threshold])))
            probs = C.predict(model, prep.transform_bio(bio[validation]), prep.transform_esm(embeddings[validation]))
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = model.cpu()
        bundle = {'model': model, 'preprocessors': prep, 'model_type': model_name}
    if not np.isfinite(probs).all():
        raise ValueError('Nonfinite model output')
    bundle.update({'feature_names': kept, 'threshold': threshold_value, 'parameters': parameters,
                   'selected_baseline_parameters': selected_parameters, 'esm_dim': C.ESM_DIM})
    return bundle, {'indices': validation, 'probabilities': probs,
                    'decisions': (probs >= threshold_value).astype(np.int8),
                    'y': y[validation], 'row_ids': frame.row_id.astype(str).to_numpy()[validation]}, {
        'threshold': float(threshold_value), 'metrics': C.evaluate(y[validation], probs, threshold_value),
        'trainable_parameters': parameter_count, 'selection': selection,
        'partition_sizes': {k:len(v) for k,v in idx.items()}}


def summarize(output: Path, frame: pd.DataFrame, protocol_record: dict) -> None:
    records = {}
    for model in ALL_MODELS:
        pieces, missing = [], []
        for fold in protocol_record['folds']:
            folder = output / 'controls' / model / f'fold_{fold}'
            manifest = folder / 'result.json'
            if not manifest.exists():
                missing.append(fold)
                continue
            record = json.loads(manifest.read_text())
            if record['fingerprint'] != protocol_record['fingerprint']:
                raise ValueError('Checkpoint protocol mismatch')
            for name, checksum in record['artifacts'].items():
                if sha256(folder/name) != checksum:
                    raise ValueError('Checkpoint artifact changed')
            with np.load(folder/'predictions.npz', allow_pickle=False) as z:
                pieces.append({k:z[k] for k in z.files})
        if missing:
            records[model] = {'status': 'incomplete', 'missing_folds': missing}
            continue
        indices = np.concatenate([p['indices'] for p in pieces])
        if len(indices) != len(frame) or set(indices) != set(range(len(frame))):
            raise ValueError('Incomplete/duplicated OOF coverage')
        order = np.argsort(indices)
        values = {k:np.concatenate([p[k] for p in pieces])[order] for k in pieces[0]}
        if not np.array_equal(values['row_ids'], frame.row_id.astype(str).to_numpy()):
            raise ValueError('OOF row identity mismatch')
        if not np.array_equal(values['y'], frame[C.LABEL_COL].to_numpy()):
            raise ValueError('OOF label identity mismatch')
        from research_evidence import rank_metrics
        records[model] = {'status':'complete','metrics':rank_metrics(values['y'], values['probabilities'])}
        np.savez_compressed(output/'controls'/f'{model}_oof.npz', **values)
    dump_json(output/'controls_status.json', {'role':'exploratory_component_and_same_input_controls',
                                             'models': records, 'all_complete': all(v['status']=='complete' for v in records.values())})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT/'outputs')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/15_research_extensions')
    parser.add_argument('--phase', choices=['prepare','train','summarize','smoke'], default='prepare')
    parser.add_argument('--model', choices=ALL_MODELS)
    parser.add_argument('--fold', type=int, choices=range(1,6), default=1)
    args = parser.parse_args()
    output = extension_dir(args.source_root, args.output_root)
    frame, embeddings, plan, original, names = read_inputs(args.source_root)
    record = protocol(args.source_root, output, frame, plan, original, names)
    if args.phase == 'prepare':
        print('Protocol locked. No training performed. Run --phase train --model NAME --fold N explicitly.')
        return
    if args.phase == 'summarize':
        summarize(output, frame, record)
        return
    if not args.model:
        parser.error('--model is required for training/smoke')
    smoke = args.phase == 'smoke'
    folder = output/('smoke' if smoke else 'controls')/args.model/f'fold_{args.fold}'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'result.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        matches = previous['fingerprint'] == record['fingerprint']
        if not matches and not smoke:
            raise ValueError('Cannot resume a different protocol')
        if matches:
            for filename, checksum in previous['artifacts'].items():
                if sha256(folder/filename) != checksum:
                    raise ValueError('Resume artifact checksum mismatch')
            print('Verified completed checkpoint; skipped:', folder)
            return
    fold = next(f for f in plan['folds'] if f['outer_fold'] == args.fold)
    started = time.monotonic()
    bundle, predictions, result = fit_fold(frame, embeddings, fold, original, names, args.model, smoke=smoke,
                                          max_epochs=plan['final_epochs'], patience=plan['final_patience'])
    with (folder/'model.pkl.tmp').open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (folder/'model.pkl.tmp').replace(folder/'model.pkl')
    with (folder/'model.pkl').open('rb') as stream:
        restored = pickle.load(stream)
    checked = predict_bundle(restored, frame.iloc[predictions['indices']], embeddings[predictions['indices']])
    if not np.allclose(checked, predictions['probabilities'], rtol=1e-5, atol=1e-6):
        raise RuntimeError('Saved model does not reproduce pre-save predictions')
    with (folder/'predictions.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **predictions)
    (folder/'predictions.npz.tmp').replace(folder/'predictions.npz')
    result.update({'fingerprint':record['fingerprint'], 'model':args.model, 'fold':args.fold,
        'engineering_only':smoke, 'role':'engineering_only' if smoke else 'exploratory',
                   'saved_model_prediction_roundtrip_verified': True,
                   'elapsed_seconds':time.monotonic()-started, 'created_utc':datetime.now(timezone.utc).isoformat(),
                   'artifacts':{n:sha256(folder/n) for n in ['model.pkl','predictions.npz']}})
    dump_json(manifest,result)
    print('Completed:', manifest)


if __name__ == '__main__':
    main()
