from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import schema


stage13 = importlib.import_module("13_generate_figures")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refresh_output_record(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_id = f"14_tuning/{output_path.name}"
    record = stage13.artifact_record(output_path)
    record["artifact_id"] = artifact_id
    manifest["outputs"][artifact_id] = record
    _write_json(manifest_path, manifest)


def _stage14_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    stage10 = tmp_path / "outputs" / "10_esm_features"
    stage14 = tmp_path / "outputs" / "14_tuning"
    stage10.mkdir(parents=True)
    stage14.mkdir(parents=True)

    table_path = stage10 / "internal_with_esm.parquet"
    embedding_path = stage10 / "internal_esm_embeddings.npy"
    status_path = stage10 / "internal_esm_extraction.parquet"
    result_path = stage14 / "architecture_selection.json"
    prediction_path = stage14 / "nested_tuning_oof.npz"
    split_path = stage14 / "nested_inner_splits.json"
    manifest_path = stage14 / "run_manifest.json"

    row_ids = np.asarray([f"r{index}" for index in range(10)])
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 5)
    groups = np.repeat(np.asarray([f"H{index}" for index in range(5)]), 2)
    genes = np.repeat(np.asarray([f"G{index}" for index in range(5)]), 2)
    folds = np.repeat(np.arange(1, 6, dtype=np.int16), 2)
    frame = pd.DataFrame(
        {
            schema.ROW_ID_COL: row_ids,
            schema.LABEL_COL: labels,
            schema.GENE_COL: genes,
            "split_group": groups,
        }
    )
    frame.to_parquet(table_path, index=False)
    pd.DataFrame({schema.ROW_ID_COL: row_ids}).to_parquet(status_path, index=False)
    np.save(embedding_path, np.zeros((len(frame), 4), dtype=np.float32))

    table_hash = stage13.file_sha256(table_path)
    embedding_hash = stage13.file_sha256(embedding_path)
    status_hash = stage13.file_sha256(status_path)
    assert table_hash and embedding_hash and status_hash

    split_payload: dict[str, Any] = {
        "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
        "n_rows": len(frame),
        "row_order_sha256": stage13._ordered_text_sha256(row_ids),
        "split_group_column": "split_group",
        "input_sha256": table_hash,
        "embedding_sha256": embedding_hash,
        "outer_folds": 5,
        "inner_folds": 3,
        "folds": [
            {
                "outer_fold": fold,
                "outer_validation": {
                    "indices": np.flatnonzero(folds == fold).tolist(),
                    "n": int((folds == fold).sum()),
                    "row_ids_sha256": stage13._ordered_text_sha256(
                        row_ids[folds == fold]
                    ),
                },
            }
            for fold in range(1, 6)
        ],
    }
    split_payload["canonical_sha256"] = stage13._canonical_sha256(split_payload)
    _write_json(split_path, split_payload)
    split_file_hash = stage13.file_sha256(split_path)
    assert split_file_hash

    probabilities = np.where(labels == 1, 0.85, 0.15).astype(np.float64)
    decisions = labels.copy()
    thresholds = np.full(len(frame), 0.5, dtype=np.float64)
    metric = stage13.C.evaluate(
        labels, probabilities, thresholds, predictions=decisions
    )
    metric["ci95"] = {"mcc": [1.0, 1.0]}
    search = {
        "trials_target_per_outer_fold": 40,
        "outer_folds": 5,
        "inner_folds": 3,
        "search_epochs": 25,
        "final_ensemble": 3,
        "final_epochs": 60,
    }
    architectures = {
        name: {
            "schema_version": 2,
            "model_tag": stage13.MODEL_TAG,
            "architecture": name,
            "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
            "external_validation_touched": False,
            "search": search,
            "nested_outer_metrics": json.loads(json.dumps(metric)),
        }
        for name in sorted(stage13.REQUIRED_STAGE14_ARCHITECTURES)
    }
    references = {
        name: json.loads(json.dumps(metric))
        for name in sorted(stage13.REQUIRED_STAGE14_REFERENCES)
    }
    result_payload = {
        "schema_version": 2,
        "model_tag": stage13.MODEL_TAG,
        "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
        "selection_protocol": (
            "fixed_nested_split_group_disjoint_cross_validation"
        ),
        "architectures_evaluated": list(architectures),
        "architectures": architectures,
        "production_params_by_architecture": {
            name: {} for name in architectures
        },
        "reference_baselines": references,
        "primary_reference_baseline": "esm_conservation_logistic",
        "reproducibility": {
            "deterministic": True,
            "input_sha256": table_hash,
            "embedding_sha256": embedding_hash,
            "split_plan_canonical_sha256": split_payload["canonical_sha256"],
            "split_plan_file_sha256": split_file_hash,
        },
        "external_validation_touched": False,
    }
    _write_json(result_path, result_payload)

    npz_values: dict[str, np.ndarray] = {
        "y": labels,
        "groups": groups,
        "row_ids": row_ids,
        "fold_ids": folds,
    }
    for prefix in [*architectures, *(f"reference__{name}" for name in references)]:
        npz_values[f"{prefix}__probabilities"] = probabilities
        npz_values[f"{prefix}__decisions"] = decisions
        npz_values[f"{prefix}__thresholds"] = thresholds
    np.savez_compressed(prediction_path, **npz_values)

    source_dir = Path(stage13.__file__).resolve().parent
    source_hashes = {
        name: stage13.file_sha256(source_dir / name)
        for name in stage13.STAGE14_PROVENANCE_SOURCES
    }
    manifest_payload = {
        "artifact_manifest_version": 2,
        "stage": "14_tune_cross_attention",
        "label_task": "clinical",
        "model_tag": stage13.MODEL_TAG,
        "source_files": source_hashes,
        "inputs": {
            str(table_path): {"exists": True, "sha256": table_hash},
            str(embedding_path): {"exists": True, "sha256": embedding_hash},
            str(status_path): {"exists": True, "sha256": status_hash},
        },
        "outputs": {
            artifact_id: {
                **stage13.artifact_record(path),
                "artifact_id": artifact_id,
            }
            for path in (result_path, prediction_path, split_path)
            for artifact_id in [f"14_tuning/{path.name}"]
        },
        "extra": {
            "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
            "external_validation_touched": False,
            "architectures": sorted(architectures),
            "reference_baselines": sorted(references),
            "split_plan_canonical_sha256": split_payload["canonical_sha256"],
            "split_plan_file_sha256": split_file_hash,
        },
    }
    _write_json(manifest_path, manifest_payload)

    monkeypatch.setattr(stage13, "INTERNAL_CSV", table_path)
    monkeypatch.setattr(stage13, "INTERNAL_EMBEDDINGS", embedding_path)
    monkeypatch.setattr(stage13, "INTERNAL_STATUS", status_path)
    monkeypatch.setattr(stage13, "INTERNAL_RESULTS", result_path)
    monkeypatch.setattr(stage13, "INTERNAL_PREDICTIONS", prediction_path)
    monkeypatch.setattr(stage13, "INTERNAL_SPLIT_PLAN", split_path)
    monkeypatch.setattr(stage13, "INTERNAL_RUN_MANIFEST", manifest_path)
    return {
        "stage14_dir": stage14,
        "frame": frame,
        "row_ids": row_ids,
        "labels": labels,
        "groups": groups,
        "result_path": result_path,
        "prediction_path": prediction_path,
        "split_path": split_path,
        "manifest_path": manifest_path,
        "result": result_payload,
        "npz": npz_values,
    }


def _add_confirmatory_fixture(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    stage14_dir = fixture["stage14_dir"]
    plan_path = stage14_dir / "confirmatory_seed_plan.json"
    results_path = stage14_dir / "confirmatory_repeated_cv_results.json"
    predictions_path = stage14_dir / "confirmatory_repeated_cv_predictions.npz"
    folds_path = stage14_dir / "confirmatory_repeated_cv_folds.csv"
    checkpoint_dir = stage14_dir / "confirmatory_repeats"
    checkpoint_dir.mkdir()
    for name, path in (
        ("CONFIRMATORY_SEED_PLAN", plan_path),
        ("CONFIRMATORY_RESULTS", results_path),
        ("CONFIRMATORY_PREDICTIONS", predictions_path),
        ("CONFIRMATORY_FOLD_ASSIGNMENTS", folds_path),
        ("CONFIRMATORY_CHECKPOINT_DIR", checkpoint_dir),
    ):
        monkeypatch.setattr(stage13, name, path)

    frame = fixture["frame"]
    row_ids = fixture["row_ids"]
    labels = fixture["labels"]
    groups = fixture["groups"]
    split_payload = json.loads(fixture["split_path"].read_text(encoding="utf-8"))
    split_payload["split_seed"] = 42
    split_payload["canonical_sha256"] = stage13._canonical_sha256(
        {key: value for key, value in split_payload.items() if key != "canonical_sha256"}
    )
    _write_json(fixture["split_path"], split_payload)
    production = {
        "N_ENSEMBLE": 3,
        "DEEP_MAX_EPOCHS": 60,
        "DEEP_PATIENCE": 10,
    }
    fixed_references = {
        "raw_esm_zero_shot": {
            "parameters": {},
            "selection": "prespecified_direction_preserving_calibration",
        },
        "esm_conservation_logistic": {
            "parameters": {"regularization_C": 1.0},
            "selection": {"rule": "fixture"},
        },
        "lightgbm": {
            "parameters": {"num_leaves": 15, "min_child_samples": 30},
            "selection": {"rule": "fixture"},
        },
    }
    feature_names = ["esm_variant_score", "LOCAL_CONTACT_COUNT_8A"]
    source_dir = Path(stage13.__file__).resolve().parent
    execution_context = {
        "outer_folds": 5,
        "inner_folds": 3,
        "deterministic": True,
        "feature_names": feature_names,
        "feature_names_sha256": stage13._canonical_sha256(feature_names),
        "n_rows": len(frame),
        "row_order_sha256": stage13._ordered_text_sha256(row_ids),
        "input_sha256": stage13.file_sha256(stage13.INTERNAL_CSV),
        "embedding_sha256": stage13.file_sha256(stage13.INTERNAL_EMBEDDINGS),
        "primary_split_plan_canonical_sha256": split_payload["canonical_sha256"],
        "primary_split_plan_file_sha256": stage13.file_sha256(
            fixture["split_path"]
        ),
        "source_sha256": {
            "stage14": stage13.file_sha256(
                source_dir / "14_tune_cross_attention.py"
            ),
            "common": stage13.file_sha256(source_dir / "common.py"),
            "gpu_runtime": stage13.file_sha256(source_dir / "gpu_runtime.py"),
            "schema": stage13.file_sha256(source_dir / "schema.py"),
            "config": stage13.file_sha256(source_dir / "config.py"),
        },
    }
    repeat_specs = [(1, 101, 100_045), (2, 202, 200_048)]
    repeats = [
        {
            "repeat": repeat,
            "split_seed": split_seed,
            "training_seed": training_seed,
            "sampler_seed": None,
            "rehpo": False,
            "output_namespace": f"repeat_{repeat:03d}_seed_{split_seed}",
        }
        for repeat, split_seed, training_seed in repeat_specs
    ]
    plan = {
        "schema_version": 2,
        "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
        "status": "scheduled_for_execution",
        "scientific_role": stage13.CONFIRMATORY_SCIENTIFIC_ROLE,
        "execution_policy": (
            "reuse_primary_nested_cv_selected_fixed_configuration_without_rehpo"
        ),
        "architecture": "reliability_residual",
        "strongest_prespecified_references": list(
            stage13.CONFIRMATORY_MODEL_NAMES[1:]
        ),
        "primary_split_seed": 42,
        "production_parameters": production,
        "production_parameters_sha256": stage13._canonical_sha256(production),
        "fixed_reference_parameters": fixed_references,
        "fixed_reference_parameters_sha256": stage13._canonical_sha256(
            fixed_references
        ),
        "execution_context": execution_context,
        "repeats": repeats,
        "inference_requirement": (
            "report every prespecified repeat and aggregate paired effects; "
            "never select seeds or configurations by confirmatory performance"
        ),
        "uncertainty_scope": (
            "joint_split_and_training_instability_for_one_primary_selected_"
            "fixed_configuration_not_additional_hyperparameter_uncertainty"
        ),
        "authoritative_primary_artifacts_unchanged": True,
    }
    plan["canonical_sha256"] = stage13._canonical_sha256(plan)
    _write_json(plan_path, plan)

    stacked: dict[str, dict[str, list[np.ndarray]]] = {
        model: {role: [] for role in ("probabilities", "decisions", "thresholds")}
        for model in stage13.CONFIRMATORY_MODEL_NAMES
    }
    checkpoints: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    for repeat, split_seed, training_seed in repeat_specs:
        splits = stage13.C.make_group_splits(labels, groups, 5, split_seed)
        fold_ids = np.full(len(frame), -1, dtype=np.int16)
        fold_records: list[dict[str, Any]] = []
        for fold, (_, validation) in enumerate(splits, 1):
            validation = np.asarray(validation, dtype=np.int64)
            fold_ids[validation] = fold
            fold_records.append(
                {
                    "outer_fold": fold,
                    "outer_validation": {
                        "indices": validation.tolist(),
                        "n": len(validation),
                        "row_ids_sha256": stage13._ordered_text_sha256(
                            row_ids[validation]
                        ),
                    },
                }
            )
        repeat_plan = {
            "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
            "n_rows": len(frame),
            "row_order_sha256": stage13._ordered_text_sha256(row_ids),
            "split_group_column": "split_group",
            "input_sha256": execution_context["input_sha256"],
            "embedding_sha256": execution_context["embedding_sha256"],
            "outer_folds": 5,
            "inner_folds": 3,
            "split_seed": split_seed,
            "training_seed": training_seed,
            "search_ensemble": 1,
            "final_ensemble": 3,
            "final_epochs": 60,
            "final_patience": 10,
            "deterministic": True,
            "protocol_fingerprint": {
                "scientific_role": stage13.CONFIRMATORY_SCIENTIFIC_ROLE,
                "confirmatory_plan_sha256": plan["canonical_sha256"],
                "rehpo": False,
            },
            "folds": fold_records,
        }
        repeat_plan["canonical_sha256"] = stage13._canonical_sha256(repeat_plan)
        predictions: dict[str, dict[str, list[Any]]] = {}
        model_results: dict[str, Any] = {}
        for model_index, model in enumerate(stage13.CONFIRMATORY_MODEL_NAMES):
            pathogenic = 0.80 - model_index * 0.03 + repeat * 0.005
            benign = 0.20 + model_index * 0.03 - repeat * 0.005
            probabilities = np.where(labels == 1, pathogenic, benign).astype(float)
            thresholds = np.full(len(frame), 0.5, dtype=float)
            decisions = (probabilities >= thresholds).astype(np.int8)
            for role, values in (
                ("probabilities", probabilities),
                ("decisions", decisions),
                ("thresholds", thresholds),
            ):
                stacked[model][role].append(values)
            metrics = stage13.C.evaluate(
                labels, probabilities, thresholds, predictions=decisions
            )
            model_results[model] = {
                "metrics": metrics,
                "ci95": {},
                "fold_results": [{} for _ in range(5)],
                "evaluation_role": (
                    "confirmatory_fixed_primary_selected_configuration"
                    if model == "reliability_residual"
                    else "fixed_configuration_confirmatory_reference"
                ),
                "rehpo": False,
            }
            predictions[model] = {
                "probabilities": probabilities.tolist(),
                "decisions": decisions.tolist(),
                "thresholds": thresholds.tolist(),
            }
        comparisons = {
            f"reliability_residual_minus_{reference}": {
                metric: {
                    "mean_difference_second_minus_first": 0.0,
                    "randomization_iterations": 1000,
                    "inference_method": "paired_split_group_randomization",
                    "exchangeability_unit": "split_group",
                }
                for metric in ("mcc", "auroc", "auprc")
            }
            for reference in stage13.CONFIRMATORY_MODEL_NAMES[1:]
        }
        checkpoint = {
            "schema_version": 1,
            "status": "complete",
            "confirmatory_plan_sha256": plan["canonical_sha256"],
            "identity": {
                "repeat": repeat,
                "split_seed": split_seed,
                "training_seed": training_seed,
            },
            "row_order_sha256": stage13._ordered_text_sha256(row_ids),
            "repeat_split_plan": repeat_plan,
            "fold_ids": fold_ids.tolist(),
            "predictions": predictions,
            "model_results": model_results,
            "comparisons": comparisons,
            "external_validation_touched": False,
        }
        checkpoint_path = checkpoint_dir / f"repeat_{repeat:03d}_seed_{split_seed}.json"
        _write_json(checkpoint_path, checkpoint)
        checkpoints.append(checkpoint)
        fold_frames.append(
            pd.DataFrame(
                {
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "training_seed": training_seed,
                    "row_index": np.arange(len(frame)),
                    schema.ROW_ID_COL: row_ids,
                    schema.GENE_COL: frame[schema.GENE_COL].astype(str),
                    "split_group": groups,
                    schema.LABEL_COL: labels,
                    "outer_fold": fold_ids,
                }
            )
        )
    pd.concat(fold_frames, ignore_index=True).to_csv(folds_path, index=False)

    npz_values: dict[str, np.ndarray] = {
        "y": labels,
        "groups": groups,
        "row_ids": row_ids,
        "repeat_ids": np.asarray([1, 2], dtype=np.int16),
        "split_seeds": np.asarray([101, 202], dtype=np.int64),
        "training_seeds": np.asarray([100_045, 200_048], dtype=np.int64),
        "fold_ids": np.stack(
            [np.asarray(checkpoint["fold_ids"]) for checkpoint in checkpoints]
        ),
    }
    for model in stage13.CONFIRMATORY_MODEL_NAMES:
        prefix = stage13._confirmatory_prefix(model)
        for role in ("probabilities", "decisions", "thresholds"):
            npz_values[f"{prefix}__{role}"] = np.stack(stacked[model][role])
    np.savez_compressed(predictions_path, **npz_values)

    per_seed = [
        {
            "identity": checkpoint["identity"],
            "repeat_split_plan_canonical_sha256": checkpoint[
                "repeat_split_plan"
            ]["canonical_sha256"],
            "models": checkpoint["model_results"],
            "paired_comparisons": checkpoint["comparisons"],
        }
        for checkpoint in checkpoints
    ]
    recomputed_metrics = {
        model: [
            checkpoint["model_results"][model]["metrics"]
            for checkpoint in checkpoints
        ]
        for model in stage13.CONFIRMATORY_MODEL_NAMES
    }
    across_seed = {
        model: stage13._confirmatory_scalar_metric_summary(metrics)
        for model, metrics in recomputed_metrics.items()
    }
    paired_summary = {
        f"reliability_residual_minus_{reference}": (
            stage13._confirmatory_scalar_metric_summary(
                [
                    {
                        metric: float(proposed[metric]) - float(comparator[metric])
                        for metric in ("mcc", "auroc", "auprc", "brier")
                    }
                    for proposed, comparator in zip(
                        recomputed_metrics["reliability_residual"],
                        recomputed_metrics[reference],
                    )
                ]
            )
        )
        for reference in stage13.CONFIRMATORY_MODEL_NAMES[1:]
    }
    pooled: dict[str, Any] = {}
    instability: dict[str, Any] = {}
    for model in stage13.CONFIRMATORY_MODEL_NAMES:
        probabilities = np.stack(stacked[model]["probabilities"])
        decisions = np.stack(stacked[model]["decisions"])
        thresholds = np.stack(stacked[model]["thresholds"])
        pooled[model] = {
            "stacked_repeated_oof_descriptive": stage13.C.evaluate(
                np.tile(labels, 2),
                probabilities.reshape(-1),
                thresholds.reshape(-1),
                predictions=decisions.reshape(-1),
            ),
            "mean_probability_repeated_oof_descriptive": stage13.C.evaluate(
                labels,
                probabilities.mean(axis=0),
                thresholds.mean(axis=0),
                predictions=(decisions.mean(axis=0) >= 0.5).astype(np.int8),
            ),
        }
        row_sd = probabilities.std(axis=0, ddof=0)
        instability[model] = {
            "per_row_probability_standard_deviation": {
                "mean": round(float(row_sd.mean()), 6),
                "median": round(float(np.median(row_sd)), 6),
                "p90": round(float(np.quantile(row_sd, 0.9)), 6),
                "maximum": round(float(row_sd.max()), 6),
            },
            "metric_variation": across_seed[model],
        }
    results = {
        "schema_version": 1,
        "status": "complete",
        "protocol_version": stage13.REQUIRED_STAGE14_PROTOCOL,
        "scientific_role": stage13.CONFIRMATORY_SCIENTIFIC_ROLE,
        "authoritative_primary_nested_oof_replaced": False,
        "hyperparameter_selection_uses_confirmatory_results": False,
        "external_validation_touched": False,
        "confirmatory_plan_sha256": plan["canonical_sha256"],
        "confirmatory_plan_file_sha256": stage13.file_sha256(plan_path),
        "model_order": list(stage13.CONFIRMATORY_MODEL_NAMES),
        "repeat_count": 2,
        "per_seed": per_seed,
        "across_seed_metric_summary": across_seed,
        "pooled_descriptive_metrics": pooled,
        "paired_metric_difference_summary": paired_summary,
        "split_and_training_instability": instability,
        "inference_scope": plan["uncertainty_scope"],
        "pooled_metrics_warning": (
            "Repeated OOF rows are correlated copies of the same variants; pooled "
            "metrics are descriptive and must not be reported as an independent "
            "larger sample. Per-seed paired effects are the confirmatory unit."
        ),
        "artifacts": {
            "predictions_npz": str(predictions_path),
            "predictions_npz_sha256": stage13.file_sha256(predictions_path),
            "fold_assignments_csv": str(folds_path),
            "fold_assignments_csv_sha256": stage13.file_sha256(folds_path),
            "checkpoint_directory": str(checkpoint_dir),
        },
    }
    _write_json(results_path, results)

    architecture_result = fixture["result"]
    architecture_result["architectures"]["reliability_residual"][
        "production_params"
    ] = production
    architecture_result["production_params_by_architecture"][
        "reliability_residual"
    ] = production
    architecture_result["feature_names_by_architecture"] = {
        name: feature_names for name in architecture_result["architectures"]
    }
    architecture_result["reproducibility"].update(
        {
            "split_seed": 42,
            "training_seed": 42,
            "split_plan_canonical_sha256": split_payload["canonical_sha256"],
            "split_plan_file_sha256": stage13.file_sha256(fixture["split_path"]),
        }
    )
    architecture_result["confirmatory_repeated_cv"] = {
        "requested": True,
        "scientific_role": "separate_fixed_configuration_confirmation",
        "seed_plan": str(plan_path),
        "results": str(results_path),
        "authoritative_primary_nested_oof_replaced": False,
    }
    _write_json(fixture["result_path"], architecture_result)

    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["extra"].update(
        {
            "split_plan_canonical_sha256": split_payload["canonical_sha256"],
            "split_plan_file_sha256": stage13.file_sha256(fixture["split_path"]),
            "confirmatory_seed_plan": str(plan_path),
            "confirmatory_repeated_cv_results": str(results_path),
            "confirmatory_repeated_cv_predictions": str(predictions_path),
            "confirmatory_scientific_role": (
                "separate_fixed_configuration_split_and_training_instability"
            ),
            "confirmatory_results_used_for_model_selection": False,
        }
    )
    for path in (
        fixture["result_path"],
        fixture["split_path"],
        plan_path,
        results_path,
        predictions_path,
        folds_path,
        checkpoint_dir,
    ):
        artifact_id = f"14_tuning/{path.name}"
        manifest["outputs"][artifact_id] = {
            **stage13.artifact_record(path),
            "artifact_id": artifact_id,
        }
    _write_json(fixture["manifest_path"], manifest)
    return {
        "plan_path": plan_path,
        "results_path": results_path,
        "predictions_path": predictions_path,
        "folds_path": folds_path,
        "checkpoint_dir": checkpoint_dir,
        "results": results,
        "npz": npz_values,
    }


def test_stage13_accepts_coherent_publication_stage14_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage14_fixture(tmp_path, monkeypatch)

    report = stage13._validate_stage14_artifacts()

    assert report["status"] == "validated"
    assert report["protocol_version"] == stage13.REQUIRED_STAGE14_PROTOCOL
    assert report["n_rows"] == 10
    assert report["outer_folds"] == 5
    assert set(report["architectures"]) == stage13.REQUIRED_STAGE14_ARCHITECTURES


def test_stage13_authenticates_requested_confirmatory_repeated_cv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    _add_confirmatory_fixture(fixture, monkeypatch)

    report = stage13._validate_stage14_artifacts()
    confirmation = report["confirmatory_repeated_cv"]

    assert confirmation["status"] == "validated"
    assert confirmation["repeat_count"] == 2
    assert confirmation["split_seeds"] == [101, 202]
    assert confirmation["results_used_for_model_selection"] is False
    assert set(confirmation["checkpoint_sha256"]) == {
        "repeat_001_seed_101.json",
        "repeat_002_seed_202.json",
    }

    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    monkeypatch.setattr(stage13, "STAGE13_OUT", figure_dir)
    table_path = stage13._write_confirmatory_stability_table(report)
    assert table_path is not None
    table = pd.read_csv(table_path)
    assert set(table["record_type"]) == {
        "model_metric_across_prespecified_seeds",
        "paired_metric_difference_across_seeds",
    }
    assert table["used_for_model_selection"].eq(False).all()
    assert set(table["metric"]) == {"mcc", "auroc", "auprc", "brier"}


def test_stage13_rejects_mixed_confirmatory_npz_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    confirmation = _add_confirmatory_fixture(fixture, monkeypatch)
    arrays = confirmation["npz"]
    arrays["reliability_residual__probabilities"] = arrays[
        "reliability_residual__probabilities"
    ].copy()
    arrays["reliability_residual__probabilities"][0, 0] = 0.25
    np.savez_compressed(confirmation["predictions_path"], **arrays)
    confirmation["results"]["artifacts"]["predictions_npz_sha256"] = (
        stage13.file_sha256(confirmation["predictions_path"])
    )
    _write_json(confirmation["results_path"], confirmation["results"])
    _refresh_output_record(
        fixture["manifest_path"], confirmation["predictions_path"]
    )
    _refresh_output_record(fixture["manifest_path"], confirmation["results_path"])

    with pytest.raises(RuntimeError, match="checkpoint and NPZ differ"):
        stage13._validate_stage14_artifacts()


def test_stage13_rejects_stale_confirmatory_per_seed_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    confirmation = _add_confirmatory_fixture(fixture, monkeypatch)
    results = confirmation["results"]
    results["per_seed"][0]["models"]["reliability_residual"]["metrics"][
        "auroc"
    ] = 0.1234
    _write_json(confirmation["results_path"], results)
    _refresh_output_record(fixture["manifest_path"], confirmation["results_path"])

    with pytest.raises(RuntimeError, match="aggregate and checkpoint contents differ"):
        stage13._validate_stage14_artifacts()


def test_stage13_rejects_confirmatory_fold_table_seed_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    confirmation = _add_confirmatory_fixture(fixture, monkeypatch)
    folds = pd.read_csv(confirmation["folds_path"])
    folds.loc[0, "split_seed"] = 999
    folds.to_csv(confirmation["folds_path"], index=False)
    confirmation["results"]["artifacts"]["fold_assignments_csv_sha256"] = (
        stage13.file_sha256(confirmation["folds_path"])
    )
    _write_json(confirmation["results_path"], confirmation["results"])
    _refresh_output_record(fixture["manifest_path"], confirmation["folds_path"])
    _refresh_output_record(fixture["manifest_path"], confirmation["results_path"])

    with pytest.raises(RuntimeError, match="long fold table differs"):
        stage13._validate_stage14_artifacts()


@pytest.mark.parametrize("corruption", ["row_ids", "fold_ids"])
def test_stage13_rejects_oof_identity_or_fold_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    values = fixture["npz"]
    if corruption == "row_ids":
        values["row_ids"] = values["row_ids"][::-1]
    else:
        values["fold_ids"] = np.roll(values["fold_ids"], 2)
    np.savez_compressed(fixture["prediction_path"], **values)
    _refresh_output_record(fixture["manifest_path"], fixture["prediction_path"])

    with pytest.raises(RuntimeError, match="row order|fold IDs"):
        stage13._validate_stage14_artifacts()


def test_stage13_rejects_reported_metric_that_differs_from_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    payload = fixture["result"]
    payload["architectures"]["gated_fusion"]["nested_outer_metrics"][
        "auroc"
    ] = 0.1234
    _write_json(fixture["result_path"], payload)
    _refresh_output_record(fixture["manifest_path"], fixture["result_path"])

    with pytest.raises(RuntimeError, match="gated_fusion.auroc differs"):
        stage13._validate_stage14_artifacts()


def test_stage13_rejects_nonpublication_protocol_or_input_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage14_fixture(tmp_path, monkeypatch)
    payload = fixture["result"]
    payload["reproducibility"]["input_sha256"] = "0" * 64
    _write_json(fixture["result_path"], payload)
    _refresh_output_record(fixture["manifest_path"], fixture["result_path"])

    with pytest.raises(RuntimeError, match="input table hash differs"):
        stage13._validate_stage14_artifacts()

    payload["protocol_version"] = "legacy"
    _write_json(fixture["result_path"], payload)
    _refresh_output_record(fixture["manifest_path"], fixture["result_path"])
    with pytest.raises(RuntimeError, match="not the publication protocol"):
        stage13._validate_stage14_artifacts()
