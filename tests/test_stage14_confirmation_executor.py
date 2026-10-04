from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import common as C


STAGE14_PATH = Path(__file__).resolve().parents[1] / "src" / "14_tune_cross_attention.py"
SPEC = importlib.util.spec_from_file_location("stage14_confirmation", STAGE14_PATH)
assert SPEC is not None and SPEC.loader is not None
stage14 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage14)


def _primary_reference_results() -> dict[str, Any]:
    return {
        "esm_conservation_logistic": {
            "fold_metadata": [
                {"regularization_C": 0.1},
                {"regularization_C": 1.0},
                {"regularization_C": 0.1},
            ]
        },
        "lightgbm": {
            "fold_metadata": [
                {"selected_hyperparameters": dict(stage14.LGBM_NESTED_GRID[0])},
                {"selected_hyperparameters": dict(stage14.LGBM_NESTED_GRID[1])},
                {"selected_hyperparameters": dict(stage14.LGBM_NESTED_GRID[0])},
            ]
        },
    }


def _patch_confirmation_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage14, "CONFIRMATORY_SEED_PLAN", tmp_path / "confirmatory_plan.json"
    )
    monkeypatch.setattr(
        stage14, "CONFIRMATORY_RESULTS_JSON", tmp_path / "confirmatory_results.json"
    )
    monkeypatch.setattr(
        stage14, "CONFIRMATORY_PREDICTIONS", tmp_path / "confirmatory_predictions.npz"
    )
    monkeypatch.setattr(
        stage14, "CONFIRMATORY_FOLD_ASSIGNMENTS", tmp_path / "confirmatory_folds.csv"
    )
    monkeypatch.setattr(
        stage14, "CONFIRMATORY_CHECKPOINT_DIR", tmp_path / "checkpoints"
    )


def test_fixed_reference_consensus_never_uses_outer_performance() -> None:
    fixed = stage14._fixed_confirmatory_reference_parameters(
        _primary_reference_results()
    )
    assert fixed["esm_conservation_logistic"]["parameters"] == {
        "regularization_C": 0.1
    }
    assert fixed["lightgbm"]["parameters"] == dict(stage14.LGBM_NESTED_GRID[0])
    assert not fixed["esm_conservation_logistic"]["selection"][
        "uses_primary_outer_validation_performance"
    ]
    assert not fixed["lightgbm"]["selection"][
        "uses_primary_outer_validation_performance"
    ]


def test_confirmation_executor_persists_aligned_outputs_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_confirmation_paths(tmp_path, monkeypatch)
    input_path = tmp_path / "input.parquet"
    embedding_path = tmp_path / "embedding.npy"
    input_path.write_bytes(b"confirmation-input")
    embedding_path.write_bytes(b"confirmation-embedding")
    monkeypatch.setattr(stage14, "INPUT_CSV", input_path)
    monkeypatch.setattr(stage14, "INPUT_NPY", embedding_path)

    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    groups = np.asarray([f"H{index}" for index in range(len(labels))], dtype=object)
    frame = pd.DataFrame(
        {
            C.ROW_ID_COL: [f"row_{index}" for index in range(len(labels))],
            C.GENE_COL: [f"GENE_{index}" for index in range(len(labels))],
            C.LABEL_COL: labels,
        }
    )
    embeddings = np.zeros((len(frame), 2), dtype=np.float32)
    context = {
        "outer_folds": 2,
        "inner_folds": 2,
        "deterministic": True,
        "feature_names": [],
        "n_rows": len(frame),
        "row_order_sha256": stage14._ordered_text_sha256(
            frame[C.ROW_ID_COL].astype(str)
        ),
        "input_sha256": stage14.file_sha256(input_path),
        "embedding_sha256": stage14.file_sha256(embedding_path),
        "primary_split_plan_canonical_sha256": "primary-canonical",
        "primary_split_plan_file_sha256": "primary-file",
    }
    production = {
        "D_MODEL": 64,
        "N_ENSEMBLE": 1,
        "DEEP_MAX_EPOCHS": 1,
        "DEEP_PATIENCE": 1,
    }
    plan = stage14._confirmatory_seed_plan(
        [101, 202],
        {"split_seed": 7, "training_seed": 11, "sampler_seed": 13},
        {C.RELIABILITY_ARCHITECTURE: {"production_params": production}},
        _primary_reference_results(),
        context,
    )
    stage14._persist_confirmatory_plan(plan)

    def repeat_plan(
        frame: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        plan: dict[str, Any],
        repeat_record: dict[str, Any],
    ) -> dict[str, Any]:
        del frame, labels, groups, plan
        payload = {
            "split_seed": int(repeat_record["split_seed"]),
            "folds": [],
        }
        payload["canonical_sha256"] = stage14._canonical_sha256(payload)
        return payload

    monkeypatch.setattr(stage14, "_confirmation_repeat_split_plan", repeat_plan)
    calls: list[int] = []

    def run_repeat(
        frame: pd.DataFrame,
        embeddings: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        base_feature_names: list[str],
        plan: dict[str, Any],
        repeat_record: dict[str, Any],
        repeat_split_plan: dict[str, Any],
    ) -> dict[str, Any]:
        del embeddings, groups, base_feature_names
        calls.append(int(repeat_record["repeat"]))
        offset = 0.01 * int(repeat_record["repeat"])
        probabilities = np.where(labels == 1, 0.80 - offset, 0.20 + offset)
        decisions = (probabilities >= 0.5).astype(np.int8)
        thresholds = np.full(len(frame), 0.5)
        metrics = C.evaluate(labels, probabilities, thresholds, predictions=decisions)
        predictions = {
            name: {
                "probabilities": probabilities.tolist(),
                "decisions": decisions.tolist(),
                "thresholds": thresholds.tolist(),
            }
            for name in stage14.CONFIRMATORY_MODEL_NAMES
        }
        model_results = {
            name: {
                "metrics": metrics,
                "ci95": {},
                "fold_results": [],
                "evaluation_role": "test",
                "rehpo": False,
            }
            for name in stage14.CONFIRMATORY_MODEL_NAMES
        }
        return {
            "schema_version": 1,
            "status": "complete",
            "confirmatory_plan_sha256": plan["canonical_sha256"],
            "identity": {
                "repeat": int(repeat_record["repeat"]),
                "split_seed": int(repeat_record["split_seed"]),
                "training_seed": int(repeat_record["training_seed"]),
            },
            "row_order_sha256": stage14._ordered_text_sha256(
                frame[C.ROW_ID_COL].astype(str)
            ),
            "repeat_split_plan": repeat_split_plan,
            "fold_ids": [1, 1, 1, 1, 2, 2, 2, 2],
            "predictions": predictions,
            "model_results": model_results,
            "comparisons": {},
            "external_validation_touched": False,
        }

    monkeypatch.setattr(stage14, "_run_single_confirmation_repeat", run_repeat)
    first = stage14._execute_confirmatory_repeats(
        frame, embeddings, labels, groups, [], plan
    )
    assert calls == [1, 2]
    assert first["repeat_count"] == 2
    assert first["authoritative_primary_nested_oof_replaced"] is False
    assert "correlated copies" in first["pooled_metrics_warning"]
    with np.load(stage14.CONFIRMATORY_PREDICTIONS) as values:
        assert values["fold_ids"].shape == (2, len(frame))
        assert values[
            f"{C.RELIABILITY_ARCHITECTURE}__probabilities"
        ].shape == (2, len(frame))
        assert values[
            "reference__esm_conservation_logistic__probabilities"
        ].shape == (2, len(frame))
    folds = pd.read_csv(stage14.CONFIRMATORY_FOLD_ASSIGNMENTS)
    assert len(folds) == 2 * len(frame)
    assert folds.groupby("split_group")["outer_fold"].nunique().eq(1).all()

    def must_not_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise AssertionError("authenticated checkpoints should resume")

    monkeypatch.setattr(stage14, "_run_single_confirmation_repeat", must_not_train)
    resumed = stage14._execute_confirmatory_repeats(
        frame, embeddings, labels, groups, [], plan
    )
    assert resumed["repeat_count"] == 2


def test_confirmation_resume_fails_closed_on_prediction_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_confirmation_paths(tmp_path, monkeypatch)
    labels = np.asarray([0, 1], dtype=np.int8)
    groups = np.asarray(["H0", "H1"], dtype=object)
    frame = pd.DataFrame(
        {
            C.ROW_ID_COL: ["row_0", "row_1"],
            C.GENE_COL: ["G0", "G1"],
            C.LABEL_COL: labels,
        }
    )
    repeat = {
        "repeat": 1,
        "split_seed": 101,
        "training_seed": 202,
        "output_namespace": "repeat_001_seed_101",
    }
    split_plan = {"canonical_sha256": "split", "folds": []}
    probabilities = np.asarray([0.2, 0.8])
    decisions = np.asarray([0, 1], dtype=np.int8)
    thresholds = np.asarray([0.5, 0.5])
    metrics = C.evaluate(labels, probabilities, thresholds, predictions=decisions)
    plan = {
        "canonical_sha256": "plan",
    }
    checkpoint = {
        "schema_version": 1,
        "status": "complete",
        "confirmatory_plan_sha256": "plan",
        "identity": {"repeat": 1, "split_seed": 101, "training_seed": 202},
        "row_order_sha256": stage14._ordered_text_sha256(
            frame[C.ROW_ID_COL].astype(str)
        ),
        "repeat_split_plan": split_plan,
        "fold_ids": [1, 2],
        "predictions": {
            name: {
                "probabilities": probabilities.tolist(),
                "decisions": decisions.tolist(),
                "thresholds": thresholds.tolist(),
            }
            for name in stage14.CONFIRMATORY_MODEL_NAMES
        },
        "model_results": {
            name: {"metrics": metrics} for name in stage14.CONFIRMATORY_MODEL_NAMES
        },
    }
    checkpoint["predictions"][C.RELIABILITY_ARCHITECTURE]["probabilities"][0] = 0.9
    with pytest.raises(RuntimeError, match="metrics are stale"):
        stage14._validate_confirmation_checkpoint(
            checkpoint, plan, repeat, split_plan, frame, labels, groups
        )


def test_confirmation_seeds_must_be_additional_to_primary() -> None:
    arguments = stage14.argparse.Namespace(
        architectures=[C.RELIABILITY_ARCHITECTURE],
        trials=1,
        outer_folds=3,
        inner_folds=2,
        search_epochs=1,
        final_epochs=1,
        search_patience=1,
        final_patience=1,
        search_ensemble=1,
        final_ensemble=1,
        seed=42,
        split_seed=101,
        confirmation_split_seeds=[101],
    )
    with pytest.raises(ValueError, match="additional to the primary"):
        stage14._validate_arguments(arguments)
