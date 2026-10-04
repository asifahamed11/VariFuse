from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import common as C


STAGE14_PATH = Path(__file__).resolve().parents[1] / "src" / "14_tune_cross_attention.py"
SPEC = importlib.util.spec_from_file_location("stage14_fair_nested", STAGE14_PATH)
assert SPEC is not None and SPEC.loader is not None
stage14 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage14)

STAGE11_PATH = Path(__file__).resolve().parents[1] / "src" / "11_train_and_evaluate.py"
STAGE11_SPEC = importlib.util.spec_from_file_location(
    "stage11_persisted_folds", STAGE11_PATH
)
assert STAGE11_SPEC is not None and STAGE11_SPEC.loader is not None
stage11 = importlib.util.module_from_spec(STAGE11_SPEC)
STAGE11_SPEC.loader.exec_module(stage11)


def _synthetic_frame(n_rows: int = 360) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), n_rows // 2)
    groups = np.asarray([f"GENE_{index:04d}" for index in range(n_rows)], dtype=object)
    frame = pd.DataFrame(
        {
            C.ROW_ID_COL: [f"row_{index:04d}" for index in range(n_rows)],
            C.GENE_COL: groups,
            C.LABEL_COL: labels,
        }
    )
    return frame, labels, groups


def _group_set(record: dict[str, Any], groups: np.ndarray) -> set[Any]:
    return set(groups[np.asarray(record["indices"], dtype=int)])


def test_partition_provenance_distinguishes_genes_from_homology_groups() -> None:
    frame = pd.DataFrame(
        {
            C.ROW_ID_COL: ["row_a", "row_b", "row_c"],
            C.GENE_COL: ["GENE_A", "GENE_B", "GENE_C"],
        }
    )
    labels = np.asarray([0, 1, 0], dtype=np.int8)
    split_groups = np.asarray(["cluster_1", "cluster_1", "cluster_2"], dtype=object)
    record = stage14._partition_record(
        np.asarray([0, 1]), frame, labels, split_groups
    )
    assert record["genes"] == 2
    assert record["split_groups"] == 1
    assert record["genes_sha256"] != record["split_groups_sha256"]


def test_stage14_authenticates_stage10_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise RuntimeError("upstream authentication sentinel")

    monkeypatch.setattr(stage14, "validate_upstream_manifest", reject)
    with pytest.raises(RuntimeError, match="authentication sentinel"):
        stage14._load_data()


@pytest.mark.parametrize(
    ("module", "status_attribute", "message"),
    [
        (stage11, "STATUS_CSV", "Publication Stage 11 requires Stage 08b"),
        (stage14, "INPUT_STATUS", "Publication Stage 14 requires Stage 08b"),
    ],
)
def test_publication_model_loaders_require_homology_split_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    status_attribute: str,
    message: str,
) -> None:
    frame = pd.DataFrame(
        {
            C.ROW_ID_COL: ["row_0", "row_1"],
            C.GENE_COL: ["GENE_0", "GENE_1"],
            C.LABEL_COL: [0, 1],
        }
    )
    input_table = tmp_path / f"{module.__name__}_internal.parquet"
    status_table = tmp_path / f"{module.__name__}_status.parquet"
    embedding_file = tmp_path / f"{module.__name__}_embeddings.npy"
    frame.to_parquet(input_table, index=False)
    frame[[C.ROW_ID_COL]].to_parquet(status_table, index=False)
    np.save(embedding_file, np.zeros((len(frame), C.ESM_DIM), dtype=np.float32))

    monkeypatch.setattr(module, "INPUT_CSV", input_table)
    monkeypatch.setattr(module, "INPUT_NPY", embedding_file)
    monkeypatch.setattr(module, status_attribute, status_table)
    monkeypatch.setattr(module, "REQUIRE_HOMOLOGY_GROUPS", True)
    monkeypatch.setattr(
        module, "validate_upstream_manifest", lambda *args, **kwargs: {}
    )

    with pytest.raises(RuntimeError, match=message):
        module._load_data()


def test_paired_group_randomization_is_null_based_and_deterministic() -> None:
    labels = np.tile(np.asarray([0, 1, 0, 1], dtype=int), 12)
    groups = np.repeat(np.asarray([f"H{index}" for index in range(12)]), 4)
    first_probability = np.where(labels == 1, 0.25, 0.75)
    second_probability = np.where(labels == 1, 0.90, 0.10)
    first_prediction = (first_probability >= 0.5).astype(int)
    second_prediction = (second_probability >= 0.5).astype(int)
    first = C.clustered_model_comparison(
        labels,
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
        groups,
        iterations=199,
        seed=71,
    )
    second = C.clustered_model_comparison(
        labels,
        first_probability,
        second_probability,
        first_prediction,
        second_prediction,
        groups,
        iterations=199,
        seed=71,
    )
    assert first == second
    for metric in ("mcc", "auroc", "auprc"):
        assert first[metric]["inference_method"] == (
            "paired_split_group_randomization"
        )
        assert first[metric]["ci_method"].startswith("descriptive_")
        assert first[metric]["two_sided_probability"] < 0.05

    null = C.paired_group_randomization_test(
        labels,
        second_probability,
        second_probability,
        second_prediction,
        second_prediction,
        groups,
        iterations=49,
        seed=72,
    )
    assert all(details["two_sided_probability"] == 1.0 for details in null.values())


def test_reference_baselines_share_outer_folds_and_emit_complete_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, labels, groups = _synthetic_frame(240)
    random_state = np.random.RandomState(17)
    signal = labels.astype(float) + random_state.normal(0.0, 0.3, len(labels))
    frame["esm_variant_score"] = -signal
    frame["GERP++_RS"] = signal
    frame["phyloP100way_vertebrate"] = signal * 0.5
    frame["phastCons100way_vertebrate"] = signal * 0.25
    frame["SASA"] = random_state.normal(size=len(labels))
    frame["aa_ref_is_A"] = (np.arange(len(labels)) % 3 == 0).astype(float)
    frame["aa_alt_is_V"] = (np.arange(len(labels)) % 4 == 0).astype(float)
    frame["aa_hydrophobicity_delta"] = random_state.normal(size=len(labels))
    frame["HAS_STRUCTURE"] = (np.arange(len(labels)) % 2 == 0).astype(float)
    embeddings = random_state.normal(size=(len(labels), 8)).astype(np.float32)
    input_path = tmp_path / "input.parquet"
    embedding_path = tmp_path / "embeddings.npy"
    input_path.write_bytes(b"input")
    embedding_path.write_bytes(b"embeddings")
    monkeypatch.setattr(stage14, "INPUT_CSV", input_path)
    monkeypatch.setattr(stage14, "INPUT_NPY", embedding_path)
    outer = C.make_group_splits(labels, groups, 3, seed=17)
    plan = stage14._build_split_plan(
        frame,
        labels,
        groups,
        outer,
        inner_folds=2,
        split_seed=17,
        training_seed=23,
        search_ensemble=1,
        final_ensemble=1,
    )
    monkeypatch.setattr(
        stage14.C,
        "LGBM_PARAMS",
        {
            **stage14.C.LGBM_PARAMS,
            "n_estimators": 20,
            "min_child_samples": 5,
            "n_jobs": 1,
        },
    )
    monkeypatch.setattr(
        stage14.C,
        "group_bootstrap_intervals",
        lambda *args, **kwargs: {"auprc": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        stage14,
        "LGBM_NESTED_GRID",
        (
            {"num_leaves": 15, "min_child_samples": 5},
            {"num_leaves": 31, "min_child_samples": 10},
        ),
    )

    predictions, results = stage14._reference_baseline_oof(
        frame,
        embeddings,
        labels,
        groups,
        plan,
        seed=23,
    )

    expected = {
        "raw_esm_zero_shot",
        "esm_score_logistic",
        "conservation_logistic",
        "esm_conservation_logistic",
        "mutation_logistic",
        "availability_logistic",
        "esm_embedding_mutation",
        "lightgbm",
    }
    assert set(predictions) == expected
    assert set(results) == expected
    for name in expected:
        assert np.isfinite(predictions[name]["probabilities"]).all()
        assert np.isfinite(predictions[name]["thresholds"]).all()
        assert set(np.unique(predictions[name]["decisions"])).issubset({0, 1})
    assert results["lightgbm"]["model_family"].startswith(
        "lightgbm_compact_grid_nested_inner_tuned"
    )
    assert len(results["lightgbm"]["fold_metadata"]) == 3
    for metadata in results["lightgbm"]["fold_metadata"]:
        assert metadata["nested_inner_selection"][
            "selection_uses_outer_validation"
        ] is False
        assert metadata["selected_hyperparameters"] in stage14.LGBM_NESTED_GRID


def test_publication_protocol_fingerprint_freezes_final_budget() -> None:
    base = dict(
        trials=40,
        search_ensemble=1,
        search_epochs=25,
        search_patience=5,
        final_ensemble=3,
        final_epochs=60,
        final_patience=10,
        allow_nondeterministic=False,
    )
    first = stage14._publication_protocol_fingerprint(
        stage14.argparse.Namespace(**base)
    )
    second = stage14._publication_protocol_fingerprint(
        stage14.argparse.Namespace(**{**base, "final_epochs": 61})
    )
    assert first != second
    assert first["deterministic"] is True
    assert first["source_sha256"]["stage14"]
    assert first["lightgbm_nested_grid"]
    assert "torch_cuda" in first["accelerator"]
    assert "cudnn" in first["accelerator"]


def test_split_plan_is_deterministic_persisted_and_group_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, labels, groups = _synthetic_frame()
    input_path = tmp_path / "input.parquet"
    embedding_path = tmp_path / "embeddings.npy"
    input_path.write_bytes(b"fixed-input")
    embedding_path.write_bytes(b"fixed-embeddings")
    monkeypatch.setattr(stage14, "INPUT_CSV", input_path)
    monkeypatch.setattr(stage14, "INPUT_NPY", embedding_path)
    monkeypatch.setattr(stage14, "SPLIT_PLAN_FILE", tmp_path / "splits.json")
    outer = C.make_group_splits(labels, groups, 3, seed=101)

    first = stage14._build_split_plan(
        frame,
        labels,
        groups,
        outer,
        inner_folds=2,
        split_seed=101,
        training_seed=202,
        search_ensemble=2,
        final_ensemble=3,
    )
    second = stage14._build_split_plan(
        frame,
        labels,
        groups,
        outer,
        inner_folds=2,
        split_seed=101,
        training_seed=202,
        search_ensemble=2,
        final_ensemble=3,
    )
    assert first == second
    unhashed = dict(first)
    assert unhashed.pop("canonical_sha256") == stage14._canonical_sha256(unhashed)
    file_hash = stage14._persist_and_validate_split_plan(first)
    assert file_hash == stage14.file_sha256(stage14.SPLIT_PLAN_FILE)
    assert stage14._persist_and_validate_split_plan(second) == file_hash

    for outer_fold in first["folds"]:
        outer_train_groups = _group_set(outer_fold["outer_train"], groups)
        outer_validation_groups = _group_set(
            outer_fold["outer_validation"], groups
        )
        assert outer_train_groups.isdisjoint(outer_validation_groups)
        seen_validation: set[int] = set()
        for inner_fold in outer_fold["inner_folds"]:
            records = [
                inner_fold[name]
                for name in (
                    "fit",
                    "early_stopping",
                    "temperature",
                    "threshold",
                    "validation",
                )
            ]
            group_sets = [_group_set(record, groups) for record in records]
            for first_index, first_groups in enumerate(group_sets):
                for second_groups in group_sets[first_index + 1 :]:
                    assert first_groups.isdisjoint(second_groups)
            validation = set(inner_fold["validation"]["indices"])
            assert seen_validation.isdisjoint(validation)
            seen_validation.update(validation)
            assert inner_fold["training_seeds"] == [
                202
                + int(outer_fold["outer_fold"]) * 1_000_000
                + int(inner_fold["inner_fold"]) * 10_000
                + member * 100
                for member in (1, 2)
            ]
        assert seen_validation == set(outer_fold["outer_train"]["indices"])

    conflicting = dict(first)
    conflicting["training_seed"] = 999
    with pytest.raises(RuntimeError, match="different protocol"):
        stage14._persist_and_validate_split_plan(conflicting)


def test_stage11_reuses_exact_stage14_final_partitions_and_training_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, labels, groups = _synthetic_frame()
    input_path = tmp_path / "internal.parquet"
    embedding_path = tmp_path / "embeddings.npy"
    split_path = tmp_path / "nested_inner_splits.json"
    tuning_path = tmp_path / "best_cross_attention_params.json"
    architecture_path = tmp_path / "architecture_selection.json"
    input_path.write_bytes(b"fixed-stage10-frame")
    embedding_path.write_bytes(b"fixed-stage10-embeddings")
    monkeypatch.setattr(stage14, "INPUT_CSV", input_path)
    monkeypatch.setattr(stage14, "INPUT_NPY", embedding_path)
    outer = C.make_group_splits(labels, groups, 5, seed=101)
    plan = stage14._build_split_plan(
        frame,
        labels,
        groups,
        outer,
        inner_folds=2,
        split_seed=101,
        training_seed=202,
        search_ensemble=1,
        final_ensemble=2,
    )
    split_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    tuning_path.write_text(
        json.dumps(
            {
                "reproducibility": {
                    "split_plan_canonical_sha256": plan["canonical_sha256"],
                    "split_plan_file_sha256": stage11.file_sha256(split_path),
                }
            }
        ),
        encoding="utf-8",
    )
    architecture_path.write_text(
        json.dumps(
            {
                "protocol_version": stage14.PROTOCOL_VERSION,
                "reproducibility": {
                    "split_plan_canonical_sha256": plan["canonical_sha256"],
                    "split_plan_file_sha256": stage11.file_sha256(split_path),
                },
                "reference_baselines": {
                    "lightgbm": {
                        "fold_metadata": [
                            {
                                "outer_fold": fold,
                                "selected_hyperparameters": {
                                    "num_leaves": 15,
                                    "min_child_samples": 30,
                                },
                            }
                            for fold in range(1, 6)
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage11, "INPUT_CSV", input_path)
    monkeypatch.setattr(stage11, "INPUT_NPY", embedding_path)
    monkeypatch.setattr(stage11, "TUNING_SPLIT_PLAN", split_path)
    monkeypatch.setattr(stage11, "TUNING_BEST_JSON", tuning_path)
    monkeypatch.setattr(stage11, "ARCHITECTURE_SELECTION", architecture_path)
    monkeypatch.setattr(
        stage11.C,
        "split_fit_stop_temperature_threshold",
        lambda *args, **kwargs: pytest.fail("Stage11 regenerated final partitions"),
    )

    records = stage11._stage14_training_records(frame, labels, groups)

    assert len(records) == 5
    for expected, observed in zip(plan["folds"], records):
        assert observed["outer_fold"] == expected["outer_fold"]
        for output_name, plan_name in (
            ("fit", "fit"),
            ("early_stopping", "early_stopping"),
            ("temperature", "temperature"),
            ("threshold", "threshold"),
        ):
            assert observed[output_name].tolist() == expected["final_partitions"][
                plan_name
            ]["indices"]
        assert observed["training_seeds"] == expected["final_training_seeds"]
        assert observed["lightgbm_parameters"] == {
            "num_leaves": 15,
            "min_child_samples": 30,
        }


class _DummyTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return False


def test_inner_evaluation_uses_only_persisted_indices_and_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    values = np.zeros((len(labels), 2), dtype=np.float32)
    embeddings = np.zeros((len(labels), 3), dtype=np.float32)
    plan = [
        {
            "inner_fold": 1,
            "fit": {"indices": [4]},
            "early_stopping": {"indices": [5]},
            "temperature": {"indices": [6]},
            "threshold": {"indices": [7]},
            "validation": {"indices": [0, 1]},
            "training_seeds": [1101, 1102],
        },
        {
            "inner_fold": 2,
            "fit": {"indices": [0]},
            "early_stopping": {"indices": [1]},
            "temperature": {"indices": [2]},
            "threshold": {"indices": [3]},
            "validation": {"indices": [4, 5]},
            "training_seeds": [1201, 1202],
        },
    ]
    calls: list[tuple[Any, ...]] = []

    def fake_fit_and_predict(
        values: np.ndarray,
        embeddings: np.ndarray,
        labels: np.ndarray,
        fit: np.ndarray,
        stop: np.ndarray,
        temperature: np.ndarray,
        threshold_set: np.ndarray,
        validation: np.ndarray,
        architecture: str,
        training_seeds: list[int],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        del values, embeddings, stop, temperature, threshold_set
        calls.append(
            (
                tuple(fit.tolist()),
                tuple(validation.tolist()),
                architecture,
                tuple(training_seeds),
            )
        )
        probabilities = np.where(labels[validation] == 1, 0.8, 0.2)
        return probabilities, (probabilities >= 0.5).astype(np.int8), 0.5

    monkeypatch.setattr(stage14, "_fit_and_predict", fake_fit_and_predict)
    monkeypatch.setattr(
        stage14.C,
        "make_group_splits",
        lambda *args, **kwargs: pytest.fail("inner splits were regenerated"),
    )
    first_trial = _DummyTrial(number=1)
    stage14._evaluate_inner_trial(
        first_trial,
        plan,
        values,
        embeddings,
        labels,
        "concatenation",
        "composite",
    )
    first_calls = list(calls)
    calls.clear()
    second_trial = _DummyTrial(number=9999)
    stage14._evaluate_inner_trial(
        second_trial,
        plan,
        values,
        embeddings,
        labels,
        "concatenation",
        "composite",
    )
    assert calls == first_calls
    assert first_trial.reports == second_trial.reports
    assert "trial.number" not in inspect.getsource(stage14._evaluate_inner_trial)


class _SuggestionTrial:
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        del name
        return choices[0]

    def suggest_float(self, name: str, lower: float, upper: float, **kwargs: Any) -> float:
        del name, upper, kwargs
        return lower

    def suggest_int(self, name: str, lower: int, upper: int) -> int:
        del name, upper
        return lower


def test_search_spaces_and_production_configs_are_architecture_scoped() -> None:
    trial = _SuggestionTrial()
    concatenation = stage14._suggest_parameters(trial, "concatenation")
    gated_fusion = stage14._suggest_parameters(trial, "gated_fusion")
    cross_attention = stage14._suggest_parameters(trial, "cross_attention")
    assert "N_HEADS" not in concatenation
    assert "N_CROSS_BLOCKS" not in concatenation
    assert set(gated_fusion) == set(concatenation)
    assert stage14._search_space_for("gated_fusion")["D_MODEL"] == [
        96,
        128,
        160,
        192,
    ]
    assert max(stage14._search_space_for("gated_fusion")["D_MODEL"]) <= 192
    assert "N_HEADS" in cross_attention
    assert "N_CROSS_BLOCKS" in cross_attention

    common = {
        "D_MODEL": 128,
        "DROPOUT": 0.1,
        "DEEP_LR": 0.001,
        "DEEP_WD": 0.0001,
        "WARMUP_EPOCHS": 4,
        "EMA_DECAY": 0.995,
        "MIXUP_ALPHA": 0.1,
        "LABEL_SMOOTH": 0.01,
        "BATCH_SIZE": 256,
    }
    distant = {
        **common,
        "D_MODEL": 320,
        "DROPOUT": 0.35,
        "DEEP_LR": 0.0001,
        "BATCH_SIZE": 512,
    }
    records = [
        {"outer_fold": 1, "best_parameters": common},
        {"outer_fold": 2, "best_parameters": dict(common)},
        {"outer_fold": 3, "best_parameters": distant},
    ]
    arguments = type(
        "Arguments",
        (),
        {"final_ensemble": 3, "final_epochs": 60, "final_patience": 10},
    )()
    production, selection = stage14._select_production_parameters(
        records, "concatenation", arguments
    )
    assert {name: production[name] for name in common} == common
    assert production["N_ENSEMBLE"] == 3
    assert selection["uses_outer_validation_metrics"] is False
    assert selection["source_outer_fold"] in {1, 2}


def test_run_search_emits_architecture_scoped_unbiased_oof_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, labels, groups = _synthetic_frame()
    frame["feature_one"] = np.linspace(-1.0, 1.0, len(frame))
    frame["feature_two"] = labels.astype(float)
    embeddings = np.zeros((len(frame), 4), dtype=np.float32)
    paths = {
        "INPUT_CSV": tmp_path / "input.parquet",
        "INPUT_NPY": tmp_path / "embeddings.npy",
        "INPUT_STATUS": tmp_path / "status.parquet",
        "TRIALS_CSV": tmp_path / "trials.csv",
        "FOLD_RESULTS_CSV": tmp_path / "fold_results.csv",
        "FOLD_ASSIGNMENTS": tmp_path / "assignments.csv",
        "OOF_FILE": tmp_path / "oof.npz",
        "MANIFEST_FILE": tmp_path / "manifest.json",
        "SPLIT_PLAN_FILE": tmp_path / "split_plan.json",
        "ARCHITECTURE_RESULTS_JSON": tmp_path / "architectures.json",
        "CONCATENATION_BEST_JSON": tmp_path / "concat.json",
        "GATED_FUSION_BEST_JSON": tmp_path / "gated.json",
        "RELIABILITY_RESIDUAL_BEST_JSON": tmp_path / "reliability.json",
        "EVIDENTIAL_RESIDUAL_BEST_JSON": tmp_path / "evidential.json",
        "TUNING_BEST_JSON": tmp_path / "legacy_cross.json",
        "ARCHITECTURE_CHECKPOINT_DIR": tmp_path / "architecture_checkpoints",
        "PRIMARY_CHECKPOINT_DIR": tmp_path / "primary_outer_folds",
    }
    for name, path in paths.items():
        monkeypatch.setattr(stage14, name, path)
        if name.startswith("INPUT_"):
            path.write_bytes(name.encode("utf-8"))
    monkeypatch.setattr(
        stage14,
        "_load_data",
        lambda: (
            frame,
            embeddings,
            labels,
            groups,
            ["feature_one", "feature_two"],
        ),
    )
    monkeypatch.setattr(stage14, "ensure_directories", lambda *args: None)
    manifest_call: dict[str, Any] = {}

    def capture_manifest(*args: Any, **kwargs: Any) -> None:
        manifest_call["args"] = args
        manifest_call["kwargs"] = kwargs

    monkeypatch.setattr(stage14, "write_run_manifest", capture_manifest)
    monkeypatch.setattr(
        stage14.C,
        "group_bootstrap_intervals",
        lambda *args, **kwargs: {"mcc": [1.0, 1.0]},
    )
    monkeypatch.setattr(
        stage14.C,
        "clustered_model_comparison",
        lambda *args, **kwargs: {"mcc": {"mean_difference": 0.0}},
    )

    def fake_fit_and_predict(
        values: np.ndarray,
        embeddings: np.ndarray,
        labels: np.ndarray,
        fit: np.ndarray,
        stop: np.ndarray,
        temperature: np.ndarray,
        threshold_set: np.ndarray,
        validation: np.ndarray,
        architecture: str,
        training_seeds: list[int],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        del (
            values,
            embeddings,
            fit,
            stop,
            temperature,
            threshold_set,
            architecture,
            training_seeds,
        )
        probabilities = np.where(labels[validation] == 1, 0.8, 0.2)
        return probabilities, (probabilities >= 0.5).astype(np.int8), 0.5

    monkeypatch.setattr(stage14, "_fit_and_predict", fake_fit_and_predict)
    reference_arrays = {
        name: {
            "probabilities": np.where(labels == 1, 0.7, 0.3),
            "decisions": labels.astype(np.int8),
            "thresholds": np.full(len(labels), 0.5),
        }
        for name in (
            "raw_esm_zero_shot",
            "esm_score_logistic",
            "conservation_logistic",
            "esm_conservation_logistic",
            "mutation_logistic",
            "availability_logistic",
            "esm_embedding_mutation",
            "lightgbm",
        )
    }
    monkeypatch.setattr(
        stage14,
        "_reference_baseline_oof",
        lambda *args, **kwargs: (
            reference_arrays,
            {name: {"auprc": 1.0} for name in reference_arrays},
        ),
    )
    arguments = stage14.argparse.Namespace(
        architectures=list(stage14.SUPPORTED_ARCHITECTURES),
        trials=1,
        objective="composite",
        outer_folds=3,
        inner_folds=2,
        search_ensemble=1,
        search_epochs=1,
        search_patience=1,
        final_ensemble=1,
        final_epochs=1,
        final_patience=1,
        seed=11,
        split_seed=101,
        training_seed=202,
        sampler_seed=303,
        storage=f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}",
        allow_nondeterministic=False,
        reproduce=False,
    )
    stage14.run_search(arguments)

    combined = json.loads(paths["ARCHITECTURE_RESULTS_JSON"].read_text())
    history = pd.read_csv(paths["TRIALS_CSV"])
    assert len(history) == 9
    assert set(zip(history["architecture"], history["outer_fold"])) == {
        (architecture, fold)
        for architecture in stage14.SUPPORTED_ARCHITECTURES
        for fold in (1, 2, 3)
    }
    assert set(combined["architectures"]) == {
        "concatenation",
        "gated_fusion",
        "cross_attention",
    }
    assert set(combined["production_params_by_architecture"]) == {
        "concatenation",
        "gated_fusion",
        "cross_attention",
    }
    assert set(combined["reference_baselines"]) == {
        "raw_esm_zero_shot",
        "esm_score_logistic",
        "conservation_logistic",
        "esm_conservation_logistic",
        "mutation_logistic",
        "availability_logistic",
        "esm_embedding_mutation",
        "lightgbm",
    }
    assert combined["primary_reference_baseline"] == "esm_conservation_logistic"
    assert "N_HEADS" not in combined["production_params_by_architecture"][
        "concatenation"
    ]
    assert "N_HEADS" in combined["production_params_by_architecture"][
        "cross_attention"
    ]
    assert "N_HEADS" not in combined["production_params_by_architecture"][
        "gated_fusion"
    ]
    for details in combined["architectures"].values():
        assert len(details["fold_results"]) == 3
        assert details["production_selection"][
            "uses_outer_validation_metrics"
        ] is False
    legacy = json.loads(paths["TUNING_BEST_JSON"].read_text())
    declared_outputs = set(manifest_call["kwargs"]["outputs"])
    assert {
        paths["TRIALS_CSV"],
        paths["FOLD_RESULTS_CSV"],
        paths["FOLD_ASSIGNMENTS"],
        paths["OOF_FILE"],
        paths["SPLIT_PLAN_FILE"],
        paths["ARCHITECTURE_RESULTS_JSON"],
        paths["CONCATENATION_BEST_JSON"],
        paths["GATED_FUSION_BEST_JSON"],
        paths["TUNING_BEST_JSON"],
        paths["ARCHITECTURE_CHECKPOINT_DIR"],
    } == declared_outputs
    assert legacy["production_params"] == combined[
        "production_params_by_architecture"
    ]["cross_attention"]
    with np.load(paths["OOF_FILE"], allow_pickle=True) as saved:
        assert "concatenation__probabilities" in saved
        assert "gated_fusion__probabilities" in saved
        assert "cross_attention__probabilities" in saved
        assert "reference__raw_esm_zero_shot__probabilities" in saved
        assert "reference__esm_conservation_logistic__probabilities" in saved
        assert "reference__lightgbm__probabilities" in saved
        assert np.array_equal(
            saved["probabilities"], saved["cross_attention__probabilities"]
        )
