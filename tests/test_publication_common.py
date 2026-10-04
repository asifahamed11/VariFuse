from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import common


stage11 = __import__("11_train_and_evaluate")


def test_affine_logit_calibration_can_correct_prior_shift() -> None:
    labels = np.asarray([0] * 90 + [1] * 10)
    logits = np.asarray([-0.5] * 90 + [0.5] * 10, dtype=np.float64)
    calibrator = common.fit_logit_calibrator(logits, labels)
    probabilities = calibrator.predict_from_logits(logits)
    assert calibrator.intercept < 0
    assert np.isfinite(probabilities).all()
    assert 0.0 < probabilities.min() < probabilities.max() < 1.0


def test_probability_calibration_never_reverses_prespecified_risk_order() -> None:
    labels = np.asarray([1, 1, 0, 0])
    reversed_calibration_sample = np.asarray([0.1, 0.2, 0.8, 0.9])
    calibrator = common.fit_probability_calibrator(
        labels, reversed_calibration_sample
    )
    transformed = calibrator.predict(np.asarray([0.1, 0.5, 0.9]))
    assert calibrator.coefficient > 0
    assert np.all(np.diff(transformed) > 0)


def test_evaluate_reports_calibration_and_threshold_scale_safe_risk_coverage() -> None:
    labels = np.asarray([0, 0, 0, 1, 1])
    probabilities = np.asarray([0.001, 0.01, 0.02, 0.04, 0.20])
    metrics = common.evaluate(labels, probabilities, 0.03)
    assert "brier_skill" in metrics["calibration"]
    assert set(metrics["risk_coverage"]) == {
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_100",
    }
    assert metrics["selective"] == metrics["risk_coverage"]["coverage_80"]


def test_evaluate_can_use_classifier_vote_margin_for_selective_risk() -> None:
    labels = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray([0.9, 0.1, 0.4, 0.6])
    decisions = np.asarray([0, 1, 0, 1])
    vote_margin = np.asarray([1.0, 1.0, 0.2, 0.2])
    metrics = common.evaluate(
        labels,
        probabilities,
        0.5,
        predictions=decisions,
        decision_confidence=vote_margin,
    )
    assert metrics["risk_coverage_confidence_source"] == (
        "provided_classifier_decision_confidence"
    )
    assert metrics["risk_coverage"]["coverage_50"]["risk"] == 0.0


def test_group_split_rejects_a_realized_single_class_fold(monkeypatch) -> None:
    class InvalidStratifiedGroupKFold:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def split(self, indices, labels, groups):
            del labels, groups
            yield indices[1:], indices[:1]
            yield indices[:1], indices[1:]

    labels = np.asarray([0, 0, 1, 1])
    groups = np.asarray(["A", "B", "C", "D"], dtype=object)
    monkeypatch.setattr(
        common, "StratifiedGroupKFold", InvalidStratifiedGroupKFold
    )
    with pytest.raises(ValueError, match="group split 1 test labels"):
        common.make_group_splits(labels, groups, 2, seed=1)


def test_gated_fusion_has_valid_shape_and_fewer_parameters_than_legacy_attention() -> None:
    gated = common.GatedFusionNet(12, esm_dim=32)
    legacy = common.CrossAttnFusionNet(12, esm_dim=32)
    bio = torch.randn(4, 12)
    esm = torch.randn(4, 32)
    assert gated(bio, esm).shape == (4,)
    gated_parameters = sum(parameter.numel() for parameter in gated.parameters())
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    assert gated_parameters < legacy_parameters


def test_nonconstant_feature_mask_drops_constant_and_all_missing_columns() -> None:
    values = np.asarray(
        [[1.0, np.nan, 0.0], [1.0, np.nan, 1.0], [1.0, np.nan, 2.0]],
        dtype=np.float32,
    )
    assert common.nonconstant_feature_mask(values).tolist() == [False, False, True]


def test_publication_baselines_keep_zero_shot_and_conservation_separate() -> None:
    frame = pd.DataFrame(
        {
            "esm_variant_score": [-1.0, 1.0],
            "GERP++_RS": [1.0, 2.0],
            "phyloP100way_vertebrate": [0.2, 0.8],
            "phastCons100way_vertebrate": [0.1, 0.9],
            "HAS_STRUCTURE": [0, 1],
        }
    )
    specs = stage11._publication_baseline_specs(frame)
    assert specs["esm_score_logistic"] == ["esm_variant_score"]
    assert specs["conservation_logistic"] == [
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
    ]
    assert specs["esm_conservation_logistic"][0] == "esm_variant_score"


def test_stage11_rejects_legacy_or_partial_stage14_contract(
    tmp_path, monkeypatch
) -> None:
    tuning = tmp_path / "best_cross_attention_params.json"
    tuning.write_text(json.dumps({"protocol_version": "legacy"}), encoding="utf-8")
    monkeypatch.setattr(stage11, "TUNING_BEST_JSON", tuning)
    monkeypatch.setattr(stage11, "REQUIRE_TUNING_ARTIFACT", True)
    monkeypatch.setattr(
        stage11, "validate_upstream_manifest", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        stage11.C,
        "read_tuned_parameter_sets",
        lambda: {"cross_attention": {"D_MODEL": 128}},
    )
    try:
        stage11._validated_tuned_parameter_sets()
    except RuntimeError as error:
        assert "legacy/incompatible" in str(error)
    else:
        raise AssertionError("Legacy tuning artifacts must be rejected")


def test_stage11_authenticates_stage10_before_loading_data(monkeypatch) -> None:
    def reject(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("upstream authentication sentinel")

    monkeypatch.setattr(stage11, "validate_upstream_manifest", reject)
    with pytest.raises(RuntimeError, match="authentication sentinel"):
        stage11._load_data()


def test_stage11_reuses_validated_stage14_outer_folds(
    tmp_path, monkeypatch
) -> None:
    n_rows = 40
    labels = np.tile(np.array([0, 1], dtype=int), n_rows // 2)
    groups = np.array([f"G{index}" for index in range(n_rows)], dtype=object)
    frame = pd.DataFrame(
        {
            common.ROW_ID_COL: [f"r{index}" for index in range(n_rows)],
            common.GENE_COL: groups,
            common.LABEL_COL: labels,
        }
    )
    input_path = tmp_path / "internal.parquet"
    input_path.write_bytes(b"same-stage10-input")
    splits = common.make_group_splits(labels, groups, 5, seed=42)
    plan = {
        "protocol_version": stage11.REQUIRED_TUNING_PROTOCOL,
        "n_rows": n_rows,
        "row_order_sha256": stage11._ordered_text_sha256(
            frame[common.ROW_ID_COL]
        ),
        "input_sha256": stage11.file_sha256(input_path),
        "split_group_column": common.GENE_COL,
        "folds": [
            {
                "outer_train": {"indices": train.tolist()},
                "outer_validation": {"indices": validation.tolist()},
            }
            for train, validation in splits
        ],
    }
    plan_path = tmp_path / "nested_inner_splits.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(stage11, "INPUT_CSV", input_path)
    monkeypatch.setattr(stage11, "TUNING_SPLIT_PLAN", plan_path)

    observed = stage11._stage14_outer_splits(frame, labels, groups)
    assert len(observed) == 5
    for expected, actual in zip(splits, observed):
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])


def test_oof_rewrite_removes_stale_models_for_same_configuration(tmp_path) -> None:
    path = tmp_path / "oof.npz"
    np.savez_compressed(
        path,
        predictor_free__stale_model=np.array([0.9, 0.1]),
        another_config__model=np.array([0.2, 0.8]),
    )
    common.save_oof_artifacts(
        path,
        np.array([0, 1]),
        {"current_model": np.array([0.1, 0.9])},
        "predictor_free",
    )
    with np.load(path) as stored:
        assert "predictor_free__stale_model" not in stored
        assert "predictor_free__current_model" in stored
        assert "another_config__model" in stored
