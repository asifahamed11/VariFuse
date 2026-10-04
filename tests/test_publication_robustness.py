from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import publication_robustness as robustness
import schema


stage12 = importlib.import_module("12_external_validation")
stage13 = importlib.import_module("13_generate_figures")


def test_group_masks_are_deterministic_nested_and_never_split_groups() -> None:
    groups = np.asarray(["A", "A", "B", "C", "C", "D", "E", "E"])
    plans = [
        robustness.deterministic_group_mask(
            groups,
            level,
            seed=19,
            namespace="clinvar:structure",
            grouping_unit="gene",
        )
        for level in robustness.STRESS_LEVELS
    ]
    repeated = robustness.deterministic_group_mask(
        groups,
        0.5,
        seed=19,
        namespace="clinvar:structure",
        grouping_unit="gene",
    )
    np.testing.assert_array_equal(plans[2].row_mask, repeated.row_mask)
    for previous, current in zip(plans, plans[1:]):
        assert np.all(~previous.row_mask | current.row_mask)
    for plan in plans:
        for group in np.unique(groups):
            selected = plan.row_mask[groups == group]
            assert selected.all() or not selected.any()
        assert len(plan.audit()["selected_group_ids_sha256"]) == 64


def test_modality_mask_removes_only_selected_source_values() -> None:
    frame = pd.DataFrame(
        {
            "SASA": [1.0, 2.0, 3.0],
            "PLDDT_SCORE": [90.0, 80.0, 70.0],
            "SASA__missing": [0, 0, 0],
            "HAS_STRUCTURE": [1, 1, 1],
            "STRUCTURE_FILE_AVAILABLE": [1, 1, 1],
            "LOW_CONFIDENCE_STRUCTURE": [0, 0, 1],
            "esm_variant_score": [-1.0, -2.0, -3.0],
        }
    )
    masked, audit = robustness.apply_modality_mask(
        frame, np.asarray([False, True, False]), robustness.STRUCTURE_COLUMNS
    )
    assert np.isnan(masked.loc[1, "SASA"])
    assert np.isnan(masked.loc[1, "PLDDT_SCORE"])
    assert masked.loc[1, "SASA__missing"] == 1
    assert masked.loc[1, "HAS_STRUCTURE"] == 0
    assert masked.loc[1, "STRUCTURE_FILE_AVAILABLE"] == 0
    assert masked.loc[1, "LOW_CONFIDENCE_STRUCTURE"] == 0
    np.testing.assert_array_equal(masked["esm_variant_score"], frame["esm_variant_score"])
    assert audit["finite_source_cells_removed"] == 2
    assert audit["external_statistics_fitted"] is False


def test_tiny_clinvar_is_underpowered_and_blocks_claims() -> None:
    labels = np.asarray([1] * 6 + [0] * 25)
    groups = np.asarray([f"G{index % 20}" for index in range(31)])
    report = robustness.clinical_evidence_assessment(labels, groups)
    assert report["status"] == "underpowered"
    assert report["reporting_mode"] == "descriptive_only"
    assert report["inferential_model_comparison_allowed"] is False
    assert report["clinical_calibration_claim_allowed"] is False
    assert report["clinical_utility_claim_allowed"] is False
    assert "inferential_model_superiority" in report["prohibited_claims"]


def test_modality_audit_reports_partial_coverage_without_requiring_completeness() -> None:
    frame = pd.DataFrame(
        {
            "ESM_EXTRACTION_SUCCESS": [1, 1, 1],
            stage12.MASKED_MARGINAL_COL: [-1.0, -2.0, -3.0],
            "HAS_STRUCTURE": [1, 0, 1],
            "SASA": [2.0, np.nan, 1.0],
            "GERP++_RS": [3.0, np.nan, np.nan],
            "phyloP100way_vertebrate": [np.nan, np.nan, 2.0],
        }
    )
    report = stage12._modality_coverage(frame)
    assert report["structure"]["row_coverage"] == round(2 / 3, 6)
    assert report["conservation"]["row_coverage"] == round(2 / 3, 6)
    assert report["conservation"]["feature_coverage"]["GERP++_RS"] == {
        "available_column": True,
        "finite_n": 1,
        "row_coverage": round(1 / 3, 6),
    }
    assert report["conservation"]["feature_coverage"][
        "phastCons100way_vertebrate"
    ]["available_column"] is False
    assert report["full_multimodal_validation"] is False


def test_conformal_reporting_fails_closed_without_exchangeable_calibration() -> None:
    absent = robustness.conformal_availability_report(
        calibration_labels=None,
        exchangeability_justification=None,
        evaluation_shift="temporal",
    )
    assert absent["status"] == "not_estimated"
    assert absent["coverage_guarantee_claimed"] is False

    labels = np.tile([0, 1], 100)
    probabilities = np.where(labels == 1, 0.8, 0.2)
    withheld = robustness.mondrian_split_conformal_binary(
        labels,
        probabilities,
        [0.1, 0.5, 0.9],
        exchangeability_justification=None,
    )
    assert withheld["status"] == "not_estimated"
    assert "exchangeability_not_justified" in withheld["guard_failures"]

    estimated = robustness.mondrian_split_conformal_binary(
        labels,
        probabilities,
        [0.1, 0.5, 0.9],
        exchangeability_justification="prospectively sampled from the same population",
    )
    assert estimated["status"] == "estimated"
    assert estimated["coverage_guarantee_claimed"] is True
    assert estimated["set_size"].shape == (3,)


def test_stage12_no_source_stress_is_non_informative_but_fallback_is_safe() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    frame = pd.DataFrame(
        {
            schema.ROW_ID_COL: [f"r{index}" for index in range(8)],
            schema.GENE_COL: ["G"] * 8,
            schema.LABEL_COL: labels,
            "ASSAY_ID": ["A"] * 4 + ["B"] * 4,
            "DMS_SCORE": [3.0, 0.0, 2.0, 1.0] * 2,
        }
    )
    pair = stage12.DatasetPair(
        name="dms",
        frame=frame,
        embeddings=np.zeros((8, 2), dtype=np.float32),
        extraction_coverage=1.0,
    )
    anchor = np.asarray([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    candidate = np.asarray([0.2, 0.8, 0.3, 0.7, 0.2, 0.8, 0.3, 0.7])
    fold_probabilities = {
        "raw_esm_zero_shot": np.stack([anchor] * 5),
        "gated_fusion": np.stack([candidate] * 5),
    }
    fold_thresholds = {
        "raw_esm_zero_shot": np.repeat(0.5, 5),
        "gated_fusion": np.repeat(0.5, 5),
    }
    report, rows = stage12._run_missing_modality_stress(
        "dms",
        pair,
        np.ones(8, dtype=bool),
        [],
        [],
        ["raw_esm_zero_shot", "gated_fusion"],
        fold_probabilities,
        fold_thresholds,
    )
    assert report["status"] == "evaluated"
    for scenario in robustness.STRESS_SCENARIOS:
        scenario_report = report["scenarios"][scenario]
        assert scenario_report["status"] == "not_applicable_no_source_values"
        complete = scenario_report["levels"]["mask_100pct"]
        fallback = complete["safe_fallback_models"]["gated_fusion"]
        assert fallback["anchor_identity_on_masked_units"]["satisfied"] is True
        assert fallback["auroc"] == complete["models"]["raw_esm_zero_shot"]["auroc"]
    assert len(rows) == len(robustness.STRESS_SCENARIOS) * 5 * 3
    assert all(row["inferential_claim_allowed"] is False for row in rows)

    results = {
        "models": ["raw_esm_zero_shot", "gated_fusion"],
        "uncertainty_reporting_policy": {
            "external_label_tuning": False,
            "conformal": "withheld_without_a_distinct_calibration_sample",
            "clinical_utility": "never_inferred_from_discrimination_alone",
        },
        "robustness": {
            "protocol_version": robustness.STRESS_PROTOCOL_VERSION,
            "sources": {"dms": report},
        },
    }
    manifest = {
        "extra": {
            "robustness_protocol_version": robustness.STRESS_PROTOCOL_VERSION,
            "robustness_sources": ["dms"],
        }
    }
    table = pd.DataFrame(rows)
    validated = stage13._validate_robustness_reporting(results, table, manifest)
    assert validated["status"] == "validated"
    table.loc[table.index[0], "auroc"] = 0.1234
    with pytest.raises(RuntimeError, match="table.auroc"):
        stage13._validate_robustness_reporting(results, table, manifest)


def test_production_clinvar_guard_skips_inference(monkeypatch) -> None:
    n_rows = 8
    labels = np.asarray([0, 1] * 4)
    frame = pd.DataFrame(
        {
            "chr": [str(index + 1) for index in range(n_rows)],
            "pos": np.arange(1, n_rows + 1),
            "ref": ["A"] * n_rows,
            "alt": ["G"] * n_rows,
            schema.ROW_ID_COL: [f"r{index}" for index in range(n_rows)],
            schema.GENE_COL: [f"G{index // 2}" for index in range(n_rows)],
            schema.LABEL_COL: labels,
            "variant_id": [f"v{index}" for index in range(n_rows)],
            "ESM_EXTRACTION_SUCCESS": np.ones(n_rows),
        }
    )
    probability = np.asarray([0.2, 0.8] * 4)
    pair = stage12.DatasetPair(
        name="clinvar",
        frame=frame,
        embeddings=np.zeros((n_rows, 2), dtype=np.float32),
        extraction_coverage=1.0,
    )
    monkeypatch.setattr(
        stage12,
        "_hierarchical_intervals",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("underpowered interval must not run")
        ),
    )
    models = {
        "esm_conservation_logistic": probability,
        "gated_fusion": probability,
    }
    fold_probabilities = {name: np.stack([probability] * 5) for name in models}
    fold_thresholds = {name: np.repeat(0.5, 5) for name in models}
    result, _, _ = stage12._evaluate_set(
        "clinvar",
        "exact_variant_disjoint",
        pair,
        np.ones(n_rows, dtype=bool),
        models,
        {name: (values >= 0.5).astype(int) for name, values in models.items()},
        {name: 0.5 for name in models},
        {},
        fold_probabilities=fold_probabilities,
        fold_thresholds=fold_thresholds,
    )
    assert result["clinical_evidence_assessment"]["status"] == "underpowered"
    assert result["comparisons"] == {
        "status": "skipped",
        "reason": "underpowered_clinical_cohort_reporting_guard",
    }
    assert all(metrics["ci95"] is None for metrics in result["models"].values())
    assert result["conformal_prediction"]["status"] == "not_estimated"
