from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import schema


stage12 = importlib.import_module("12_external_validation")
stage13 = importlib.import_module("13_generate_figures")


def _clinvar_annotation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chr": ["1", "1", "2", "2"],
            "pos": [10, 10, 20, 20],
            "ref": ["A", "A", "C", "C"],
            "alt": ["G", "G", "T", "T"],
            schema.ROW_ID_COL: ["r1", "r2", "r3", "r4"],
            schema.GENE_COL: ["BGENE", "AGENE", "CGENE", "CGENE"],
            schema.LABEL_COL: [1, 1, 0, 0],
            "variant_id": ["v1", "v1", "v2", "v2"],
            "ESM_EXTRACTION_SUCCESS": [1, 1, 1, 1],
            stage12.MASKED_MARGINAL_COL: [-1.0, -3.0, -0.5, -1.5],
        }
    )


def test_clinvar_is_aggregated_to_unique_genomic_variants() -> None:
    frame = _clinvar_annotation_rows()
    probabilities = {"model": np.array([0.2, 0.6, 0.1, 0.3])}
    rankings = {
        stage12.ZERO_SHOT_MODEL: np.array([1.0, 3.0, 0.5, 1.5]),
        "context_SIFT_score": np.array([0.1, np.nan, 0.2, 0.4]),
    }
    units = stage12._aggregate_clinvar_variants(
        frame,
        probabilities,
        {"model": np.ones(4, dtype=int)},
        rankings,
        {"model": 0.3},
    )

    assert units.frame["genomic_variant_key"].tolist() == [
        "1:10:A:G",
        "2:20:C:T",
    ]
    assert units.frame[schema.GENE_COL].tolist() == ["AGENE", "CGENE"]
    assert units.frame["all_genes"].tolist() == ["AGENE;BGENE", "CGENE"]
    np.testing.assert_allclose(units.probabilities["model"], [0.4, 0.2])
    np.testing.assert_allclose(units.ranking_scores[stage12.ZERO_SHOT_MODEL], [2.0, 1.0])
    np.testing.assert_allclose(units.ranking_scores["context_SIFT_score"], [0.1, 0.3])
    assert units.decisions["model"].tolist() == [1, 0]
    assert units.audit["input_annotation_rows"] == 4
    assert units.audit["unique_genomic_variants"] == 2
    assert units.audit["multi_annotation_variants"] == 2
    assert units.audit["multi_gene_variants"] == 1


def test_clinvar_decision_aggregates_annotations_before_frozen_fold_vote() -> None:
    frame = _clinvar_annotation_rows().iloc[:2].copy()
    fold_probabilities = {
        "model": np.asarray(
            [
                [0.49, 0.49],
                [0.49, 0.49],
                [0.90, 0.90],
            ]
        )
    }
    fold_thresholds = {"model": np.asarray([0.50, 0.50, 0.10])}
    probabilities, row_decisions, thresholds = stage12._aggregate_predictions(
        {"model": list(fold_probabilities["model"])},
        {"model": list(fold_thresholds["model"])},
    )
    # The old mean-probability/mean-threshold rule would classify this positive.
    assert probabilities["model"].mean() > thresholds["model"]
    units = stage12._aggregate_clinvar_variants(
        frame,
        probabilities,
        row_decisions,
        {},
        thresholds,
        fold_probabilities,
        fold_thresholds,
    )
    assert units.decisions["model"].tolist() == [0]
    np.testing.assert_allclose(
        units.decision_confidence["model"], [1.0 / 3.0]
    )
    assert units.audit["decision_aggregation"].startswith("within_each_fold")


def test_zero_shot_baseline_is_raw_masked_marginal_with_safe_orientation() -> None:
    frame = pd.DataFrame(
        {stage12.MASKED_MARGINAL_COL: [-2.0, 0.5, np.nan]}
    )
    scores = stage12._esm_zero_shot_scores(frame)
    assert scores is not None
    np.testing.assert_allclose(scores[:2], [2.0, -0.5])
    assert np.isnan(scores[2])


def test_contextual_predictors_are_oriented_and_never_declared_model_inputs() -> None:
    frame = pd.DataFrame(
        {
            "SIFT_score": [0.1, 0.9, np.nan],
            "CADD_phred": [5.0, 25.0, 10.0],
        }
    )
    scores, status = stage12._contextual_predictor_scores(frame)

    np.testing.assert_allclose(
        scores["context_SIFT_score"][:2], [-0.1, -0.9]
    )
    np.testing.assert_allclose(scores["context_CADD_phred"], [5.0, 25.0, 10.0])
    assert status["SIFT_score"]["source_orientation"] == "lower_is_more_deleterious"
    assert status["SIFT_score"]["row_coverage"] == 0.666667
    assert status["SIFT_score"]["model_input"] is False
    assert status["REVEL_score"]["available"] is False


def test_modern_contextual_predictor_directions_are_supported() -> None:
    frame = pd.DataFrame(
        {
            "AlphaMissense_score": [0.1, 0.9],
            "EVE_score": [0.2, 0.8],
            "PrimateAI-3D_score": [0.3, 0.7],
        }
    )
    scores, status = stage12._contextual_predictor_scores(frame)
    for column in frame:
        np.testing.assert_allclose(scores[f"context_{column}"], frame[column])
        assert status[column]["model_input"] is False
        assert status[column]["source_orientation"] == (
            "higher_is_more_deleterious"
        )


def test_contextual_benchmark_reports_available_and_common_coverage(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            schema.ROW_ID_COL: ["v1", "v2", "v3", "v4"],
            schema.GENE_COL: ["A", "A", "B", "B"],
            schema.LABEL_COL: [0, 1, 0, 1],
        }
    )
    contextual = {
        "context_SIFT_score": np.array([-0.9, -0.1, np.nan, -0.2]),
        "context_CADD_phred": np.array([1.0, 4.0, 2.0, 3.0]),
    }
    monkeypatch.setattr(
        stage12,
        "_hierarchical_rank_intervals",
        lambda labels, scores, groups: {"auroc": [0.0, 1.0], "auprc": [0.0, 1.0]},
    )
    artifact: dict[str, np.ndarray] = {}
    result = stage12._contextual_predictor_benchmark(
        frame,
        contextual,
        {"model": np.array([0.1, 0.9, 0.2, 0.8])},
        {stage12.ZERO_SHOT_MODEL: np.array([0.0, 1.0, 0.2, 0.8])},
        "clinvar_exact_variant_disjoint",
        artifact,
    )

    assert result["available"] is True
    assert result["individual_coverage"]["context_SIFT_score"]["n"] == 3
    common = result["common_coverage"]
    assert common["n"] == 3
    assert set(common["models"]) == {
        "context_SIFT_score",
        "context_CADD_phred",
        "model",
        stage12.ZERO_SHOT_MODEL,
    }
    prefix = "clinvar_exact_variant_disjoint_contextual_common_coverage"
    assert artifact[f"{prefix}__y"].shape == (3,)
    assert result["model_input"] is False


def test_gene_then_variant_bootstrap_is_reproducible() -> None:
    groups = np.array(["A", "A", "B", "C", "C", "C"], dtype=object)
    first = stage12._hierarchical_bootstrap_indices(groups, iterations=5, seed=7)
    second = stage12._hierarchical_bootstrap_indices(groups, iterations=5, seed=7)
    assert len(first) == 5
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert np.isin(left, np.arange(len(groups))).all()


def test_missing_dms_modalities_are_flagged_as_not_full_multimodal() -> None:
    frame = pd.DataFrame(
        {
            "ESM_EXTRACTION_SUCCESS": [1, 1],
            stage12.MASKED_MARGINAL_COL: [-1.0, -2.0],
            "HAS_STRUCTURE": [np.nan, np.nan],
            "SASA": [np.nan, np.nan],
            "PLDDT_SCORE": [np.nan, np.nan],
            "GERP++_RS": [np.nan, np.nan],
            "phyloP100way_vertebrate": [np.nan, np.nan],
            "phastCons100way_vertebrate": [np.nan, np.nan],
        }
    )
    report = stage12._modality_coverage(frame)
    assert report["esm_masked_marginal"]["row_coverage"] == 1.0
    assert report["structure"]["row_coverage"] == 0.0
    assert report["conservation"]["row_coverage"] == 0.0
    assert report["full_multimodal_validation"] is False
    assert report["missing_modalities"] == ["structure", "conservation"]


def test_dms_primary_endpoints_are_assay_macro_with_paired_bootstrap() -> None:
    frame = pd.DataFrame(
        {
            "ASSAY_ID": ["A"] * 4 + ["B"] * 4,
            schema.LABEL_COL: [1, 1, 0, 0, 1, 1, 0, 0],
            "DMS_SCORE": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        }
    )
    probabilities = {
        "model": np.array([0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1])
    }
    decisions = {"model": (probabilities["model"] >= 0.5).astype(int)}
    rankings = {
        stage12.ZERO_SHOT_MODEL: np.array([3.0, 2.0, 1.0, 0.0] * 2)
    }
    result = stage12._dms_assay_results(
        frame,
        probabilities,
        decisions,
        {"model": 0.5},
        rankings,
        {"is_full_proteingym_benchmark": False},
    )

    assert result["analysis_unit"] == "assay"
    assert result["assay_count"] == 2
    assert result["macro"]["model"]["functional_spearman"] == 1.0
    assert result["macro"]["model"]["auroc"] == 1.0
    assert result["macro"][stage12.ZERO_SHOT_MODEL]["functional_spearman"] == 1.0
    assert result["macro_ci95"]["model"]["auroc"] == [1.0, 1.0]
    comparison = result["paired_comparisons"][
        f"{stage12.ZERO_SHOT_MODEL}_minus_model"
    ]
    assert comparison["metrics"]["auroc"]["paired_assays"] == 2
    assert comparison["metrics"]["auroc"]["two_sided_probability"] > 0
    assert "holm_adjusted_probability" in comparison["metrics"]["auroc"]
    assert result["sampling_caveats"]["is_full_proteingym_benchmark"] is False


def test_dms_sampling_audit_reads_uniform_probabilities_and_legacy_fallback(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "extra": {
                    "dms_sampling_policy": "hash_uniform",
                    "dms_sampling_uses_label": False,
                    "dms_max_rows_per_assay": 500,
                    "assay_rows": {"A": 500, "B": 100},
                    "assay_candidate_rows": {"A": 1000, "B": 100},
                    "assay_sampling": {
                        "A": {
                            "sampling_probability": 0.5,
                            "sample_weight": 2.0,
                        },
                        "B": {
                            "sampling_probability": 1.0,
                            "sample_weight": 1.0,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    audit = stage12._dms_sampling_audit(manifest)
    assert audit["sampling_policy"] == "hash_uniform"
    assert audit["sampling_uses_label"] is False
    assert audit["legacy_fallback"] is False
    assert audit["sampling_probability_range"] == [0.5, 1.0]
    assert audit["sample_weight_range"] == [1.0, 2.0]
    assert audit["inverse_probability_weight_inconsistencies"] == 0
    assert audit["assays_downsampled"] == 1

    manifest.write_text(
        json.dumps(
            {
                "extra": {
                    "assay_rows": {"A": 500},
                    "assay_candidate_rows": {"A": 1000},
                }
            }
        ),
        encoding="utf-8",
    )
    legacy = stage12._dms_sampling_audit(manifest)
    assert legacy["legacy_fallback"] is True
    assert legacy["sampling_uses_label"] is True
    assert legacy["sampling_policy"].startswith("legacy_")


def test_figures_use_predeclared_exact_variant_set_not_tiny_gene_set(
    tmp_path: Path, monkeypatch
) -> None:
    prediction_path = tmp_path / "external_predictions.npz"
    result_path = tmp_path / "external_validation.json"
    np.savez_compressed(
        prediction_path,
        clinvar_gene_disjoint__y=np.array([0, 1]),
        clinvar_gene_disjoint__groups=np.array(["A", "B"], dtype=object),
        clinvar_gene_disjoint__lightgbm=np.array([0.1, 0.9]),
        clinvar_gene_disjoint__lightgbm__decisions=np.array([0, 1]),
    )
    result_path.write_text(json.dumps({"sets": {}}), encoding="utf-8")
    monkeypatch.setattr(stage13, "EXTERNAL_PREDICTIONS", prediction_path)
    monkeypatch.setattr(stage13, "EXTERNAL_RESULTS", result_path)

    assert stage13._external_vectors("clinvar") is None

    np.savez_compressed(
        prediction_path,
        clinvar_gene_disjoint__y=np.array([0, 1]),
        clinvar_gene_disjoint__groups=np.array(["A", "B"], dtype=object),
        clinvar_exact_variant_disjoint__y=np.array([0, 1, 1]),
        clinvar_exact_variant_disjoint__groups=np.array(
            ["A", "B", "B"], dtype=object
        ),
        clinvar_exact_variant_disjoint__lightgbm=np.array([0.1, 0.8, 0.9]),
        clinvar_exact_variant_disjoint__lightgbm__decisions=np.array([0, 1, 1]),
    )
    result_path.write_text(
        json.dumps(
            {
                "sets": {
                    "clinvar_exact_variant_disjoint": {
                        "n": 3,
                        "positives": 2,
                        "genes": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    vectors = stage13._external_vectors("clinvar")
    assert vectors is not None
    assert vectors.prefix == "clinvar_exact_variant_disjoint"
    title = stage13._external_panel_title(vectors)
    assert "exact-variant-disjoint" in title
    assert "n=3 unique variants" in title
    assert "pos=2" in title
    assert "genes=2" in title


def test_stage12_discovers_new_and_legacy_stage11_models(
    tmp_path: Path, monkeypatch
) -> None:
    row_ids = np.array([f"r{index}" for index in range(5)], dtype=object)
    oof = tmp_path / "oof_predictions.npz"
    prefix = stage12.MODEL_TAG
    np.savez_compressed(
        oof,
        **{
            f"{prefix}__row_ids": row_ids,
            f"{prefix}__fold_ids": np.arange(1, 6),
            f"{prefix}__y": np.array([0, 1, 0, 1, 0]),
            f"{prefix}__lightgbm": np.linspace(0.1, 0.5, 5),
            f"{prefix}__raw_esm_zero_shot": np.linspace(0.1, 0.5, 5),
            f"{prefix}__esm_score_logistic": np.linspace(0.1, 0.5, 5),
            f"{prefix}__esm_embedding_mutation": np.linspace(0.1, 0.5, 5),
            f"{prefix}__concatenation": np.linspace(0.1, 0.5, 5),
            f"{prefix}__gated_fusion": np.linspace(0.1, 0.5, 5),
            f"{prefix}__cross_attention": np.linspace(0.1, 0.5, 5),
            f"{prefix}__cross_attention__thresholds": np.repeat(0.5, 5),
        },
    )
    monkeypatch.setattr(stage12, "INTERNAL_OOF", oof)
    frame = pd.DataFrame({schema.ROW_ID_COL: row_ids})
    _, names = stage12._validate_internal_reference(frame)

    assert names == [
        "lightgbm",
        "raw_esm_zero_shot",
        "esm_score_logistic",
        "esm_embedding_mutation",
        "concatenation",
        "gated_fusion",
        "cross_attention",
    ]


def test_external_ensemble_uses_majority_of_frozen_fold_decisions() -> None:
    probabilities, decisions, thresholds = stage12._aggregate_predictions(
        {"model": [np.array([0.49]), np.array([0.49]), np.array([0.60])]},
        {"model": [0.50, 0.50, 0.10]},
    )
    assert probabilities["model"][0] > thresholds["model"]
    assert decisions["model"].tolist() == [0]


@pytest.mark.parametrize("include_dms_sequences", [False, True])
def test_stage12_authenticates_internal_external_and_model_artifacts(
    tmp_path: Path, monkeypatch, include_dms_sequences: bool
) -> None:
    model_dir = tmp_path / "models" / "fold_1"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "gated_fusion.pt"
    model_file.write_bytes(b"model")
    calls: list[tuple[Path, str, list[Path], dict[str, object]]] = []
    input_binding_calls: list[tuple[Path, str, list[Path]]] = []

    def capture(manifest, stage, artifacts, **kwargs):
        calls.append(
            (
                Path(manifest),
                stage,
                [Path(value) for value in artifacts],
                kwargs,
            )
        )
        return {}

    def capture_inputs(manifest, stage, artifacts):
        input_binding_calls.append(
            (Path(manifest), stage, [Path(value) for value in artifacts])
        )
        return {}

    monkeypatch.setattr(stage12, "MODEL_DIR", tmp_path / "models")
    # This unit test must not depend on datasets installed beside the source.
    sequence_file = tmp_path / "dms_sequences.parquet"
    if include_dms_sequences:
        sequence_file.write_bytes(b"sequence fixture")
    monkeypatch.setattr(stage12, "DMS_SEQUENCE_OUTPUT", sequence_file)
    monkeypatch.setattr(stage12, "validate_upstream_manifest", capture)
    monkeypatch.setattr(
        stage12, "validate_manifest_input_bindings", capture_inputs
    )
    observed_models = stage12._validate_upstream_chain(["clinvar", "dms"])

    assert observed_models == [model_file]
    assert [stage for _, stage, _, _ in calls] == [
        "07_dataset_balancing",
        "10_extract_esm_features:internal",
        "11_train_and_evaluate",
        "14_tune_cross_attention",
        "09_prepare_external_esm_dataset",
        "10_extract_esm_features:clinvar",
        "10_extract_esm_features:dms",
    ]
    assert calls[0][2] == [stage12.INTERNAL_UNIVERSE]
    assert model_file in calls[2][2]
    assert calls[4][2] == [
        stage12.STAGE09_PREPARED_OUTPUTS["clinvar"],
        stage12.STAGE09_PREPARED_OUTPUTS["dms"],
    ] + ([sequence_file] if include_dms_sequences else [])
    assert list(stage12.EXTERNAL_INPUTS["clinvar"]) == calls[5][2]
    assert list(stage12.EXTERNAL_INPUTS["dms"]) == calls[6][2]
    assert calls[2][3]["required_source_files"] == (
        "common.py",
        "gpu_runtime.py",
    )
    assert calls[3][3]["required_source_files"] == (
        "common.py",
        "gpu_runtime.py",
    )
    assert calls[4][3]["required_source_files"] == (
        "01_dbnsfp_processor.py",
        "04_feature_engineering.py",
        "08_prepare_esm_dataset.py",
    )
    assert [stage for _, stage, _ in input_binding_calls] == [
        "10_extract_esm_features:clinvar",
        "10_extract_esm_features:dms",
    ]
    assert input_binding_calls[0][2] == [
        stage12.EXTERNAL_PREP_MANIFEST,
        stage12.STAGE09_PREPARED_OUTPUTS["clinvar"],
    ]


def test_contextual_figure_states_policy_counts_and_evidence_warning(
    tmp_path: Path, monkeypatch
) -> None:
    result_path = tmp_path / "external_validation.json"
    common = {
        "n": 40,
        "positives": 10,
        "genes": 8,
        "models": {
            "context_SIFT_score": {
                "auroc": 0.7,
                "auprc": 0.4,
                "ci95": {"auroc": [0.6, 0.8], "auprc": [0.3, 0.5]},
            },
            "lightgbm": {
                "auroc": 0.75,
                "auprc": 0.45,
                "ci95": {"auroc": [0.65, 0.85], "auprc": [0.35, 0.55]},
            },
        },
    }
    result_path.write_text(
        json.dumps(
            {
                "sets": {
                    "clinvar_exact_variant_disjoint": {
                        "contextual_predictor_benchmark": {
                            "available": True,
                            "common_coverage": common,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage13, "EXTERNAL_RESULTS", result_path)
    figure = stage13.figure_external_contextual_predictors()
    title = figure._suptitle.get_text()
    assert "exact-variant-disjoint" in title
    assert "n=40 unique variants, pos=10, genes=8" in title
    assert "not independent ACMG evidence" in title
    plt.close(figure)


def test_internal_figures_use_authoritative_stage14_nested_oof(
    tmp_path: Path, monkeypatch
) -> None:
    prediction_path = tmp_path / "nested_tuning_oof.npz"
    result_path = tmp_path / "architecture_selection.json"
    labels = np.array([0, 1, 0, 1], dtype=int)
    groups = np.array(["H1", "H2", "H3", "H4"], dtype=object)
    np.savez_compressed(
        prediction_path,
        y=labels,
        groups=groups,
        reference__raw_esm_zero_shot__probabilities=np.array(
            [0.1, 0.8, 0.2, 0.7]
        ),
        reference__raw_esm_zero_shot__decisions=labels,
        reference__raw_esm_zero_shot__thresholds=np.repeat(0.5, 4),
        gated_fusion__probabilities=np.array([0.2, 0.9, 0.1, 0.8]),
        gated_fusion__decisions=labels,
        gated_fusion__thresholds=np.repeat(0.5, 4),
    )
    metric = {
        "mcc": 1.0,
        "auroc": 1.0,
        "auprc": 1.0,
        "ci95": {
            "mcc": [1.0, 1.0],
            "auroc": [1.0, 1.0],
            "auprc": [1.0, 1.0],
        },
    }
    result_path.write_text(
        json.dumps(
            {
                "architectures": {
                    "gated_fusion": {"nested_outer_metrics": metric}
                },
                "reference_baselines": {"raw_esm_zero_shot": metric},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage13, "INTERNAL_PREDICTIONS", prediction_path)
    monkeypatch.setattr(stage13, "INTERNAL_RESULTS", result_path)

    observed_labels, observed_groups, probabilities, decisions = (
        stage13._internal_vectors()
    )
    np.testing.assert_array_equal(observed_labels, labels)
    np.testing.assert_array_equal(observed_groups, groups)
    assert set(probabilities) == {"raw_esm_zero_shot", "gated_fusion"}
    assert set(decisions) == set(probabilities)
    figure = stage13.figure_model_comparison()
    assert "Primary internal nested outer-fold" in figure._suptitle.get_text()
    plt.close(figure)


def test_clinvar_model_comparison_uses_cluster_bootstrap_not_row_mcnemar(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "chr": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "pos": np.arange(1, 9),
            "ref": ["A"] * 8,
            "alt": ["G"] * 8,
            schema.ROW_ID_COL: [f"r{index}" for index in range(8)],
            schema.GENE_COL: ["A", "A", "B", "B", "C", "C", "D", "D"],
            schema.LABEL_COL: [0, 1, 0, 1, 0, 1, 0, 1],
            "variant_id": [f"v{index}" for index in range(8)],
            "ESM_EXTRACTION_SUCCESS": np.ones(8),
        }
    )
    baseline = np.array([0.2, 0.7, 0.3, 0.6, 0.4, 0.8, 0.1, 0.9])
    gated = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.2, 0.8])
    pair = stage12.DatasetPair(
        name="clinvar",
        frame=frame,
        embeddings=np.zeros((8, 2), dtype=np.float32),
        extraction_coverage=1.0,
    )
    monkeypatch.setattr(
        stage12,
        "_hierarchical_model_comparison",
        lambda *args, **kwargs: {
            "mcc": {"two_sided_probability": 0.5},
            "auroc": {"two_sided_probability": 0.5},
            "auprc": {"two_sided_probability": 0.5},
        },
    )
    monkeypatch.setattr(
        stage12,
        "_hierarchical_intervals",
        lambda *args, **kwargs: {"mcc": [0.0, 1.0]},
    )
    result, _, _ = stage12._evaluate_set(
        "clinvar",
        "exact_variant_disjoint",
        pair,
        np.ones(8, dtype=bool),
        {
            "esm_conservation_logistic": baseline,
            "gated_fusion": gated,
        },
        {
            "esm_conservation_logistic": (baseline >= 0.5).astype(int),
            "gated_fusion": (gated >= 0.5).astype(int),
        },
        {"esm_conservation_logistic": 0.5, "gated_fusion": 0.5},
        {},
    )

    comparisons = result["comparisons"]
    assert comparisons["primary_baseline"] == "esm_conservation_logistic"
    assert "gated_fusion_minus_esm_conservation_logistic" in comparisons[
        "clustered_paired"
    ]
    assert comparisons["rowwise_mcnemar_policy"] == (
        "omitted_because_row_independence_is_invalid_for_clustered_variants"
    )
