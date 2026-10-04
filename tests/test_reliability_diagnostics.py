from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import common as C


stage12 = importlib.import_module("12_external_validation")

FEATURE_NAMES = [
    "esm_variant_score",
    "GERP++_RS",
    "SASA",
    "LOCAL_MEAN_PLDDT_8A",
    "LOCAL_MIN_PLDDT_8A",
    "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
    "HAS_STRUCTURE",
    "LOW_CONFIDENCE_STRUCTURE",
    "GERP++_RS__missing",
    "phyloP100way_vertebrate__missing",
    "phastCons100way_vertebrate__missing",
]


def _diagnostic_model() -> C.EnsembleModel:
    with C.temporary_model_config(
        {
            "D_MODEL": 32,
            "DROPOUT": 0.0,
            "MODALITY_DROPOUT": 0.0,
            "RELIABILITY_RESIDUAL_SCALE": 1.25,
        }
    ):
        members = [
            C.ReliabilityResidualNet(len(FEATURE_NAMES), FEATURE_NAMES, esm_dim=8)
            for _ in range(2)
        ]
    return C.EnsembleModel(
        members,
        calibration_slope=1.2,
        calibration_intercept=-0.1,
    ).to(C.DEVICE)


def _availability_matrix() -> np.ndarray:
    values = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    values[:, FEATURE_NAMES.index("esm_variant_score")] = [-1.0, -0.5, 0.5, 1.0]
    for name in (
        "GERP++_RS__missing",
        "phyloP100way_vertebrate__missing",
        "phastCons100way_vertebrate__missing",
    ):
        values[:, FEATURE_NAMES.index(name)] = 1.0

    # structure_only
    values[1, FEATURE_NAMES.index("HAS_STRUCTURE")] = 1.0
    values[1, FEATURE_NAMES.index("LOCAL_MEAN_PLDDT_8A")] = 90.0
    values[1, FEATURE_NAMES.index("LOCAL_MIN_PLDDT_8A")] = 80.0
    values[1, FEATURE_NAMES.index("LOCAL_CONFIDENT_CONTACT_FRACTION_8A")] = 0.9
    # conservation_only
    values[2, FEATURE_NAMES.index("GERP++_RS__missing")] = 0.0
    values[2, FEATURE_NAMES.index("phyloP100way_vertebrate__missing")] = 0.0
    values[2, FEATURE_NAMES.index("phastCons100way_vertebrate__missing")] = 0.0
    # structure_and_conservation
    values[3] = values[1]
    values[3, FEATURE_NAMES.index("esm_variant_score")] = 1.0
    values[3, FEATURE_NAMES.index("GERP++_RS__missing")] = 0.0
    values[3, FEATURE_NAMES.index("phyloP100way_vertebrate__missing")] = 0.0
    values[3, FEATURE_NAMES.index("phastCons100way_vertebrate__missing")] = 0.0
    return values


def test_component_extraction_matches_deployment_and_prespecified_strata() -> None:
    C.set_seeds(17)
    model = _diagnostic_model()
    bio = _availability_matrix()
    esm = np.random.RandomState(17).normal(size=(4, 8)).astype(np.float32)

    expected = C.predict(model, bio, esm, batch_size=2)
    components = C.predict_reliability_components(model, bio, esm, batch_size=2)

    assert np.allclose(components["model_probability"], expected, atol=1e-6)
    assert components["model_probability"][0] == pytest.approx(
        components["anchor_probability"][0], abs=1e-7
    )
    assert components["gate"][0] == 0.0
    assert np.abs(components["bounded_residual"]).max() <= 1.25 + 1e-6
    assert C.reliability_availability_codes(components).tolist() == [0, 1, 2, 3]


def _summary_components() -> dict[str, np.ndarray]:
    structure = np.asarray([0, 0, 0.8, 0.9, 0, 0, 0.7, 0.8], dtype=np.float32)
    conservation = np.asarray([0, 0, 0, 0, 1, 1, 0.8, 0.9], dtype=np.float32)
    gate = np.asarray([0, 0, 0.3, 0.4, 0.2, 0.3, 0.6, 0.7], dtype=np.float32)
    anchor_probability = np.asarray(
        [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.25, 0.75], dtype=np.float32
    )
    return {
        "anchor_logit": np.log(anchor_probability / (1.0 - anchor_probability)),
        "anchor_probability": anchor_probability,
        "gate": gate,
        "bounded_residual": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        "hard_availability": (gate > 0).astype(np.float32),
        "structure_reliability": structure,
        "conservation_reliability": conservation,
    }


def test_summary_strata_are_label_independent_and_explicitly_descriptive() -> None:
    components = _summary_components()
    labels = np.asarray([0, 1] * 4)
    model_probability = components["anchor_probability"].copy()
    model_probability[2:] = np.asarray([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
    decisions = (model_probability >= 0.5).astype(np.int8)

    first = C.summarize_reliability_diagnostics(
        components, labels, model_probability, 0.5, decisions
    )
    second = C.summarize_reliability_diagnostics(
        components, 1 - labels, model_probability, 0.5, decisions
    )

    assert first["status"] == "descriptive_post_selection_not_primary"
    assert first["stratification"]["uses_labels"] is False
    assert first["selection_or_refitting"] == {
        "used_for_model_selection": False,
        "used_for_threshold_selection": False,
        "external_labels_used_for_refitting": False,
        "inference_claim_allowed": False,
    }
    for name in C.RELIABILITY_AVAILABILITY_STRATA.values():
        assert first["strata"][name]["n"] == second["strata"][name]["n"] == 2
        assert first["strata"][name]["coverage"] == second["strata"][name][
            "coverage"
        ]
    assert first["gate_behavior"]["maximum_anchor_identity_error_on_hard_fallback"] == 0
    rows = C.reliability_diagnostic_rows(first, {"scope": "test"})
    assert len(rows) == 4
    assert all(row["strata_use_labels"] is False for row in rows)
    assert all(row["used_for_model_selection"] is False for row in rows)


def test_hard_fallback_identity_check_fails_closed() -> None:
    components = _summary_components()
    labels = np.asarray([0, 1] * 4)
    invalid = components["anchor_probability"].copy()
    invalid[0] += 0.01
    with pytest.raises(RuntimeError, match="differ from the anchor"):
        C.summarize_reliability_diagnostics(
            components,
            labels,
            invalid,
            0.5,
            (invalid >= 0.5).astype(np.int8),
        )


def test_oof_extra_component_arrays_are_namespaced_and_aligned(tmp_path: Path) -> None:
    path = tmp_path / "oof.npz"
    C.save_oof_artifacts(
        path,
        np.asarray([0, 1]),
        {"model": np.asarray([0.2, 0.8])},
        "cfg",
        extra_arrays={"reliability_components__gate": np.asarray([0.0, 0.5])},
    )
    with np.load(path) as stored:
        assert "cfg__model" in stored
        assert "cfg__reliability_components__gate" in stored
    with pytest.raises(ValueError, match="misaligned"):
        C.save_oof_artifacts(
            path,
            np.asarray([0, 1]),
            {"model": np.asarray([0.2, 0.8])},
            "cfg",
            extra_arrays={"bad": np.asarray([0.0])},
        )


def _component_fold_tensor(rows: int, folds: int = 3) -> dict[str, np.ndarray]:
    anchor = np.asarray(
        [
            [0.2, 0.4, 0.1],
            [0.8, 0.8, 0.1],
            [0.9, 0.9, 0.1],
        ],
        dtype=np.float32,
    )[:folds, :rows]
    structure = np.asarray([0.8, 0.6, 0.0], dtype=np.float32)[:rows]
    conservation = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)[:rows]
    gate = np.asarray([0.4, 0.3, 0.2], dtype=np.float32)[:rows]
    base = {
        "anchor_logit": np.log(anchor / (1.0 - anchor)),
        "anchor_probability": anchor,
        "gate": np.stack([gate] * folds),
        "bounded_residual": np.zeros((folds, rows), dtype=np.float32),
        "hard_availability": np.ones((folds, rows), dtype=np.float32),
        "structure_reliability": np.stack([structure] * folds),
        "conservation_reliability": np.stack([conservation] * folds),
    }
    return base


def test_clinvar_component_and_anchor_vote_aggregation_follow_fold_contract() -> None:
    frame = pd.DataFrame(
        {
            "chr": [1, 1, 2],
            "pos": [10, 10, 20],
            "ref": ["A", "A", "C"],
            "alt": ["G", "G", "T"],
            C.ROW_ID_COL: ["a1", "a2", "b1"],
            C.GENE_COL: ["GENE1", "GENE1", "GENE2"],
            C.LABEL_COL: [1, 1, 0],
            "variant_id": ["v1", "v1", "v2"],
        }
    )
    fold_components = _component_fold_tensor(3)
    components = {
        name: values.mean(axis=0) for name, values in fold_components.items()
    }
    fold_probability = fold_components["anchor_probability"].copy()
    probability = fold_probability.mean(axis=0)
    units = stage12._aggregate_clinvar_variants(
        frame,
        {C.RELIABILITY_ARCHITECTURE: probability},
        {C.RELIABILITY_ARCHITECTURE: (probability >= 0.5).astype(np.int8)},
        {},
        {C.RELIABILITY_ARCHITECTURE: 0.5},
        {C.RELIABILITY_ARCHITECTURE: fold_probability},
        {C.RELIABILITY_ARCHITECTURE: np.repeat(0.5, 3)},
        components,
        fold_components,
    )

    assert units.frame["annotation_row_count"].tolist() == [2, 1]
    assert units.reliability_components is not None
    assert units.reliability_components["anchor_probability"].tolist() == pytest.approx(
        [2.0 / 3.0, 0.1]
    )
    # Variant 1 gets fold votes 0,1,1 after averaging its two annotations.
    assert units.reliability_anchor_decisions is not None
    assert units.reliability_anchor_decisions.tolist() == [1, 0]


def test_external_set_persists_compact_reliability_components_and_summary() -> None:
    rows = 8
    labels = np.asarray([0, 1] * 4)
    frame = pd.DataFrame(
        {
            "chr": np.arange(1, rows + 1),
            "pos": np.arange(100, 100 + rows),
            "ref": ["A"] * rows,
            "alt": ["G"] * rows,
            C.ROW_ID_COL: [f"r{index}" for index in range(rows)],
            C.GENE_COL: [f"G{index}" for index in range(rows)],
            C.LABEL_COL: labels,
            "variant_id": [f"v{index}" for index in range(rows)],
            "ESM_EXTRACTION_SUCCESS": np.ones(rows),
            "esm_variant_score_masked_marginal": np.linspace(-1, 1, rows),
            "HAS_STRUCTURE": np.ones(rows),
            "SASA": np.ones(rows),
            "GERP++_RS": np.ones(rows),
        }
    )
    pair = stage12.DatasetPair(
        "clinvar", frame, np.zeros((rows, 2), dtype=np.float32), 1.0
    )
    components = _summary_components()
    probability = components["anchor_probability"].copy()
    probability[2:] = np.asarray([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
    fold_components = {
        name: np.stack([values] * 5) for name, values in components.items()
    }
    fold_probability = np.stack([probability] * 5)
    artifact: dict[str, np.ndarray] = {}
    result, _, _ = stage12._evaluate_set(
        "clinvar",
        "exact_variant_disjoint",
        pair,
        np.ones(rows, dtype=bool),
        {C.RELIABILITY_ARCHITECTURE: probability},
        {C.RELIABILITY_ARCHITECTURE: (probability >= 0.5).astype(np.int8)},
        {C.RELIABILITY_ARCHITECTURE: 0.5},
        artifact,
        fold_probabilities={C.RELIABILITY_ARCHITECTURE: fold_probability},
        fold_thresholds={C.RELIABILITY_ARCHITECTURE: np.repeat(0.5, 5)},
        reliability_components=components,
        fold_reliability_components=fold_components,
    )

    diagnostic = result["reliability_diagnostics"]
    assert diagnostic["stratification"]["uses_labels"] is False
    prefix = "clinvar_exact_variant_disjoint__reliability_components"
    for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS:
        assert artifact[f"{prefix}__{name}"].shape == (rows,)
        assert artifact[f"{prefix}__{name}"].dtype == np.float32
    assert artifact[f"{prefix}__availability_stratum_code"].dtype == np.int8
    assert artifact[f"{prefix}__anchor_decisions"].dtype == np.int8
