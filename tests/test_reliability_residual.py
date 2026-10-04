from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import common


stage11 = importlib.import_module("11_train_and_evaluate")
stage14 = importlib.import_module("14_tune_cross_attention")


FEATURE_NAMES = [
    "esm_variant_score",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SASA",
    "LOCAL_CONTACT_COUNT_8A",
    "LOCAL_MEAN_PLDDT_8A",
    "LOCAL_MIN_PLDDT_8A",
    "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
    "aa_charge_delta",
    "HAS_STRUCTURE",
    "LOW_CONFIDENCE_STRUCTURE",
    "HAS_DOMAIN_ANNOTATION",
    "GERP++_RS__missing",
    "phyloP100way_vertebrate__missing",
    "phastCons100way_vertebrate__missing",
]


def _model(dropout: float = 0.0) -> common.ReliabilityResidualNet:
    with common.temporary_model_config(
        {
            "D_MODEL": 64,
            "DROPOUT": 0.0,
            "MODALITY_DROPOUT": dropout,
            "RELIABILITY_RESIDUAL_SCALE": 1.5,
        }
    ):
        return common.ReliabilityResidualNet(
            len(FEATURE_NAMES), FEATURE_NAMES, esm_dim=16
        )


def _bio(rows: int = 3) -> torch.Tensor:
    values = torch.zeros((rows, len(FEATURE_NAMES)), dtype=torch.float32)
    values[:, FEATURE_NAMES.index("esm_variant_score")] = torch.tensor(
        [-1.0, 0.0, 1.0][:rows]
    )
    values[:, FEATURE_NAMES.index("GERP++_RS__missing")] = 1.0
    values[:, FEATURE_NAMES.index("phyloP100way_vertebrate__missing")] = 1.0
    values[:, FEATURE_NAMES.index("phastCons100way_vertebrate__missing")] = 1.0
    return values


def test_safe_fallback_is_exact_anchor_for_missing_or_low_confidence_auxiliary() -> None:
    model = _model()
    model.eval()
    bio = _bio()
    # Row 1 has a structure, but it is explicitly low confidence.  Conservation
    # is missing for every row, so both row 0 and row 1 must use the anchor only.
    bio[1, FEATURE_NAMES.index("HAS_STRUCTURE")] = 1.0
    bio[1, FEATURE_NAMES.index("LOW_CONFIDENCE_STRUCTURE")] = 1.0
    bio[1, FEATURE_NAMES.index("LOCAL_MEAN_PLDDT_8A")] = 45.0
    components = model.forward_components(bio, torch.randn(3, 16))
    assert torch.equal(components["gate"][:2], torch.zeros(2))
    assert torch.equal(
        components["logits"][:2], components["anchor_logit"][:2]
    )
    assert torch.equal(
        components["structure_embedding"][:2],
        torch.zeros_like(components["structure_embedding"][:2]),
    )


def test_anchor_direction_is_monotonic_when_fallback_is_active() -> None:
    model = _model()
    model.eval()
    logits = model(_bio(), torch.randn(3, 16))
    assert logits[0] > logits[1] > logits[2]


def test_reliable_local_structure_activates_dedicated_structural_expert() -> None:
    model = _model()
    model.eval()
    bio = _bio(1)
    bio[0, FEATURE_NAMES.index("HAS_STRUCTURE")] = 1.0
    bio[0, FEATURE_NAMES.index("LOCAL_CONTACT_COUNT_8A")] = 12.0
    bio[0, FEATURE_NAMES.index("LOCAL_MEAN_PLDDT_8A")] = 92.0
    bio[0, FEATURE_NAMES.index("LOCAL_MIN_PLDDT_8A")] = 80.0
    bio[0, FEATURE_NAMES.index("LOCAL_CONFIDENT_CONTACT_FRACTION_8A")] = 0.9
    components = model.forward_components(bio, torch.randn(1, 16))
    assert components["structure_reliability"].item() > 0.0
    assert components["hard_availability"].item() == 1.0
    assert components["gate"].item() > 0.0
    assert components["structure_embedding"].abs().sum().item() > 0.0
    assert "LOCAL_CONTACT_COUNT_8A" in model.structure_names


def test_full_modality_dropout_preserves_anchor_during_training() -> None:
    model = _model(dropout=1.0)
    model.train()
    bio = _bio(1)
    bio[0, FEATURE_NAMES.index("HAS_STRUCTURE")] = 1.0
    bio[0, FEATURE_NAMES.index("LOCAL_MEAN_PLDDT_8A")] = 95.0
    bio[0, FEATURE_NAMES.index("GERP++_RS__missing")] = 0.0
    components = model.forward_components(bio, torch.randn(1, 16))
    assert components["gate"].item() == 0.0
    assert torch.equal(components["logits"], components["anchor_logit"])


def test_availability_flags_are_gate_only_not_residual_predictors() -> None:
    model = _model()
    predictive = {
        *model.structure_names,
        *model.evolution_names,
        *model.context_names,
    }
    assert not predictive & set(common.RELIABILITY_GATE_ONLY_FEATURES)
    assert set(common.RELIABILITY_REQUIRED_GATE_FEATURES) <= set(model.gate_names)


def test_architecture_specific_schema_does_not_broaden_comparator_features() -> None:
    frame = pd.DataFrame({name: [0.0, 1.0] for name in FEATURE_NAMES})
    base = [
        "esm_variant_score",
        "GERP++_RS",
        "SASA",
        "LOCAL_CONTACT_COUNT_8A",
    ]
    comparator = common.architecture_feature_names("gated_fusion", frame, base)
    reliability = common.architecture_feature_names(
        common.RELIABILITY_ARCHITECTURE, frame, base
    )
    assert comparator == base
    assert "HAS_STRUCTURE" not in comparator
    assert "HAS_STRUCTURE" in reliability
    assert "GERP++_RS__missing" in reliability


def test_reliability_preprocessor_preserves_gate_scales() -> None:
    values = np.asarray(
        [[-2.0, 0.0, 40.0], [2.0, 1.0, 95.0]], dtype=np.float32
    )
    fitted = common.ArrayPreprocessor.fit_with_passthrough(values, [1, 2])
    transformed = fitted.transform(values)
    assert transformed[:, 1:].tolist() == values[:, 1:].tolist()
    assert transformed[:, 0].tolist() == pytest.approx([-1.0, 1.0])


def test_hpo_and_deployment_contracts_include_proposed_architecture() -> None:
    class Trial:
        @staticmethod
        def suggest_categorical(name, choices):
            del name
            return choices[0]

        @staticmethod
        def suggest_float(name, lower, upper, **kwargs):
            del name, upper, kwargs
            return lower

        @staticmethod
        def suggest_int(name, lower, upper):
            del name, upper
            return lower

    parameters = stage14._suggest_parameters(
        Trial(), common.RELIABILITY_ARCHITECTURE
    )
    assert parameters["MIXUP_ALPHA"] == 0.0
    assert "MODALITY_DROPOUT" in parameters
    assert "RELIABILITY_RESIDUAL_SCALE" in parameters
    assert common.RELIABILITY_ARCHITECTURE in stage14.PUBLICATION_ARCHITECTURES
    assert common.RELIABILITY_ARCHITECTURE in stage11.REQUIRED_ARCHITECTURES
    assert stage11.REQUIRED_TUNING_PROTOCOL == stage14.PROTOCOL_VERSION
    assert stage11._model_reporting_role(common.RELIABILITY_ARCHITECTURE).startswith(
        "proposed_"
    )


def test_confirmatory_seed_plan_reuses_fixed_hpo_configuration() -> None:
    parameters = {"D_MODEL": 64, "MODALITY_DROPOUT": 0.25}
    plan = stage14._confirmatory_seed_plan(
        [101, 202],
        {"split_seed": 7, "training_seed": 11, "sampler_seed": 13},
        {
            common.RELIABILITY_ARCHITECTURE: {
                "production_params": parameters
            }
        },
    )
    assert [record["split_seed"] for record in plan["repeats"]] == [101, 202]
    assert all(record["rehpo"] is False for record in plan["repeats"])
    assert plan["production_parameters"] == parameters
    assert "without_rehpo" in plan["execution_policy"]


def test_bundle_round_trip_preserves_reliability_contract(tmp_path: Path) -> None:
    # Use the production ESM width because load_deep_bundle reconstructs from
    # the globally configured ESM dimension.
    with common.temporary_model_config(
        {
            "D_MODEL": 64,
            "DROPOUT": 0.0,
            "MODALITY_DROPOUT": 0.2,
            "RELIABILITY_RESIDUAL_SCALE": 1.25,
        }
    ):
        member = common.ReliabilityResidualNet(
            len(FEATURE_NAMES), FEATURE_NAMES
        )
        ensemble = common.EnsembleModel([member])
        preprocessors = common.fit_preprocessors(
            np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((2, common.ESM_DIM), dtype=np.float32),
            common.reliability_passthrough_indices(FEATURE_NAMES),
        )
        path = tmp_path / "reliability_residual.pt"
        common.save_deep_bundle(
            path,
            ensemble,
            preprocessors,
            FEATURE_NAMES,
            0.5,
            common.RELIABILITY_ARCHITECTURE,
        )
    loaded, _, names, threshold, metadata = common.load_deep_bundle(path)
    assert names == FEATURE_NAMES
    assert threshold == 0.5
    assert metadata["reliability_protocol"]["safe_fallback"].startswith("exact_")
    assert loaded.members[0].modality_dropout == pytest.approx(0.2)
    assert loaded.members[0].residual_scale == pytest.approx(1.25)
