"""Regression specifications from the static review; no pretrained downloads."""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef, recall_score

import common as C


stage01 = importlib.import_module("01_dbnsfp_processor")
stage09 = importlib.import_module("09_prepare_external_esm_dataset")
stage10 = importlib.import_module("10_extract_esm_features")
stage12 = importlib.import_module("12_external_validation")
stage13 = importlib.import_module("13_generate_figures")

FEATURES = [
    "esm_variant_score", "GERP++_RS", "phyloP100way_vertebrate",
    "phastCons100way_vertebrate", "SASA", "LOCAL_MEAN_PLDDT_8A",
    "aa_charge_delta", "HAS_STRUCTURE", "LOW_CONFIDENCE_STRUCTURE",
    "HAS_DOMAIN_ANNOTATION", "GERP++_RS__missing",
    "phyloP100way_vertebrate__missing", "phastCons100way_vertebrate__missing",
]


def test_entirely_missing_training_features_use_zero_without_warnings() -> None:
    values = np.array([[1., np.nan, np.inf], [3., np.nan, -np.inf]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fitted = C.ArrayPreprocessor.fit(values)
    np.testing.assert_array_equal(fitted.median, [2., 0., 0.])
    np.testing.assert_allclose(fitted.transform(values), [[-1., 0., 0.], [1., 0., 0.]])
    assert np.isnan(values[:, 1]).all()  # The default preserves caller data.


@pytest.mark.parametrize("validate_upstream", [True, False])
def test_extraction_manifest_reports_actual_upstream_validation(tmp_path, monkeypatch, validate_upstream) -> None:
    frame = pd.DataFrame({
        "row_id": ["example"], "protein_sequence": ["ACD"],
        "aa_pos": [2], "aa_ref": ["C"], "aa_alt": ["A"],
        "sequence_hash": ["fixture"], "genename": ["G"],
        "split_group": ["G"], "homology_cluster": ["G"], "LABEL_PATHOGENIC": [1],
    })
    input_path = tmp_path / "input.parquet"
    frame.to_parquet(input_path, index=False)
    task = stage10.ExtractionTask(
        "internal", input_path, tmp_path / "output.parquet", tmp_path / "embeddings.npy",
        tmp_path / "status.parquet", tmp_path / "manifest.json", tmp_path / "cache",
    )
    calls, recorded = [], {}
    monkeypatch.setattr(stage10, "ESM_EMBED_DIM", 2)
    monkeypatch.setattr(stage10, "STAGE10_OUT", tmp_path)
    monkeypatch.setattr(stage10, "_hardware_summary", lambda: {})
    monkeypatch.setattr(stage10, "_validate_task_upstream", lambda task: calls.append(task.name))

    def cached(task, item, scoring_mode, embeddings, wt, masked, statuses):
        embeddings[:] = 1.
        wt[:] = masked[:] = -1.
        statuses["extraction_status"] = "cached"
        return True

    def manifest(path, stage, inputs, extra, **kwargs):
        recorded.update(extra)

    monkeypatch.setattr(stage10, "_load_cache", cached)
    monkeypatch.setattr(stage10, "write_run_manifest", manifest)
    stage10.extract_task(task, object(), SimpleNamespace(get_batch_converter=lambda: None), "both",
                         validate_upstream=validate_upstream)
    assert calls == (["internal"] if validate_upstream else [])
    assert recorded["validated_upstream_stage"] == (
        "08b_build_homology_groups" if validate_upstream else "explicitly_skipped_nonpublication"
    )


@pytest.mark.parametrize("values", [[0, 0.5, 1], [0, 1.9], [0, np.inf], [[0], [1]]])
def test_labels_are_validated_before_integer_conversion(values) -> None:
    with pytest.raises(ValueError):
        C.validate_binary_labels(np.asarray(values))


def test_single_class_evaluation_is_explicit() -> None:
    with pytest.raises(ValueError):
        C.validate_binary_labels(np.ones(3))
    assert C.validate_binary_labels(np.ones(3), require_both_classes=False).tolist() == [1, 1, 1]


def test_signed_conservation_minus_one_is_observed() -> None:
    cleaned = stage01._clean_numeric(pd.DataFrame({
        "GERP++_RS": ["-1", ".", "inf"],
        "phyloP100way_vertebrate": ["-1", ".", "-inf"],
        "gnomAD4.1_joint_AF": ["-1", ".", "0.1"],
    }))
    for name in ("GERP++_RS", "phyloP100way_vertebrate"):
        assert cleaned.loc[0, name] == -1.0
        assert cleaned.loc[1:, name].isna().all()
    assert np.isnan(cleaned.loc[0, "gnomAD4.1_joint_AF"])


@pytest.mark.parametrize("architecture", C.RELIABILITY_FAMILY_ARCHITECTURES)
def test_sequence_only_schema_cannot_inherit_training_availability(architecture) -> None:
    training = np.ones((2, len(FEATURES)), dtype=np.float32)
    training[:, FEATURES.index("LOCAL_MEAN_PLDDT_8A")] = 95.0
    preprocessor = C.ArrayPreprocessor.fit_with_passthrough(
        training, C.reliability_passthrough_indices(FEATURES)
    )
    external = stage09._align_external_schema(pd.DataFrame({
        "esm_variant_score": [-3.0, 2.0], "aa_charge_delta": [0.0, 1.0],
    }))
    transformed = preprocessor.transform(external[FEATURES].to_numpy(dtype=np.float32))
    assert transformed[:, FEATURES.index("HAS_STRUCTURE")].tolist() == [0.0, 0.0]
    assert transformed[:, FEATURES.index("esm_variant_score")].tolist() == [-3.0, 2.0]
    with C.temporary_model_config({"D_MODEL": 64, "DROPOUT": 0.0}):
        model_type = C.EvidentialResidualNet if architecture == C.EVIDENTIAL_RESIDUAL_ARCHITECTURE else C.ReliabilityResidualNet
        model = model_type(len(FEATURES), FEATURES, esm_dim=4).eval()
    parts = model.forward_components(torch.from_numpy(transformed), torch.zeros(2, 4))
    assert torch.equal(parts["gate"], torch.zeros(2))
    assert torch.equal(parts["logits"], parts["anchor_logit"])


def test_expert_gate_is_bounded_by_observed_auxiliary_quality() -> None:
    with C.temporary_model_config({
        "D_MODEL": 64, "DROPOUT": 0.0, "EVIDENTIAL_RELIABILITY_POWER": 1.0,
    }):
        model = C.EvidentialResidualNet(len(FEATURES), FEATURES, esm_dim=4).eval()
    bio = torch.zeros(2, len(FEATURES))
    for name in FEATURES:
        if name.endswith("__missing"):
            bio[:, FEATURES.index(name)] = 1.0
    bio[1, FEATURES.index("HAS_STRUCTURE")] = 1.0
    bio[1, FEATURES.index("LOCAL_MEAN_PLDDT_8A")] = 0.5
    parts = model.forward_components(bio, torch.zeros(2, 4))
    quality = torch.maximum(parts["structure_reliability"], parts["conservation_reliability"])
    assert torch.all(parts["gate"] <= quality)
    assert 0.0 < parts["gate"][1].item() <= 0.005
    assert torch.equal(parts["expert_weights"][:, 1], torch.zeros(2))
    assert torch.isfinite(parts["logits"]).all()
    degraded = model.rcdi_components(bio[:1], torch.zeros(1, 4), mask_probability=1.0)
    assert degraded["consistency_weight"].item() == 0.0


@pytest.mark.parametrize("stored_power", [None, 1.0])
def test_expert_bundle_restores_legacy_or_explicit_gate_rule(stored_power, monkeypatch) -> None:
    expected_power = 0.0 if stored_power is None else stored_power
    settings = {
        "D_MODEL": 64, "DROPOUT": 0.0, "MODALITY_DROPOUT": 0.0,
        "EVIDENTIAL_RELIABILITY_POWER": expected_power,
    }
    with C.temporary_model_config(settings):
        original = C.EvidentialResidualNet(len(FEATURES), FEATURES).eval()
        configuration = {name: getattr(C, name) for name in C.TUNABLE_CONSTANTS}
    if stored_power is None:
        configuration.pop("EVIDENTIAL_RELIABILITY_POWER")
    preprocessors = C.fit_preprocessors(
        np.zeros((2, len(FEATURES)), dtype=np.float32),
        np.zeros((2, C.ESM_DIM), dtype=np.float32),
        C.reliability_passthrough_indices(FEATURES),
    )
    payload = {
        "architecture": C.EVIDENTIAL_RESIDUAL_ARCHITECTURE,
        "feature_names": FEATURES, "model_config": configuration,
        "model_states": [original.state_dict()], "preprocessors": preprocessors,
        "threshold": 0.5,
    }
    monkeypatch.setattr(C, "_torch_load", lambda _: payload)
    monkeypatch.setattr(C, "DEVICE", "cpu")
    monkeypatch.setattr(C, "EVIDENTIAL_RELIABILITY_POWER", 2.0)
    restored, *_ = C.load_deep_bundle(Path("unused.pt"))
    restored.eval()
    assert restored.members[0].reliability_power == expected_power
    assert C.EVIDENTIAL_RELIABILITY_POWER == 2.0
    bio = torch.zeros(2, len(FEATURES))
    bio[:, FEATURES.index("HAS_STRUCTURE")] = 1.0
    bio[:, FEATURES.index("LOCAL_MEAN_PLDDT_8A")] = 50.0
    for name in FEATURES:
        if name.endswith("__missing"):
            bio[:, FEATURES.index(name)] = 1.0
    esm = torch.zeros(2, C.ESM_DIM)
    torch.testing.assert_close(restored(bio, esm), original(bio, esm), rtol=0.0, atol=0.0)


def test_plain_bce_ablation_is_described_as_plain_bce() -> None:
    protocol = C.reliability_architecture_protocol(
        FEATURES, {"RCDI_ANCHOR_WEIGHT": 0.0, "RCDI_CONSISTENCY_WEIGHT": 0.0},
        architecture=C.EVIDENTIAL_RESIDUAL_ARCHITECTURE,
    )
    assert protocol["training_objective"] == "binary_cross_entropy"


@pytest.mark.parametrize("scores", [
    [0.0, 0.0, 0.2, 0.2, 0.8, 0.8, 1.0, 1.0],
    [0.5] * 8,
    [0.9, 0.1, 0.7, 0.3, 0.6, 0.2, 0.4, 0.8],
])
def test_threshold_selection_matches_exhaustive_reference(scores) -> None:
    labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
    scores = np.asarray(scores)
    grid = np.unique(np.r_[np.linspace(0.01, 0.99, 393), scores])
    feasible = [cut for cut in grid if recall_score(labels, scores >= cut) >= C.RECALL_FLOOR]
    reference = max(feasible, key=lambda cut: matthews_corrcoef(labels, scores >= cut))
    assert C.select_threshold(labels, scores) == reference

    grid = np.unique(np.r_[0.0, 1.0, scores])
    reference_mcc = max(grid, key=lambda cut: matthews_corrcoef(labels, scores >= cut))
    actual = C.select_operating_points(labels, scores, sensitivity_targets=(0.8,))
    candidates = []
    for cut in grid:
        tn, fp, fn, tp = confusion_matrix(labels, scores >= cut, labels=[0, 1]).ravel()
        if tp / (tp + fn) + 1e-12 >= 0.8:
            candidates.append((tn / (tn + fp), cut))
    assert actual == {"max_mcc": reference_mcc, "sensitivity_80": max(candidates)[1]}


@pytest.mark.parametrize("scores", [[0.2, np.nan], [-0.1, 0.8], [[0.2], [0.8]]])
def test_thresholds_reject_malformed_probabilities(scores) -> None:
    with pytest.raises(ValueError):
        C.select_threshold(np.asarray([0, 1]), np.asarray(scores))


def test_preprocessor_copy_false_accepts_dtype_conversion_and_readonly_arrays() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    processor = C.ArrayPreprocessor.fit(values, copy=False)
    values.setflags(write=False)
    actual = processor.transform(values, copy=False)
    np.testing.assert_array_equal(actual, [[-1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(values, [[1.0, 2.0], [3.0, 4.0]])


def test_training_uses_scoped_epoch_budget_without_real_training(monkeypatch) -> None:
    seen = []

    def fake_train(*args):
        seen.append((args[8], args[9]))
        return torch.nn.Linear(1, 1), 0.5, 1

    monkeypatch.setattr(C, "_train_single", fake_train)
    monkeypatch.setattr(C, "DEVICE", "cpu")
    bio = np.zeros((2, 1), dtype=np.float32)
    esm = np.zeros((2, C.ESM_DIM), dtype=np.float32)
    labels = np.asarray([0, 1])
    with C.temporary_model_config({"DEEP_MAX_EPOCHS": 7, "DEEP_PATIENCE": 2}):
        C.train_deep_model(bio, esm, labels, bio, esm, labels, seeds=[17])
    assert seen == [(7, 2)]
    with pytest.raises(ValueError, match="misaligned"):
        C.train_deep_model(bio[:1], esm, labels, bio, esm, labels, seeds=[17])
    assert len(seen) == 1


def test_inference_checks_prebuilt_loader_row_count_and_finiteness(monkeypatch) -> None:
    class Logits(torch.nn.Module):
        def forward(self, bio, esm):
            return bio[:, 0]

    monkeypatch.setattr(C, "DEVICE", "cpu")
    monkeypatch.setattr(C, "_parallel_model", lambda model: model)
    batch = (torch.zeros(2, 1), torch.zeros(2, 4), torch.zeros(2))
    placeholder = np.empty((0, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 3"):
        C.predict_logits(Logits(), placeholder, placeholder, loader=[batch], num_rows=3)
    batch[0][1, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        C.predict_logits(Logits(), placeholder, placeholder, loader=[batch], num_rows=2)


def test_mask_attention_override_is_a_quadratic_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(stage10, "ESM_MAX_MASKS_PER_BATCH", 8)
    monkeypatch.setattr(stage10, "ESM_MAX_BATCH_TOKENS", 512)
    monkeypatch.setattr(stage10, "ESM_MASK_BATCH_ATTENTION", 32768)
    assert stage10._mask_batch_capacity(128) == 2
    with pytest.raises(ValueError, match="budget"):
        stage10._mask_batch_capacity(256)


def test_cache_authenticates_content_and_remaps_current_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(stage10, "DEVICE", "cpu")
    monkeypatch.setattr(stage10, "ESM_EMBED_DIM", 4)
    task = SimpleNamespace(cache_dir=tmp_path)
    original = stage10.ContextItem("ACD", 1, 3, [(0, 1, "A", "V"), (1, 2, "C", "D")])
    remapped = stage10.ContextItem("ACD", 1, 3, [(5, 1, "A", "V"), (3, 2, "C", "D")])
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    stage10._save_cache(task, original, "masked-marginal", source, np.full(2, np.nan), np.asarray([-0.3, 0.8]))
    embeddings = np.full((6, 4), np.nan, dtype=np.float32)
    wt = np.full(6, np.nan)
    masked = np.full(6, np.nan)
    status = pd.DataFrame({"extraction_status": ["pending"] * 6, "error_type": [""] * 6})
    assert stage10._load_cache(task, remapped, "masked-marginal", embeddings, wt, masked, status)
    np.testing.assert_array_equal(embeddings[[5, 3]], source)
    np.testing.assert_allclose(masked[[5, 3]], [-0.3, 0.8])
    assert np.isnan(embeddings[0]).all()
    changed = stage10.ContextItem("ACD", 1, 3, [(5, 1, "A", "L"), (3, 2, "C", "D")])
    assert not stage10._load_cache_from(
        stage10._cache_path(task, original, "masked-marginal"), changed,
        "masked-marginal", embeddings, wt, masked, status,
    )


def test_failed_legacy_cache_migration_keeps_original(tmp_path, monkeypatch) -> None:
    task = SimpleNamespace(cache_dir=tmp_path)
    item = stage10.ContextItem("A", 1, 1, [(0, 1, "A", "V")])
    legacy = stage10._legacy_cache_path(task, item, "masked-marginal")
    legacy.write_bytes(b"legacy")
    monkeypatch.setattr(stage10, "_load_cache_from", lambda *args: True)

    def fail_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr(stage10, "_save_cache", fail_save)
    assert stage10._load_cache(task, item, "masked-marginal", None, None, None, None)
    assert legacy.read_bytes() == b"legacy"


def test_multiple_variant_errors_are_recorded_without_ragged_assignment() -> None:
    item = stage10.ContextItem("AC", 1, 2, [(0, 1, "A", "V"), (1, 2, "C", "D")])

    def missing_token(_):
        raise KeyError("token")

    status = pd.DataFrame({"extraction_status": ["pending"] * 2, "error_type": [""] * 2})
    stage10._fill_item(
        item, None, None, None, SimpleNamespace(get_idx=missing_token), None, 0,
        "wt-marginal", np.zeros((2, 4)), np.zeros(2), np.zeros(2), status,
    )
    assert status["extraction_status"].tolist() == ["variant_failed"] * 2
    assert status["error_type"].tolist() == ["KeyError"] * 2


def test_both_family_models_are_routed_to_external_validation_and_figures() -> None:
    for architecture in C.RELIABILITY_FAMILY_ARCHITECTURES:
        assert architecture in stage12.DEEP_MODEL_NAMES
        assert architecture in stage12.MODEL_DISCOVERY_ORDER
        assert architecture in stage13.MODEL_LABELS
        assert architecture in stage13.MODEL_COLORS
        assert architecture in stage13.REQUIRED_STAGE14_ARCHITECTURES | stage13.OPTIONAL_STAGE14_ARCHITECTURES


def test_lora_loader_rejects_missing_head_weights_without_downloading(monkeypatch) -> None:
    alphabet = SimpleNamespace(prepend_bos=True)

    def backbone():
        module = torch.nn.Module()
        module.embed_dim = 4
        module.q_proj = torch.nn.Linear(4, 4)
        return module

    source = backbone()
    C.inject_lora(source)
    model = C.ESMVariantClassifier(source, alphabet)
    state = {name: value for name, value in model.state_dict().items() if not name.startswith("head.")}
    payload = {"state": state, "temperature": 1.0, "threshold": 0.5}
    monkeypatch.setitem(sys.modules, "esm", SimpleNamespace(pretrained=SimpleNamespace(
        load_model_and_alphabet=lambda _: (backbone(), alphabet)
    )))
    monkeypatch.setattr(C, "ENABLE_LORA", True)
    monkeypatch.setattr(C, "DEVICE", "cpu")
    monkeypatch.setattr(C, "_torch_load", lambda _: payload)
    with pytest.raises(ValueError, match="Missing LoRA adapter/head"):
        C.load_lora_bundle(Path("unused.pt"))


def test_lora_uses_projection_forward_and_preserves_base_dtype() -> None:
    attention = torch.nn.Module()
    attention.q_proj = torch.nn.Linear(4, 4, dtype=torch.float64)
    attention.enable_torch_version = True
    assert C.inject_lora(attention) == 1
    assert attention.enable_torch_version is False
    assert attention.q_proj.adapter_a.weight.dtype == torch.float64
    assert attention.q_proj.adapter_b.weight.dtype == torch.float64
