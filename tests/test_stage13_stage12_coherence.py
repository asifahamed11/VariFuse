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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refresh_output_record(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        artifact_id
        for artifact_id in manifest["outputs"]
        if artifact_id.endswith("/" + output_path.name)
    ]
    assert len(matches) == 1
    artifact_id = matches[0]
    record = stage13.artifact_record(output_path)
    record["artifact_id"] = artifact_id
    manifest["outputs"][artifact_id] = record
    _write_json(manifest_path, manifest)


def _source_hashes(names: tuple[str, ...]) -> dict[str, str | None]:
    source_dir = Path(stage13.__file__).resolve().parent
    return {name: stage13.file_sha256(source_dir / name) for name in names}


def _probability_metrics(labels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    decisions = (values >= 0.5).astype(np.int8)
    metrics = stage13.C.evaluate(
        labels,
        values,
        0.5,
        predictions=decisions,
        decision_confidence=np.ones(len(labels), dtype=float),
    )
    metrics["ci95"] = None
    metrics["output_scale"] = "calibrated_probability"
    metrics["model_role"] = "reference_baseline"
    return metrics


def _reliability_fixture(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    source: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pattern = np.resize(np.asarray([0, 1, 2, 3], dtype=np.int8), len(labels))
    structure = np.isin(pattern, (1, 3)).astype(np.float32)
    conservation = np.isin(pattern, (2, 3)).astype(np.float32)
    hard_availability = ((structure + conservation) > 0).astype(np.float32)
    gate = (0.2 * hard_availability).astype(np.float32)
    anchor_probability = probabilities.astype(np.float32)
    components = {
        "anchor_logit": np.log(
            anchor_probability / (1.0 - anchor_probability)
        ).astype(np.float32),
        "anchor_probability": anchor_probability,
        "gate": gate,
        "bounded_residual": np.zeros(len(labels), dtype=np.float32),
        "hard_availability": hard_availability,
        "structure_reliability": structure,
        "conservation_reliability": conservation,
    }
    decisions = (probabilities >= 0.5).astype(np.int8)
    summary = stage13.C.summarize_reliability_diagnostics(
        components,
        labels,
        probabilities,
        0.5,
        decisions,
        anchor_decisions=decisions,
    )
    summary["evaluation_scope"] = (
        "external_frozen_deployment_folds_descriptive_only"
    )
    summary["fold_aggregation"] = (
        "arithmetic_mean_component_and_calibrated_probability_across_five_"
        "frozen_internal_deployment_folds"
    )
    summary["analysis_unit_aggregation"] = (
        "mean_across_retained_annotation_rows_after_fold_prediction"
        if source == "clinvar"
        else "none_assay_variant_row"
    )
    return components, summary


def _stage12_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    outputs = tmp_path / "outputs"
    stage09_dir = outputs / "09_prepare_external_esm"
    stage07_dir = outputs / "07_natural_prevalence"
    stage10_dir = outputs / "10_esm_features"
    stage11_dir = outputs / "11_train_and_evaluate"
    stage12_dir = outputs / "12_external_validation"
    stage14_dir = outputs / "14_tuning"
    for directory in (
        stage07_dir,
        stage09_dir,
        stage10_dir,
        stage11_dir,
        stage12_dir,
        stage14_dir,
    ):
        directory.mkdir(parents=True)

    internal_universe = stage07_dir / "Final_Dataset_Natural_Prevalence.parquet"
    pd.DataFrame({schema.ROW_ID_COL: ["internal0"]}).to_parquet(
        internal_universe, index=False
    )
    stage07_manifest = stage07_dir / "run_manifest.json"
    _write_json(stage07_manifest, {"stage": "07_dataset_balancing"})

    cv_ids = np.asarray([f"cv{index}" for index in range(4)])
    cv_labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    cv_groups = np.asarray(["G1", "G1", "G2", "G2"])
    cv_scores = np.asarray([0.1, 0.9, 0.2, 0.8])
    cv_decisions = (cv_scores >= 0.5).astype(np.int8)

    dms_ids = np.asarray([f"dms{index}" for index in range(8)])
    dms_labels = np.asarray([1, 1, 0, 0] * 2, dtype=np.int8)
    dms_groups = np.asarray(["A"] * 4 + ["B"] * 4)
    dms_scores = np.asarray([0.9, 0.8, 0.2, 0.1] * 2)
    dms_decisions = (dms_scores >= 0.5).astype(np.int8)
    functional_scores = np.asarray([0.0, 1.0, 2.0, 3.0] * 2)

    external_inputs: dict[str, tuple[Path, Path, Path]] = {}
    for source, row_ids, labels, groups in (
        ("clinvar", cv_ids, cv_labels, cv_groups),
        ("dms", dms_ids, dms_labels, dms_groups),
    ):
        table = stage10_dir / f"{source}_with_esm.parquet"
        embeddings = stage10_dir / f"{source}_esm_embeddings.npy"
        status = stage10_dir / f"{source}_esm_extraction.parquet"
        if source == "clinvar":
            frame = pd.DataFrame(
                {
                    schema.ROW_ID_COL: row_ids,
                    schema.LABEL_COL: labels,
                    schema.GENE_COL: groups,
                }
            )
        else:
            frame = pd.DataFrame(
                {
                    schema.ROW_ID_COL: row_ids,
                    schema.LABEL_COL: labels,
                    schema.GENE_COL: ["DMS_GENE"] * len(row_ids),
                    "ASSAY_ID": groups,
                    "variant_id": [f"v{index}" for index in range(len(row_ids))],
                    "DMS_SCORE": functional_scores,
                }
            )
        frame.to_parquet(table, index=False)
        np.save(embeddings, np.zeros((len(frame), 3), dtype=np.float32))
        pd.DataFrame(
            {schema.ROW_ID_COL: row_ids, "success": np.ones(len(frame), dtype=int)}
        ).to_parquet(status, index=False)
        external_inputs[source] = (table, embeddings, status)

    deployment_results = stage11_dir / "results.json"
    deployment_manifest = stage11_dir / "run_manifest.json"
    model_names = ["lightgbm", stage13.C.RELIABILITY_ARCHITECTURE]
    _write_json(
        deployment_results,
        {"primary": {name: {} for name in model_names}},
    )
    _write_json(
        deployment_manifest,
        {
            "stage": "11_train_and_evaluate",
            "label_task": "clinical",
            "model_tag": stage13.MODEL_TAG,
            "source_files": _source_hashes(stage13.STAGE11_PROVENANCE_SOURCES),
            "extra": {"models": model_names},
        },
    )

    stage14_results = stage14_dir / "architecture_selection.json"
    stage14_split = stage14_dir / "nested_inner_splits.json"
    stage14_manifest = stage14_dir / "run_manifest.json"
    _write_json(stage14_results, {"fixture": "result"})
    _write_json(stage14_split, {"fixture": "split"})
    _write_json(stage14_manifest, {"fixture": "manifest"})

    prep_manifest = stage09_dir / "run_manifest.json"
    _write_json(
        prep_manifest,
        {
            "stage": "09_prepare_external_esm_dataset",
            "label_task": "clinical",
            "source_files": _source_hashes(stage13.STAGE09_PROVENANCE_SOURCES),
            "extra": {
                "dms_sampling_policy": "hash_uniform",
                "dms_sampling_uses_label": False,
            },
        },
    )
    prepared_outputs: dict[str, Path] = {}
    stage10_manifests: dict[str, Path] = {}
    for source, ids in (("clinvar", cv_ids), ("dms", dms_ids)):
        prepared = stage09_dir / f"{source}_esm_ready.csv"
        pd.DataFrame({schema.ROW_ID_COL: ids}).to_csv(prepared, index=False)
        prepared_outputs[source] = prepared
        stage10_manifest = stage10_dir / f"{source}_esm_manifest.json"
        _write_json(stage10_manifest, {"stage": f"10_extract_esm_features:{source}"})
        stage10_manifests[source] = stage10_manifest

    cv_metrics = _probability_metrics(cv_labels, cv_scores)
    dms_metrics = _probability_metrics(dms_labels, dms_scores)
    assay_metrics: dict[str, Any] = {}
    for assay in ("A", "B"):
        index = np.flatnonzero(dms_groups == assay)
        metrics = stage13.C.evaluate(
            dms_labels[index],
            dms_scores[index],
            0.5,
            predictions=dms_decisions[index],
            decision_confidence=np.ones(len(index), dtype=float),
        )
        metrics["functional_spearman"] = 1.0
        assay_metrics[assay] = {
            "n": 4,
            "positives": 2,
            "models": {name: json.loads(json.dumps(metrics)) for name in model_names},
        }
    macro = {
        name: {
            metric: assay_metrics["A"]["models"][name].get(metric)
            for metric in ("mcc", "auroc", "auprc", "f1", "functional_spearman")
        }
        for name in model_names
    }
    cv_components, cv_diagnostics = _reliability_fixture(
        cv_labels, cv_scores, source="clinvar"
    )
    dms_components, dms_diagnostics = _reliability_fixture(
        dms_labels, dms_scores, source="dms"
    )
    cv_result = {
        "status": "evaluated",
        "source": "clinvar",
        "policy": "exact_variant_disjoint",
        "role": "primary",
        "analysis_unit": "unique_genomic_variant",
        "n": 4,
        "row_n": 4,
        "positives": 2,
        "genes": 2,
        "assays": None,
        "aggregation_audit": {"unique_genomic_variants": 4},
        "models": {
            name: json.loads(json.dumps(cv_metrics)) for name in model_names
        },
        "reliability_diagnostics": cv_diagnostics,
    }
    dms_result = {
        "status": "evaluated",
        "source": "dms",
        "policy": "exact_variant_disjoint",
        "role": "primary",
        "analysis_unit": "assay_variant_row",
        "n": 8,
        "row_n": 8,
        "positives": 4,
        "genes": 1,
        "assays": 2,
        "models": {
            name: json.loads(json.dumps(dms_metrics)) for name in model_names
        },
        "reliability_diagnostics": dms_diagnostics,
        "per_assay": {
            "available": True,
            "analysis_unit": "assay",
            "assay_count": 2,
            "assays": assay_metrics,
            "macro": macro,
            "macro_ci95": {
                name: {
                    "functional_spearman": [1.0, 1.0],
                    "auroc": [1.0, 1.0],
                }
                for name in model_names
            },
            "sampling_caveats": {
                "manifest_available": True,
                "legacy_fallback": False,
                "sampling_uses_label": False,
                "sampling_policy": "hash_uniform",
            },
        },
        "pooled_row_metrics_are_primary": False,
    }
    results = {
        "model_tag": stage13.MODEL_TAG,
        "models": model_names,
        "prediction_aggregation": (
            "mean_calibrated_probability_with_majority_vote_of_frozen_fold_thresholds"
        ),
        "threshold_source": "dedicated_internal_fold_partitions",
        "primary_evaluation_policy": {
            "clinvar": {"set": "clinvar_exact_variant_disjoint"},
            "dms": {"set": "dms_exact_variant_disjoint"},
        },
        "reliability_diagnostic_policy": {
            "status": "descriptive_post_selection_not_primary",
            "strata_use_labels": False,
            "used_for_model_or_threshold_selection": False,
            "external_label_refitting": False,
            "fold_aggregation": (
                "mean_components_across_frozen_internal_deployment_folds"
            ),
            "component_names": list(stage13.C.RELIABILITY_DIAGNOSTIC_COMPONENTS),
        },
        "sets": {
            "clinvar_exact_variant_disjoint": cv_result,
            "dms_exact_variant_disjoint": dms_result,
        },
    }
    result_path = stage12_dir / "external_validation.json"
    prediction_path = stage12_dir / "external_predictions.npz"
    summary_path = stage12_dir / "external_validation_table.csv"
    prediction_table_path = stage12_dir / "external_predictions.csv"
    manifest_path = stage12_dir / "run_manifest.json"
    _write_json(result_path, results)
    np.savez_compressed(
        prediction_path,
        clinvar_exact_variant_disjoint__y=cv_labels,
        clinvar_exact_variant_disjoint__groups=cv_groups,
        clinvar_exact_variant_disjoint__row_ids=cv_ids,
        clinvar_exact_variant_disjoint__variant_ids=np.asarray(
            [f"1:{index}:A:G" for index in range(4)]
        ),
        clinvar_exact_variant_disjoint__annotation_row_counts=np.ones(
            4, dtype=np.int32
        ),
        clinvar_exact_variant_disjoint__lightgbm=cv_scores,
        clinvar_exact_variant_disjoint__lightgbm__decisions=cv_decisions,
        clinvar_exact_variant_disjoint__lightgbm__decision_confidence=np.ones(4),
        clinvar_exact_variant_disjoint__lightgbm__threshold=np.asarray([0.5]),
        clinvar_exact_variant_disjoint__reliability_residual=cv_scores,
        clinvar_exact_variant_disjoint__reliability_residual__decisions=cv_decisions,
        clinvar_exact_variant_disjoint__reliability_residual__decision_confidence=np.ones(
            4
        ),
        clinvar_exact_variant_disjoint__reliability_residual__threshold=np.asarray(
            [0.5]
        ),
        dms_exact_variant_disjoint__y=dms_labels,
        dms_exact_variant_disjoint__groups=dms_groups,
        dms_exact_variant_disjoint__row_ids=dms_ids,
        dms_exact_variant_disjoint__lightgbm=dms_scores,
        dms_exact_variant_disjoint__lightgbm__decisions=dms_decisions,
        dms_exact_variant_disjoint__lightgbm__decision_confidence=np.ones(8),
        dms_exact_variant_disjoint__lightgbm__threshold=np.asarray([0.5]),
        dms_exact_variant_disjoint__reliability_residual=dms_scores,
        dms_exact_variant_disjoint__reliability_residual__decisions=dms_decisions,
        dms_exact_variant_disjoint__reliability_residual__decision_confidence=np.ones(
            8
        ),
        dms_exact_variant_disjoint__reliability_residual__threshold=np.asarray([0.5]),
        **{
            f"clinvar_exact_variant_disjoint__reliability_components__{name}": values
            for name, values in {
                **cv_components,
                "availability_stratum_code": stage13.C.reliability_availability_codes(
                    cv_components
                ),
                "anchor_decisions": cv_decisions,
            }.items()
        },
        **{
            f"dms_exact_variant_disjoint__reliability_components__{name}": values
            for name, values in {
                **dms_components,
                "availability_stratum_code": stage13.C.reliability_availability_codes(
                    dms_components
                ),
                "anchor_decisions": dms_decisions,
            }.items()
        },
    )
    pd.DataFrame(
        [
            {
                "source": source,
                "policy": "exact_variant_disjoint",
                "model": model,
                "n": n,
                "positives": positives,
            }
            for source, n, positives in (("clinvar", 4, 2), ("dms", 8, 4))
            for model in model_names
        ]
    ).to_csv(summary_path, index=False)
    prediction_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "evaluation_set": "clinvar_exact_variant_disjoint",
                    schema.ROW_ID_COL: cv_ids,
                    schema.LABEL_COL: cv_labels,
                    schema.GENE_COL: cv_groups,
                    "ASSAY_ID": [None] * 4,
                    "lightgbm_probability": cv_scores,
                    "lightgbm_decision": cv_decisions,
                    "reliability_residual_probability": cv_scores,
                    "reliability_residual_decision": cv_decisions,
                }
            ),
            pd.DataFrame(
                {
                    "evaluation_set": "dms_exact_variant_disjoint",
                    schema.ROW_ID_COL: dms_ids,
                    schema.LABEL_COL: dms_labels,
                    schema.GENE_COL: ["DMS_GENE"] * 8,
                    "ASSAY_ID": dms_groups,
                    "lightgbm_probability": dms_scores,
                    "lightgbm_decision": dms_decisions,
                    "reliability_residual_probability": dms_scores,
                    "reliability_residual_decision": dms_decisions,
                }
            ),
        ],
        ignore_index=True,
    )
    prediction_rows.to_csv(prediction_table_path, index=False)
    reliability_path = stage12_dir / "reliability_diagnostics.csv"
    reliability_rows = [
        *stage13.C.reliability_diagnostic_rows(
            cv_diagnostics,
            {
                "scope": "external_validation",
                "source": "clinvar",
                "policy": "exact_variant_disjoint",
                "role": "primary",
                "analysis_unit": "unique_genomic_variant",
                "model": stage13.C.RELIABILITY_ARCHITECTURE,
            },
        ),
        *stage13.C.reliability_diagnostic_rows(
            dms_diagnostics,
            {
                "scope": "external_validation",
                "source": "dms",
                "policy": "exact_variant_disjoint",
                "role": "primary",
                "analysis_unit": "assay_variant_row",
                "model": stage13.C.RELIABILITY_ARCHITECTURE,
            },
        ),
    ]
    pd.DataFrame(reliability_rows).to_csv(reliability_path, index=False)

    input_paths = [
        stage07_manifest,
        internal_universe,
        prep_manifest,
        deployment_results,
        deployment_manifest,
        stage14_results,
        stage14_split,
        stage14_manifest,
        *(path for paths in external_inputs.values() for path in paths),
        *prepared_outputs.values(),
        *stage10_manifests.values(),
    ]
    _write_json(
        manifest_path,
        {
            "artifact_manifest_version": 2,
            "stage": "12_external_validation",
            "label_task": "clinical",
            "model_tag": stage13.MODEL_TAG,
            "external_contract": {"require_clinvar": True, "require_dms": True},
            "source_files": _source_hashes(stage13.STAGE12_PROVENANCE_SOURCES),
            "inputs": {
                str(path): stage13.artifact_record(path) for path in input_paths
            },
            "outputs": {
                artifact_id: {
                    **stage13.artifact_record(path),
                    "artifact_id": artifact_id,
                }
                for path in (
                    result_path,
                    prediction_path,
                    summary_path,
                    prediction_table_path,
                    reliability_path,
                )
                for artifact_id in [f"12_external_validation/{path.name}"]
            },
            "extra": {
                "models": model_names,
                "clinvar_and_dms_pooled": False,
                "exact_variant_deoverlap": True,
                "primary_sets": [
                    "clinvar_exact_variant_disjoint",
                    "dms_exact_variant_disjoint",
                ],
                "sets": {
                    "clinvar_exact_variant_disjoint": 4,
                    "dms_exact_variant_disjoint": 8,
                },
                "reliability_diagnostics": {
                    "status": "descriptive_post_selection_not_primary",
                    "strata_use_labels": False,
                    "used_for_model_or_threshold_selection": False,
                    "component_names": list(
                        stage13.C.RELIABILITY_DIAGNOSTIC_COMPONENTS
                    ),
                },
            },
        },
    )

    monkeypatch.setattr(stage13, "REQUIRE_EXTERNAL_CLINVAR", True)
    monkeypatch.setattr(stage13, "REQUIRE_EXTERNAL_DMS", True)
    monkeypatch.setattr(stage13, "EXTERNAL_RESULTS", result_path)
    monkeypatch.setattr(stage13, "EXTERNAL_PREDICTIONS", prediction_path)
    monkeypatch.setattr(stage13, "EXTERNAL_SUMMARY_TABLE", summary_path)
    monkeypatch.setattr(stage13, "EXTERNAL_PREDICTION_TABLE", prediction_table_path)
    monkeypatch.setattr(
        stage13, "EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE", reliability_path
    )
    monkeypatch.setattr(stage13, "EXTERNAL_RUN_MANIFEST", manifest_path)
    monkeypatch.setattr(stage13, "EXTERNAL_PREP_MANIFEST", prep_manifest)
    monkeypatch.setattr(stage13, "STAGE07_MANIFEST", stage07_manifest)
    monkeypatch.setattr(stage13, "INTERNAL_UNIVERSE", internal_universe)
    monkeypatch.setattr(stage13, "STAGE09_PREPARED_OUTPUTS", prepared_outputs)
    monkeypatch.setattr(stage13, "STAGE10_MANIFESTS", stage10_manifests)
    monkeypatch.setattr(stage13, "DEPLOYMENT_RESULTS", deployment_results)
    monkeypatch.setattr(stage13, "DEPLOYMENT_RUN_MANIFEST", deployment_manifest)
    monkeypatch.setattr(stage13, "INTERNAL_RESULTS", stage14_results)
    monkeypatch.setattr(stage13, "INTERNAL_SPLIT_PLAN", stage14_split)
    monkeypatch.setattr(stage13, "INTERNAL_RUN_MANIFEST", stage14_manifest)
    monkeypatch.setattr(stage13, "EXTERNAL_STAGE10_INPUTS", external_inputs)
    stage14_report = {
        "architecture_results_sha256": stage13.file_sha256(stage14_results),
        "split_plan_file_sha256": stage13.file_sha256(stage14_split),
    }
    return {
        "report": stage14_report,
        "results": results,
        "result_path": result_path,
        "prediction_path": prediction_path,
        "reliability_path": reliability_path,
        "npz": dict(np.load(prediction_path)),
        "prep_path": prep_manifest,
        "manifest_path": manifest_path,
    }


def test_stage13_accepts_coherent_stage12_publication_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)

    report = stage13._validate_stage12_artifacts(fixture["report"])

    assert report["status"] == "validated"
    assert report["publication_complete"] is True
    assert report["validated_sources"] == ["clinvar", "dms"]
    assert report["sets"]["clinvar"]["n"] == 4
    assert report["sets"]["dms"]["assays"] == 2
    reliability = report["reliability_diagnostic_reporting"]
    assert reliability["status"] == (
        "validated_descriptive_post_selection_not_primary"
    )
    assert reliability["strata_use_labels"] is False
    assert reliability["not_primary_or_confirmatory"] is True
    assert set(reliability["sets"]) == {
        "clinvar_exact_variant_disjoint",
        "dms_exact_variant_disjoint",
    }

    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    monkeypatch.setattr(stage13, "STAGE13_OUT", figure_dir)
    table_path = stage13._write_mechanistic_reliability_table(report)
    assert table_path is not None
    table = pd.read_csv(table_path)
    assert len(table) == 8
    assert table["reporting_scope"].eq(
        "descriptive_post_selection_not_primary"
    ).all()
    assert table["not_primary_or_confirmatory"].eq(True).all()
    assert table["primary_endpoint"].eq(False).all()
    assert table["confirmatory_evidence"].eq(False).all()
    assert table["strata_use_labels"].eq(False).all()
    assert table["inferential_claim_allowed"].eq(False).all()


@pytest.mark.parametrize(
    "unsafe_guard",
    ["external_refit", "label_based_strata", "model_selection"],
)
def test_stage13_rejects_label_based_or_refitted_reliability_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_guard: str,
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)
    if unsafe_guard == "external_refit":
        fixture["results"]["reliability_diagnostic_policy"][
            "external_label_refitting"
        ] = True
    elif unsafe_guard == "label_based_strata":
        fixture["results"]["sets"]["clinvar_exact_variant_disjoint"][
            "reliability_diagnostics"
        ]["stratification"]["uses_labels"] = True
    else:
        fixture["results"]["sets"]["clinvar_exact_variant_disjoint"][
            "reliability_diagnostics"
        ]["selection_or_refitting"]["used_for_model_selection"] = True
    _write_json(fixture["result_path"], fixture["results"])
    _refresh_output_record(fixture["manifest_path"], fixture["result_path"])

    with pytest.raises(RuntimeError, match="publication-safe|reported metric"):
        stage13._validate_stage12_artifacts(fixture["report"])


def test_stage13_rejects_label_incoherent_reliability_stratum_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)
    arrays = fixture["npz"]
    key = (
        "clinvar_exact_variant_disjoint__reliability_components__"
        "availability_stratum_code"
    )
    arrays[key] = arrays[key].copy()
    arrays[key][0] = 3
    np.savez_compressed(fixture["prediction_path"], **arrays)
    _refresh_output_record(fixture["manifest_path"], fixture["prediction_path"])

    with pytest.raises(RuntimeError, match="not label-independent"):
        stage13._validate_stage12_artifacts(fixture["report"])


def test_stage13_rejects_stale_reliability_csv_or_manifest_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)
    table = pd.read_csv(fixture["reliability_path"])
    table.loc[0, "n"] = 999
    table.to_csv(fixture["reliability_path"], index=False)
    _refresh_output_record(fixture["manifest_path"], fixture["reliability_path"])
    with pytest.raises(RuntimeError, match="CSV differs from JSON/NPZ"):
        stage13._validate_stage12_artifacts(fixture["report"])

    fixture = _stage12_fixture(tmp_path / "binding", monkeypatch)
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["outputs"].pop(
        "12_external_validation/reliability_diagnostics.csv"
    )
    _write_json(fixture["manifest_path"], manifest)
    with pytest.raises(RuntimeError, match="does not bind portable output"):
        stage13._validate_stage12_artifacts(fixture["report"])


def test_stage13_rejects_external_json_npz_metric_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)
    fixture["results"]["sets"]["clinvar_exact_variant_disjoint"]["models"][
        "lightgbm"
    ]["auroc"] = 0.1234
    _write_json(fixture["result_path"], fixture["results"])
    _refresh_output_record(fixture["manifest_path"], fixture["result_path"])

    with pytest.raises(RuntimeError, match="clinvar_exact_variant_disjoint.lightgbm.auroc"):
        stage13._validate_stage12_artifacts(fixture["report"])


def test_stage13_requires_fold_vote_confidence_for_risk_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)
    arrays = fixture["npz"]
    arrays.pop(
        "clinvar_exact_variant_disjoint__lightgbm__decision_confidence"
    )
    np.savez_compressed(fixture["prediction_path"], **arrays)
    _refresh_output_record(fixture["manifest_path"], fixture["prediction_path"])

    with pytest.raises(RuntimeError, match="misses fold-vote decision confidence"):
        stage13._validate_stage12_artifacts(fixture["report"])


def test_stage13_rejects_duplicate_clinvar_variant_or_unsafe_dms_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _stage12_fixture(tmp_path, monkeypatch)
    arrays = fixture["npz"]
    arrays["clinvar_exact_variant_disjoint__variant_ids"][1] = arrays[
        "clinvar_exact_variant_disjoint__variant_ids"
    ][0]
    np.savez_compressed(fixture["prediction_path"], **arrays)
    _refresh_output_record(fixture["manifest_path"], fixture["prediction_path"])
    with pytest.raises(RuntimeError, match="ClinVar primary variants"):
        stage13._validate_stage12_artifacts(fixture["report"])

    fixture = _stage12_fixture(tmp_path / "second", monkeypatch)
    prep = json.loads(fixture["prep_path"].read_text(encoding="utf-8"))
    prep["extra"]["dms_sampling_uses_label"] = True
    _write_json(fixture["prep_path"], prep)
    with pytest.raises(RuntimeError, match="DMS sampling is not publication-safe"):
        stage13._validate_stage12_artifacts(fixture["report"])


@pytest.mark.parametrize("fail_required", [False, True])
def test_stage13_manifest_truthfully_records_completion_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_required: bool,
) -> None:
    figure_dir = tmp_path / "figures"
    figure_manifest = figure_dir / "figure_manifest.json"
    run_manifest = figure_dir / "run_manifest.json"
    monkeypatch.setattr(stage13, "STAGE13_OUT", figure_dir)
    monkeypatch.setattr(stage13, "FIGURE_MANIFEST", figure_manifest)
    monkeypatch.setattr(stage13, "RUN_MANIFEST", run_manifest)
    monkeypatch.setattr(
        stage13,
        "_validate_stage14_artifacts",
        lambda: {"status": "validated"},
    )
    monkeypatch.setattr(
        stage13,
        "_validate_stage12_artifacts",
        lambda report: {"status": "validated", "publication_complete": True},
    )
    monkeypatch.setattr(
        stage13,
        "_load_npz",
        lambda path: {"clinvar_exact_variant_disjoint__y": np.asarray([0, 1])},
    )
    monkeypatch.setattr(
        stage13,
        "_load_json_object",
        lambda path, description: {"sets": {}},
    )

    def fake_figure(
        manifest: dict[str, Any], name: str, required: bool, builder: Any
    ) -> None:
        if fail_required and required and name == "external_roc_pr":
            manifest[name] = {"status": "failure", "required": True}
            return
        output = figure_dir / f"{name}.png"
        output.write_bytes(b"figure")
        manifest[name] = {
            "status": "success",
            "required": required,
            "paths": [str(output)],
        }

    monkeypatch.setattr(stage13, "_run_figure", fake_figure)
    captured: dict[str, Any] = {}

    def fake_run_manifest(
        path: Path,
        stage: str,
        inputs: list[Path],
        extra: dict[str, Any],
        outputs: list[Path],
    ) -> None:
        captured.update(
            {"stage": stage, "extra": extra, "outputs": list(outputs)}
        )

    monkeypatch.setattr(stage13, "write_run_manifest", fake_run_manifest)
    if fail_required:
        with pytest.raises(RuntimeError, match="Required figures failed"):
            stage13.main()
    else:
        stage13.main()

    figure_payload = json.loads(figure_manifest.read_text(encoding="utf-8"))
    assert captured["stage"] == "13_generate_figures"
    assert captured["extra"]["completed"] is (not fail_required)
    assert captured["extra"]["publication_complete"] is (not fail_required)
    assert figure_payload["completed"] is (not fail_required)
    assert figure_payload["publication_complete"] is (not fail_required)
    assert figure_manifest in captured["outputs"]
    assert all(Path(path).is_file() for path in captured["outputs"])


def test_stage13_refuses_nonempty_figure_directory_without_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "prior_valid_figure.pdf").write_bytes(b"prior")
    monkeypatch.setattr(stage13, "STAGE13_OUT", figure_dir)
    monkeypatch.setattr(stage13, "ALLOW_EXISTING_FIGURE_DIR", False)

    with pytest.raises(RuntimeError, match="Refusing to write Stage 13"):
        stage13.main()

    assert (figure_dir / "prior_valid_figure.pdf").read_bytes() == b"prior"


def test_reported_metric_accepts_only_one_serialized_rounding_unit() -> None:
    stage13._assert_reported_value(
        0.0629,
        0.0628,
        "dms_exact_variant_disjoint.assay.model.functional_spearman",
    )
    with pytest.raises(RuntimeError, match="reported metric"):
        stage13._assert_reported_value(
            0.0630,
            0.0628,
            "dms_exact_variant_disjoint.assay.model.functional_spearman",
        )
    with pytest.raises(RuntimeError, match="reported metric"):
        stage13._assert_reported_value(1.0001, 1.0, "artifact.identity")
