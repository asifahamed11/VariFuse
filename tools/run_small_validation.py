"""Real-data engineering validation; never produces publication-certified artifacts.

Starts from existing Stage 08/09 tables, freshly computes ESM features, and runs
five group-disjoint folds with separate stop/calibration/threshold partitions.
Historical preprocessing provenance is inherited, not re-certified. Selection
favours short proteins and mixed-label groups to keep GPU/runtime costs bounded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_inputs(output: Path, *, extracted: bool = False) -> None:
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["datasets"].items():
        if sha(output / f"{name}_input.parquet") != record["sha256"]:
            raise ValueError(f"Small dataset changed after selection: {name}")
        if extracted:
            extraction = json.loads((output / f"{name}_esm_manifest.json").read_text(encoding="utf-8"))
            input_record = extraction["inputs"].get(str(output / f"{name}_input.parquet"), {})
            if input_record.get("sha256") != record["sha256"]:
                raise ValueError(f"Extraction belongs to different input rows: {name}")
            bound = {item["sha256"] for item in extraction["outputs"].values()}
            for suffix in ("with_esm.parquet", "embeddings.npy", "status.parquet"):
                if sha(output / f"{name}_{suffix}") not in bound:
                    raise ValueError(f"Extracted artifact changed: {name}_{suffix}")
            for filename, digest in extraction["source_files"].items():
                if sha(ROOT / "src" / filename) != digest:
                    raise ValueError(f"Extraction code changed: {filename}; rerun --phase extract")


def prepare(source: Path, output: Path) -> dict:
    import pandas as pd
    from schema import LABEL_COL

    external = module("small_stage09", "09_prepare_external_esm_dataset.py")
    internal_path = source / "08_prepare_esm/internal_esm_ready_homology.parquet"
    sequence_path = source / "08_prepare_esm/internal_sequences.parquet"
    universe = pd.read_parquet(internal_path)
    frame = universe.merge(
        pd.read_parquet(sequence_path), on="sequence_hash", validate="many_to_one"
    )
    frame = frame.loc[frame.protein_sequence.str.len().le(200)].copy()
    counts = frame.groupby("split_group")[LABEL_COL].nunique()
    eligible = sorted(
        counts[counts.eq(2)].index, key=lambda x: hashlib.sha256(str(x).encode()).hexdigest()
    )[:64]
    if len(eligible) < 60:
        raise ValueError("Need at least 60 mixed-label short-protein groups")
    internal = (
        frame.loc[frame.split_group.isin(eligible)]
        .sort_values("row_id")
        .groupby(["split_group", LABEL_COL], sort=True)
        .head(2)
        .reset_index(drop=True)
    )
    candidates = []
    clinvar_path = source / "09_prepare_external_esm/clinvar_esm_ready.csv"
    for chunk in pd.read_csv(clinvar_path, chunksize=10000, low_memory=False, dtype={"chr": str}):
        keep = chunk.protein_sequence.str.len().le(250)
        keep &= ~chunk.genename.isin(universe.genename)
        keep &= ~chunk.sequence_hash.isin(universe.sequence_hash)
        keep &= ~chunk.variant_id.isin(universe.variant_id)
        keep &= ~chunk.CLINVAR_VARIATION_ID.astype(str).isin(
            universe.CLINVAR_VARIATION_ID.astype(str)
        )
        keep &= chunk.SCV_EVIDENCE_PASS.eq(1)
        candidates.append(chunk.loc[keep])
    clinvar = (
        pd.concat(candidates)
        .sort_values("row_id")
        .groupby(LABEL_COL)
        .head(16)
        .reset_index(drop=True)
    )
    if set(clinvar[LABEL_COL]) != {0, 1} or len(clinvar) < 16:
        raise ValueError("Insufficient disjoint external clinical rows")
    dms_path = source / "09_prepare_external_esm/dms_esm_ready.csv"
    dms_sequences = source / "09_prepare_external_esm/dms_sequences.parquet"
    seq = pd.read_parquet(dms_sequences)
    seq = seq.loc[seq.protein_sequence.str.len().le(200)]
    dms_chunks = []
    for chunk in pd.read_csv(dms_path, chunksize=20000, low_memory=False):
        keep = chunk.sequence_hash.isin(seq.sequence_hash)
        keep &= ~chunk.sequence_hash.isin(universe.sequence_hash)
        dms_chunks.append(chunk.loc[keep])
    dms = pd.concat(dms_chunks).sort_values("row_id")
    assay_counts = dms.groupby("ASSAY_ID")[LABEL_COL].nunique()
    assays = sorted(assay_counts[assay_counts.eq(2)].index)[:2]
    if len(assays) != 2:
        raise ValueError("Need two short-protein DMS assays with both functional labels")
    dms = (
        dms.loc[dms.ASSAY_ID.isin(assays)]
        .groupby(["ASSAY_ID", LABEL_COL])
        .head(8)
        .merge(seq, on="sequence_hash", validate="many_to_one")
        .reset_index(drop=True)
    )
    manifest = {
        "purpose": "engineering_validation_only",
        "publication_eligible": False,
        "selection": "SHA-ordered 64 mixed-label homology groups, proteins <=200 aa, up to 2 rows/class/group; external clinical <=250 aa, max16/class; DMS 2 assays <=200 aa, max8/class/assay",
        "upstream": "Historical Stage 08/09 outputs; raw-source stages not rerun or re-certified",
        "clinical_external_exclusions": "All original internal genes, exact sequences, genomic variant IDs and ClinVar VariationIDs; inherited SCV_EVIDENCE_PASS required. No new external homology clustering.",
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [internal_path, sequence_path, clinvar_path, dms_path, dms_sequences]
        },
        "datasets": {},
    }
    for name, table in [
        ("internal", internal),
        ("clinvar", external._align_external_schema(clinvar)),
        ("dms", external._align_external_schema(dms)),
    ]:
        if table.row_id.duplicated().any():
            raise ValueError(f"Duplicate rows in {name}")
        path = output / f"{name}_input.parquet"
        table.to_parquet(path, index=False)
        manifest["datasets"][name] = {
            "rows": len(table),
            "class_counts": {str(k): int(v) for k, v in table[LABEL_COL].value_counts().items()},
            "groups": int(table.split_group.nunique()),
            "sha256": sha(path),
        }
    dump(output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest["datasets"], indent=2), flush=True)
    return manifest


def extract(output: Path) -> None:
    import esm
    import numpy as np
    import pandas as pd
    import torch

    stage = module("small_stage10", "10_extract_esm_features.py")
    model, alphabet = stage._load_esm_model(esm)
    for name in ("internal", "clinvar", "dms"):
        task = stage.ExtractionTask(
            name,
            output / f"{name}_input.parquet",
            output / f"{name}_with_esm.parquet",
            output / f"{name}_embeddings.npy",
            output / f"{name}_status.parquet",
            output / f"{name}_esm_manifest.json",
            output / "cache" / name,
        )
        stage.extract_task(task, model, alphabet, "both", validate_upstream=False)
        values = np.load(task.embedding_file)
        frame = pd.read_parquet(task.output_table)
        assert len(frame) == len(values) and np.isfinite(values).all()
        assert frame.ESM_EXTRACTION_SUCCESS.eq(1).all(), name
        assert (
            json.loads(task.manifest_file.read_text())["extra"]["validated_upstream_stage"]
            == "explicitly_skipped_nonpublication"
        )
        print(f"ESM complete: {name}, {len(frame)} rows, dimension {values.shape[1]}", flush=True)
    del model, alphabet
    gc.collect()
    torch.cuda.empty_cache()


def train_and_evaluate(output: Path, epochs: int) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    import common as C
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from capture_environment import capture

    torch.set_num_threads(4)
    dump(output / "environment-lock.json", capture())
    stage12 = module("small_stage12", "12_external_validation.py")
    stage12.BOOTSTRAP_ITERATIONS = 100
    frames = {
        name: pd.read_parquet(output / f"{name}_with_esm.parquet")
        for name in ("internal", "clinvar", "dms")
    }
    embeddings = {name: np.load(output / f"{name}_embeddings.npy") for name in frames}
    frame = frames["internal"]
    labels = C.validate_binary_labels(frame[C.LABEL_COL].to_numpy())
    groups = frame.split_group.astype(str).to_numpy()
    splits = C.make_group_splits(labels, groups, 5, 42)
    base = C.select_features(frame)
    configurations = [
        (name, name, {})
        for name in (
            "concatenation",
            "gated_fusion",
            "cross_attention",
            "reliability_residual",
            "evidential_residual",
        )
    ]
    configurations += [
        (
            "evidential_no_rcdi",
            "evidential_residual",
            {"RCDI_ANCHOR_WEIGHT": 0.0, "RCDI_CONSISTENCY_WEIGHT": 0.0},
        ),
        ("evidential_legacy_gate", "evidential_residual", {"EVIDENTIAL_RELIABILITY_POWER": 0.0}),
    ]
    model_names = [name for name, _, _ in configurations] + ["esm_logistic", "lightgbm"]
    oof = {name: np.full(len(frame), np.nan) for name in model_names}
    cutoffs = {name: np.full(len(frame), np.nan) for name in model_names}
    external = {dataset: {name: [] for name in model_names} for dataset in ("clinvar", "dms")}
    thresholds = {name: [] for name in model_names}
    partition_records, stress = [], []
    small_config = {
        "D_MODEL": 32,
        "N_HEADS": 4,
        "N_CROSS_BLOCKS": 1,
        "N_FUSION_LAYERS": 1,
        "N_ESM_SLOTS": 4,
        "BATCH_SIZE": 32,
        "WARMUP_EPOCHS": 1,
        "DEEP_MAX_EPOCHS": epochs,
        "DEEP_PATIENCE": 2,
        "N_ENSEMBLE": 1,
    }
    for fold, (outer_train, test) in enumerate(splits, 1):
        fit, stop, calibration, threshold_set = C.split_fit_stop_temperature_threshold(
            outer_train, labels, groups, 100 + fold
        )
        parts = {
            "fit": fit,
            "stop": stop,
            "calibration": calibration,
            "threshold": threshold_set,
            "test": test,
        }
        for i, values in enumerate(parts.values()):
            for other in list(parts.values())[i + 1 :]:
                assert set(groups[values]).isdisjoint(groups[other])
        partition_records.append(
            {
                "fold": fold,
                "partitions": {
                    key: {
                        "rows": frame.iloc[value].row_id.tolist(),
                        "groups": sorted(set(groups[value])),
                    }
                    for key, value in parts.items()
                },
            }
        )
        for name, architecture, overrides in configurations:
            names = C.architecture_feature_names(architecture, frame, base)
            raw = frame[names].to_numpy(dtype=np.float32)
            passed = (
                C.reliability_passthrough_indices(names)
                if C.is_reliability_family(architecture)
                else []
            )
            processors = C.fit_preprocessors(raw[fit], embeddings["internal"][fit], passed)
            bio = processors.transform_bio(raw)
            esm_values = processors.transform_esm(embeddings["internal"])
            with C.temporary_model_config({**small_config, **overrides}):
                model = C.train_deep_model(
                    bio[fit],
                    esm_values[fit],
                    labels[fit],
                    bio[stop],
                    esm_values[stop],
                    labels[stop],
                    bio[calibration],
                    esm_values[calibration],
                    labels[calibration],
                    architecture=architecture,
                    seeds=[142],
                    feature_names=names,
                )
                threshold = C.select_threshold(
                    labels[threshold_set],
                    C.predict(model, bio[threshold_set], esm_values[threshold_set]),
                )
                predicted = C.predict(model, bio[test], esm_values[test])
                bundle = output / "models" / f"fold{fold}_{name}.pt"
                C.save_deep_bundle(
                    bundle,
                    model,
                    processors,
                    names,
                    threshold,
                    architecture,
                    {"publication_eligible": False, "fold": fold},
                )
                restored, restored_processors, restored_names, restored_threshold, _ = (
                    C.load_deep_bundle(bundle)
                )
                np.testing.assert_allclose(
                    C.predict(
                        restored,
                        restored_processors.transform_bio(raw[test]),
                        restored_processors.transform_esm(embeddings["internal"][test]),
                    ),
                    predicted,
                    atol=1e-6,
                    rtol=1e-6,
                )
                assert restored_names == names and restored_threshold == threshold
                del restored
                oof[name][test], cutoffs[name][test] = predicted, threshold
                thresholds[name].append(threshold)
                for dataset in external:
                    ext_bio = processors.transform_bio(
                        frames[dataset][names].to_numpy(dtype=np.float32)
                    )
                    ext_esm = processors.transform_esm(embeddings[dataset])
                    external[dataset][name].append(C.predict(model, ext_bio, ext_esm))
                if C.is_reliability_family(architecture):
                    degraded = frame.iloc[test].copy()
                    degraded["HAS_STRUCTURE"] = 0.0
                    degraded["LOW_CONFIDENCE_STRUCTURE"] = 1.0
                    for feature in C.RELIABILITY_CONSERVATION_MISSING_FEATURES:
                        degraded[feature] = 1.0
                    components = C.predict_reliability_components(
                        model,
                        processors.transform_bio(degraded[names].to_numpy(dtype=np.float32)),
                        esm_values[test],
                    )
                    assert np.allclose(components["gate"], 0, atol=1e-7)
                    stress.append(
                        {
                            "fold": fold,
                            "model": name,
                            "absent_auxiliary_gate_max": float(np.max(np.abs(components["gate"]))),
                        }
                    )
                del model
            print(f"Training complete: fold {fold}/5 {name}", flush=True)
        # Fit both simple baselines only on the fitting partition and calibrate
        # on the same reserved calibration partition as the neural models.
        for name in ("esm_logistic", "lightgbm"):
            names = ["esm_variant_score"] if name == "esm_logistic" else base
            raw = frame[names].to_numpy(dtype=np.float32)
            prep = C.ArrayPreprocessor.fit(raw[fit])
            values = pd.DataFrame(prep.transform(raw), columns=names)
            estimator = (
                LogisticRegression(C=1.0, max_iter=500, random_state=142)
                if name == "esm_logistic"
                else LGBMClassifier(
                    n_estimators=40,
                    num_leaves=7,
                    min_child_samples=5,
                    n_jobs=4,
                    verbosity=-1,
                    random_state=142,
                )
            )
            estimator.fit(values.iloc[fit], labels[fit])
            calibrator = C.fit_probability_calibrator(
                labels[calibration], estimator.predict_proba(values.iloc[calibration])[:, 1]
            )
            probability = calibrator.predict(estimator.predict_proba(values)[:, 1])
            threshold = C.select_threshold(labels[threshold_set], probability[threshold_set])
            oof[name][test], cutoffs[name][test] = probability[test], threshold
            thresholds[name].append(threshold)
            for dataset in external:
                external[dataset][name].append(
                    calibrator.predict(
                        estimator.predict_proba(
                            pd.DataFrame(prep.transform(frames[dataset][names].to_numpy(dtype=np.float32)), columns=names)
                        )[:, 1]
                    )
                )
    dump(output / "partitions.json", partition_records)
    dump(output / "absent_modality_checks.json", stress)
    metrics = {name: C.evaluate(labels, oof[name], cutoffs[name]) for name in model_names}
    prediction_frame = frame[[C.ROW_ID_COL, C.LABEL_COL, "split_group"]].copy()
    for name in model_names:
        assert np.isfinite(oof[name]).all()
        prediction_frame[name] = oof[name]
        prediction_frame[f"{name}_threshold"] = cutoffs[name]
    prediction_frame.to_csv(output / "oof_predictions.csv", index=False)
    clinical, functional = {}, {}
    for dataset, predictions in external.items():
        probabilities = {name: np.mean(values, axis=0) for name, values in predictions.items()}
        # Majority vote freezes each fold's threshold; labels never tune these.
        decisions = {
            name: (
                np.mean(np.asarray(values) >= np.asarray(thresholds[name])[:, None], axis=0) >= 0.5
            ).astype(int)
            for name, values in predictions.items()
        }
        mean_thresholds = {name: float(np.mean(thresholds[name])) for name in model_names}
        table = frames[dataset][[C.ROW_ID_COL, C.LABEL_COL]].copy()
        for name, values in probabilities.items():
            table[name], table[f"{name}_decision"] = values, decisions[name]
        table.to_csv(output / f"{dataset}_predictions.csv", index=False)
        if dataset == "clinvar":
            clinical = {
                name: C.evaluate(
                    frames[dataset][C.LABEL_COL].to_numpy(),
                    values,
                    mean_thresholds[name],
                    predictions=decisions[name],
                )
                for name, values in probabilities.items()
            }
        else:
            functional = stage12._dms_assay_results(
                frames[dataset],
                probabilities,
                decisions,
                mean_thresholds,
                sampling_audit={
                    "publication_eligible": False,
                    "selection_uses_labels": True,
                    "purpose": "balanced engineering fixture",
                },
            )
    summary = {
        "status": "passed",
        "publication_eligible": False,
        "epochs_per_model_max": epochs,
        "outer_folds": 5,
        "ensemble_members_per_fold": 1,
        "hyperparameter_search": "not performed; fixed small architecture budget",
        "nested_partition_leakage_checks": "passed",
        "checkpoint_reload_checks": 35,
        "absent_modality_checks": len(stress),
        "internal_oof": metrics,
        "external_clinvar": clinical,
        "external_dms": functional,
    }
    rows = [
        {"model": name, **{key: metrics[name][key] for key in ("auroc", "auprc", "mcc", "brier")}}
        for name in model_names
    ]
    pd.DataFrame(rows).to_csv(output / "ablation_comparison.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh([row["model"] for row in rows], [row["auprc"] for row in rows])
    axis.set(
        xlabel="OOF AUPRC",
        title="Small engineering validation — not publication results",
        xlim=(0, 1),
    )
    figure.tight_layout()
    figure.savefig(output / "ablation_comparison.png", dpi=160)
    plt.close(figure)
    dump(output / "results.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "publication_runs/reliability_temporal_v1/outputs"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "validation_runs/small_20260909")
    parser.add_argument("--phase", choices=["all", "prepare", "extract", "train"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    args.output = args.output.resolve()
    allowed = (ROOT / "validation_runs").resolve()
    if not args.output.is_relative_to(allowed) or args.output == allowed:
        parser.error("Output must be a named child of validation_runs")
    if args.epochs < 1:
        parser.error("Epochs must be positive")
    if args.phase in {"all", "prepare"} and (args.output / "results.json").exists():
        parser.error("Completed validation exists; choose a fresh --output")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TORCH_HOME": str(ROOT / "model_cache"),
            "VARIANT_OUTPUT_DIR": str(args.output / "outputs"),
            "VARIANT_FIGURE_DIR": str(args.output / "figures"),
            "ESM_DEVICE": "cuda",
            "ESM_USE_FP16": "1",
            "ESM_MAX_BATCH_PROTEINS": "1",
            "ESM_MAX_MASKS_PER_BATCH": "2",
            "ESM_MAX_BATCH_TOKENS": "1024",
            "ESM_MAX_BATCH_ATTENTION": "1048576",
            "ESM_INTERNAL_MAX_ROWS": "1000",
            "MMSEQS_EXECUTABLE": str(ROOT / "tools/mmseqs2/mmseqs/bin/mmseqs.exe"),
            "VARIANT_REPRODUCIBLE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    logging.basicConfig(level=logging.INFO)
    started = time.monotonic()
    status_path = args.output / f"{args.phase}_status.json"
    dump(status_path, {"status": "running", "phase": args.phase})
    try:
        if args.phase in {"all", "prepare"}:
            prepare(args.source, args.output)
        if args.phase in {"all", "extract"}:
            verify_inputs(args.output)
            extract(args.output)
        if args.phase in {"all", "train"}:
            verify_inputs(args.output, extracted=True)
            train_and_evaluate(args.output, args.epochs)
    except BaseException as error:
        dump(status_path, {"status": "failed", "phase": args.phase, "error": str(error)})
        raise
    dump(status_path, {"status": "passed", "phase": args.phase, "seconds": time.monotonic() - started})
    print(f"Completed {args.phase} in {time.monotonic() - started:.1f}s: {args.output}", flush=True)


if __name__ == "__main__":
    main()
