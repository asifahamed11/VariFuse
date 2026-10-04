from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests

import common as C
from config import (
    MODEL_TAG,
    RANDOM_STATE,
    REQUIRE_HOMOLOGY_GROUPS,
    STAGE10_OUT,
    STAGE14_OUT,
    TUNING_BEST_JSON,
    TUNING_STORAGE,
    ensure_directories,
    file_sha256,
    json_default,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import (
    MUTATION_FEATURE_COLS,
    select_availability_features,
    select_tabular_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage14_tuning")

INPUT_CSV = STAGE10_OUT / "internal_with_esm.parquet"
INPUT_NPY = STAGE10_OUT / "internal_esm_embeddings.npy"
INPUT_STATUS = STAGE10_OUT / "internal_esm_extraction.parquet"
TRIALS_CSV = STAGE14_OUT / "trials_history.csv"
FOLD_RESULTS_CSV = STAGE14_OUT / "nested_outer_results.csv"
FOLD_ASSIGNMENTS = STAGE14_OUT / "nested_outer_folds.csv"
OOF_FILE = STAGE14_OUT / "nested_tuning_oof.npz"
MANIFEST_FILE = STAGE14_OUT / "run_manifest.json"
STAGE10_MANIFEST = STAGE10_OUT / "internal_esm_manifest.json"
SPLIT_PLAN_FILE = STAGE14_OUT / "nested_inner_splits.json"
ARCHITECTURE_RESULTS_JSON = STAGE14_OUT / "architecture_selection.json"
CONCATENATION_BEST_JSON = STAGE14_OUT / "best_concatenation_params.json"
GATED_FUSION_BEST_JSON = STAGE14_OUT / "best_gated_fusion_params.json"
RELIABILITY_RESIDUAL_BEST_JSON = (
    STAGE14_OUT / "best_reliability_residual_params.json"
)
# CHANGELOG 2026-09 (novelty): per-family-member best-payload file.
EVIDENTIAL_RESIDUAL_BEST_JSON = (
    STAGE14_OUT / "best_evidential_residual_params.json"
)
CONFIRMATORY_SEED_PLAN = STAGE14_OUT / "confirmatory_seed_plan.json"
CONFIRMATORY_RESULTS_JSON = STAGE14_OUT / "confirmatory_repeated_cv_results.json"
CONFIRMATORY_PREDICTIONS = STAGE14_OUT / "confirmatory_repeated_cv_predictions.npz"
CONFIRMATORY_FOLD_ASSIGNMENTS = STAGE14_OUT / "confirmatory_repeated_cv_folds.csv"
CONFIRMATORY_CHECKPOINT_DIR = STAGE14_OUT / "confirmatory_repeats"
PRIMARY_CHECKPOINT_DIR = STAGE14_OUT / "primary_outer_folds"
ARCHITECTURE_CHECKPOINT_DIR = STAGE14_OUT / "architecture_checkpoints"
SESSION_STATUS_FILE = STAGE14_OUT / "session_status.json"
CONFIRMATORY_MODEL_NAMES = (
    C.RELIABILITY_ARCHITECTURE,
    "raw_esm_zero_shot",
    "esm_conservation_logistic",
    "lightgbm",
)
PROTOCOL_VERSION = "fixed_nested_group_cv_v4_reliability_residual"
# Retained for callers/tests that explicitly request the historical comparator
# panel.  The publication default below additionally evaluates the proposed
# reliability-conditioned residual architecture.
SUPPORTED_ARCHITECTURES = ("concatenation", "gated_fusion", "cross_attention")
# CHANGELOG 2026-09 (novelty): the proposed model became a two-member family
# (reliability_residual, evidential_residual).  The family tuple is spliced in
# LAST and appends new members at the end, which is load-bearing: the
# per-architecture confirmatory/tuning seed offset is computed as
# ``PUBLICATION_ARCHITECTURES.index(architecture) * 1_000_000``, so inserting a
# name anywhere but the end would silently re-seed every existing study.
PUBLICATION_ARCHITECTURES = (
    *SUPPORTED_ARCHITECTURES,
    *C.RELIABILITY_FAMILY_ARCHITECTURES,
)

SEARCH_SPACE = {
    "N_HEADS": [4, 8],
    "HEAD_WIDTH": [24, 32, 40],
    "N_CROSS_BLOCKS": [2, 5],
    "N_FUSION_LAYERS": [1, 3],
    "N_ESM_SLOTS": [8, 16, 24, 32],
    "DROPOUT": [0.05, 0.35],
    "DEEP_LR": [1e-4, 2e-3],
    "DEEP_WD": [1e-6, 2e-3],
    "WARMUP_EPOCHS": [2, 12],
    "EMA_DECAY": [0.99, 0.995, 0.999],
    "MIXUP_ALPHA": [0.0, 0.4],
    "LABEL_SMOOTH": [0.0, 0.05],
    "BATCH_SIZE": [128, 256, 512],
}

CONCATENATION_SEARCH_SPACE = {
    "D_MODEL": [96, 128, 160, 192, 256, 320],
    "DROPOUT": SEARCH_SPACE["DROPOUT"],
    "DEEP_LR": SEARCH_SPACE["DEEP_LR"],
    "DEEP_WD": SEARCH_SPACE["DEEP_WD"],
    "WARMUP_EPOCHS": SEARCH_SPACE["WARMUP_EPOCHS"],
    "EMA_DECAY": SEARCH_SPACE["EMA_DECAY"],
    "MIXUP_ALPHA": SEARCH_SPACE["MIXUP_ALPHA"],
    "LABEL_SMOOTH": SEARCH_SPACE["LABEL_SMOOTH"],
    "BATCH_SIZE": SEARCH_SPACE["BATCH_SIZE"],
}

GATED_FUSION_SEARCH_SPACE = {
    **CONCATENATION_SEARCH_SPACE,
    # GatedFusionNet caps its effective width at 192, so larger aliases would
    # waste trials while constructing exactly the same network.
    "D_MODEL": [96, 128, 160, 192],
}

RELIABILITY_RESIDUAL_SEARCH_SPACE = {
    **CONCATENATION_SEARCH_SPACE,
    # Both reliability encoders cap width at 128; 160 was an identical alias.
    "D_MODEL": [64, 96, 128],
    "MODALITY_DROPOUT": [0.10, 0.50],
    "RELIABILITY_RESIDUAL_SCALE": [0.5, 3.0],
    # Mixup would interpolate binary availability semantics, so the proposed
    # architecture uses modality dropout instead.
    "MIXUP_ALPHA": [0.0, 0.0],
}

# Expert fusion adds two search dimensions under the same trial budget.
# Quality attenuation and auxiliary-loss coefficients stay prespecified and
# are persisted with each run. Their effect requires separately trained
# ablations; this search does not produce those comparisons automatically.
EVIDENTIAL_RESIDUAL_SEARCH_SPACE = {
    **RELIABILITY_RESIDUAL_SEARCH_SPACE,
    "EVIDENTIAL_PRECISION_FLOOR": [0.01, 0.25],
    "EVIDENTIAL_GATE_TEMPERATURE": [0.5, 4.0],
}

# Maps each reliability-family member to its search space.  Keyed by name so a
# future family member cannot be added without also declaring its space.
RELIABILITY_FAMILY_SEARCH_SPACES = {
    C.RELIABILITY_RESIDUAL_ARCHITECTURE: RELIABILITY_RESIDUAL_SEARCH_SPACE,
    C.EVIDENTIAL_RESIDUAL_ARCHITECTURE: EVIDENTIAL_RESIDUAL_SEARCH_SPACE,
}

# Compact, prespecified tree grid evaluated on the exact same inner folds as
# the neural architectures.  This removes the former asymmetry where the deep
# models received HPO but LightGBM received only one hand-set configuration.
LGBM_NESTED_GRID = (
    {"num_leaves": 15, "min_child_samples": 30},
    {"num_leaves": 15, "min_child_samples": 90},
    {"num_leaves": 31, "min_child_samples": 30},
    {"num_leaves": 31, "min_child_samples": 90},
    {"num_leaves": 63, "min_child_samples": 30},
    {"num_leaves": 63, "min_child_samples": 90},
)


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(values: dict[str, np.ndarray], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _architecture_checkpoint_npz(architecture: str) -> Path:
    return ARCHITECTURE_CHECKPOINT_DIR / f"{architecture}_checkpoint.npz"


def _architecture_checkpoint_meta(architecture: str) -> Path:
    return ARCHITECTURE_CHECKPOINT_DIR / f"{architecture}_checkpoint.json"


def _save_architecture_checkpoint(
    architecture: str,
    oof_probabilities: np.ndarray,
    oof_decisions: np.ndarray,
    oof_thresholds: np.ndarray,
    fold_results: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """Persist a completed architecture's OOF predictions and results.

    Called after all outer folds for one architecture finish, so that a
    subsequent ``--resume`` invocation can skip the architecture entirely.
    """
    ARCHITECTURE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        {
            "probabilities": np.asarray(oof_probabilities, dtype=np.float64),
            "decisions": np.asarray(oof_decisions, dtype=np.int8),
            "thresholds": np.asarray(oof_thresholds, dtype=np.float64),
        },
        _architecture_checkpoint_npz(architecture),
    )
    _atomic_json(
        {"fold_results": fold_results, "payload": payload},
        _architecture_checkpoint_meta(architecture),
    )
    logger.info("Saved architecture checkpoint for %s", architecture)


def _load_architecture_checkpoint(
    architecture: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]] | None:
    """Load a previously completed architecture checkpoint, or ``None``."""
    npz_path = _architecture_checkpoint_npz(architecture)
    meta_path = _architecture_checkpoint_meta(architecture)
    if not npz_path.exists() or not meta_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as stored:
        predictions = {
            "probabilities": np.asarray(stored["probabilities"], dtype=np.float64),
            "decisions": np.asarray(stored["decisions"], dtype=np.int8),
            "thresholds": np.asarray(stored["thresholds"], dtype=np.float64),
        }
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return predictions, meta["fold_results"], meta["payload"]


def _primary_checkpoint_path(architecture: str, outer_fold: int) -> Path:
    return PRIMARY_CHECKPOINT_DIR / f"{architecture}_outer_{outer_fold}.npz"


def _save_primary_checkpoint(
    architecture: str,
    outer_fold: int,
    study_name: str,
    study_protocol: dict[str, Any],
    target_trials: int,
    record: dict[str, Any],
    probabilities: np.ndarray,
    decisions: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    PRIMARY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "architecture": architecture,
        "outer_fold": int(outer_fold),
        "study_name": study_name,
        "study_protocol_sha256": _canonical_sha256(study_protocol),
        "target_trials": int(target_trials),
        "record": record,
    }
    _atomic_npz(
        {
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, default=json_default)
            ),
            "probabilities": np.asarray(probabilities, dtype=np.float64),
            "decisions": np.asarray(decisions, dtype=np.int8),
            "thresholds": np.asarray(thresholds, dtype=np.float64),
        },
        _primary_checkpoint_path(architecture, outer_fold),
    )


def _load_primary_checkpoint(
    architecture: str,
    outer_fold: int,
    study_name: str,
    study_protocol: dict[str, Any],
    target_trials: int,
    expected_rows: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None:
    path = _primary_checkpoint_path(architecture, outer_fold)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata_json"].item()))
        probabilities = np.asarray(stored["probabilities"], dtype=np.float64)
        decisions = np.asarray(stored["decisions"], dtype=np.int8)
        thresholds = np.asarray(stored["thresholds"], dtype=np.float64)
    expected = {
        "schema_version": 1,
        "architecture": architecture,
        "outer_fold": int(outer_fold),
        "study_name": study_name,
        "target_trials": int(target_trials),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Primary fold checkpoint mismatch for {path}: {key}")
    record = metadata.get("record")
    if not isinstance(record, dict) or record.get("outer_validation_rows") != expected_rows:
        raise RuntimeError(f"Primary fold checkpoint row identity mismatch: {path}")
    size = len(expected_rows)
    if thresholds.ndim == 0:
        thresholds = np.full(size, float(thresholds), dtype=np.float64)
    elif thresholds.shape != (size,):
        raise RuntimeError(f"Primary fold checkpoint thresholds array has invalid shape: {thresholds.shape}")
    if (
        probabilities.shape != (size,)
        or decisions.shape != (size,)
        or thresholds.shape != (size,)
        or not np.isfinite(probabilities).all()
        or not np.isfinite(thresholds).all()
        or not np.isin(decisions, (0, 1)).all()
    ):
        raise RuntimeError(f"Primary fold checkpoint arrays are invalid: {path}")
    return record, probabilities, decisions, thresholds


def _write_session_status(
    *,
    complete: bool,
    session_seconds: int,
    architecture: str | None = None,
    outer_fold: int | None = None,
    finished_trials: int | None = None,
    target_trials: int | None = None,
) -> None:
    _atomic_json(
        {
            "schema_version": 1,
            "complete": bool(complete),
            "session_seconds": int(session_seconds),
            "architecture": architecture,
            "outer_fold": outer_fold,
            "finished_trials": finished_trials,
            "target_trials": target_trials,
            "resume_command": "python kaggle/kaggle_runner.py stage14",
        },
        SESSION_STATUS_FILE,
    )


def _canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol data independently of file formatting."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_text_sha256(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _publication_protocol_fingerprint(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Fingerprint every setting that can change a persisted trial outcome."""
    source_paths = {
        "stage14": Path(__file__).resolve(),
        "common": Path(C.__file__).resolve(),
        "gpu_runtime": Path(C.__file__).resolve().with_name("gpu_runtime.py"),
        "config": Path(__file__).resolve().with_name("config.py"),
        "schema": Path(__file__).resolve().with_name("schema.py"),
    }
    return {
        "source_sha256": {
            name: file_sha256(path) for name, path in source_paths.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "lightgbm": _package_version("lightgbm"),
            "scikit_learn": _package_version("scikit-learn"),
            "optuna": _package_version("optuna"),
        },
        "accelerator": {
            **_gpu_metadata(),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
        "deterministic": not bool(arguments.allow_nondeterministic),
        "early_stopping_metric": C.DEEP_EARLY_STOP_METRIC,
        "threshold_policy": {
            "recall_floor": float(C.RECALL_FLOOR),
            "fallback_fbeta": float(C.FBETA_BETA),
        },
        "search_budget": {
            "trials": int(arguments.trials),
            "ensemble": int(arguments.search_ensemble),
            "epochs": int(arguments.search_epochs),
            "patience": int(arguments.search_patience),
        },
        "final_budget": {
            "ensemble": int(arguments.final_ensemble),
            "epochs": int(arguments.final_epochs),
            "patience": int(arguments.final_patience),
        },
        "lightgbm_nested_grid": [dict(value) for value in LGBM_NESTED_GRID],
        "reliability_residual_contract": C.reliability_architecture_protocol(),
    }


def _partition_record(
    indices: np.ndarray,
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    selected_genes = frame.iloc[selected][C.GENE_COL].astype(str)
    return {
        "indices": selected.tolist(),
        "n": int(len(selected)),
        "positives": int(labels[selected].sum()),
        "split_groups": int(pd.Series(groups[selected]).nunique()),
        "genes": int(selected_genes.nunique()),
        "row_ids_sha256": _ordered_text_sha256(
            frame.iloc[selected][C.ROW_ID_COL].astype(str)
        ),
        "genes_sha256": _ordered_text_sha256(selected_genes),
        "split_groups_sha256": _ordered_text_sha256(groups[selected]),
    }


def _record_indices(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["indices"], dtype=np.int64)


def _build_split_plan(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    inner_folds: int,
    split_seed: int,
    training_seed: int,
    search_ensemble: int,
    final_ensemble: int,
    final_epochs: int | None = None,
    final_patience: int | None = None,
    deterministic: bool | None = None,
    protocol_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all nested partitions before any model configuration is sampled."""
    folds: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_validation) in enumerate(outer_splits, 1):
        inner_splits = C.make_group_splits(
            labels[outer_train],
            groups[outer_train],
            inner_folds,
            split_seed + outer_fold * 10_000,
        )
        inner_records: list[dict[str, Any]] = []
        for inner_fold, (local_train, local_validation) in enumerate(inner_splits, 1):
            inner_train = outer_train[local_train]
            validation = outer_train[local_validation]
            fit, stop, temperature, threshold = (
                C.split_fit_stop_temperature_threshold(
                    inner_train,
                    labels,
                    groups,
                    split_seed + outer_fold * 100_000 + inner_fold * 1_000,
                )
            )
            seed_base = training_seed + outer_fold * 1_000_000 + inner_fold * 10_000
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "fit": _partition_record(fit, frame, labels, groups),
                    "early_stopping": _partition_record(stop, frame, labels, groups),
                    "temperature": _partition_record(
                        temperature, frame, labels, groups
                    ),
                    "threshold": _partition_record(threshold, frame, labels, groups),
                    "validation": _partition_record(
                        validation, frame, labels, groups
                    ),
                    "training_seeds": [
                        seed_base + member * 100
                        for member in range(1, search_ensemble + 1)
                    ],
                }
            )
        outer_fit, outer_stop, outer_temperature, outer_threshold = (
            C.split_fit_stop_temperature_threshold(
                outer_train,
                labels,
                groups,
                split_seed + outer_fold * 10_000_000,
            )
        )
        outer_seed_base = training_seed + outer_fold * 100_000_000
        folds.append(
            {
                "outer_fold": outer_fold,
                "outer_train": _partition_record(
                    outer_train, frame, labels, groups
                ),
                "outer_validation": _partition_record(
                    outer_validation, frame, labels, groups
                ),
                "inner_folds": inner_records,
                "final_partitions": {
                    "fit": _partition_record(outer_fit, frame, labels, groups),
                    "early_stopping": _partition_record(
                        outer_stop, frame, labels, groups
                    ),
                    "temperature": _partition_record(
                        outer_temperature, frame, labels, groups
                    ),
                    "threshold": _partition_record(
                        outer_threshold, frame, labels, groups
                    ),
                },
                "final_training_seeds": [
                    outer_seed_base + member * 100
                    for member in range(1, final_ensemble + 1)
                ],
            }
        )
    plan: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "n_rows": int(len(frame)),
        "row_order_sha256": _ordered_text_sha256(
            frame[C.ROW_ID_COL].astype(str)
        ),
        "split_group_column": (
            "split_group" if "split_group" in frame else C.GENE_COL
        ),
        "input_sha256": file_sha256(INPUT_CSV),
        "embedding_sha256": file_sha256(INPUT_NPY),
        "outer_folds": len(outer_splits),
        "inner_folds": int(inner_folds),
        "split_seed": int(split_seed),
        "training_seed": int(training_seed),
        "search_ensemble": int(search_ensemble),
        "final_ensemble": int(final_ensemble),
        "final_epochs": None if final_epochs is None else int(final_epochs),
        "final_patience": (
            None if final_patience is None else int(final_patience)
        ),
        "deterministic": deterministic,
        "protocol_fingerprint": protocol_fingerprint,
        "folds": folds,
    }
    plan["canonical_sha256"] = _canonical_sha256(plan)
    return plan


def _persist_and_validate_split_plan(
    plan: dict[str, Any], resume: bool = False
) -> str:
    """Persist the deterministic plan or reject a conflicting existing plan."""
    if SPLIT_PLAN_FILE.exists():
        stored = json.loads(SPLIT_PLAN_FILE.read_text(encoding="utf-8"))
        if resume:
            critical_keys = (
                "protocol_version",
                "n_rows",
                "row_order_sha256",
                "split_group_column",
                "input_sha256",
                "embedding_sha256",
                "outer_folds",
                "inner_folds",
                "split_seed",
                "training_seed",
                "search_ensemble",
                "final_ensemble",
                "final_epochs",
                "final_patience",
                "deterministic",
            )
            mismatches = [
                k for k in critical_keys if stored.get(k) != plan.get(k)
            ]
            if mismatches:
                raise RuntimeError(
                    f"Existing split plan conflicts with resume settings: {mismatches}"
                )
            logger.info("Resuming with validated existing split plan: %s", SPLIT_PLAN_FILE)
            plan.clear()
            plan.update(stored)
        elif stored != plan:
            raise RuntimeError(
                f"Existing split plan uses a different protocol: {SPLIT_PLAN_FILE}. "
                "Use a clean Stage 14 output directory for the new run."
            )
    else:
        _atomic_json(plan, SPLIT_PLAN_FILE)
    digest = file_sha256(SPLIT_PLAN_FILE)
    if digest is None:
        raise RuntimeError("Could not hash the persisted nested split plan")
    return digest


def _load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    validate_upstream_manifest(
        STAGE10_MANIFEST,
        "10_extract_esm_features:internal",
        [INPUT_CSV, INPUT_NPY, INPUT_STATUS],
        required_source_files=("gpu_runtime.py",),
    )
    for path in (INPUT_CSV, INPUT_NPY, INPUT_STATUS):
        if not path.exists():
            raise FileNotFoundError(path)
    frame = pd.read_parquet(INPUT_CSV).reset_index(drop=True)
    embeddings = np.load(INPUT_NPY, mmap_mode="r")
    statuses = pd.read_parquet(INPUT_STATUS)
    if len(frame) != len(embeddings) or len(frame) != len(statuses):
        raise ValueError(
            f"Input row mismatch: CSV={len(frame)}, NPY={len(embeddings)}, "
            f"status={len(statuses)}"
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != C.ESM_DIM:
        raise ValueError(f"Unexpected embedding shape {embeddings.shape}")
    if list(frame[C.ROW_ID_COL].astype(str)) != list(
        statuses[C.ROW_ID_COL].astype(str)
    ):
        raise ValueError("ESM status order differs from the modelling CSV")
    labels = C.validate_binary_labels(frame[C.LABEL_COL].to_numpy())
    if C.GENE_COL not in frame:
        raise KeyError(f"Missing group column {C.GENE_COL}")
    split_column = "split_group" if "split_group" in frame else C.GENE_COL
    if split_column == C.GENE_COL and REQUIRE_HOMOLOGY_GROUPS:
        raise RuntimeError(
            "Publication Stage 14 requires Stage 08b homology split_group values; "
            "rerun Stages 08b and 10 before tuning"
        )
    groups = frame[split_column].to_numpy(dtype=object)
    if split_column == C.GENE_COL:
        logger.warning("Stage14 is using gene-only groups; run Stage 08b for homology groups")
    if frame[C.ROW_ID_COL].duplicated().any():
        raise ValueError("Internal row identifiers are duplicated")
    feature_names = C.select_features(frame)
    return frame, embeddings, labels, groups, feature_names


def _snapshot_constants() -> dict[str, Any]:
    return {name: getattr(C, name) for name in C.TUNABLE_CONSTANTS}


def _apply_parameters(parameters: dict[str, Any]) -> None:
    unknown = set(parameters) - set(C.TUNABLE_CONSTANTS)
    if unknown:
        raise KeyError(f"Unknown tuning parameters: {sorted(unknown)}")
    for name, value in parameters.items():
        setattr(C, name, value)
    if C.D_MODEL % C.N_HEADS:
        raise ValueError("D_MODEL must divide evenly across N_HEADS")


def _search_space_for(architecture: str) -> dict[str, Any]:
    if architecture == "cross_attention":
        return SEARCH_SPACE
    if architecture == "concatenation":
        return CONCATENATION_SEARCH_SPACE
    if architecture == "gated_fusion":
        return GATED_FUSION_SEARCH_SPACE
    # CHANGELOG 2026-09 (novelty): family-based lookup replaces the equality
    # test against the single proposed architecture, so both family members are
    # tunable regardless of which one config.PROPOSED_ARCHITECTURE points at.
    if C.is_reliability_family(architecture):
        return RELIABILITY_FAMILY_SEARCH_SPACES[architecture]
    raise ValueError(f"Unknown architecture: {architecture}")


def _suggest_parameters(trial: Any, architecture: str) -> dict[str, Any]:
    # CHANGELOG 2026-09 (novelty): membership test widened from the single
    # proposed architecture to the whole reliability family.  The order of
    # trial.suggest_* calls for the pre-existing architectures is unchanged, so
    # already-recorded Optuna trials decode to the same parameter values.
    reliability_family = C.is_reliability_family(architecture)
    if reliability_family or architecture in {"concatenation", "gated_fusion"}:
        architecture_space = _search_space_for(architecture)
        parameters = {
            "D_MODEL": trial.suggest_categorical(
                "D_MODEL", architecture_space["D_MODEL"]
            ),
            "DROPOUT": trial.suggest_float(
                "DROPOUT", *architecture_space["DROPOUT"]
            ),
            "DEEP_LR": trial.suggest_float(
                "DEEP_LR", *architecture_space["DEEP_LR"], log=True
            ),
            "DEEP_WD": trial.suggest_float(
                "DEEP_WD", *architecture_space["DEEP_WD"], log=True
            ),
            "WARMUP_EPOCHS": trial.suggest_int(
                "WARMUP_EPOCHS",
                *architecture_space["WARMUP_EPOCHS"],
            ),
            "EMA_DECAY": trial.suggest_categorical(
                "EMA_DECAY", architecture_space["EMA_DECAY"]
            ),
            "MIXUP_ALPHA": (
                0.0
                if reliability_family
                else trial.suggest_float(
                    "MIXUP_ALPHA", *architecture_space["MIXUP_ALPHA"]
                )
            ),
            "LABEL_SMOOTH": trial.suggest_float(
                "LABEL_SMOOTH", *architecture_space["LABEL_SMOOTH"]
            ),
            "BATCH_SIZE": trial.suggest_categorical(
                "BATCH_SIZE", architecture_space["BATCH_SIZE"]
            ),
        }
        if reliability_family:
            parameters.update(
                {
                    "MODALITY_DROPOUT": trial.suggest_float(
                        "MODALITY_DROPOUT",
                        *architecture_space["MODALITY_DROPOUT"],
                    ),
                    "RELIABILITY_RESIDUAL_SCALE": trial.suggest_float(
                        "RELIABILITY_RESIDUAL_SCALE",
                        *architecture_space["RELIABILITY_RESIDUAL_SCALE"],
                    ),
                }
            )
        # CHANGELOG 2026-09 (novelty): the two evidential-fusion
        # hyperparameters are suggested AFTER the shared reliability block so
        # the shared prefix of the suggestion sequence is byte-identical to the
        # reliability_residual study.  Guarded on presence in the architecture's
        # own space rather than on the architecture name, so a family member
        # that does not declare them simply skips the suggestion.
        if "EVIDENTIAL_PRECISION_FLOOR" in architecture_space:
            parameters["EVIDENTIAL_PRECISION_FLOOR"] = trial.suggest_float(
                "EVIDENTIAL_PRECISION_FLOOR",
                *architecture_space["EVIDENTIAL_PRECISION_FLOOR"],
            )
        if "EVIDENTIAL_GATE_TEMPERATURE" in architecture_space:
            parameters["EVIDENTIAL_GATE_TEMPERATURE"] = trial.suggest_float(
                "EVIDENTIAL_GATE_TEMPERATURE",
                *architecture_space["EVIDENTIAL_GATE_TEMPERATURE"],
                log=True,
            )
        return parameters
    if architecture != "cross_attention":
        raise ValueError(f"Unknown architecture: {architecture}")
    heads = trial.suggest_categorical("N_HEADS", SEARCH_SPACE["N_HEADS"])
    width = trial.suggest_categorical("HEAD_WIDTH", SEARCH_SPACE["HEAD_WIDTH"])
    return {
        "N_HEADS": heads,
        "D_MODEL": heads * width,
        "N_CROSS_BLOCKS": trial.suggest_int(
            "N_CROSS_BLOCKS", *SEARCH_SPACE["N_CROSS_BLOCKS"]
        ),
        "N_FUSION_LAYERS": trial.suggest_int(
            "N_FUSION_LAYERS", *SEARCH_SPACE["N_FUSION_LAYERS"]
        ),
        "N_ESM_SLOTS": trial.suggest_categorical(
            "N_ESM_SLOTS", SEARCH_SPACE["N_ESM_SLOTS"]
        ),
        "DROPOUT": trial.suggest_float("DROPOUT", *SEARCH_SPACE["DROPOUT"]),
        "DEEP_LR": trial.suggest_float(
            "DEEP_LR", *SEARCH_SPACE["DEEP_LR"], log=True
        ),
        "DEEP_WD": trial.suggest_float(
            "DEEP_WD", *SEARCH_SPACE["DEEP_WD"], log=True
        ),
        "WARMUP_EPOCHS": trial.suggest_int(
            "WARMUP_EPOCHS", *SEARCH_SPACE["WARMUP_EPOCHS"]
        ),
        "EMA_DECAY": trial.suggest_categorical(
            "EMA_DECAY", SEARCH_SPACE["EMA_DECAY"]
        ),
        "MIXUP_ALPHA": trial.suggest_float(
            "MIXUP_ALPHA", *SEARCH_SPACE["MIXUP_ALPHA"]
        ),
        "LABEL_SMOOTH": trial.suggest_float(
            "LABEL_SMOOTH", *SEARCH_SPACE["LABEL_SMOOTH"]
        ),
        "BATCH_SIZE": trial.suggest_categorical(
            "BATCH_SIZE", SEARCH_SPACE["BATCH_SIZE"]
        ),
    }


def _decode_study_parameters(
    parameters: dict[str, Any], architecture: str
) -> dict[str, Any]:
    decoded = dict(parameters)
    if architecture == "cross_attention":
        width = int(decoded.pop("HEAD_WIDTH"))
        decoded["D_MODEL"] = int(decoded["N_HEADS"]) * width
    # CHANGELOG 2026-09 (novelty): family predicate, not proposed-model
    # equality.  MIXUP_ALPHA is never suggested for family members (mixup would
    # interpolate binary availability semantics), so it must be reinstated here
    # when decoding a stored trial for any family member.
    elif C.is_reliability_family(architecture):
        decoded["MIXUP_ALPHA"] = 0.0
    elif architecture not in {"concatenation", "gated_fusion"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    return decoded


def _objective_value(metrics: dict[str, Any], objective: str) -> float:
    if objective != "composite":
        value = metrics.get(objective)
        return -1.0 if value is None else float(value)
    auroc = -1.0 if metrics["auroc"] is None else float(metrics["auroc"])
    auprc = -1.0 if metrics["auprc"] is None else float(metrics["auprc"])
    mcc = -1.0 if metrics["mcc"] is None else float(metrics["mcc"])
    return (
        0.40 * mcc
        + 0.30 * auroc
        + 0.20 * auprc
        - 0.10 * float(metrics["brier"])
    )


def _fit_and_predict(
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
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    # CHANGELOG 2026-09 (novelty): family predicate replaces the equality test.
    # Every reliability-family member needs the augmented schema, the
    # architecture-specific constant-column mask and the unscaled passthrough of
    # anchor/gate columns; without this the new architecture would be handed the
    # comparator schema and ReliabilityFamilyNet.__init__ would raise on the
    # missing gate features.
    reliability_family = C.is_reliability_family(architecture)
    if reliability_family:
        if feature_names is None:
            raise ValueError("Reliability training requires ordered feature names")
        feature_mask = C.architecture_feature_mask(
            values[fit], feature_names, architecture
        )
        feature_names = [
            name for name, keep in zip(feature_names, feature_mask) if keep
        ]
        values = values[:, feature_mask]
    passthrough = (
        C.reliability_passthrough_indices(feature_names or [])
        if reliability_family
        else []
    )
    preprocessors = C.fit_preprocessors(
        values[fit],
        embeddings[fit],
        bio_passthrough_indices=passthrough,
    )
    if len(training_seeds) != C.N_ENSEMBLE:
        raise ValueError(
            f"Expected {C.N_ENSEMBLE} fixed training seeds; found {len(training_seeds)}"
        )
    model = C.train_deep_model(
        preprocessors.transform_bio(values[fit], copy=False),
        preprocessors.transform_esm(embeddings[fit], copy=False),
        labels[fit],
        preprocessors.transform_bio(values[stop], copy=False),
        preprocessors.transform_esm(embeddings[stop], copy=False),
        labels[stop],
        preprocessors.transform_bio(values[temperature], copy=False),
        preprocessors.transform_esm(embeddings[temperature], copy=False),
        labels[temperature],
        max_epochs=C.DEEP_MAX_EPOCHS,
        patience=C.DEEP_PATIENCE,
        architecture=architecture,
        seeds=training_seeds,
        feature_names=feature_names,
    )
    threshold_probabilities = C.predict(
        model,
        preprocessors.transform_bio(values[threshold_set], copy=False),
        preprocessors.transform_esm(embeddings[threshold_set], copy=False),
    )
    threshold = C.select_threshold(labels[threshold_set], threshold_probabilities)
    probabilities = C.predict(
        model,
        preprocessors.transform_bio(values[validation], copy=False),
        preprocessors.transform_esm(embeddings[validation], copy=False),
    )
    decisions = (probabilities >= threshold).astype(np.int8)
    del model
    if C.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return probabilities, decisions, threshold


def _evaluate_inner_trial(
    trial: Any,
    inner_plan: list[dict[str, Any]],
    values: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    architecture: str,
    objective: str,
    feature_names: list[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    validation_rows: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    decisions: list[np.ndarray] = []
    thresholds: list[np.ndarray] = []
    for fold_record in inner_plan:
        fold = int(fold_record["inner_fold"])
        fit = _record_indices(fold_record["fit"])
        stop = _record_indices(fold_record["early_stopping"])
        temperature = _record_indices(fold_record["temperature"])
        threshold_set = _record_indices(fold_record["threshold"])
        validation = _record_indices(fold_record["validation"])
        # CHANGELOG 2026-09 (novelty): family predicate, so both reliability
        # family members receive their ordered feature names.
        reliability_options = (
            {"feature_names": feature_names}
            if C.is_reliability_family(architecture)
            else {}
        )
        predicted, classified, threshold = _fit_and_predict(
            values,
            embeddings,
            labels,
            fit,
            stop,
            temperature,
            threshold_set,
            validation,
            architecture,
            [int(seed) for seed in fold_record["training_seeds"]],
            **reliability_options,
        )
        validation_rows.append(validation)
        probabilities.append(predicted)
        decisions.append(classified)
        thresholds.append(np.full(len(validation), threshold))
        partial_labels = labels[np.concatenate(validation_rows)]
        partial_probabilities = np.concatenate(probabilities)
        partial_decisions = np.concatenate(decisions)
        partial_thresholds = np.concatenate(thresholds)
        metrics = C.evaluate(
            partial_labels,
            partial_probabilities,
            partial_thresholds,
            predictions=partial_decisions,
        )
        trial.report(_objective_value(metrics, objective), fold)
        if trial.should_prune():
            import optuna

            raise optuna.TrialPruned(f"Pruned after inner fold {fold}")
    rows = np.concatenate(validation_rows)
    all_probabilities = np.concatenate(probabilities)
    all_decisions = np.concatenate(decisions)
    all_thresholds = np.concatenate(thresholds)
    metrics = C.evaluate(
        labels[rows],
        all_probabilities,
        all_thresholds,
        predictions=all_decisions,
    )
    return metrics, rows, all_probabilities, all_decisions


def _evaluate_outer_fold(
    outer_plan: dict[str, Any],
    values: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    architecture: str,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    final = outer_plan["final_partitions"]
    fit = _record_indices(final["fit"])
    stop = _record_indices(final["early_stopping"])
    temperature = _record_indices(final["temperature"])
    threshold_set = _record_indices(final["threshold"])
    outer_validation = _record_indices(outer_plan["outer_validation"])
    # CHANGELOG 2026-09 (novelty): family predicate, so both reliability family
    # members receive their ordered feature names.
    reliability_options = (
        {"feature_names": feature_names}
        if C.is_reliability_family(architecture)
        else {}
    )
    probabilities, decisions, threshold = _fit_and_predict(
        values,
        embeddings,
        labels,
        fit,
        stop,
        temperature,
        threshold_set,
        outer_validation,
        architecture,
        [int(seed) for seed in outer_plan["final_training_seeds"]],
        **reliability_options,
    )
    metrics = C.evaluate(
        labels[outer_validation],
        probabilities,
        threshold,
        predictions=decisions,
    )
    metrics["partition_sizes"] = {
        "fit": len(fit),
        "early_stopping": len(stop),
        "temperature": len(temperature),
        "threshold": len(threshold_set),
        "outer_validation": len(outer_validation),
    }
    return probabilities, decisions, threshold, metrics


def _gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    summary = C.gpu_runtime_summary()
    summary["available"] = True
    for device in summary["devices"]:
        device["capability"] = list(
            torch.cuda.get_device_capability(int(device["index"]))
        )
    return summary


def _study_history(study: Any, architecture: str, outer_fold: int) -> pd.DataFrame:
    history = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "state",
        )
    )
    history.insert(0, "outer_fold", outer_fold)
    history.insert(0, "architecture", architecture)
    return history


def _persist_complete_study_history(
    optuna: Any,
    storage: str,
    architectures: Iterable[str],
    outer_folds: int,
) -> pd.DataFrame:
    """Rebuild the CSV from persistent storage, including resumed studies.

    Incrementally concatenating only studies evaluated by the current process
    loses earlier architectures when ``--resume`` skips their checkpoints.
    SQLite is the authoritative trial ledger, so every refresh reads all
    expected studies that currently exist and writes one deterministic table.
    """
    expected = {
        (
            f"{architecture}_{MODEL_TAG}_{PROTOCOL_VERSION}_outer_{outer_fold}"
        ): (architecture, outer_fold)
        for architecture in architectures
        for outer_fold in range(1, int(outer_folds) + 1)
    }
    available = {
        summary.study_name
        for summary in optuna.get_all_study_summaries(storage=storage)
    }
    frames = []
    for study_name, (architecture, outer_fold) in expected.items():
        if study_name not in available:
            continue
        study = optuna.load_study(study_name=study_name, storage=storage)
        frames.append(_study_history(study, architecture, outer_fold))
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(
        ["architecture", "outer_fold", "number"], kind="stable"
    ).reset_index(drop=True)
    _atomic_csv(history, TRIALS_CSV)
    return history


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.trials < 1:
        raise ValueError("Trials must be positive")
    if arguments.outer_folds < 3:
        raise ValueError("At least three outer folds are required")
    if arguments.inner_folds < 2:
        raise ValueError("At least two inner folds are required")
    if arguments.search_epochs < 1 or arguments.final_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    if arguments.search_patience < 1 or arguments.final_patience < 1:
        raise ValueError("Patience values must be positive")
    if arguments.search_ensemble < 1 or arguments.final_ensemble < 1:
        raise ValueError("Ensemble counts must be positive")
    if not arguments.architectures:
        raise ValueError("At least one architecture is required")
    unknown = set(arguments.architectures) - set(PUBLICATION_ARCHITECTURES)
    if unknown:
        raise ValueError(f"Unknown architectures: {sorted(unknown)}")
    confirmation_seeds = [
        int(seed) for seed in getattr(arguments, "confirmation_split_seeds", [])
    ]
    if len(confirmation_seeds) != len(set(confirmation_seeds)):
        raise ValueError("Confirmatory split seeds must be unique")
    if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in confirmation_seeds):
        raise ValueError("Confirmatory split seeds must be unsigned 32-bit integers")
    primary_split_seed = int(
        arguments.seed
        if getattr(arguments, "split_seed", None) is None
        else arguments.split_seed
    )
    if primary_split_seed in confirmation_seeds:
        raise ValueError(
            "Confirmatory split seeds must be additional to the primary split seed"
        )
    if (
        confirmation_seeds
        and C.RELIABILITY_ARCHITECTURE not in arguments.architectures
    ):
        raise ValueError(
            "Confirmatory repeats require reliability_residual in --architectures"
        )


def _create_study(
    optuna: Any,
    storage: str,
    name: str,
    seed: int,
) -> Any:
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )


def _resolved_seeds(arguments: argparse.Namespace) -> dict[str, int]:
    """Resolve independent random streams while retaining --seed compatibility."""
    return {
        "legacy_seed": int(arguments.seed),
        "split_seed": int(
            arguments.seed if arguments.split_seed is None else arguments.split_seed
        ),
        "training_seed": int(
            arguments.seed
            if arguments.training_seed is None
            else arguments.training_seed
        ),
        "sampler_seed": int(
            arguments.seed if arguments.sampler_seed is None else arguments.sampler_seed
        ),
    }


def _consensus_reference_value(
    values: Iterable[Any],
) -> tuple[Any, dict[str, Any]]:
    """Select a deterministic mode without consulting outer-fold performance."""
    candidates = list(values)
    if not candidates:
        raise RuntimeError("Primary reference metadata contains no fixed candidates")
    by_hash: dict[str, dict[str, Any]] = {}
    for value in candidates:
        digest = _canonical_sha256(value)
        record = by_hash.setdefault(
            digest, {"value": value, "count": 0, "sha256": digest}
        )
        record["count"] += 1
    selected = min(
        by_hash.values(), key=lambda record: (-int(record["count"]), record["sha256"])
    )
    return selected["value"], {
        "rule": (
            "most_frequent_primary_fold_training_selected_value_then_"
            "lexicographically_smallest_canonical_sha256"
        ),
        "uses_primary_outer_validation_performance": False,
        "candidates": sorted(by_hash.values(), key=lambda record: record["sha256"]),
        "selected_sha256": selected["sha256"],
    }


def _fixed_confirmatory_reference_parameters(
    reference_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Freeze reference settings learned only inside the primary training folds."""
    if reference_results is None:
        return {}
    logistic = reference_results.get("esm_conservation_logistic", {})
    lightgbm = reference_results.get("lightgbm", {})
    logistic_values = [
        float(record["regularization_C"])
        for record in logistic.get("fold_metadata", [])
        if record.get("regularization_C") is not None
    ]
    lightgbm_values = [
        dict(record["selected_hyperparameters"])
        for record in lightgbm.get("fold_metadata", [])
        if record.get("selected_hyperparameters") is not None
    ]
    regularization, logistic_selection = _consensus_reference_value(
        logistic_values
    )
    tree_parameters, lightgbm_selection = _consensus_reference_value(
        lightgbm_values
    )
    if dict(tree_parameters) not in LGBM_NESTED_GRID:
        raise RuntimeError(
            "Primary LightGBM consensus is outside the prespecified nested grid"
        )
    return {
        "raw_esm_zero_shot": {
            "parameters": {},
            "selection": "prespecified_direction_preserving_calibration",
        },
        "esm_conservation_logistic": {
            "parameters": {"regularization_C": float(regularization)},
            "selection": logistic_selection,
        },
        "lightgbm": {
            "parameters": dict(tree_parameters),
            "selection": lightgbm_selection,
        },
    }


def _confirmatory_seed_plan(
    requested_split_seeds: Iterable[int],
    primary_seeds: dict[str, int],
    architecture_results: dict[str, dict[str, Any]],
    reference_results: dict[str, Any] | None = None,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable fixed-configuration repeated-evaluation protocol."""
    split_seeds = [int(seed) for seed in requested_split_seeds]
    production = architecture_results.get(C.RELIABILITY_ARCHITECTURE, {}).get(
        "production_params", {}
    )
    if split_seeds and not production:
        raise RuntimeError(
            "Confirmatory repeats require reliability_residual in the primary HPO run"
        )
    fixed_references = _fixed_confirmatory_reference_parameters(reference_results)
    if split_seeds and reference_results is not None and not fixed_references:
        raise RuntimeError("Confirmatory reference configurations could not be frozen")
    plan: dict[str, Any] = {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "status": "scheduled_for_execution",
        "scientific_role": (
            "separate_confirmatory_fixed_configuration_repeated_group_cv"
        ),
        "execution_policy": (
            "reuse_primary_nested_cv_selected_fixed_configuration_without_rehpo"
        ),
        "architecture": C.RELIABILITY_ARCHITECTURE,
        "strongest_prespecified_references": [
            "raw_esm_zero_shot",
            "esm_conservation_logistic",
            "lightgbm",
        ],
        "primary_split_seed": int(primary_seeds["split_seed"]),
        "production_parameters": dict(production),
        "production_parameters_sha256": _canonical_sha256(production),
        "fixed_reference_parameters": fixed_references,
        "fixed_reference_parameters_sha256": _canonical_sha256(fixed_references),
        "execution_context": dict(execution_context or {}),
        "repeats": [
            {
                "repeat": index,
                "split_seed": split_seed,
                "training_seed": int(primary_seeds["training_seed"] + index * 100_003),
                "sampler_seed": None,
                "rehpo": False,
                "output_namespace": f"repeat_{index:03d}_seed_{split_seed}",
            }
            for index, split_seed in enumerate(split_seeds, 1)
        ],
        "inference_requirement": (
            "report every prespecified repeat and aggregate paired effects; "
            "never select seeds or configurations by confirmatory performance"
        ),
        "uncertainty_scope": (
            "joint_split_and_training_instability_for_one_primary_selected_"
            "fixed_configuration_not_additional_hyperparameter_uncertainty"
        ),
        "authoritative_primary_artifacts_unchanged": True,
    }
    plan["canonical_sha256"] = _canonical_sha256(plan)
    return plan


def _parameter_distance(
    first: dict[str, Any],
    second: dict[str, Any],
    architecture: str,
) -> float:
    """Return a normalized distance used only for deterministic consensus."""
    space = _search_space_for(architecture)
    categorical = {"N_HEADS", "N_ESM_SLOTS", "EMA_DECAY", "BATCH_SIZE", "D_MODEL"}
    log_scaled = {"DEEP_LR", "DEEP_WD"}
    distances: list[float] = []
    for name in sorted(set(first) & set(second)):
        if name not in C.TUNABLE_CONSTANTS:
            continue
        one = first[name]
        two = second[name]
        if name in categorical:
            distances.append(float(one != two))
            continue
        bounds = space.get(name)
        if not isinstance(bounds, list) or len(bounds) != 2:
            distances.append(float(one != two))
            continue
        lower, upper = float(bounds[0]), float(bounds[1])
        one_value, two_value = float(one), float(two)
        if name in log_scaled:
            lower, upper = math.log(lower), math.log(upper)
            one_value, two_value = math.log(one_value), math.log(two_value)
        scale = upper - lower
        distances.append(abs(one_value - two_value) / scale if scale else 0.0)
    return float(np.mean(distances)) if distances else 0.0


def _select_production_parameters(
    fold_results: list[dict[str, Any]],
    architecture: str,
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the hyperparameter medoid without consulting outer-fold outcomes."""
    if not fold_results:
        raise ValueError(f"No fold results available for {architecture}")
    candidates = [dict(record["best_parameters"]) for record in fold_results]
    totals = [
        sum(
            _parameter_distance(candidate, other, architecture)
            for other in candidates
        )
        for candidate in candidates
    ]
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (totals[index], _canonical_sha256(candidates[index])),
    )
    selected_index = ranked[0]
    selected = dict(candidates[selected_index])
    selected.update(
        {
            "N_ENSEMBLE": int(arguments.final_ensemble),
            "DEEP_MAX_EPOCHS": int(arguments.final_epochs),
            "DEEP_PATIENCE": int(arguments.final_patience),
        }
    )
    metadata = {
        "rule": "hyperparameter_medoid_of_outer_fold_inner_cv_winners",
        "uses_outer_validation_metrics": False,
        "source_outer_fold": int(fold_results[selected_index]["outer_fold"]),
        "total_normalized_distance": float(totals[selected_index]),
        "candidate_parameter_sha256": [
            _canonical_sha256(candidate) for candidate in candidates
        ],
        "tie_break": "lexicographically_smallest_canonical_parameter_sha256",
        "note": (
            "This is a deployable consensus candidate. Nested OOF metrics estimate "
            "the tuning procedure, not this single production configuration."
        ),
    }
    return selected, metadata


def _fit_lightgbm_fold(
    values: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    stop: np.ndarray,
    candidate: dict[str, Any],
    seed: int,
) -> tuple[LGBMClassifier, np.ndarray]:
    """Fit one deterministic LightGBM candidate using fold-local features."""
    feature_mask = C.nonconstant_feature_mask(values[fit])
    if not feature_mask.any():
        raise RuntimeError("LightGBM candidate has no varying fit-fold features")
    parameters = dict(C.LGBM_PARAMS)
    parameters.update(candidate)
    parameters["random_state"] = int(seed)
    parameters["scale_pos_weight"] = C.fold_class_weight(labels[fit])
    model = LGBMClassifier(**parameters)
    model.fit(
        values[fit][:, feature_mask],
        labels[fit],
        eval_set=[(values[stop][:, feature_mask], labels[stop])],
        eval_metric="aucpr",
        callbacks=[
            early_stopping(C.LGBM_EARLY_STOP, verbose=False),
            log_evaluation(0),
        ],
    )
    return model, feature_mask


def _select_nested_lightgbm_parameters(
    values: np.ndarray,
    labels: np.ndarray,
    outer_plan: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a tree configuration on the persisted inner validation folds."""
    candidate_records: list[dict[str, Any]] = []
    outer_fold = int(outer_plan["outer_fold"])
    for candidate in LGBM_NESTED_GRID:
        validation_labels: list[np.ndarray] = []
        validation_probabilities: list[np.ndarray] = []
        fold_details: list[dict[str, Any]] = []
        for inner_record in outer_plan["inner_folds"]:
            inner_fold = int(inner_record["inner_fold"])
            fit = _record_indices(inner_record["fit"])
            stop = _record_indices(inner_record["early_stopping"])
            validation = _record_indices(inner_record["validation"])
            model, feature_mask = _fit_lightgbm_fold(
                values,
                labels,
                fit,
                stop,
                dict(candidate),
                seed + outer_fold * 10_000 + inner_fold * 100,
            )
            predicted = model.booster_.predict(values[validation][:, feature_mask])
            validation_labels.append(labels[validation])
            validation_probabilities.append(np.asarray(predicted, dtype=float))
            fold_details.append(
                {
                    "inner_fold": inner_fold,
                    "best_iteration": int(model.best_iteration_ or 0),
                    "retained_features": int(feature_mask.sum()),
                    "validation_rows": int(len(validation)),
                }
            )
        pooled_labels = np.concatenate(validation_labels)
        pooled_probabilities = np.concatenate(validation_probabilities)
        score = float(
            average_precision_score(pooled_labels, pooled_probabilities)
        )
        candidate_records.append(
            {
                "parameters": dict(candidate),
                "inner_oof_auprc": score,
                "inner_folds": fold_details,
                "parameter_sha256": _canonical_sha256(candidate),
            }
        )
    selected = min(
        candidate_records,
        key=lambda record: (
            -float(record["inner_oof_auprc"]),
            str(record["parameter_sha256"]),
        ),
    )
    return dict(selected["parameters"]), {
        "selection_metric": "pooled_inner_oof_auprc",
        "selection_uses_outer_validation": False,
        "selected_inner_oof_auprc": float(selected["inner_oof_auprc"]),
        "selected_parameter_sha256": selected["parameter_sha256"],
        "candidate_results": candidate_records,
        "grid": [dict(value) for value in LGBM_NESTED_GRID],
    }


def _reference_baseline_oof(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    split_plan: dict[str, Any],
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Evaluate every prespecified non-deep baseline on identical outer folds."""
    raw_score = -pd.to_numeric(
        frame["esm_variant_score"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(raw_score).all():
        raise ValueError("Reference raw ESM score contains nonfinite values")
    conservation_columns = [
        column
        for column in (
            "GERP++_RS",
            "phyloP100way_vertebrate",
            "phastCons100way_vertebrate",
        )
        if column in frame
    ]
    if not conservation_columns:
        raise RuntimeError("ESM+conservation reference columns are unavailable")
    mutation_columns = [column for column in MUTATION_FEATURE_COLS if column in frame]
    availability_columns = select_availability_features(frame)
    biological_columns = C.select_features(frame)
    tabular_columns = select_tabular_features(biological_columns)
    if not tabular_columns:
        raise RuntimeError("LightGBM reference features are unavailable")
    logistic_specs: dict[str, tuple[np.ndarray, list[str]]] = {
        "esm_score_logistic": (
            frame[["esm_variant_score"]].to_numpy(dtype=np.float32),
            ["esm_variant_score"],
        ),
        "conservation_logistic": (
            frame[conservation_columns].to_numpy(dtype=np.float32),
            conservation_columns,
        ),
        "esm_conservation_logistic": (
            frame[["esm_variant_score", *conservation_columns]].to_numpy(
                dtype=np.float32
            ),
            ["esm_variant_score", *conservation_columns],
        ),
    }
    if mutation_columns:
        logistic_specs["mutation_logistic"] = (
            frame[mutation_columns].to_numpy(dtype=np.float32),
            mutation_columns,
        )
    if availability_columns:
        logistic_specs["availability_logistic"] = (
            frame[availability_columns].to_numpy(dtype=np.float32),
            availability_columns,
        )
    embedding_names = [
        f"esm_embedding_{index}" for index in range(embeddings.shape[1])
    ]
    embedding_extras = ["esm_variant_score", *mutation_columns]
    logistic_specs["esm_embedding_mutation"] = (
        np.column_stack(
            [
                np.asarray(embeddings, dtype=np.float32),
                frame[embedding_extras].to_numpy(dtype=np.float32),
            ]
        ),
        [*embedding_names, *embedding_extras],
    )
    lightgbm_values = frame[tabular_columns].to_numpy(dtype=np.float32)
    names = (
        "raw_esm_zero_shot",
        *logistic_specs,
        "lightgbm",
    )
    predictions = {
        name: {
            "probabilities": np.full(len(frame), np.nan, dtype=np.float64),
            "decisions": np.full(len(frame), -1, dtype=np.int8),
            "thresholds": np.full(len(frame), np.nan, dtype=np.float64),
        }
        for name in names
    }
    fold_metadata: dict[str, list[dict[str, Any]]] = {
        name: [] for name in names
    }
    for outer_plan in split_plan["folds"]:
        fold = int(outer_plan["outer_fold"])
        partitions = outer_plan["final_partitions"]
        fit = _record_indices(partitions["fit"])
        stop = _record_indices(partitions["early_stopping"])
        calibration = _record_indices(partitions["temperature"])
        threshold_set = _record_indices(partitions["threshold"])
        validation = _record_indices(outer_plan["outer_validation"])

        raw_calibrator = C.fit_direction_preserving_calibrator(
            labels[calibration], raw_score[calibration]
        )
        raw_threshold_probabilities = raw_calibrator.predict_from_logits(
            raw_score[threshold_set]
        )
        raw_threshold = C.select_threshold(
            labels[threshold_set], raw_threshold_probabilities
        )
        raw_probabilities = raw_calibrator.predict_from_logits(raw_score[validation])
        predictions["raw_esm_zero_shot"]["probabilities"][validation] = (
            raw_probabilities
        )
        predictions["raw_esm_zero_shot"]["thresholds"][validation] = raw_threshold
        predictions["raw_esm_zero_shot"]["decisions"][validation] = (
            raw_probabilities >= raw_threshold
        ).astype(np.int8)
        fold_metadata["raw_esm_zero_shot"].append(
            {
                "outer_fold": fold,
                "feature_names": ["esm_variant_score"],
                "calibration_slope": raw_calibrator.slope,
                "calibration_intercept": raw_calibrator.intercept,
                "threshold": raw_threshold,
            }
        )

        for offset, (name, (source_values, requested_names)) in enumerate(
            logistic_specs.items(), 1
        ):
            feature_mask = C.nonconstant_feature_mask(source_values[fit])
            if not feature_mask.any():
                raise RuntimeError(
                    f"Reference baseline {name} has no varying features in fold {fold}"
                )
            fold_values = source_values[:, feature_mask]
            preprocessor = C.ArrayPreprocessor.fit(fold_values[fit], copy=False)
            fit_values = preprocessor.transform(fold_values[fit], copy=False)
            stop_values = preprocessor.transform(fold_values[stop], copy=False)
            model = C.fit_regularized_logistic(
                fit_values,
                stop_values,
                labels[fit],
                labels[stop],
                seed=seed + fold * 100 + offset,
                primary_metric="auprc",
            )
            calibration_values = preprocessor.transform(
                fold_values[calibration], copy=False
            )
            probability_calibrator = C.fit_probability_calibrator(
                labels[calibration], model.predict_proba(calibration_values)[:, 1]
            )
            threshold_values = preprocessor.transform(
                fold_values[threshold_set], copy=False
            )
            threshold_probabilities = probability_calibrator.predict(
                model.predict_proba(threshold_values)[:, 1]
            )
            operating_threshold = C.select_threshold(
                labels[threshold_set], threshold_probabilities
            )
            validation_values = preprocessor.transform(
                fold_values[validation], copy=False
            )
            validation_probabilities = probability_calibrator.predict(
                model.predict_proba(validation_values)[:, 1]
            )
            predictions[name]["probabilities"][validation] = validation_probabilities
            predictions[name]["thresholds"][validation] = operating_threshold
            predictions[name]["decisions"][validation] = (
                validation_probabilities >= operating_threshold
            ).astype(np.int8)
            fold_metadata[name].append(
                {
                    "outer_fold": fold,
                    "feature_names": [
                        feature
                        for feature, keep in zip(requested_names, feature_mask)
                        if keep
                    ],
                    "regularization_C": float(model.C),
                    "calibration_slope": probability_calibrator.coefficient,
                    "calibration_intercept": probability_calibrator.intercept,
                    "threshold": operating_threshold,
                }
            )

        selected_lightgbm, lightgbm_selection = (
            _select_nested_lightgbm_parameters(
                lightgbm_values,
                labels,
                outer_plan,
                seed,
            )
        )
        lightgbm, lightgbm_mask = _fit_lightgbm_fold(
            lightgbm_values,
            labels,
            fit,
            stop,
            selected_lightgbm,
            seed + fold,
        )
        fold_lightgbm = lightgbm_values[:, lightgbm_mask]
        lightgbm_calibrator = C.fit_probability_calibrator(
            labels[calibration],
            lightgbm.booster_.predict(fold_lightgbm[calibration]),
        )
        lightgbm_threshold_probabilities = lightgbm_calibrator.predict(
            lightgbm.booster_.predict(fold_lightgbm[threshold_set])
        )
        lightgbm_threshold = C.select_threshold(
            labels[threshold_set], lightgbm_threshold_probabilities
        )
        lightgbm_probabilities = lightgbm_calibrator.predict(
            lightgbm.booster_.predict(fold_lightgbm[validation])
        )
        predictions["lightgbm"]["probabilities"][validation] = (
            lightgbm_probabilities
        )
        predictions["lightgbm"]["thresholds"][validation] = lightgbm_threshold
        predictions["lightgbm"]["decisions"][validation] = (
            lightgbm_probabilities >= lightgbm_threshold
        ).astype(np.int8)
        fold_metadata["lightgbm"].append(
            {
                "outer_fold": fold,
                "feature_names": [
                    feature
                    for feature, keep in zip(tabular_columns, lightgbm_mask)
                    if keep
                ],
                "best_iteration": int(lightgbm.best_iteration_ or 0),
                "selected_hyperparameters": selected_lightgbm,
                "nested_inner_selection": lightgbm_selection,
                "calibration_slope": lightgbm_calibrator.coefficient,
                "calibration_intercept": lightgbm_calibrator.intercept,
                "threshold": lightgbm_threshold,
            }
        )
    results: dict[str, Any] = {}
    for name, arrays in predictions.items():
        if (
            not np.isfinite(arrays["probabilities"]).all()
            or not np.isfinite(arrays["thresholds"]).all()
            or (arrays["decisions"] < 0).any()
        ):
            raise RuntimeError(f"Reference baseline {name} OOF output is incomplete")
        metrics = C.evaluate(
            labels,
            arrays["probabilities"],
            arrays["thresholds"],
            predictions=arrays["decisions"],
        )
        metrics["ci95"] = C.group_bootstrap_intervals(
            labels,
            arrays["probabilities"],
            arrays["decisions"],
            groups,
            seed=seed,
        )
        if name in logistic_specs:
            metrics["feature_names"] = logistic_specs[name][1]
            metrics["model_family"] = "regularized_logistic"
        elif name == "lightgbm":
            metrics["feature_names"] = tabular_columns
            metrics["model_family"] = (
                "lightgbm_compact_grid_nested_inner_tuned_and_early_stopped"
            )
            metrics["hyperparameter_selection"] = (
                "same_persisted_inner_group_folds_as_neural_architectures"
            )
        else:
            metrics["feature_names"] = ["esm_variant_score"]
            metrics["model_family"] = "direction_preserving_raw_score_calibration"
        metrics["evaluation_role"] = "prespecified_nested_outer_reference"
        if name == "availability_logistic":
            metrics["evaluation_role"] = (
                "diagnostic_missingness_availability_negative_control"
            )
        metrics["fold_metadata"] = fold_metadata[name]
        results[name] = metrics
    return predictions, results


def _validate_confirmatory_plan(plan: dict[str, Any]) -> None:
    """Reject edited or incomplete confirmation protocols before any training."""
    if plan.get("schema_version") != 2:
        raise RuntimeError("Unsupported confirmatory seed-plan schema")
    expected = plan.get("canonical_sha256")
    unhashed = dict(plan)
    unhashed.pop("canonical_sha256", None)
    if expected != _canonical_sha256(unhashed):
        raise RuntimeError("Confirmatory seed-plan canonical hash is invalid")
    if plan.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Confirmatory seed plan uses a different Stage 14 protocol")
    if plan.get("architecture") != C.RELIABILITY_ARCHITECTURE:
        raise RuntimeError("Confirmatory seed plan does not target reliability_residual")
    if plan.get("production_parameters_sha256") != _canonical_sha256(
        plan.get("production_parameters", {})
    ):
        raise RuntimeError("Confirmatory production-parameter hash is invalid")
    if plan.get("fixed_reference_parameters_sha256") != _canonical_sha256(
        plan.get("fixed_reference_parameters", {})
    ):
        raise RuntimeError("Confirmatory fixed-reference hash is invalid")
    repeats = plan.get("repeats", [])
    if not repeats:
        raise RuntimeError("Confirmatory seed plan contains no repeats")
    split_seeds = [int(record["split_seed"]) for record in repeats]
    if len(split_seeds) != len(set(split_seeds)):
        raise RuntimeError("Confirmatory seed plan repeats a split seed")
    if int(plan["primary_split_seed"]) in split_seeds:
        raise RuntimeError("Confirmatory seed plan reuses the primary split seed")
    required_references = {
        "raw_esm_zero_shot",
        "esm_conservation_logistic",
        "lightgbm",
    }
    if set(plan.get("fixed_reference_parameters", {})) != required_references:
        raise RuntimeError("Confirmatory fixed-reference configuration is incomplete")


def _persist_confirmatory_plan(plan: dict[str, Any]) -> str:
    """Persist an immutable plan or fail closed on a conflicting prior run."""
    _validate_confirmatory_plan(plan)
    if CONFIRMATORY_SEED_PLAN.exists():
        stored = json.loads(CONFIRMATORY_SEED_PLAN.read_text(encoding="utf-8"))
        if stored != plan:
            raise RuntimeError(
                "Existing confirmatory seed plan uses a different protocol; "
                "use a clean Stage 14 output directory"
            )
    else:
        _atomic_json(plan, CONFIRMATORY_SEED_PLAN)
    digest = file_sha256(CONFIRMATORY_SEED_PLAN)
    if digest is None:
        raise RuntimeError("Could not hash the persisted confirmatory seed plan")
    return digest


def _confirmatory_reference_oof(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    split_plan: dict[str, Any],
    training_seed: int,
    fixed_parameters: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Evaluate three fixed reference procedures without confirmation-set HPO."""
    raw_score = -pd.to_numeric(
        frame["esm_variant_score"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(raw_score).all():
        raise ValueError("Confirmatory raw ESM score contains nonfinite values")
    conservation_columns = [
        column
        for column in (
            "GERP++_RS",
            "phyloP100way_vertebrate",
            "phastCons100way_vertebrate",
        )
        if column in frame
    ]
    if not conservation_columns:
        raise RuntimeError("Confirmatory ESM+conservation columns are unavailable")
    logistic_names = ["esm_variant_score", *conservation_columns]
    logistic_values = frame[logistic_names].to_numpy(dtype=np.float32)
    biological_columns = C.select_features(frame)
    tabular_columns = select_tabular_features(biological_columns)
    if not tabular_columns:
        raise RuntimeError("Confirmatory LightGBM features are unavailable")
    lightgbm_values = frame[tabular_columns].to_numpy(dtype=np.float32)
    regularization = float(
        fixed_parameters["esm_conservation_logistic"]["parameters"][
            "regularization_C"
        ]
    )
    tree_parameters = dict(fixed_parameters["lightgbm"]["parameters"])
    if regularization <= 0 or not np.isfinite(regularization):
        raise RuntimeError("Confirmatory logistic regularization is invalid")
    if tree_parameters not in LGBM_NESTED_GRID:
        raise RuntimeError("Confirmatory LightGBM parameters are outside the grid")

    reference_names = CONFIRMATORY_MODEL_NAMES[1:]
    predictions = {
        name: {
            "probabilities": np.full(len(frame), np.nan, dtype=np.float64),
            "decisions": np.full(len(frame), -1, dtype=np.int8),
            "thresholds": np.full(len(frame), np.nan, dtype=np.float64),
        }
        for name in reference_names
    }
    fold_metadata: dict[str, list[dict[str, Any]]] = {
        name: [] for name in reference_names
    }
    for outer_plan in split_plan["folds"]:
        fold = int(outer_plan["outer_fold"])
        partitions = outer_plan["final_partitions"]
        fit = _record_indices(partitions["fit"])
        stop = _record_indices(partitions["early_stopping"])
        calibration = _record_indices(partitions["temperature"])
        threshold_set = _record_indices(partitions["threshold"])
        validation = _record_indices(outer_plan["outer_validation"])

        raw_calibrator = C.fit_direction_preserving_calibrator(
            labels[calibration], raw_score[calibration]
        )
        raw_threshold_probabilities = raw_calibrator.predict_from_logits(
            raw_score[threshold_set]
        )
        raw_threshold = C.select_threshold(
            labels[threshold_set], raw_threshold_probabilities
        )
        raw_probabilities = raw_calibrator.predict_from_logits(raw_score[validation])
        raw_arrays = predictions["raw_esm_zero_shot"]
        raw_arrays["probabilities"][validation] = raw_probabilities
        raw_arrays["thresholds"][validation] = raw_threshold
        raw_arrays["decisions"][validation] = (
            raw_probabilities >= raw_threshold
        ).astype(np.int8)
        fold_metadata["raw_esm_zero_shot"].append(
            {
                "outer_fold": fold,
                "calibration_slope": raw_calibrator.slope,
                "calibration_intercept": raw_calibrator.intercept,
                "threshold": raw_threshold,
            }
        )

        logistic_mask = C.nonconstant_feature_mask(logistic_values[fit])
        if not logistic_mask.any():
            raise RuntimeError(
                f"Confirmatory logistic has no varying features in fold {fold}"
            )
        fold_logistic = logistic_values[:, logistic_mask]
        preprocessor = C.ArrayPreprocessor.fit(fold_logistic[fit], copy=False)
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=3000,
            random_state=int(training_seed + fold * 100 + 1),
            solver="lbfgs",
        )
        model.fit(
            preprocessor.transform(fold_logistic[fit], copy=False), labels[fit]
        )
        calibration_probabilities = model.predict_proba(
            preprocessor.transform(fold_logistic[calibration], copy=False)
        )[:, 1]
        logistic_calibrator = C.fit_probability_calibrator(
            labels[calibration], calibration_probabilities
        )
        logistic_threshold_probabilities = logistic_calibrator.predict(
            model.predict_proba(
                preprocessor.transform(fold_logistic[threshold_set], copy=False)
            )[:, 1]
        )
        logistic_threshold = C.select_threshold(
            labels[threshold_set], logistic_threshold_probabilities
        )
        logistic_probabilities = logistic_calibrator.predict(
            model.predict_proba(
                preprocessor.transform(fold_logistic[validation], copy=False)
            )[:, 1]
        )
        logistic_arrays = predictions["esm_conservation_logistic"]
        logistic_arrays["probabilities"][validation] = logistic_probabilities
        logistic_arrays["thresholds"][validation] = logistic_threshold
        logistic_arrays["decisions"][validation] = (
            logistic_probabilities >= logistic_threshold
        ).astype(np.int8)
        fold_metadata["esm_conservation_logistic"].append(
            {
                "outer_fold": fold,
                "feature_names": [
                    name for name, keep in zip(logistic_names, logistic_mask) if keep
                ],
                "regularization_C": regularization,
                "configuration_source": "fixed_primary_fold_consensus",
                "calibration_slope": logistic_calibrator.coefficient,
                "calibration_intercept": logistic_calibrator.intercept,
                "threshold": logistic_threshold,
            }
        )

        lightgbm, lightgbm_mask = _fit_lightgbm_fold(
            lightgbm_values,
            labels,
            fit,
            stop,
            tree_parameters,
            training_seed + fold * 100 + 2,
        )
        fold_lightgbm = lightgbm_values[:, lightgbm_mask]
        lightgbm_calibrator = C.fit_probability_calibrator(
            labels[calibration],
            lightgbm.booster_.predict(fold_lightgbm[calibration]),
        )
        lightgbm_threshold_probabilities = lightgbm_calibrator.predict(
            lightgbm.booster_.predict(fold_lightgbm[threshold_set])
        )
        lightgbm_threshold = C.select_threshold(
            labels[threshold_set], lightgbm_threshold_probabilities
        )
        lightgbm_probabilities = lightgbm_calibrator.predict(
            lightgbm.booster_.predict(fold_lightgbm[validation])
        )
        lightgbm_arrays = predictions["lightgbm"]
        lightgbm_arrays["probabilities"][validation] = lightgbm_probabilities
        lightgbm_arrays["thresholds"][validation] = lightgbm_threshold
        lightgbm_arrays["decisions"][validation] = (
            lightgbm_probabilities >= lightgbm_threshold
        ).astype(np.int8)
        fold_metadata["lightgbm"].append(
            {
                "outer_fold": fold,
                "feature_names": [
                    name
                    for name, keep in zip(tabular_columns, lightgbm_mask)
                    if keep
                ],
                "best_iteration": int(lightgbm.best_iteration_ or 0),
                "selected_hyperparameters": tree_parameters,
                "configuration_source": "fixed_primary_fold_consensus",
                "calibration_slope": lightgbm_calibrator.coefficient,
                "calibration_intercept": lightgbm_calibrator.intercept,
                "threshold": lightgbm_threshold,
            }
        )

    results: dict[str, Any] = {}
    for name, arrays in predictions.items():
        if (
            not np.isfinite(arrays["probabilities"]).all()
            or not np.isfinite(arrays["thresholds"]).all()
            or not np.isin(arrays["decisions"], (0, 1)).all()
        ):
            raise RuntimeError(f"Confirmatory reference {name} output is incomplete")
        metrics = C.evaluate(
            labels,
            arrays["probabilities"],
            arrays["thresholds"],
            predictions=arrays["decisions"],
        )
        metrics["ci95"] = C.group_bootstrap_intervals(
            labels,
            arrays["probabilities"],
            arrays["decisions"],
            groups,
            seed=training_seed,
        )
        metrics["evaluation_role"] = "fixed_configuration_confirmatory_reference"
        metrics["rehpo"] = False
        metrics["fold_metadata"] = fold_metadata[name]
        results[name] = metrics
    return predictions, results


def _checkpoint_path(repeat_record: dict[str, Any]) -> Path:
    return CONFIRMATORY_CHECKPOINT_DIR / (
        f"{repeat_record['output_namespace']}.json"
    )


def _checkpoint_arrays(
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {
            "probabilities": np.asarray(
                checkpoint["predictions"][name]["probabilities"], dtype=np.float64
            ),
            "decisions": np.asarray(
                checkpoint["predictions"][name]["decisions"], dtype=np.int8
            ),
            "thresholds": np.asarray(
                checkpoint["predictions"][name]["thresholds"], dtype=np.float64
            ),
        }
        for name in CONFIRMATORY_MODEL_NAMES
    }


def _validate_confirmation_checkpoint(
    checkpoint: dict[str, Any],
    plan: dict[str, Any],
    repeat_record: dict[str, Any],
    repeat_split_plan: dict[str, Any],
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Authenticate a completed seed checkpoint before resuming from it."""
    expected_identity = {
        "repeat": int(repeat_record["repeat"]),
        "split_seed": int(repeat_record["split_seed"]),
        "training_seed": int(repeat_record["training_seed"]),
    }
    if checkpoint.get("schema_version") != 1 or checkpoint.get("status") != "complete":
        raise RuntimeError("Confirmatory repeat checkpoint is incomplete or legacy")
    if checkpoint.get("confirmatory_plan_sha256") != plan["canonical_sha256"]:
        raise RuntimeError("Confirmatory checkpoint belongs to a different seed plan")
    if checkpoint.get("identity") != expected_identity:
        raise RuntimeError("Confirmatory checkpoint identity has changed")
    if checkpoint.get("repeat_split_plan") != repeat_split_plan:
        raise RuntimeError("Confirmatory checkpoint split partitions have changed")
    if checkpoint.get("row_order_sha256") != _ordered_text_sha256(
        frame[C.ROW_ID_COL].astype(str)
    ):
        raise RuntimeError("Confirmatory checkpoint row order has changed")
    if set(checkpoint.get("predictions", {})) != set(CONFIRMATORY_MODEL_NAMES):
        raise RuntimeError("Confirmatory checkpoint model panel is incomplete")
    fold_ids = np.asarray(checkpoint.get("fold_ids", []), dtype=np.int16)
    if fold_ids.shape != (len(frame),) or (fold_ids < 1).any():
        raise RuntimeError("Confirmatory checkpoint fold assignments are incomplete")
    group_fold_counts = pd.DataFrame(
        {"group": groups.astype(str), "fold": fold_ids}
    ).groupby("group", sort=False)["fold"].nunique()
    if (group_fold_counts != 1).any():
        raise RuntimeError("Confirmatory checkpoint splits a homology group")
    arrays = _checkpoint_arrays(checkpoint)
    for name, model_arrays in arrays.items():
        if any(array.shape != (len(frame),) for array in model_arrays.values()):
            raise RuntimeError(f"Confirmatory checkpoint {name} arrays are misaligned")
        if (
            not np.isfinite(model_arrays["probabilities"]).all()
            or not np.isfinite(model_arrays["thresholds"]).all()
            or not np.isin(model_arrays["decisions"], (0, 1)).all()
        ):
            raise RuntimeError(f"Confirmatory checkpoint {name} arrays are invalid")
        recomputed = C.evaluate(
            labels,
            model_arrays["probabilities"],
            model_arrays["thresholds"],
            predictions=model_arrays["decisions"],
        )
        stored = checkpoint.get("model_results", {}).get(name, {}).get("metrics")
        if stored != recomputed:
            raise RuntimeError(f"Confirmatory checkpoint {name} metrics are stale")


def _confirmation_repeat_split_plan(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    plan: dict[str, Any],
    repeat_record: dict[str, Any],
) -> dict[str, Any]:
    context = plan["execution_context"]
    outer_splits = C.make_group_splits(
        labels,
        groups,
        int(context["outer_folds"]),
        int(repeat_record["split_seed"]),
    )
    return _build_split_plan(
        frame,
        labels,
        groups,
        outer_splits,
        inner_folds=int(context["inner_folds"]),
        split_seed=int(repeat_record["split_seed"]),
        training_seed=int(repeat_record["training_seed"]),
        search_ensemble=1,
        final_ensemble=int(plan["production_parameters"]["N_ENSEMBLE"]),
        final_epochs=int(plan["production_parameters"]["DEEP_MAX_EPOCHS"]),
        final_patience=int(plan["production_parameters"]["DEEP_PATIENCE"]),
        deterministic=bool(context["deterministic"]),
        protocol_fingerprint={
            "scientific_role": plan["scientific_role"],
            "confirmatory_plan_sha256": plan["canonical_sha256"],
            "rehpo": False,
        },
    )


def _run_single_confirmation_repeat(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    base_feature_names: list[str],
    plan: dict[str, Any],
    repeat_record: dict[str, Any],
    repeat_split_plan: dict[str, Any],
) -> dict[str, Any]:
    """Train one fixed-configuration repeat and return an atomic checkpoint."""
    architecture = C.RELIABILITY_ARCHITECTURE
    feature_names = C.architecture_feature_names(
        architecture, frame, base_feature_names
    )
    expected_features = plan["execution_context"]["feature_names"]
    if feature_names != expected_features:
        raise RuntimeError("Confirmatory reliability feature schema has changed")
    values = frame[feature_names].to_numpy(dtype=np.float32)
    probabilities = np.full(len(frame), np.nan, dtype=np.float64)
    decisions = np.full(len(frame), -1, dtype=np.int8)
    thresholds = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_metrics: list[dict[str, Any]] = []
    baseline = _snapshot_constants()
    try:
        _apply_parameters(baseline)
        _apply_parameters(plan["production_parameters"])
        for outer_plan in repeat_split_plan["folds"]:
            fold = int(outer_plan["outer_fold"])
            validation = _record_indices(outer_plan["outer_validation"])
            C.set_seeds(
                int(outer_plan["final_training_seeds"][0]),
                deterministic=bool(plan["execution_context"]["deterministic"]),
            )
            predicted, classified, threshold, metrics = _evaluate_outer_fold(
                outer_plan,
                values,
                embeddings,
                labels,
                architecture,
                feature_names,
            )
            probabilities[validation] = predicted
            decisions[validation] = classified
            thresholds[validation] = threshold
            fold_ids[validation] = fold
            fold_metrics.append({"outer_fold": fold, "metrics": metrics})
    finally:
        _apply_parameters(baseline)
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(thresholds).all()
        or not np.isin(decisions, (0, 1)).all()
        or (fold_ids < 1).any()
    ):
        raise RuntimeError("Confirmatory reliability predictions are incomplete")
    model_metrics = C.evaluate(
        labels, probabilities, thresholds, predictions=decisions
    )
    reliability_results = {
        "metrics": model_metrics,
        "ci95": C.group_bootstrap_intervals(
            labels,
            probabilities,
            decisions,
            groups,
            seed=int(repeat_record["split_seed"]),
        ),
        "fold_results": fold_metrics,
        "evaluation_role": "confirmatory_fixed_primary_selected_configuration",
        "rehpo": False,
    }
    reference_predictions, reference_results = _confirmatory_reference_oof(
        frame,
        labels,
        groups,
        repeat_split_plan,
        int(repeat_record["training_seed"]),
        plan["fixed_reference_parameters"],
    )
    model_predictions = {
        architecture: {
            "probabilities": probabilities,
            "decisions": decisions,
            "thresholds": thresholds,
        },
        **reference_predictions,
    }
    model_results = {
        architecture: reliability_results,
        **{
            name: {
                "metrics": {
                    key: value
                    for key, value in result.items()
                    if key not in {"ci95", "fold_metadata", "evaluation_role", "rehpo"}
                },
                "ci95": result["ci95"],
                "fold_results": result["fold_metadata"],
                "evaluation_role": result["evaluation_role"],
                "rehpo": False,
            }
            for name, result in reference_results.items()
        },
    }
    comparisons = {
        f"{architecture}_minus_{reference}": C.clustered_model_comparison(
            labels,
            model_predictions[reference]["probabilities"],
            probabilities,
            model_predictions[reference]["decisions"],
            decisions,
            groups,
            seed=int(repeat_record["split_seed"]),
        )
        for reference in CONFIRMATORY_MODEL_NAMES[1:]
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
        "row_order_sha256": _ordered_text_sha256(
            frame[C.ROW_ID_COL].astype(str)
        ),
        "repeat_split_plan": repeat_split_plan,
        "fold_ids": fold_ids.tolist(),
        "predictions": {
            name: {key: value.tolist() for key, value in arrays.items()}
            for name, arrays in model_predictions.items()
        },
        "model_results": model_results,
        "comparisons": comparisons,
        "external_validation_touched": False,
    }


def _scalar_metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "mcc",
        "auroc",
        "auprc",
        "brier",
        "precision",
        "recall",
        "f1",
        "specificity",
        "npv",
    )
    summary: dict[str, Any] = {}
    for name in names:
        values = [float(record[name]) for record in records if record.get(name) is not None]
        if not values:
            summary[name] = None
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "values": values,
            "mean": round(float(array.mean()), 6),
            "standard_deviation": round(
                float(array.std(ddof=1)) if len(array) > 1 else 0.0, 6
            ),
            "minimum": round(float(array.min()), 6),
            "maximum": round(float(array.max()), 6),
        }
    return summary


def _execute_confirmatory_repeats(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    base_feature_names: list[str],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Execute or safely resume every prespecified fixed-config repeat."""
    _validate_confirmatory_plan(plan)
    context = plan["execution_context"]
    required_context = {
        "outer_folds",
        "inner_folds",
        "deterministic",
        "feature_names",
        "n_rows",
        "row_order_sha256",
        "input_sha256",
        "embedding_sha256",
        "primary_split_plan_canonical_sha256",
        "primary_split_plan_file_sha256",
    }
    if not required_context <= set(context):
        raise RuntimeError("Confirmatory execution context is incomplete")
    if int(context["n_rows"]) != len(frame):
        raise RuntimeError("Confirmatory input row count has changed")
    if context["row_order_sha256"] != _ordered_text_sha256(
        frame[C.ROW_ID_COL].astype(str)
    ):
        raise RuntimeError("Confirmatory input row order has changed")
    if context["input_sha256"] != file_sha256(INPUT_CSV):
        raise RuntimeError("Confirmatory modelling table content has changed")
    if context["embedding_sha256"] != file_sha256(INPUT_NPY):
        raise RuntimeError("Confirmatory embedding content has changed")
    CONFIRMATORY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict[str, Any]] = []
    for repeat_record in plan["repeats"]:
        repeat_split_plan = _confirmation_repeat_split_plan(
            frame, labels, groups, plan, repeat_record
        )
        checkpoint_file = _checkpoint_path(repeat_record)
        if checkpoint_file.exists():
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            _validate_confirmation_checkpoint(
                checkpoint,
                plan,
                repeat_record,
                repeat_split_plan,
                frame,
                labels,
                groups,
            )
            logger.info(
                "Resumed authenticated confirmation repeat %d (split seed %d)",
                int(repeat_record["repeat"]),
                int(repeat_record["split_seed"]),
            )
        else:
            checkpoint = _run_single_confirmation_repeat(
                frame,
                embeddings,
                labels,
                groups,
                base_feature_names,
                plan,
                repeat_record,
                repeat_split_plan,
            )
            _validate_confirmation_checkpoint(
                checkpoint,
                plan,
                repeat_record,
                repeat_split_plan,
                frame,
                labels,
                groups,
            )
            _atomic_json(checkpoint, checkpoint_file)
            logger.info(
                "Completed confirmation repeat %d/%d (split seed %d)",
                int(repeat_record["repeat"]),
                len(plan["repeats"]),
                int(repeat_record["split_seed"]),
            )
        checkpoints.append(checkpoint)

    repeat_ids = np.asarray(
        [record["identity"]["repeat"] for record in checkpoints], dtype=np.int16
    )
    split_seeds = np.asarray(
        [record["identity"]["split_seed"] for record in checkpoints], dtype=np.int64
    )
    training_seeds = np.asarray(
        [record["identity"]["training_seed"] for record in checkpoints],
        dtype=np.int64,
    )
    fold_ids = np.stack(
        [np.asarray(record["fold_ids"], dtype=np.int16) for record in checkpoints]
    )
    stacked_predictions = {
        name: {
            array_name: np.stack(
                [
                    np.asarray(
                        record["predictions"][name][array_name],
                        dtype=np.int8 if array_name == "decisions" else np.float64,
                    )
                    for record in checkpoints
                ]
            )
            for array_name in ("probabilities", "decisions", "thresholds")
        }
        for name in CONFIRMATORY_MODEL_NAMES
    }
    npz_values: dict[str, np.ndarray] = {
        "y": labels,
        "groups": groups,
        "row_ids": frame[C.ROW_ID_COL].astype(str).to_numpy(),
        "repeat_ids": repeat_ids,
        "split_seeds": split_seeds,
        "training_seeds": training_seeds,
        "fold_ids": fold_ids,
    }
    for name, arrays in stacked_predictions.items():
        prefix = name if name == C.RELIABILITY_ARCHITECTURE else f"reference__{name}"
        for array_name, values in arrays.items():
            npz_values[f"{prefix}__{array_name}"] = values
    _atomic_npz(npz_values, CONFIRMATORY_PREDICTIONS)

    fold_frames = []
    for index, checkpoint in enumerate(checkpoints):
        fold_frames.append(
            pd.DataFrame(
                {
                    "repeat": int(checkpoint["identity"]["repeat"]),
                    "split_seed": int(checkpoint["identity"]["split_seed"]),
                    "training_seed": int(checkpoint["identity"]["training_seed"]),
                    "row_index": np.arange(len(frame), dtype=np.int64),
                    C.ROW_ID_COL: frame[C.ROW_ID_COL].astype(str),
                    C.GENE_COL: frame[C.GENE_COL].astype(str),
                    "split_group": groups.astype(str),
                    C.LABEL_COL: labels,
                    "outer_fold": fold_ids[index],
                }
            )
        )
    _atomic_csv(pd.concat(fold_frames, ignore_index=True), CONFIRMATORY_FOLD_ASSIGNMENTS)

    per_seed: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        per_seed.append(
            {
                "identity": checkpoint["identity"],
                "repeat_split_plan_canonical_sha256": checkpoint[
                    "repeat_split_plan"
                ]["canonical_sha256"],
                "models": checkpoint["model_results"],
                "paired_comparisons": checkpoint["comparisons"],
            }
        )
    across_seed: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    instability: dict[str, Any] = {}
    for name, arrays in stacked_predictions.items():
        seed_metrics = [
            record["model_results"][name]["metrics"] for record in checkpoints
        ]
        across_seed[name] = _scalar_metric_summary(seed_metrics)
        flattened_probabilities = arrays["probabilities"].reshape(-1)
        flattened_decisions = arrays["decisions"].reshape(-1)
        flattened_thresholds = arrays["thresholds"].reshape(-1)
        repeated_labels = np.tile(labels, len(checkpoints))
        mean_probabilities = arrays["probabilities"].mean(axis=0)
        mean_thresholds = arrays["thresholds"].mean(axis=0)
        majority_decisions = (
            arrays["decisions"].mean(axis=0) >= 0.5
        ).astype(np.int8)
        pooled[name] = {
            "stacked_repeated_oof_descriptive": C.evaluate(
                repeated_labels,
                flattened_probabilities,
                flattened_thresholds,
                predictions=flattened_decisions,
            ),
            "mean_probability_repeated_oof_descriptive": C.evaluate(
                labels,
                mean_probabilities,
                mean_thresholds,
                predictions=majority_decisions,
            ),
        }
        row_standard_deviation = arrays["probabilities"].std(axis=0, ddof=0)
        instability[name] = {
            "per_row_probability_standard_deviation": {
                "mean": round(float(row_standard_deviation.mean()), 6),
                "median": round(float(np.median(row_standard_deviation)), 6),
                "p90": round(float(np.quantile(row_standard_deviation, 0.90)), 6),
                "maximum": round(float(row_standard_deviation.max()), 6),
            },
            "metric_variation": across_seed[name],
        }
    comparison_summaries: dict[str, Any] = {}
    proposed_seed_metrics = [
        record["model_results"][C.RELIABILITY_ARCHITECTURE]["metrics"]
        for record in checkpoints
    ]
    for reference in CONFIRMATORY_MODEL_NAMES[1:]:
        reference_seed_metrics = [
            record["model_results"][reference]["metrics"] for record in checkpoints
        ]
        metric_differences = []
        for proposed, comparator in zip(
            proposed_seed_metrics, reference_seed_metrics
        ):
            metric_differences.append(
                {
                    name: (
                        None
                        if proposed.get(name) is None or comparator.get(name) is None
                        else float(proposed[name]) - float(comparator[name])
                    )
                    for name in ("mcc", "auroc", "auprc", "brier")
                }
            )
        comparison_summaries[
            f"{C.RELIABILITY_ARCHITECTURE}_minus_{reference}"
        ] = _scalar_metric_summary(metric_differences)

    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "scientific_role": plan["scientific_role"],
        "authoritative_primary_nested_oof_replaced": False,
        "hyperparameter_selection_uses_confirmatory_results": False,
        "external_validation_touched": False,
        "confirmatory_plan_sha256": plan["canonical_sha256"],
        "confirmatory_plan_file_sha256": file_sha256(CONFIRMATORY_SEED_PLAN),
        "model_order": list(CONFIRMATORY_MODEL_NAMES),
        "repeat_count": len(checkpoints),
        "per_seed": per_seed,
        "across_seed_metric_summary": across_seed,
        "pooled_descriptive_metrics": pooled,
        "paired_metric_difference_summary": comparison_summaries,
        "split_and_training_instability": instability,
        "inference_scope": plan["uncertainty_scope"],
        "pooled_metrics_warning": (
            "Repeated OOF rows are correlated copies of the same variants; pooled "
            "metrics are descriptive and must not be reported as an independent "
            "larger sample. Per-seed paired effects are the confirmatory unit."
        ),
        "artifacts": {
            "predictions_npz": str(CONFIRMATORY_PREDICTIONS),
            "predictions_npz_sha256": file_sha256(CONFIRMATORY_PREDICTIONS),
            "fold_assignments_csv": str(CONFIRMATORY_FOLD_ASSIGNMENTS),
            "fold_assignments_csv_sha256": file_sha256(
                CONFIRMATORY_FOLD_ASSIGNMENTS
            ),
            "checkpoint_directory": str(CONFIRMATORY_CHECKPOINT_DIR),
        },
    }
    _atomic_json(result, CONFIRMATORY_RESULTS_JSON)
    return result


def _architecture_payload(
    architecture: str,
    fold_results: list[dict[str, Any]],
    probabilities: np.ndarray,
    decisions: np.ndarray,
    thresholds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    arguments: argparse.Namespace,
    seeds: dict[str, int],
    split_plan: dict[str, Any],
    split_plan_file_sha256: str,
) -> dict[str, Any]:
    metrics = C.evaluate(
        labels, probabilities, thresholds, predictions=decisions
    )
    metrics["ci95"] = C.group_bootstrap_intervals(
        labels,
        probabilities,
        decisions,
        groups,
        seed=seeds["split_seed"],
    )
    production_parameters, production_selection = _select_production_parameters(
        fold_results, architecture, arguments
    )
    return {
        "schema_version": 2,
        "model_tag": MODEL_TAG,
        "architecture": architecture,
        "architecture_interpretation": (
            "legacy_pooled_ESM_vector_projected_to_pseudo_slots_not_residue_tokens"
            if architecture == "cross_attention"
            else (
                "proposed_reliability_conditioned_bounded_residual_with_"
                "exact_sequence_fallback"
                if architecture == C.RELIABILITY_RESIDUAL_ARCHITECTURE
                else (
                    "precision_weighted_evidential_bounded_residual_with_"
                    "exact_sequence_fallback"
                    if architecture == C.EVIDENTIAL_RESIDUAL_ARCHITECTURE
                    else "valid_pooled_embedding_fusion_baseline"
                )
            )
        ),
        "reliability_protocol": (
            C.reliability_architecture_protocol(
                feature_names, production_parameters, architecture=architecture
            )
            if C.is_reliability_family(architecture)
            else None
        ),
        "selection_protocol": "nested_split_group_disjoint_cross_validation",
        "protocol_version": PROTOCOL_VERSION,
        "threshold_protocol": "fixed_dedicated_inner_threshold_partition",
        "calibration_protocol": (
            "fixed_dedicated_inner_affine_logit_calibration_partition"
        ),
        "production_parameter_rule": production_selection["rule"],
        "production_source_outer_fold": production_selection[
            "source_outer_fold"
        ],
        "production_selection": production_selection,
        "production_params": production_parameters,
        "best_params_by_outer_fold": {
            str(record["outer_fold"]): record["best_parameters"]
            for record in fold_results
        },
        "nested_outer_metrics": metrics,
        "fold_results": fold_results,
        "search": {
            "trials_target_per_outer_fold": int(arguments.trials),
            "outer_folds": int(arguments.outer_folds),
            "inner_folds": int(arguments.inner_folds),
            "search_ensemble": int(arguments.search_ensemble),
            "search_epochs": int(arguments.search_epochs),
            "search_patience": int(arguments.search_patience),
            "final_ensemble": int(arguments.final_ensemble),
            "final_epochs": int(arguments.final_epochs),
            "final_patience": int(arguments.final_patience),
            "objective": arguments.objective,
            "storage": arguments.storage,
            "search_space": _search_space_for(architecture),
            "feature_names": feature_names,
        },
        "reproducibility": {
            **seeds,
            "deterministic": not arguments.allow_nondeterministic,
            "protocol_fingerprint": split_plan.get("protocol_fingerprint"),
            "gpu": _gpu_metadata(),
            "input_sha256": file_sha256(INPUT_CSV),
            "embedding_sha256": file_sha256(INPUT_NPY),
            "split_plan_path": str(SPLIT_PLAN_FILE),
            "split_plan_canonical_sha256": split_plan["canonical_sha256"],
            "split_plan_file_sha256": split_plan_file_sha256,
        },
        "lora_policy": (
            "optional_exploratory_stage11_model_not_part_of_primary_nested_comparison"
        ),
        "external_validation_touched": False,
    }


def run_search(arguments: argparse.Namespace) -> None:
    """Run architecture-scoped HPO with fixed, persisted nested partitions."""
    _validate_arguments(arguments)
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Optuna is required for persistent tuning") from error
    ensure_directories(STAGE14_OUT)
    architectures = list(dict.fromkeys(arguments.architectures))
    confirmation_split_seeds = [
        int(seed)
        for seed in getattr(arguments, "confirmation_split_seeds", [])
    ]
    seeds = _resolved_seeds(arguments)
    protocol_fingerprint = _publication_protocol_fingerprint(arguments)
    C.set_seeds(
        seeds["training_seed"],
        deterministic=not arguments.allow_nondeterministic,
    )
    frame, embeddings, labels, groups, base_feature_names = _load_data()
    C.validate_groups(groups, labels, arguments.outer_folds)
    feature_names_by_architecture = {
        architecture: C.architecture_feature_names(
            architecture, frame, base_feature_names
        )
        for architecture in architectures
    }
    # OPTIMIZATION 2026-09: family members share one feature schema, so they
    # share one float32 matrix instead of each holding an identical copy.  With
    # the reliability family there are now five architectures but only two
    # distinct schemas; this removes three full copies from host RAM (~40 MB
    # each at 200k rows x 50 columns).  Sharing is safe: `values` is only ever
    # read or masked into a new array inside _fit_and_predict, never written in
    # place.
    _schema_matrix_cache: dict[tuple[str, ...], np.ndarray] = {}
    values_by_architecture = {}
    for architecture, names in feature_names_by_architecture.items():
        schema_key = tuple(names)
        matrix = _schema_matrix_cache.get(schema_key)
        if matrix is None:
            matrix = frame[names].to_numpy(dtype=np.float32)
            _schema_matrix_cache[schema_key] = matrix
        values_by_architecture[architecture] = matrix
    del _schema_matrix_cache
    baseline = _snapshot_constants()
    outer_splits = C.make_group_splits(
        labels, groups, arguments.outer_folds, seeds["split_seed"]
    )
    split_plan = _build_split_plan(
        frame,
        labels,
        groups,
        outer_splits,
        inner_folds=arguments.inner_folds,
        split_seed=seeds["split_seed"],
        training_seed=seeds["training_seed"],
        search_ensemble=arguments.search_ensemble,
        final_ensemble=arguments.final_ensemble,
        final_epochs=arguments.final_epochs,
        final_patience=arguments.final_patience,
        deterministic=not arguments.allow_nondeterministic,
        protocol_fingerprint=protocol_fingerprint,
    )
    split_plan_file_sha256 = _persist_and_validate_split_plan(
        split_plan, resume=getattr(arguments, "resume", False)
    )
    outer_fold_ids = np.full(len(frame), -1, dtype=np.int16)
    for outer_plan in split_plan["folds"]:
        outer_fold_ids[_record_indices(outer_plan["outer_validation"])] = int(
            outer_plan["outer_fold"]
        )
    architecture_predictions: dict[str, dict[str, np.ndarray]] = {}
    architecture_results: dict[str, dict[str, Any]] = {}
    all_fold_results: list[dict[str, Any]] = []
    _persist_complete_study_history(
        optuna, arguments.storage, architectures, arguments.outer_folds
    )
    try:
        for architecture in architectures:
            # --- Resume: skip architectures completed in a prior session ---
            if getattr(arguments, "resume", False):
                checkpoint = _load_architecture_checkpoint(architecture)
                if checkpoint is not None:
                    _ckpt_preds, _ckpt_folds, _ckpt_payload = checkpoint
                    architecture_predictions[architecture] = _ckpt_preds
                    architecture_results[architecture] = _ckpt_payload
                    all_fold_results.extend(_ckpt_folds)
                    logger.info(
                        "Resumed %s from architecture checkpoint (skipping)",
                        architecture,
                    )
                    _persist_complete_study_history(
                        optuna,
                        arguments.storage,
                        architectures,
                        arguments.outer_folds,
                    )
                    continue

            feature_names = feature_names_by_architecture[architecture]
            values = values_by_architecture[architecture]
            architecture_seed_offset = (
                PUBLICATION_ARCHITECTURES.index(architecture) * 1_000_000
            )
            oof_probabilities = np.full(len(frame), np.nan, dtype=np.float64)
            oof_decisions = np.full(len(frame), -1, dtype=np.int8)
            oof_thresholds = np.full(len(frame), np.nan, dtype=np.float64)
            fold_results: list[dict[str, Any]] = []
            for outer_plan in split_plan["folds"]:
                outer_fold = int(outer_plan["outer_fold"])
                outer_validation = _record_indices(
                    outer_plan["outer_validation"]
                )
                study_name = (
                    f"{architecture}_{MODEL_TAG}_{PROTOCOL_VERSION}_outer_"
                    f"{outer_fold}"
                )
                study = _create_study(
                    optuna,
                    arguments.storage,
                    study_name,
                    seeds["sampler_seed"]
                    + architecture_seed_offset
                    + outer_fold * 10_000,
                )
                study_protocol = {
                    "protocol_version": PROTOCOL_VERSION,
                    "architecture": architecture,
                    "outer_fold": outer_fold,
                    "outer_fold_plan_sha256": _canonical_sha256(outer_plan),
                    "split_plan_canonical_sha256": split_plan[
                        "canonical_sha256"
                    ],
                    "split_plan_file_sha256": split_plan_file_sha256,
                    "inner_folds": int(arguments.inner_folds),
                    "fixed_inner_training_seed_sha256": _canonical_sha256(
                        [
                            fold["training_seeds"]
                            for fold in outer_plan["inner_folds"]
                        ]
                    ),
                    "search_ensemble": int(arguments.search_ensemble),
                    "search_epochs": int(arguments.search_epochs),
                    "search_patience": int(arguments.search_patience),
                    "objective": arguments.objective,
                    "search_space": _search_space_for(architecture),
                    "feature_names": feature_names,
                    "reliability_protocol": (
                        C.reliability_architecture_protocol(
                            feature_names, architecture=architecture
                        )
                        if C.is_reliability_family(architecture)
                        else None
                    ),
                    "sampler_seed": seeds["sampler_seed"]
                    + architecture_seed_offset
                    + outer_fold * 10_000,
                    "final_epochs": int(arguments.final_epochs),
                    "final_patience": int(arguments.final_patience),
                    "deterministic": not arguments.allow_nondeterministic,
                    "protocol_fingerprint": protocol_fingerprint,
                }
                previous_protocol = study.user_attrs.get("protocol")
                if getattr(arguments, "resume", False) and previous_protocol is not None:
                    study_critical = (
                        "protocol_version",
                        "architecture",
                        "outer_fold",
                        "inner_folds",
                        "search_ensemble",
                        "search_epochs",
                        "search_patience",
                        "objective",
                        "search_space",
                        "feature_names",
                        "sampler_seed",
                        "final_epochs",
                        "final_patience",
                    )
                    study_mismatches = [
                        k
                        for k in study_critical
                        if previous_protocol.get(k) != study_protocol.get(k)
                    ]
                    if study_mismatches:
                        raise RuntimeError(
                            f"Study {study_name} protocol mismatch on resume: {study_mismatches}"
                        )
                    study_protocol = previous_protocol
                elif previous_protocol is None and study.trials:
                    raise RuntimeError(
                        f"Persistent study {study_name} contains unversioned trials; "
                        "use a clean Stage 14 Optuna storage for publication"
                    )
                elif (
                    previous_protocol is not None
                    and previous_protocol != study_protocol
                ):
                    raise RuntimeError(
                        f"Persistent study {study_name} uses a different protocol"
                    )
                study.set_user_attr("protocol", study_protocol)

                if getattr(arguments, "resume", False):
                    expected_rows = frame.iloc[outer_validation][
                        C.ROW_ID_COL
                    ].astype(str).tolist()
                    primary_ckpt = _load_primary_checkpoint(
                        architecture,
                        outer_fold,
                        study_name,
                        study_protocol,
                        arguments.trials,
                        expected_rows,
                    )
                    if primary_ckpt is not None:
                        record, predicted, classified, threshold = primary_ckpt
                        oof_probabilities[outer_validation] = predicted
                        oof_decisions[outer_validation] = classified
                        oof_thresholds[outer_validation] = threshold
                        fold_results.append(record)
                        all_fold_results.append(record)
                        logger.info(
                            "Resumed %s outer fold %d from primary checkpoint (skipping)",
                            architecture,
                            outer_fold,
                        )
                        _persist_complete_study_history(
                            optuna,
                            arguments.storage,
                            architectures,
                            arguments.outer_folds,
                        )
                        continue

                def objective(trial: Any) -> float:
                    # OPTIMIZATION 2026-09 (correctness under pruning): restore
                    # the module-level constants even if this trial is pruned or
                    # raises, mirroring _run_single_confirmation_repeat.  The
                    # search objective temporarily overrides N_ENSEMBLE /
                    # DEEP_MAX_EPOCHS / DEEP_PATIENCE; without a finally the
                    # search ensemble/epoch values would leak past a pruned
                    # trial into the outer-fold evaluation that follows the
                    # study, silently changing its training budget.
                    parameters = _suggest_parameters(trial, architecture)
                    trial.set_user_attr("architecture", architecture)
                    trial.set_user_attr("decoded_parameters", parameters)
                    trial.set_user_attr(
                        "outer_fold_plan_sha256",
                        study_protocol["outer_fold_plan_sha256"],
                    )
                    trial.set_user_attr(
                        "fixed_inner_training_seed_sha256",
                        study_protocol["fixed_inner_training_seed_sha256"],
                    )
                    try:
                        _apply_parameters(baseline)
                        _apply_parameters(
                            {
                                "N_ENSEMBLE": arguments.search_ensemble,
                                "DEEP_MAX_EPOCHS": arguments.search_epochs,
                                "DEEP_PATIENCE": arguments.search_patience,
                            }
                        )
                        _apply_parameters(parameters)
                        first_fixed_seed = int(
                            outer_plan["inner_folds"][0]["training_seeds"][0]
                        )
                        C.set_seeds(
                            first_fixed_seed,
                            deterministic=not arguments.allow_nondeterministic,
                        )
                        started = time.monotonic()
                        # CHANGELOG 2026-09 (fix): the evaluate call is wrapped
                        # so an out-of-memory RuntimeError is converted back to
                        # optuna.TrialPruned (instead of aborting the study) and
                        # the CUDA cache is freed.  This must live INSIDE the
                        # guarded region: the finally below restores the
                        # module-level constants regardless of the raise, while
                        # this nested except chooses which exception escapes.
                        try:
                            metrics, _, _, _ = _evaluate_inner_trial(
                                trial,
                                outer_plan["inner_folds"],
                                values,
                                embeddings,
                                labels,
                                architecture,
                                arguments.objective,
                                (
                                    feature_names
                                    if C.is_reliability_family(architecture)
                                    else None
                                ),
                            )
                        except RuntimeError as error:
                            if "out of memory" in str(error).lower():
                                if C.DEVICE == "cuda":
                                    torch.cuda.empty_cache()
                                raise optuna.TrialPruned(
                                    "GPU memory exhausted"
                                ) from error
                            raise
                        value = _objective_value(metrics, arguments.objective)
                        trial.set_user_attr("metrics", metrics)
                        trial.set_user_attr(
                            "seconds", time.monotonic() - started
                        )
                        return value
                    finally:
                        _apply_parameters(baseline)

                if getattr(arguments, "resume", False):
                    for trial in study.trials:
                        if trial.state == optuna.trial.TrialState.RUNNING:
                            try:
                                study.tell(
                                    trial.number,
                                    state=optuna.trial.TrialState.FAIL,
                                )
                                logger.info(
                                    "Marked interrupted trial %d as FAIL on resume",
                                    trial.number,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Could not mark trial %d as FAIL: %s",
                                    trial.number,
                                    exc,
                                )

                non_failed_trials = [
                    t
                    for t in study.trials
                    if t.state
                    in (
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                    )
                ]
                remaining = max(0, arguments.trials - len(non_failed_trials))
                if remaining:
                    study.optimize(
                        objective, n_trials=remaining, gc_after_trial=True
                    )
                non_failed_trials = [
                    trial
                    for trial in study.trials
                    if trial.state
                    in (
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                    )
                ]
                complete = [
                    trial
                    for trial in study.trials
                    if trial.state == optuna.trial.TrialState.COMPLETE
                    and trial.value is not None
                ]
                if not complete:
                    raise RuntimeError(
                        f"{architecture} outer fold {outer_fold} has no successful trials"
                    )
                best_parameters = _decode_study_parameters(
                    study.best_trial.params, architecture
                )
                _apply_parameters(baseline)
                _apply_parameters(
                    {
                        "N_ENSEMBLE": arguments.final_ensemble,
                        "DEEP_MAX_EPOCHS": arguments.final_epochs,
                        "DEEP_PATIENCE": arguments.final_patience,
                    }
                )
                _apply_parameters(best_parameters)
                C.set_seeds(
                    int(outer_plan["final_training_seeds"][0]),
                    deterministic=not arguments.allow_nondeterministic,
                )
                predicted, classified, threshold, metrics = (
                    _evaluate_outer_fold(
                        outer_plan,
                        values,
                        embeddings,
                        labels,
                        architecture,
                        (
                            feature_names
                            if C.is_reliability_family(architecture)
                            else None
                        ),
                    )
                )
                oof_probabilities[outer_validation] = predicted
                oof_decisions[outer_validation] = classified
                oof_thresholds[outer_validation] = threshold
                record = {
                    "architecture": architecture,
                    "outer_fold": outer_fold,
                    "study_name": study_name,
                    "successful_trials": len(complete),
                    "attempted_trials": len(non_failed_trials),
                    "inner_best_objective": float(study.best_value),
                    "best_parameters": best_parameters,
                    "outer_metrics": metrics,
                    "outer_fold_plan_sha256": _canonical_sha256(outer_plan),
                    "outer_validation_rows_sha256": outer_plan[
                        "outer_validation"
                    ]["row_ids_sha256"],
                    "outer_validation_rows": frame.iloc[outer_validation][
                        C.ROW_ID_COL
                    ].astype(str).tolist(),
                }
                fold_results.append(record)
                all_fold_results.append(record)
                _save_primary_checkpoint(
                    architecture,
                    outer_fold,
                    study_name,
                    study_protocol,
                    arguments.trials,
                    record,
                    predicted,
                    classified,
                    threshold,
                )
                _persist_complete_study_history(
                    optuna,
                    arguments.storage,
                    architectures,
                    arguments.outer_folds,
                )
                logger.info(
                    "Completed %s nested outer fold %d/%d",
                    architecture,
                    outer_fold,
                    arguments.outer_folds,
                )
            if (
                not np.isfinite(oof_probabilities).all()
                or not np.isfinite(oof_thresholds).all()
                or (oof_decisions < 0).any()
            ):
                raise RuntimeError(
                    f"Nested outer predictions are incomplete for {architecture}"
                )
            architecture_predictions[architecture] = {
                "probabilities": oof_probabilities,
                "decisions": oof_decisions,
                "thresholds": oof_thresholds,
            }
            architecture_results[architecture] = _architecture_payload(
                architecture,
                fold_results,
                oof_probabilities,
                oof_decisions,
                oof_thresholds,
                labels,
                groups,
                feature_names,
                arguments,
                seeds,
                split_plan,
                split_plan_file_sha256,
            )
            # --- Save checkpoint so a subsequent --resume can skip this arch ---
            _save_architecture_checkpoint(
                architecture,
                oof_probabilities,
                oof_decisions,
                oof_thresholds,
                fold_results,
                architecture_results[architecture],
            )
    finally:
        _apply_parameters(baseline)
    if (outer_fold_ids < 1).any():
        raise RuntimeError("Nested outer fold assignments are incomplete")
    reference_predictions, reference_results = _reference_baseline_oof(
        frame,
        embeddings,
        labels,
        groups,
        split_plan,
        seeds["training_seed"],
    )
    comparisons: dict[str, Any] = {}
    if "concatenation" in architecture_predictions:
        first = architecture_predictions["concatenation"]
        # CHANGELOG 2026-09 (novelty): iterate every trained deep architecture
        # rather than a hard-coded list, so the new reliability-family member
        # automatically receives its concatenation comparison.  "concatenation"
        # is excluded by the memberships guard below (it is the reference).
        for proposed in architecture_predictions:
            if proposed == "concatenation":
                continue
            second = architecture_predictions[proposed]
            comparisons[f"{proposed}_minus_concatenation"] = (
                C.clustered_model_comparison(
                    labels,
                    first["probabilities"],
                    second["probabilities"],
                    first["decisions"],
                    second["decisions"],
                    groups,
                    seed=seeds["split_seed"],
                )
            )
    comparison_references = [
        name
        for name in (
            "raw_esm_zero_shot",
            "esm_conservation_logistic",
            "lightgbm",
        )
        if name in reference_predictions
    ]
    for reference_name in comparison_references:
        reference = reference_predictions[reference_name]
        for architecture, predicted in architecture_predictions.items():
            comparisons[f"{architecture}_minus_{reference_name}"] = (
                C.clustered_model_comparison(
                    labels,
                    reference["probabilities"],
                    predicted["probabilities"],
                    reference["decisions"],
                    predicted["decisions"],
                    groups,
                    seed=seeds["split_seed"],
                )
            )
    for metric in ("mcc", "auroc", "auprc"):
        keys = [
            key
            for key, details in comparisons.items()
            if details.get(metric)
            and details[metric].get("two_sided_probability") is not None
        ]
        if not keys:
            continue
        adjusted = multipletests(
            [
                comparisons[key][metric]["two_sided_probability"]
                for key in keys
            ],
            method="holm",
        )[1]
        for key, value in zip(keys, adjusted):
            comparisons[key][metric]["holm_adjusted_probability"] = round(
                float(value), 4
            )
    production_params_by_architecture = {
        name: details["production_params"]
        for name, details in architecture_results.items()
    }
    combined_payload = {
        "schema_version": 2,
        "model_tag": MODEL_TAG,
        "protocol_version": PROTOCOL_VERSION,
        "selection_protocol": "fixed_nested_split_group_disjoint_cross_validation",
        "architectures_evaluated": architectures,
        "architectures": architecture_results,
        "production_params_by_architecture": production_params_by_architecture,
        "feature_names_by_architecture": feature_names_by_architecture,
        "reference_baselines": reference_results,
        "primary_reference_baseline": "esm_conservation_logistic",
        "nested_oof_comparisons": comparisons,
        "comparison_inference": (
            "paired_split_group_randomization_with_descriptive_"
            "split_group_bootstrap_ci"
        ),
        "comparison_inference_scope": (
            "conditional_on_one_prespecified_nested_cv_oof_realization"
        ),
        "training_procedure_uncertainty_included": False,
        "repeated_nested_cv_policy": (
            (
                "fixed_primary_selected_configuration_confirmation_runs_"
                "persisted_separately_and_never_used_for_selection"
            )
            if confirmation_split_seeds
            else (
                "not_run_by_default_due_to_gpu_budget; confirmatory superiority_"
                "requires_prespecified_repeated_nested_cv"
            )
        ),
        "comparison_multiplicity_correction": (
            "Holm within endpoint over paired-randomization probabilities"
        ),
        "reproducibility": {
            **seeds,
            "deterministic": not arguments.allow_nondeterministic,
            "protocol_fingerprint": protocol_fingerprint,
            "gpu": _gpu_metadata(),
            "input_sha256": file_sha256(INPUT_CSV),
            "embedding_sha256": file_sha256(INPUT_NPY),
            "split_plan_path": str(SPLIT_PLAN_FILE),
            "split_plan_canonical_sha256": split_plan["canonical_sha256"],
            "split_plan_file_sha256": split_plan_file_sha256,
        },
        "external_validation_touched": False,
        "proposed_model": C.RELIABILITY_ARCHITECTURE,
        "reliability_residual_protocol": (
            C.reliability_architecture_protocol(
                feature_names_by_architecture[C.RELIABILITY_ARCHITECTURE],
                production_params_by_architecture[C.RELIABILITY_ARCHITECTURE],
            )
            if C.RELIABILITY_ARCHITECTURE in architecture_results
            else None
        ),
        "confirmatory_repeated_cv": (
            {
                "requested": True,
                "scientific_role": "separate_fixed_configuration_confirmation",
                "seed_plan": str(CONFIRMATORY_SEED_PLAN),
                "results": str(CONFIRMATORY_RESULTS_JSON),
                "authoritative_primary_nested_oof_replaced": False,
            }
            if confirmation_split_seeds
            else {"requested": False}
        ),
    }
    _atomic_json(combined_payload, ARCHITECTURE_RESULTS_JSON)
    if "concatenation" in architecture_results:
        _atomic_json(
            architecture_results["concatenation"], CONCATENATION_BEST_JSON
        )
    if "gated_fusion" in architecture_results:
        _atomic_json(
            architecture_results["gated_fusion"], GATED_FUSION_BEST_JSON
        )
    # CHANGELOG 2026-09 (novelty): the proposed architecture became a family, so
    # every reliability-family member gets its own best-payload file.  The
    # legacy RELIABILITY_RESIDUAL_BEST_JSON name is kept for the
    # reliability_residual member -- reusing it here keeps the pre-existing
    # file path stable for downstream consumers and the kaggle completeness
    # gate -- and the new member is written to its own sibling file.  The
    # case-insensitive branch structure below is deliberately duplicated rather
    # than unified into a loop so each file keeps its well-known name.
    if C.RELIABILITY_RESIDUAL_ARCHITECTURE in architecture_results:
        _atomic_json(
            architecture_results[C.RELIABILITY_RESIDUAL_ARCHITECTURE],
            RELIABILITY_RESIDUAL_BEST_JSON,
        )
    if C.EVIDENTIAL_RESIDUAL_ARCHITECTURE in architecture_results:
        _atomic_json(
            architecture_results[C.EVIDENTIAL_RESIDUAL_ARCHITECTURE],
            EVIDENTIAL_RESIDUAL_BEST_JSON,
        )
    if "cross_attention" in architecture_results:
        legacy_payload = {
            **architecture_results["cross_attention"],
            "production_params_by_architecture": production_params_by_architecture,
            "architectures": architecture_results,
            "architecture_selection_file": str(ARCHITECTURE_RESULTS_JSON),
            "legacy_compatibility": (
                "Root production_params remains cross_attention only; use "
                "production_params_by_architecture for architecture-scoped training."
            ),
        }
        _atomic_json(legacy_payload, TUNING_BEST_JSON)
    flat_rows = []
    for record in all_fold_results:
        flat_rows.append(
            {
                "architecture": record["architecture"],
                "outer_fold": record["outer_fold"],
                "inner_best_objective": record["inner_best_objective"],
                "successful_trials": record["successful_trials"],
                **{
                    name: record["outer_metrics"].get(name)
                    for name in (
                        "mcc",
                        "auroc",
                        "auprc",
                        "brier",
                        "recall",
                        "precision",
                        "f1",
                        "threshold",
                    )
                },
            }
        )
    _atomic_csv(pd.DataFrame(flat_rows), FOLD_RESULTS_CSV)
    _atomic_csv(
        pd.DataFrame(
            {
                C.ROW_ID_COL: frame[C.ROW_ID_COL].astype(str),
                C.GENE_COL: frame[C.GENE_COL].astype(str),
                "split_group": groups,
                C.LABEL_COL: labels,
                "outer_fold": outer_fold_ids,
            }
        ),
        FOLD_ASSIGNMENTS,
    )
    oof_values: dict[str, np.ndarray] = {
        "y": labels,
        "groups": groups,
        "row_ids": frame[C.ROW_ID_COL].astype(str).to_numpy(),
        "fold_ids": outer_fold_ids,
    }
    for architecture, predictions in architecture_predictions.items():
        for name, array in predictions.items():
            oof_values[f"{architecture}__{name}"] = array
    for baseline_name, predictions in reference_predictions.items():
        for name, array in predictions.items():
            oof_values[f"reference__{baseline_name}__{name}"] = array
    if "cross_attention" in architecture_predictions:
        for name, array in architecture_predictions["cross_attention"].items():
            oof_values[name] = array
    _atomic_npz(oof_values, OOF_FILE)
    if confirmation_split_seeds:
        confirmation_feature_names = feature_names_by_architecture[
            C.RELIABILITY_ARCHITECTURE
        ]
        confirmation_context = {
            "outer_folds": int(arguments.outer_folds),
            "inner_folds": int(arguments.inner_folds),
            "deterministic": not arguments.allow_nondeterministic,
            "feature_names": confirmation_feature_names,
            "feature_names_sha256": _canonical_sha256(
                confirmation_feature_names
            ),
            "n_rows": int(len(frame)),
            "row_order_sha256": _ordered_text_sha256(
                frame[C.ROW_ID_COL].astype(str)
            ),
            "input_sha256": file_sha256(INPUT_CSV),
            "embedding_sha256": file_sha256(INPUT_NPY),
            "primary_split_plan_canonical_sha256": split_plan[
                "canonical_sha256"
            ],
            "primary_split_plan_file_sha256": split_plan_file_sha256,
            "source_sha256": {
                "stage14": file_sha256(Path(__file__).resolve()),
                "common": file_sha256(Path(C.__file__).resolve()),
                "gpu_runtime": file_sha256(
                    Path(C.__file__).resolve().with_name("gpu_runtime.py")
                ),
                "schema": file_sha256(
                    Path(__file__).resolve().with_name("schema.py")
                ),
                "config": file_sha256(
                    Path(__file__).resolve().with_name("config.py")
                ),
            },
        }
        confirmation_plan = _confirmatory_seed_plan(
            confirmation_split_seeds,
            seeds,
            architecture_results,
            reference_results,
            confirmation_context,
        )
        _persist_confirmatory_plan(confirmation_plan)
        _execute_confirmatory_repeats(
            frame,
            embeddings,
            labels,
            groups,
            base_feature_names,
            confirmation_plan,
        )
    stage_outputs = [
        path
        for path in (
            TRIALS_CSV,
            FOLD_RESULTS_CSV,
            FOLD_ASSIGNMENTS,
            OOF_FILE,
            SPLIT_PLAN_FILE,
            ARCHITECTURE_RESULTS_JSON,
            CONCATENATION_BEST_JSON,
            GATED_FUSION_BEST_JSON,
            RELIABILITY_RESIDUAL_BEST_JSON,
            EVIDENTIAL_RESIDUAL_BEST_JSON,
            CONFIRMATORY_SEED_PLAN,
            CONFIRMATORY_RESULTS_JSON,
            CONFIRMATORY_PREDICTIONS,
            CONFIRMATORY_FOLD_ASSIGNMENTS,
            CONFIRMATORY_CHECKPOINT_DIR,
            ARCHITECTURE_CHECKPOINT_DIR,
            TUNING_BEST_JSON,
        )
        if path.exists()
    ]
    if not confirmation_split_seeds:
        stage_outputs = [
            path
            for path in stage_outputs
            if path
            not in {
                CONFIRMATORY_SEED_PLAN,
                CONFIRMATORY_RESULTS_JSON,
                CONFIRMATORY_PREDICTIONS,
                CONFIRMATORY_FOLD_ASSIGNMENTS,
                CONFIRMATORY_CHECKPOINT_DIR,
            }
        ]
    write_run_manifest(
        MANIFEST_FILE,
        "14_tune_cross_attention",
        [INPUT_CSV, INPUT_NPY, INPUT_STATUS, STAGE10_MANIFEST],
        {
            "model_tag": MODEL_TAG,
            "feature_names": base_feature_names,
            "feature_names_by_architecture": feature_names_by_architecture,
            "protocol_version": PROTOCOL_VERSION,
            "architectures": architectures,
            "reference_baselines": list(reference_results),
            "primary_objective": arguments.objective,
            "early_stopping_metric": C.DEEP_EARLY_STOP_METRIC,
            "search_spaces": {
                name: _search_space_for(name) for name in architectures
            },
            "storage": arguments.storage,
            "outer_folds": arguments.outer_folds,
            "inner_folds": arguments.inner_folds,
            "fold_assignments": str(FOLD_ASSIGNMENTS),
            "split_plan": str(SPLIT_PLAN_FILE),
            "split_plan_canonical_sha256": split_plan["canonical_sha256"],
            "split_plan_file_sha256": split_plan_file_sha256,
            "seeds": seeds,
            "gpu": _gpu_metadata(),
            "esm_model": C.ESM_MODEL_NAME,
            "lora_tuned": False,
            "external_validation_touched": False,
            "reliability_residual_protocol": (
                C.reliability_architecture_protocol(
                    feature_names_by_architecture[C.RELIABILITY_ARCHITECTURE],
                    production_params_by_architecture.get(
                        C.RELIABILITY_ARCHITECTURE, {}
                    ),
                )
                if C.RELIABILITY_ARCHITECTURE in feature_names_by_architecture
                else None
            ),
            "confirmatory_seed_plan": (
                str(CONFIRMATORY_SEED_PLAN)
                if confirmation_split_seeds
                else None
            ),
            "confirmatory_repeated_cv_results": (
                str(CONFIRMATORY_RESULTS_JSON)
                if confirmation_split_seeds
                else None
            ),
            "confirmatory_repeated_cv_predictions": (
                str(CONFIRMATORY_PREDICTIONS)
                if confirmation_split_seeds
                else None
            ),
            "confirmatory_scientific_role": (
                "separate_fixed_configuration_split_and_training_instability"
                if confirmation_split_seeds
                else None
            ),
            "confirmatory_results_used_for_model_selection": False,
        },
        outputs=stage_outputs,
    )
    logger.info(
        "Saved unbiased nested tuning results for %s", ", ".join(architectures)
    )


def _run_reproduce_legacy() -> None:
    """Retain reproduction support for pre-v2 cross-attention artifacts."""
    if not TUNING_BEST_JSON.exists():
        raise FileNotFoundError(TUNING_BEST_JSON)
    payload = json.loads(TUNING_BEST_JSON.read_text(encoding="utf-8"))
    frame, embeddings, labels, groups, feature_names = _load_data()
    values = frame[feature_names].to_numpy(dtype=np.float32)
    saved_search = payload["search"]
    outer_folds = int(saved_search["outer_folds"])
    saved_seed = int(payload["reproducibility"]["seed"])
    deterministic = bool(payload["reproducibility"]["deterministic"])
    C.set_seeds(saved_seed, deterministic=deterministic)
    splits = C.make_group_splits(labels, groups, outer_folds, saved_seed)
    baseline = _snapshot_constants()
    probabilities = np.full(len(frame), np.nan)
    decisions = np.full(len(frame), -1, dtype=np.int8)
    thresholds = np.full(len(frame), np.nan)
    try:
        for record, (outer_train, outer_validation) in zip(
            payload["fold_results"], splits
        ):
            _apply_parameters(baseline)
            _apply_parameters(payload["production_params"])
            _apply_parameters(record["best_parameters"])
            fold = int(record["outer_fold"])
            legacy_seed = saved_seed + fold * 100_000
            fit, stop, temperature, threshold_set = (
                C.split_fit_stop_temperature_threshold(
                    outer_train, labels, groups, legacy_seed
                )
            )
            outer_plan = {
                "outer_validation": {"indices": outer_validation.tolist()},
                "final_partitions": {
                    "fit": {"indices": fit.tolist()},
                    "early_stopping": {"indices": stop.tolist()},
                    "temperature": {"indices": temperature.tolist()},
                    "threshold": {"indices": threshold_set.tolist()},
                },
                "final_training_seeds": [
                    legacy_seed + member * 100
                    for member in range(1, int(C.N_ENSEMBLE) + 1)
                ],
            }
            C.set_seeds(legacy_seed, deterministic=deterministic)
            predicted, classified, threshold, _ = _evaluate_outer_fold(
                outer_plan,
                values,
                embeddings,
                labels,
                "cross_attention",
            )
            probabilities[outer_validation] = predicted
            decisions[outer_validation] = classified
            thresholds[outer_validation] = threshold
    finally:
        _apply_parameters(baseline)
    reproduced = C.evaluate(
        labels, probabilities, thresholds, predictions=decisions
    )
    logger.warning(
        "Reproducing legacy cross-attention output without a persisted v2 split plan"
    )
    logger.info("Saved metrics: %s", payload["nested_outer_metrics"])
    logger.info("Reproduced metrics: %s", reproduced)


def run_reproduce(arguments: argparse.Namespace) -> None:
    """Reproduce saved untouched outer-fold results from the persisted plan."""
    if not ARCHITECTURE_RESULTS_JSON.exists():
        _run_reproduce_legacy()
        return
    if not SPLIT_PLAN_FILE.exists():
        raise FileNotFoundError(SPLIT_PLAN_FILE)
    payload = json.loads(ARCHITECTURE_RESULTS_JSON.read_text(encoding="utf-8"))
    split_plan = json.loads(SPLIT_PLAN_FILE.read_text(encoding="utf-8"))
    expected_canonical = split_plan.get("canonical_sha256")
    unhashed_plan = dict(split_plan)
    unhashed_plan.pop("canonical_sha256", None)
    if expected_canonical != _canonical_sha256(unhashed_plan):
        raise RuntimeError("Persisted nested split plan canonical hash is invalid")
    expected_file_hash = payload["reproducibility"]["split_plan_file_sha256"]
    if file_sha256(SPLIT_PLAN_FILE) != expected_file_hash:
        raise RuntimeError("Persisted nested split plan file hash has changed")
    frame, embeddings, labels, groups, base_feature_names = _load_data()
    if split_plan["row_order_sha256"] != _ordered_text_sha256(
        frame[C.ROW_ID_COL].astype(str)
    ):
        raise RuntimeError("Current input row order differs from the split plan")
    deterministic = bool(payload["reproducibility"]["deterministic"])
    baseline = _snapshot_constants()
    requested = list(dict.fromkeys(arguments.architectures))
    missing = set(requested) - set(payload["architectures"])
    if missing:
        raise ValueError(
            f"No saved nested results for architectures: {sorted(missing)}"
        )
    try:
        for architecture in requested:
            details = payload["architectures"][architecture]
            feature_names = C.architecture_feature_names(
                architecture, frame, base_feature_names
            )
            saved_feature_names = details.get("search", {}).get("feature_names")
            if saved_feature_names is not None and saved_feature_names != feature_names:
                raise RuntimeError(
                    f"Current {architecture} feature schema differs from saved HPO"
                )
            values = frame[feature_names].to_numpy(dtype=np.float32)
            records = {
                int(record["outer_fold"]): record
                for record in details["fold_results"]
            }
            probabilities = np.full(len(frame), np.nan)
            decisions = np.full(len(frame), -1, dtype=np.int8)
            thresholds = np.full(len(frame), np.nan)
            for outer_plan in split_plan["folds"]:
                fold = int(outer_plan["outer_fold"])
                record = records[fold]
                if record["outer_fold_plan_sha256"] != _canonical_sha256(
                    outer_plan
                ):
                    raise RuntimeError(
                        f"Saved {architecture} fold {fold} plan hash differs"
                    )
                _apply_parameters(baseline)
                _apply_parameters(details["production_params"])
                _apply_parameters(record["best_parameters"])
                C.set_seeds(
                    int(outer_plan["final_training_seeds"][0]),
                    deterministic=deterministic,
                )
                predicted, classified, threshold, _ = _evaluate_outer_fold(
                    outer_plan,
                    values,
                    embeddings,
                    labels,
                    architecture,
                    (
                        feature_names
                        if C.is_reliability_family(architecture)
                        else None
                    ),
                )
                outer_validation = _record_indices(
                    outer_plan["outer_validation"]
                )
                probabilities[outer_validation] = predicted
                decisions[outer_validation] = classified
                thresholds[outer_validation] = threshold
            if (
                not np.isfinite(probabilities).all()
                or not np.isfinite(thresholds).all()
                or (decisions < 0).any()
            ):
                raise RuntimeError(
                    f"Reproduced predictions are incomplete for {architecture}"
                )
            reproduced = C.evaluate(
                labels, probabilities, thresholds, predictions=decisions
            )
            logger.info(
                "%s saved metrics: %s",
                architecture,
                details["nested_outer_metrics"],
            )
            logger.info("%s reproduced metrics: %s", architecture, reproduced)
    finally:
        _apply_parameters(baseline)


def build_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=PUBLICATION_ARCHITECTURES,
        default=list(PUBLICATION_ARCHITECTURES),
        help=(
            "Architectures to tune on the identical nested partitions "
            "(default: concatenation gated_fusion cross_attention "
            "reliability_residual)."
        ),
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument(
        "--objective",
        choices=["composite", "mcc", "auroc", "auprc"],
        default="auprc",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--search-ensemble", type=int, default=1)
    parser.add_argument("--search-epochs", type=int, default=25)
    parser.add_argument("--search-patience", type=int, default=5)
    parser.add_argument("--final-ensemble", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=60)
    parser.add_argument("--final-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Partition RNG seed; defaults to --seed for compatibility.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Fixed model RNG stream seed; defaults to --seed.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Optuna sampler RNG seed; defaults to --seed.",
    )
    parser.add_argument(
        "--confirmation-split-seeds",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Execute confirmatory repeated group CV after primary HPO. Repeats "
            "reuse the fixed reliability_residual production configuration and "
            "fixed reference settings without rerunning HPO."
        ),
    )
    parser.add_argument("--storage", default=TUNING_STORAGE)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from per-architecture checkpoints saved by a previous "
            "session.  Architectures whose checkpoints exist are loaded "
            "from disk and skipped; remaining architectures run normally."
        ),
    )
    parser.add_argument("--reproduce", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_arguments()
    if args.reproduce:
        run_reproduce(args)
    else:
        run_search(args)
