from __future__ import annotations

import json
import logging
import pickle
import gc
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests

import common as C
from config import (
    ENABLE_LORA,
    MODEL_TAG,
    RANDOM_STATE,
    REQUIRE_HOMOLOGY_GROUPS,
    STAGE05_OUT,
    REQUIRE_TUNING_ARTIFACT,
    STAGE10_OUT,
    STAGE11_OUT,
    STAGE14_OUT,
    TUNING_BEST_JSON,
    ensure_directories,
    file_sha256,
    json_default,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import (
    MUTATION_FEATURE_COLS,
    PREDICTOR_COLS,
    select_availability_features,
    select_tabular_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage11_evaluation")

INPUT_CSV = STAGE10_OUT / "internal_with_esm.parquet"
INPUT_NPY = STAGE10_OUT / "internal_esm_embeddings.npy"
STATUS_CSV = STAGE10_OUT / "internal_esm_extraction.parquet"
RESULTS_JSON = STAGE11_OUT / "results.json"
COMPARISON_CSV = STAGE11_OUT / "comparison_table.csv"
FOLD_ASSIGNMENTS = STAGE11_OUT / "fold_assignments.csv"
INNER_PARTITIONS = STAGE11_OUT / "inner_partitions.json"
OOF_STORE = STAGE11_OUT / "oof_predictions.npz"
SHAP_STORE = STAGE11_OUT / f"shap_values_{MODEL_TAG}.npz"
RELIABILITY_DIAGNOSTICS_CSV = STAGE11_OUT / "reliability_diagnostics.csv"
MANIFEST_FILE = STAGE11_OUT / "run_manifest.json"
MODEL_DIR = STAGE11_OUT / "models"
AUDIT_SIDECAR = STAGE05_OUT / "somatic_variant_audit_sidecar.parquet"
TUNING_SPLIT_PLAN = STAGE14_OUT / "nested_inner_splits.json"
ARCHITECTURE_SELECTION = STAGE14_OUT / "architecture_selection.json"
STAGE10_MANIFEST = STAGE10_OUT / "internal_esm_manifest.json"
STAGE14_MANIFEST = STAGE14_OUT / "run_manifest.json"
N_SPLITS = 5
# CHANGELOG 2026-09: REQUIRED_TUNING_PROTOCOL is deliberately NOT bumped even
# though the candidate architecture set grew.  The protocol itself (fixed nested
# group CV, fixed outer/inner splits, fixed seed plan) is unchanged; only the
# set of candidates evaluated under it was extended.  Bumping the string would
# rename every Optuna study (see stage 14 study-name construction), discarding
# existing trial databases for the three pre-existing architectures for no
# scientific reason.  A stale Stage 14 run that predates the new architecture is
# still caught -- by the REQUIRED_ARCHITECTURES completeness check below, which
# reports exactly which architecture is missing.
REQUIRED_TUNING_PROTOCOL = "fixed_nested_group_cv_v4_reliability_residual"
# CHANGELOG 2026-09 (novelty): the single proposed architecture became a family
# (reliability_residual + evidential_residual).  Both members are required so the
# published comparison always contains the ablation of the novel fusion rule
# against the uniform-gate variant; whichever member config.PROPOSED_ARCHITECTURE
# names is reported as "proposed" and the other as its ablation.
REQUIRED_ARCHITECTURES = {
    "concatenation",
    "gated_fusion",
    "cross_attention",
    *C.RELIABILITY_FAMILY_ARCHITECTURES,
}

# CHANGELOG 2026-09 (novelty): reporting roles for reliability-family members.
# The family member selected by config.PROPOSED_ARCHITECTURE is reported as the
# proposed model; the remaining member is reported as a fusion-rule ablation, so
# adding the new architecture cannot silently create two "proposed" models.
_PROPOSED_FAMILY_ROLES = {
    "reliability_residual": "proposed_reliability_conditioned_residual_fusion_model",
    "evidential_residual": "proposed_precision_weighted_evidential_residual_fusion_model",
}
_ABLATION_FAMILY_ROLES = {
    "reliability_residual": "reliability_family_ablation_uniform_gate_residual_fusion",
    "evidential_residual": "reliability_family_ablation_precision_weighted_evidential_fusion",
}
# CHANGELOG 2026-09 (novelty): bundle-metadata interpretation strings, factored
# out of the deep-architecture loop so the new family member is described
# accurately instead of falling through to the pooled-fusion baseline text.
# The two pre-existing strings are reproduced verbatim so bundle metadata for
# previously supported architectures is byte-identical.
_ARCHITECTURE_INTERPRETATIONS = {
    "cross_attention": "legacy_pooled_ESM_vector_pseudo_slots_not_residue_tokens",
    "reliability_residual": (
        "proposed_reliability_conditioned_bounded_residual_"
        "with_exact_sequence_fallback"
    ),
    "evidential_residual": (
        "precision_weighted_evidential_bounded_residual_"
        "with_exact_sequence_fallback"
    ),
}


def _architecture_interpretation(architecture: str) -> str:
    return _ARCHITECTURE_INTERPRETATIONS.get(
        architecture, "valid_pooled_embedding_fusion_baseline"
    )


def _model_reporting_role(name: str) -> str:
    if name == "availability_logistic":
        return "diagnostic_missingness_availability_negative_control"
    if name == "cross_attention":
        return "legacy_pooled_vector_pseudo_slot_comparator"
    if name == "lora_esm":
        return "optional_exploratory_model"
    if name == C.RELIABILITY_ARCHITECTURE:
        return _PROPOSED_FAMILY_ROLES[name]
    if C.is_reliability_family(name):
        return _ABLATION_FAMILY_ROLES[name]
    if name in {
        "lightgbm",
        "raw_esm_zero_shot",
        "esm_score_logistic",
        "conservation_logistic",
        "esm_conservation_logistic",
        "mutation_logistic",
        "esm_embedding_mutation",
    }:
        return "prespecified_reference_baseline"
    return "candidate_fusion_model"


def _atomic_pickle(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def _load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    validate_upstream_manifest(
        STAGE10_MANIFEST,
        "10_extract_esm_features:internal",
        [INPUT_CSV, INPUT_NPY, STATUS_CSV],
        required_source_files=("gpu_runtime.py",),
    )
    for path in (INPUT_CSV, INPUT_NPY, STATUS_CSV):
        if not path.exists():
            raise FileNotFoundError(path)
    df = pd.read_parquet(INPUT_CSV).reset_index(drop=True)
    embeddings = np.load(INPUT_NPY, mmap_mode="r")
    statuses = pd.read_parquet(STATUS_CSV)
    if len(df) != len(embeddings) or len(df) != len(statuses):
        raise ValueError(
            f"Input row mismatch: CSV={len(df)}, NPY={len(embeddings)}, status={len(statuses)}"
        )
    if list(df[C.ROW_ID_COL].astype(str)) != list(statuses[C.ROW_ID_COL].astype(str)):
        raise ValueError("ESM status row order differs from the modelling CSV")
    if embeddings.ndim != 2 or embeddings.shape[1] != C.ESM_DIM:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    labels = C.validate_binary_labels(df[C.LABEL_COL].to_numpy())
    if C.GENE_COL not in df:
        raise KeyError(f"Missing required group column {C.GENE_COL}")
    split_column = "split_group" if "split_group" in df.columns else C.GENE_COL
    if split_column == C.GENE_COL and REQUIRE_HOMOLOGY_GROUPS:
        raise RuntimeError(
            "Publication Stage 11 requires Stage 08b homology split_group values; "
            "rerun Stages 08b and 10 before model training"
        )
    groups = C.validate_groups(df[split_column].to_numpy(), labels, N_SPLITS)
    if split_column == C.GENE_COL:
        logger.warning(
            "Homology groups are unavailable; evaluation is gene-disjoint only"
        )
    if df[C.ROW_ID_COL].isna().any() or df[C.ROW_ID_COL].duplicated().any():
        raise ValueError("Internal row identifiers are missing or duplicated")
    return df, embeddings, labels, groups


def _esm_extras(df: pd.DataFrame, embeddings: np.ndarray) -> tuple[np.ndarray, list[str]]:
    extra_columns = [
        column
        for column in ("esm_variant_score", *MUTATION_FEATURE_COLS)
        if column in df.columns
    ]
    extras = (
        df[extra_columns].to_numpy(dtype=np.float32)
        if extra_columns
        else np.empty((len(df), 0), dtype=np.float32)
    )
    names = [f"esm_embedding_{index}" for index in range(embeddings.shape[1])]
    return extras, [*names, *extra_columns]


def _fit_esm_baseline(
    train: np.ndarray,
    stop: np.ndarray,
    labels_train: np.ndarray,
    labels_stop: np.ndarray,
    seed: int,
) -> LogisticRegression:
    best_model: LogisticRegression | None = None
    best_score = -np.inf
    for regularization in (0.1, 1.0, 10.0):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="saga",
            tol=1e-3,
        )
        model.fit(train, labels_train)
        probabilities = model.predict_proba(stop)[:, 1]
        score = C.average_precision_score(labels_stop, probabilities)
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("ESM baseline fitting failed")
    return best_model


def _fit_logistic_baseline(
    train: np.ndarray,
    stop: np.ndarray,
    labels_train: np.ndarray,
    labels_stop: np.ndarray,
    seed: int,
) -> LogisticRegression:
    """Select regularization using the prespecified internal AUPRC endpoint."""
    return C.fit_regularized_logistic(
        train,
        stop,
        labels_train,
        labels_stop,
        seed=seed,
        primary_metric="auprc",
    )


def _publication_baseline_specs(df: pd.DataFrame) -> dict[str, list[str]]:
    conservation = [
        column
        for column in (
            "GERP++_RS",
            "phyloP100way_vertebrate",
            "phastCons100way_vertebrate",
        )
        if column in df
    ]
    mutation = [column for column in MUTATION_FEATURE_COLS if column in df]
    availability = select_availability_features(df)
    specs = {
        "esm_score_logistic": ["esm_variant_score"],
        "conservation_logistic": conservation,
        "esm_conservation_logistic": ["esm_variant_score", *conservation],
        "mutation_logistic": mutation,
        "availability_logistic": availability,
    }
    return {name: columns for name, columns in specs.items() if columns}


def _fit_raw_esm_calibrator(
    labels: np.ndarray, pathogenic_scores: np.ndarray
) -> C.LogitCalibrator:
    """Calibrate without changing the predetermined zero-shot score direction."""
    return C.fit_direction_preserving_calibrator(labels, pathogenic_scores)


def _validated_tuned_parameter_sets() -> dict[str, dict[str, Any]]:
    """Reject legacy/partial HPO artifacts for a publication deployment run."""
    if not TUNING_BEST_JSON.exists():
        if REQUIRE_TUNING_ARTIFACT:
            raise FileNotFoundError(
                f"Run Stage 14 tuning before final Stage 11 training: "
                f"{TUNING_BEST_JSON}"
            )
        return {}
    validate_upstream_manifest(
        STAGE14_MANIFEST,
        "14_tune_cross_attention",
        [TUNING_BEST_JSON, TUNING_SPLIT_PLAN, ARCHITECTURE_SELECTION],
        required_source_files=("common.py", "gpu_runtime.py"),
    )
    payload = json.loads(TUNING_BEST_JSON.read_text(encoding="utf-8"))
    protocol = payload.get("protocol_version")
    parameter_sets = C.read_tuned_parameter_sets()
    if REQUIRE_TUNING_ARTIFACT:
        if protocol != REQUIRED_TUNING_PROTOCOL:
            raise RuntimeError(
                "Stage 14 artifact uses a legacy/incompatible protocol; rerun Stage 14 "
                f"with {REQUIRED_TUNING_PROTOCOL} in the current output directory"
            )
        missing = REQUIRED_ARCHITECTURES - set(parameter_sets)
        if missing:
            raise RuntimeError(
                "Stage 14 artifact lacks architecture-scoped production parameters: "
                f"{sorted(missing)}"
            )
    return parameter_sets


def _ordered_text_sha256(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage14_outer_splits(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reuse and validate Stage 14 outer folds for deployment bundle training."""
    if not TUNING_SPLIT_PLAN.exists():
        if not REQUIRE_TUNING_ARTIFACT:
            logger.warning(
                "Stage 14 split plan is absent; using a non-publication debug split"
            )
            return C.make_group_splits(labels, groups, N_SPLITS, RANDOM_STATE)
        raise FileNotFoundError(
            f"Stage 14 split plan is required for Stage 11: {TUNING_SPLIT_PLAN}"
        )
    plan = json.loads(TUNING_SPLIT_PLAN.read_text(encoding="utf-8"))
    if plan.get("protocol_version") != REQUIRED_TUNING_PROTOCOL:
        raise RuntimeError("Stage 14 split plan uses an incompatible protocol")
    if int(plan.get("n_rows", -1)) != len(frame):
        raise RuntimeError("Stage 14 split plan row count differs from Stage 10")
    expected_row_hash = _ordered_text_sha256(frame[C.ROW_ID_COL].astype(str))
    if plan.get("row_order_sha256") != expected_row_hash:
        raise RuntimeError("Stage 14 split plan row order differs from Stage 10")
    if plan.get("input_sha256") != file_sha256(INPUT_CSV):
        raise RuntimeError("Stage 14 split plan input hash differs from Stage 10")
    expected_split_column = "split_group" if "split_group" in frame else C.GENE_COL
    if plan.get("split_group_column") != expected_split_column:
        raise RuntimeError("Stage 14 split-group column differs from Stage 11")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    validation_seen: set[int] = set()
    universe = set(range(len(frame)))
    for fold in plan.get("folds", []):
        train = np.asarray(fold["outer_train"]["indices"], dtype=np.int64)
        validation = np.asarray(
            fold["outer_validation"]["indices"], dtype=np.int64
        )
        if set(train) != universe - set(validation):
            raise RuntimeError("Stage 14 outer train is not the validation complement")
        if validation_seen & set(validation):
            raise RuntimeError("Stage 14 outer validation rows overlap across folds")
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("Stage 14 outer fold contains split-group leakage")
        C.validate_binary_labels(labels[train], "Stage 14 outer-train labels")
        C.validate_binary_labels(
            labels[validation], "Stage 14 outer-validation labels"
        )
        validation_seen.update(validation.tolist())
        splits.append((train, validation))
    if validation_seen != universe or len(splits) != N_SPLITS:
        raise RuntimeError("Stage 14 outer folds do not cover Stage 10 exactly once")
    return splits


def _stage14_training_records(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    """Load exact persisted Stage 14 outer and final-training partitions."""
    outer_splits = _stage14_outer_splits(frame, labels, groups)
    plan = json.loads(TUNING_SPLIT_PLAN.read_text(encoding="utf-8"))
    stored_canonical = plan.get("canonical_sha256")
    unhashed_plan = dict(plan)
    unhashed_plan.pop("canonical_sha256", None)
    if stored_canonical != _canonical_sha256(unhashed_plan):
        raise RuntimeError("Stage 14 split-plan canonical hash is invalid")
    if plan.get("embedding_sha256") != file_sha256(INPUT_NPY):
        raise RuntimeError("Stage 14 split plan embedding hash differs from Stage 10")
    lightgbm_by_fold: dict[int, dict[str, Any]] = {}
    if TUNING_BEST_JSON.exists():
        tuning = json.loads(TUNING_BEST_JSON.read_text(encoding="utf-8"))
        reproducibility = tuning.get("reproducibility", {})
        if (
            reproducibility.get("split_plan_canonical_sha256")
            != stored_canonical
        ):
            raise RuntimeError("Stage 14 tuning parameters and split plan disagree")
        if reproducibility.get("split_plan_file_sha256") != file_sha256(
            TUNING_SPLIT_PLAN
        ):
            raise RuntimeError("Stage 14 tuning split-plan file hash is invalid")
    if ARCHITECTURE_SELECTION.exists():
        selection = json.loads(ARCHITECTURE_SELECTION.read_text(encoding="utf-8"))
        if selection.get("protocol_version") != REQUIRED_TUNING_PROTOCOL:
            raise RuntimeError("Stage 14 architecture selection protocol is incompatible")
        selection_reproducibility = selection.get("reproducibility", {})
        if (
            selection_reproducibility.get("split_plan_canonical_sha256")
            != stored_canonical
            or selection_reproducibility.get("split_plan_file_sha256")
            != file_sha256(TUNING_SPLIT_PLAN)
        ):
            raise RuntimeError(
                "Stage 14 architecture selection and split plan disagree"
            )
        lightgbm = selection.get("reference_baselines", {}).get("lightgbm", {})
        for metadata in lightgbm.get("fold_metadata", []):
            fold = int(metadata.get("outer_fold", -1))
            parameters = metadata.get("selected_hyperparameters")
            if fold > 0 and isinstance(parameters, dict):
                lightgbm_by_fold[fold] = dict(parameters)
    if REQUIRE_TUNING_ARTIFACT and set(lightgbm_by_fold) != set(
        range(1, N_SPLITS + 1)
    ):
        raise RuntimeError(
            "Stage 14 artifact lacks nested inner-fold LightGBM selections; "
            "rerun Stage 14 under the publication protocol"
        )

    records: list[dict[str, Any]] = []
    expected_final_ensemble = int(plan.get("final_ensemble", -1))
    for expected_fold, (fold_plan, (outer_train, outer_validation)) in enumerate(
        zip(plan["folds"], outer_splits), 1
    ):
        fold = int(fold_plan.get("outer_fold", -1))
        if fold != expected_fold:
            raise RuntimeError("Stage 14 outer-fold identifiers are not sequential")
        final = fold_plan.get("final_partitions")
        if not isinstance(final, dict):
            raise RuntimeError(f"Stage 14 fold {fold} lacks final partitions")
        names = ("fit", "early_stopping", "temperature", "threshold")
        partitions: dict[str, np.ndarray] = {}
        for name in names:
            record = final.get(name)
            if not isinstance(record, dict) or "indices" not in record:
                raise RuntimeError(f"Stage 14 fold {fold} lacks final {name} rows")
            indices = np.asarray(record["indices"], dtype=np.int64)
            if len(indices) == 0 or len(np.unique(indices)) != len(indices):
                raise RuntimeError(
                    f"Stage 14 fold {fold} final {name} rows are empty or duplicated"
                )
            if not set(indices).issubset(set(outer_train)):
                raise RuntimeError(
                    f"Stage 14 fold {fold} final {name} rows leave outer training"
                )
            if record.get("row_ids_sha256") != _ordered_text_sha256(
                frame.iloc[indices][C.ROW_ID_COL].astype(str)
            ):
                raise RuntimeError(
                    f"Stage 14 fold {fold} final {name} row hash is invalid"
                )
            C.validate_binary_labels(
                labels[indices], f"Stage 14 fold {fold} final {name} labels"
            )
            partitions[name] = indices
        partition_sets = [set(partitions[name]) for name in names]
        for first_index, first in enumerate(partition_sets):
            for second in partition_sets[first_index + 1 :]:
                if first & second:
                    raise RuntimeError(
                        f"Stage 14 fold {fold} final partitions overlap"
                    )
        if set().union(*partition_sets) != set(outer_train):
            raise RuntimeError(
                f"Stage 14 fold {fold} final partitions do not cover outer training"
            )
        group_sets = [set(groups[partitions[name]]) for name in names]
        for first_index, first in enumerate(group_sets):
            for second in group_sets[first_index + 1 :]:
                if first & second:
                    raise RuntimeError(
                        f"Stage 14 fold {fold} final partitions leak split groups"
                    )
        training_seeds = [int(seed) for seed in fold_plan.get("final_training_seeds", [])]
        if (
            len(training_seeds) != expected_final_ensemble
            or len(set(training_seeds)) != len(training_seeds)
        ):
            raise RuntimeError(f"Stage 14 fold {fold} final training seeds are invalid")
        records.append(
            {
                "outer_fold": fold,
                "outer_train": outer_train,
                "outer_validation": outer_validation,
                **partitions,
                "training_seeds": training_seeds,
                "baseline_seed": int(plan["training_seed"]),
                "lightgbm_parameters": lightgbm_by_fold.get(fold, {}),
            }
        )
    return records


def _contextual_predictor_results(
    frame: pd.DataFrame, labels: np.ndarray
) -> dict[str, Any]:
    """Evaluate forbidden established scores without exposing them to training."""
    if not AUDIT_SIDECAR.exists():
        return {"available": False, "reason": "audit_sidecar_missing"}
    columns = pd.read_parquet(AUDIT_SIDECAR).columns
    selected = [column for column in PREDICTOR_COLS if column in columns]
    if not selected:
        return {"available": False, "reason": "no_contextual_predictors"}
    sidecar = pd.read_parquet(AUDIT_SIDECAR, columns=[C.ROW_ID_COL, *selected])
    if sidecar[C.ROW_ID_COL].duplicated().any():
        raise ValueError("Audit sidecar row identifiers are duplicated")
    aligned = frame[[C.ROW_ID_COL]].merge(
        sidecar, on=C.ROW_ID_COL, how="left", validate="one_to_one"
    )
    output: dict[str, Any] = {
        "available": True,
        "role": "evaluation_only_not_model_features",
        "circularity_warning": (
            "Training/version overlap may inflate internal estimates; use the "
            "strict temporal external comparison for claims."
        ),
        "predictors": {},
    }
    valid_masks: dict[str, np.ndarray] = {}
    for name in selected:
        values = pd.to_numeric(aligned[name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        valid_masks[name] = valid
        if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
            continue
        pathogenic_scores = -values[valid] if name == "SIFT_score" else values[valid]
        output["predictors"][name] = {
            "n": int(valid.sum()),
            "positives": int(labels[valid].sum()),
            "coverage": round(float(valid.mean()), 6),
            "pathogenic_direction": "lower" if name == "SIFT_score" else "higher",
            "auroc": round(float(C.roc_auc_score(labels[valid], pathogenic_scores)), 6),
            "auprc": round(
                float(C.average_precision_score(labels[valid], pathogenic_scores)), 6
            ),
        }
    contributing = [name for name in selected if name in output["predictors"]]
    if contributing:
        common = np.logical_and.reduce([valid_masks[name] for name in contributing])
        common_results: dict[str, Any] = {
            "n": int(common.sum()),
            "positives": int(labels[common].sum()),
            "predictors": {},
        }
        if common.sum() and len(np.unique(labels[common])) == 2:
            for name in contributing:
                values = pd.to_numeric(aligned[name], errors="coerce").to_numpy(
                    dtype=float
                )[common]
                scores = -values if name == "SIFT_score" else values
                common_results["predictors"][name] = {
                    "auroc": round(float(C.roc_auc_score(labels[common], scores)), 6),
                    "auprc": round(
                        float(C.average_precision_score(labels[common], scores)), 6
                    ),
                }
        output["common_coverage"] = common_results
    return output


def _save_lightgbm(model: LGBMClassifier, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    model.booster_.save_model(str(temporary))
    temporary.replace(path)


def _fold_metric(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decisions: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return C.evaluate(
        labels,
        probabilities,
        threshold,
        predictions=decisions,
    )


def run_evaluation() -> None:
    """Fit cross-validated deployment bundles and descriptive OOF predictions.

    Unbiased tuned-model estimates belong to Stage 14's nested outer folds.
    Stage 11 must not be used as the primary internal performance estimate.
    """
    ensure_directories(STAGE11_OUT, MODEL_DIR)
    C.set_seeds(RANDOM_STATE)
    tuned_parameter_sets = _validated_tuned_parameter_sets()
    df, embeddings, labels, groups = _load_data()
    feature_names = C.select_features(df)
    deep_architectures = (
        "concatenation",
        "gated_fusion",
        "cross_attention",
        # CHANGELOG 2026-09 (novelty): both reliability-family members are
        # trained so the fusion-rule ablation is always available.  Order is
        # taken from C.RELIABILITY_FAMILY_ARCHITECTURES, which appends new
        # members last, keeping Stage 14 per-architecture seed offsets stable.
        *C.RELIABILITY_FAMILY_ARCHITECTURES,
    )
    feature_names_by_architecture = {
        architecture: C.architecture_feature_names(
            architecture, df, feature_names
        )
        for architecture in deep_architectures
    }
    # OPTIMIZATION 2026-09: architectures that share a feature schema now share
    # one materialised float32 matrix instead of each holding its own identical
    # copy.  With the reliability family added there are five deep architectures
    # but only two distinct schemas, so this removes four full copies of the
    # tabular matrix from host RAM (~40 MB per copy at 200k rows x 50 columns),
    # including the separate `bio_all` copy used by the tabular baselines, whose
    # column list is exactly the non-reliability schema.
    # Sharing is safe: these arrays are only ever read (masked/indexed into new
    # arrays), never written in place.
    bio_all = df[feature_names].to_numpy(dtype=np.float32)
    _bio_matrix_cache: dict[tuple[str, ...], np.ndarray] = {
        tuple(feature_names): bio_all
    }
    bio_all_by_architecture = {}
    for architecture, names in feature_names_by_architecture.items():
        schema_key = tuple(names)
        matrix = _bio_matrix_cache.get(schema_key)
        if matrix is None:
            matrix = df[names].to_numpy(dtype=np.float32)
            _bio_matrix_cache[schema_key] = matrix
        bio_all_by_architecture[architecture] = matrix
    del _bio_matrix_cache
    tabular_features = select_tabular_features(feature_names)
    if not tabular_features:
        raise RuntimeError("No tabular-only features are available")
    tabular_all = df[tabular_features].to_numpy(dtype=np.float32)
    esm_extra_all, esm_only_names = _esm_extras(df, embeddings)
    baseline_specs = _publication_baseline_specs(df)
    model_names = [
        "lightgbm",
        "raw_esm_zero_shot",
        *baseline_specs,
        "esm_embedding_mutation",
        "concatenation",
        "gated_fusion",
        "cross_attention",
        # CHANGELOG 2026-09 (novelty): appended via the family tuple so the new
        # architecture gets its own probability/threshold/decision arrays and
        # fold metrics, exactly like every other reported model.
        *C.RELIABILITY_FAMILY_ARCHITECTURES,
    ]
    if ENABLE_LORA:
        model_names.append("lora_esm")
    probabilities = {
        name: np.full(len(df), np.nan, dtype=np.float64) for name in model_names
    }
    thresholds = {
        name: np.full(len(df), np.nan, dtype=np.float64) for name in model_names
    }
    decisions = {
        name: np.full(len(df), -1, dtype=np.int8) for name in model_names
    }
    reliability_components = {
        name: np.full(len(df), np.nan, dtype=np.float32)
        for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS
    }
    fold_ids = np.full(len(df), -1, dtype=np.int8)
    fold_metrics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in model_names
    }
    inner_records: dict[str, dict[str, list[str]]] = {}
    shap_values: list[np.ndarray] = []
    shap_inputs: list[np.ndarray] = []
    shap_folds: list[np.ndarray] = []
    training_records = _stage14_training_records(df, labels, groups)
    for training_record in training_records:
        fold = int(training_record["outer_fold"])
        outer_validation = training_record["outer_validation"]
        fit = training_record["fit"]
        stop = training_record["early_stopping"]
        temperature = training_record["temperature"]
        threshold_set = training_record["threshold"]
        final_training_seeds = training_record["training_seeds"]
        baseline_seed = int(training_record["baseline_seed"])
        fold_ids[outer_validation] = fold
        inner_records[str(fold)] = {
            "fit": df.iloc[fit][C.ROW_ID_COL].astype(str).tolist(),
            "early_stopping": df.iloc[stop][C.ROW_ID_COL].astype(str).tolist(),
            "temperature": df.iloc[temperature][C.ROW_ID_COL].astype(str).tolist(),
            "threshold": df.iloc[threshold_set][C.ROW_ID_COL].astype(str).tolist(),
            "outer_validation": df.iloc[outer_validation][C.ROW_ID_COL].astype(str).tolist(),
        }
        fold_dir = MODEL_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        bio_mask = C.nonconstant_feature_mask(bio_all[fit])
        tabular_mask = C.nonconstant_feature_mask(tabular_all[fit])
        if not bio_mask.any() or not tabular_mask.any():
            raise RuntimeError(f"Fold {fold} has no varying model features")
        fold_feature_names = [
            name for name, keep in zip(feature_names, bio_mask) if keep
        ]
        fold_tabular_features = [
            name for name, keep in zip(tabular_features, tabular_mask) if keep
        ]
        fold_bio_all = bio_all[:, bio_mask]
        fold_tabular_all = tabular_all[:, tabular_mask]
        preprocessors = C.fit_preprocessors(fold_bio_all[fit], embeddings[fit])
        _atomic_pickle(preprocessors, fold_dir / "preprocessors.pkl")
        lightgbm_parameters = dict(C.LGBM_PARAMS)
        lightgbm_parameters.update(training_record["lightgbm_parameters"])
        lightgbm_parameters["random_state"] = baseline_seed + fold
        lightgbm_parameters["scale_pos_weight"] = C.fold_class_weight(labels[fit])
        lightgbm_model = LGBMClassifier(**lightgbm_parameters)
        lightgbm_model.fit(
            fold_tabular_all[fit],
            labels[fit],
            eval_set=[(fold_tabular_all[stop], labels[stop])],
            eval_metric="aucpr",
            callbacks=[
                early_stopping(C.LGBM_EARLY_STOP, verbose=False),
                log_evaluation(0),
            ],
        )
        lightgbm_temperature_raw = lightgbm_model.predict_proba(
            fold_tabular_all[temperature]
        )[:, 1]
        lightgbm_calibrator = C.fit_probability_calibrator(
            labels[temperature], lightgbm_temperature_raw
        )
        lightgbm_threshold_prob = lightgbm_calibrator.predict(
            lightgbm_model.predict_proba(fold_tabular_all[threshold_set])[:, 1]
        )
        lightgbm_threshold = C.select_threshold(
            labels[threshold_set], lightgbm_threshold_prob
        )
        lightgbm_outer = lightgbm_calibrator.predict(
            lightgbm_model.predict_proba(fold_tabular_all[outer_validation])[:, 1]
        )
        probabilities["lightgbm"][outer_validation] = lightgbm_outer
        thresholds["lightgbm"][outer_validation] = lightgbm_threshold
        decisions["lightgbm"][outer_validation] = (
            lightgbm_outer >= lightgbm_threshold
        ).astype(np.int8)
        fold_metrics["lightgbm"].append(
            _fold_metric(
                labels[outer_validation],
                lightgbm_outer,
                decisions["lightgbm"][outer_validation],
                lightgbm_threshold,
            )
        )
        _save_lightgbm(lightgbm_model, fold_dir / "lightgbm.txt")
        _atomic_pickle(lightgbm_calibrator, fold_dir / "lightgbm_calibrator.pkl")
        shap_index = outer_validation
        if len(shap_index) > 1000:
            shap_index = np.random.RandomState(RANDOM_STATE + fold).choice(
                shap_index, 1000, replace=False
            )
        shap_result = C.compute_shap_values(
            lightgbm_model,
            fold_tabular_all[shap_index],
            fold_tabular_features,
        )
        if shap_result is not None:
            values, inputs, _ = shap_result
            expanded_values = np.zeros(
                (len(values), len(tabular_features)), dtype=np.float32
            )
            expanded_inputs = np.full(
                (len(inputs), len(tabular_features)), np.nan, dtype=np.float32
            )
            expanded_values[:, tabular_mask] = values
            expanded_inputs[:, tabular_mask] = inputs
            shap_values.append(expanded_values)
            shap_inputs.append(expanded_inputs)
            shap_folds.append(np.full(len(values), fold, dtype=np.int8))

        raw_pathogenic_score = -pd.to_numeric(
            df["esm_variant_score"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(raw_pathogenic_score).all():
            raise ValueError("Raw ESM score contains nonfinite values")
        raw_calibrator = _fit_raw_esm_calibrator(
            labels[temperature], raw_pathogenic_score[temperature]
        )
        raw_threshold_probabilities = raw_calibrator.predict_from_logits(
            raw_pathogenic_score[threshold_set]
        )
        raw_threshold = C.select_threshold(
            labels[threshold_set], raw_threshold_probabilities
        )
        raw_outer = raw_calibrator.predict_from_logits(
            raw_pathogenic_score[outer_validation]
        )
        probabilities["raw_esm_zero_shot"][outer_validation] = raw_outer
        thresholds["raw_esm_zero_shot"][outer_validation] = raw_threshold
        decisions["raw_esm_zero_shot"][outer_validation] = (
            raw_outer >= raw_threshold
        ).astype(np.int8)
        fold_metrics["raw_esm_zero_shot"].append(
            _fold_metric(
                labels[outer_validation],
                raw_outer,
                decisions["raw_esm_zero_shot"][outer_validation],
                raw_threshold,
            )
        )
        _atomic_pickle(
            {
                "score_column": "esm_variant_score",
                "pathogenic_direction": -1.0,
                "calibrator": raw_calibrator,
                "threshold": raw_threshold,
                "training_role": "zero_shot_score_calibration_only",
            },
            fold_dir / "raw_esm_zero_shot.pkl",
        )

        publication_artifacts: dict[str, dict[str, Any]] = {}
        for baseline_offset, (baseline_name, requested_columns) in enumerate(
            baseline_specs.items(), 1
        ):
            baseline_values = df[requested_columns].to_numpy(dtype=np.float32)
            baseline_mask = C.nonconstant_feature_mask(baseline_values[fit])
            if not baseline_mask.any():
                raise RuntimeError(
                    f"{baseline_name} has no varying features in fold {fold}"
                )
            baseline_columns = [
                name for name, keep in zip(requested_columns, baseline_mask) if keep
            ]
            baseline_values = baseline_values[:, baseline_mask]
            baseline_preprocessor = C.ArrayPreprocessor.fit(
                baseline_values[fit], copy=False
            )
            baseline_fit = baseline_preprocessor.transform(
                baseline_values[fit], copy=False
            )
            baseline_stop = baseline_preprocessor.transform(
                baseline_values[stop], copy=False
            )
            baseline_model = _fit_logistic_baseline(
                baseline_fit,
                baseline_stop,
                labels[fit],
                labels[stop],
                baseline_seed + fold * 100 + baseline_offset,
            )
            baseline_temperature = baseline_preprocessor.transform(
                baseline_values[temperature], copy=False
            )
            baseline_calibrator = C.fit_probability_calibrator(
                labels[temperature],
                baseline_model.predict_proba(baseline_temperature)[:, 1],
            )
            baseline_threshold_values = baseline_preprocessor.transform(
                baseline_values[threshold_set], copy=False
            )
            baseline_threshold_probabilities = baseline_calibrator.predict(
                baseline_model.predict_proba(baseline_threshold_values)[:, 1]
            )
            baseline_threshold = C.select_threshold(
                labels[threshold_set], baseline_threshold_probabilities
            )
            baseline_outer_values = baseline_preprocessor.transform(
                baseline_values[outer_validation], copy=False
            )
            baseline_outer = baseline_calibrator.predict(
                baseline_model.predict_proba(baseline_outer_values)[:, 1]
            )
            probabilities[baseline_name][outer_validation] = baseline_outer
            thresholds[baseline_name][outer_validation] = baseline_threshold
            decisions[baseline_name][outer_validation] = (
                baseline_outer >= baseline_threshold
            ).astype(np.int8)
            fold_metrics[baseline_name].append(
                _fold_metric(
                    labels[outer_validation],
                    baseline_outer,
                    decisions[baseline_name][outer_validation],
                    baseline_threshold,
                )
            )
            publication_artifacts[baseline_name] = {
                "model": baseline_model,
                "calibrator": baseline_calibrator,
                "preprocessor": baseline_preprocessor,
                "feature_names": baseline_columns,
                "threshold": baseline_threshold,
            }
        _atomic_pickle(
            publication_artifacts, fold_dir / "publication_baselines.pkl"
        )

        esm_fit = np.column_stack([embeddings[fit], esm_extra_all[fit]])
        esm_preprocessor = C.ArrayPreprocessor.fit(esm_fit, copy=False)
        esm_fit = esm_preprocessor.transform(esm_fit, copy=False)
        esm_stop = np.column_stack([embeddings[stop], esm_extra_all[stop]])
        esm_stop = esm_preprocessor.transform(esm_stop, copy=False)
        esm_model = _fit_esm_baseline(
            esm_fit,
            esm_stop,
            labels[fit],
            labels[stop],
            baseline_seed + fold * 100 + len(baseline_specs) + 1,
        )
        del esm_fit, esm_stop
        gc.collect()
        esm_temperature = np.column_stack(
            [embeddings[temperature], esm_extra_all[temperature]]
        )
        esm_temperature = esm_preprocessor.transform(esm_temperature, copy=False)
        esm_calibrator = C.fit_probability_calibrator(
            labels[temperature],
            esm_model.predict_proba(esm_temperature)[:, 1],
        )
        del esm_temperature
        esm_threshold_matrix = np.column_stack(
            [embeddings[threshold_set], esm_extra_all[threshold_set]]
        )
        esm_threshold_matrix = esm_preprocessor.transform(
            esm_threshold_matrix, copy=False
        )
        esm_threshold_prob = esm_calibrator.predict(
            esm_model.predict_proba(esm_threshold_matrix)[:, 1]
        )
        del esm_threshold_matrix
        esm_threshold = C.select_threshold(labels[threshold_set], esm_threshold_prob)
        esm_outer_matrix = np.column_stack(
            [embeddings[outer_validation], esm_extra_all[outer_validation]]
        )
        esm_outer_matrix = esm_preprocessor.transform(esm_outer_matrix, copy=False)
        esm_outer = esm_calibrator.predict(
            esm_model.predict_proba(esm_outer_matrix)[:, 1]
        )
        del esm_outer_matrix
        gc.collect()
        probabilities["esm_embedding_mutation"][outer_validation] = esm_outer
        thresholds["esm_embedding_mutation"][outer_validation] = esm_threshold
        decisions["esm_embedding_mutation"][outer_validation] = (
            esm_outer >= esm_threshold
        ).astype(np.int8)
        fold_metrics["esm_embedding_mutation"].append(
            _fold_metric(
                labels[outer_validation],
                esm_outer,
                decisions["esm_embedding_mutation"][outer_validation],
                esm_threshold,
            )
        )
        _atomic_pickle(
            {
                "model": esm_model,
                "calibrator": esm_calibrator,
                "preprocessor": esm_preprocessor,
                "feature_names": esm_only_names,
                "threshold": esm_threshold,
            },
            fold_dir / "esm_embedding_mutation.pkl",
        )
        transformed = {
            "fit_bio": preprocessors.transform_bio(fold_bio_all[fit], copy=False),
            "fit_esm": preprocessors.transform_esm(embeddings[fit], copy=False),
            "stop_bio": preprocessors.transform_bio(fold_bio_all[stop], copy=False),
            "stop_esm": preprocessors.transform_esm(embeddings[stop], copy=False),
            "temperature_bio": preprocessors.transform_bio(
                fold_bio_all[temperature], copy=False
            ),
            "temperature_esm": preprocessors.transform_esm(
                embeddings[temperature], copy=False
            ),
            "threshold_bio": preprocessors.transform_bio(
                fold_bio_all[threshold_set], copy=False
            ),
            "threshold_esm": preprocessors.transform_esm(
                embeddings[threshold_set], copy=False
            ),
            "outer_bio": preprocessors.transform_bio(
                fold_bio_all[outer_validation], copy=False
            ),
            "outer_esm": preprocessors.transform_esm(
                embeddings[outer_validation], copy=False
            ),
        }
        deep_fold_features: dict[str, list[str]] = {}
        # OPTIMIZATION 2026-09: every reliability-family member derives the
        # identical feature mask, preprocessor and transformed arrays (same
        # augmented schema, same passthrough indices, same fit rows), so the
        # ~10 transformed matrices are built once per fold and shared instead of
        # once per family member.  Sharing read-only arrays is safe because
        # train_deep_model / predict never write into their inputs.
        family_schema: dict[str, Any] = {}
        for architecture in deep_architectures:
            architecture_parameters = tuned_parameter_sets.get(architecture, {})
            # CHANGELOG 2026-09 (novelty): schema dispatch is now family-based.
            # Every reliability-family member consumes the augmented feature
            # schema (anchor + gate columns + missingness indicators) with the
            # gate columns passed through unscaled, so the test must be
            # "is this architecture in the family?" rather than "is this the
            # single architecture reported as proposed?".
            if C.is_reliability_family(architecture):
                if not family_schema:
                    architecture_names = feature_names_by_architecture[architecture]
                    architecture_values = bio_all_by_architecture[architecture]
                    architecture_mask = C.architecture_feature_mask(
                        architecture_values[fit], architecture_names, architecture
                    )
                    selected_feature_names = [
                        name
                        for name, keep in zip(architecture_names, architecture_mask)
                        if keep
                    ]
                    selected_values = architecture_values[:, architecture_mask]
                    architecture_preprocessors = C.PreprocessorBundle(
                        bio=C.ArrayPreprocessor.fit_with_passthrough(
                            selected_values[fit],
                            C.reliability_passthrough_indices(selected_feature_names),
                            copy=False,
                        ),
                        esm=preprocessors.esm,
                    )
                    architecture_transformed = {
                        "fit_bio": architecture_preprocessors.transform_bio(
                            selected_values[fit], copy=False
                        ),
                        "fit_esm": transformed["fit_esm"],
                        "stop_bio": architecture_preprocessors.transform_bio(
                            selected_values[stop], copy=False
                        ),
                        "stop_esm": transformed["stop_esm"],
                        "temperature_bio": architecture_preprocessors.transform_bio(
                            selected_values[temperature], copy=False
                        ),
                        "temperature_esm": transformed["temperature_esm"],
                        "threshold_bio": architecture_preprocessors.transform_bio(
                            selected_values[threshold_set], copy=False
                        ),
                        "threshold_esm": transformed["threshold_esm"],
                        "outer_bio": architecture_preprocessors.transform_bio(
                            selected_values[outer_validation], copy=False
                        ),
                        "outer_esm": transformed["outer_esm"],
                    }
                    # Release the masked copy immediately; the transformed
                    # arrays above are the only thing the models consume.
                    del selected_values
                    family_schema = {
                        "feature_names": selected_feature_names,
                        "preprocessors": architecture_preprocessors,
                        "transformed": architecture_transformed,
                    }
                selected_feature_names = family_schema["feature_names"]
                architecture_preprocessors = family_schema["preprocessors"]
                architecture_transformed = family_schema["transformed"]
            else:
                selected_feature_names = fold_feature_names
                architecture_preprocessors = preprocessors
                architecture_transformed = transformed
            deep_fold_features[architecture] = selected_feature_names
            bundle_path = fold_dir / f"{architecture}.pt"
            with C.temporary_model_config(architecture_parameters):
                if len(final_training_seeds) != C.N_ENSEMBLE:
                    raise RuntimeError(
                        f"Stage 14 fold {fold} seed count differs from the "
                        f"{architecture} production ensemble"
                    )
                if bundle_path.is_file():
                    logger.info(
                        "Resumed %s fold %d from existing deployment bundle (skipping training)",
                        architecture,
                        fold,
                    )
                    deep_model, architecture_preprocessors, selected_feature_names, operating_threshold, _ = (
                        C.load_deep_bundle(bundle_path)
                    )
                else:
                    deep_model = C.train_deep_model(
                        architecture_transformed["fit_bio"],
                        architecture_transformed["fit_esm"],
                        labels[fit],
                        architecture_transformed["stop_bio"],
                        architecture_transformed["stop_esm"],
                        labels[stop],
                        architecture_transformed["temperature_bio"],
                        architecture_transformed["temperature_esm"],
                        labels[temperature],
                        architecture=architecture,
                        seeds=final_training_seeds,
                        feature_names=(
                            selected_feature_names
                            if C.is_reliability_family(architecture)
                            else None
                        ),
                    )
                    threshold_probabilities = C.predict(
                        deep_model,
                        architecture_transformed["threshold_bio"],
                        architecture_transformed["threshold_esm"],
                    )
                    operating_threshold = C.select_threshold(
                        labels[threshold_set], threshold_probabilities
                    )
                    C.save_deep_bundle(
                        bundle_path,
                        deep_model,
                        architecture_preprocessors,
                        selected_feature_names,
                        operating_threshold,
                        architecture,
                        {
                            "fold": fold,
                            "outer_validation_rows": len(outer_validation),
                            "architecture_parameters": architecture_parameters,
                            "stage14_final_training_seeds": final_training_seeds,
                            "stage14_final_partitions_reused": True,
                            "architecture_interpretation": (
                                _architecture_interpretation(architecture)
                            ),
                            "reliability_feature_names": (
                                [
                                    name
                                    for name in selected_feature_names
                                    if name in C.RELIABILITY_GATE_FEATURES
                                ]
                                if C.is_reliability_family(architecture)
                                else []
                            ),
                        },
                    )
                outer_probabilities = C.predict(
                    deep_model,
                    architecture_transformed["outer_bio"],
                    architecture_transformed["outer_esm"],
                )
                if architecture == C.RELIABILITY_ARCHITECTURE:
                    fold_components = C.predict_reliability_components(
                        deep_model,
                        architecture_transformed["outer_bio"],
                        architecture_transformed["outer_esm"],
                    )
                    if not np.allclose(
                        fold_components["model_probability"],
                        outer_probabilities,
                        atol=1e-6,
                        rtol=1e-6,
                    ):
                        raise RuntimeError(
                            "Reliability component extraction differs from the "
                            "deployment prediction path"
                        )
                    for component_name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS:
                        reliability_components[component_name][outer_validation] = (
                            fold_components[component_name]
                        )
                probabilities[architecture][outer_validation] = outer_probabilities
                thresholds[architecture][outer_validation] = operating_threshold
                decisions[architecture][outer_validation] = (
                    outer_probabilities >= operating_threshold
                ).astype(np.int8)
                fold_metrics[architecture].append(
                    _fold_metric(
                        labels[outer_validation],
                        outer_probabilities,
                        decisions[architecture][outer_validation],
                        operating_threshold,
                    )
                )
                del deep_model
                if C.DEVICE == "cuda":
                    torch.cuda.empty_cache()
        # CHANGELOG 2026-09 (novelty + optimization): the shared family schema
        # is released once, after the loop, rather than after each family
        # member -- freeing it inside the loop would destroy the arrays the
        # next family member reuses.  Peak host memory is unchanged versus the
        # previous per-architecture rebuild because only one copy of the family
        # arrays ever exists at a time.
        family_schema.clear()
        del architecture_transformed, architecture_preprocessors
        del transformed, preprocessors, lightgbm_model, esm_model
        gc.collect()
        if ENABLE_LORA:
            lora_model = C.train_lora_model(
                df.iloc[fit],
                df.iloc[stop],
                df.iloc[temperature],
                seed=C.LORA_SEED + fold,
            )
            lora_threshold_prob = C.predict_lora(lora_model, df.iloc[threshold_set])
            lora_threshold = C.select_threshold(
                labels[threshold_set], lora_threshold_prob
            )
            lora_outer = C.predict_lora(lora_model, df.iloc[outer_validation])
            probabilities["lora_esm"][outer_validation] = lora_outer
            thresholds["lora_esm"][outer_validation] = lora_threshold
            decisions["lora_esm"][outer_validation] = (
                lora_outer >= lora_threshold
            ).astype(np.int8)
            fold_metrics["lora_esm"].append(
                _fold_metric(
                    labels[outer_validation],
                    lora_outer,
                    decisions["lora_esm"][outer_validation],
                    lora_threshold,
                )
            )
            trainable_state = {
                name: parameter.detach().cpu()
                for name, parameter in lora_model.named_parameters()
                if parameter.requires_grad
            }
            lora_path = fold_dir / "lora_esm.pt"
            lora_temporary = lora_path.with_suffix(lora_path.suffix + ".tmp")
            torch.save(
                {
                    "state": trainable_state,
                    "trainable_parameter_names": sorted(trainable_state),
                    "temperature": lora_model.T,
                    "threshold": lora_threshold,
                    "fold": fold,
                    "esm_model": C.ESM_MODEL_NAME,
                    "esm_layer": C.ESM_LAYER,
                    "target_modules": C.LORA_TARGET_MODULES,
                    "rank": C.LORA_RANK,
                    "alpha": C.LORA_ALPHA,
                    "dropout": C.LORA_DROPOUT,
                },
                lora_temporary,
            )
            lora_temporary.replace(lora_path)
            del lora_model
            if C.DEVICE == "cuda":
                torch.cuda.empty_cache()
        _atomic_json(
            {
                "fold": fold,
                "feature_names": fold_feature_names,
                "deep_feature_names_by_architecture": deep_fold_features,
                "tabular_features": fold_tabular_features,
                "dropped_constant_features": [
                    name for name, keep in zip(feature_names, bio_mask) if not keep
                ],
                "thresholds": {
                    name: float(thresholds[name][outer_validation][0])
                    for name in model_names
                },
                "lightgbm_nested_selected_hyperparameters": training_record[
                    "lightgbm_parameters"
                ],
            },
            fold_dir / "metadata.json",
        )
        logger.info("Completed outer fold %d/%d", fold, N_SPLITS)
    if (fold_ids < 1).any():
        raise RuntimeError("Some rows never received an outer prediction")
    results: dict[str, Any] = {
        "tag": MODEL_TAG,
        "n_rows": len(df),
        "n_genes": int(df[C.GENE_COL].astype(str).nunique()),
        "n_split_groups": int(pd.Series(groups).nunique()),
        "split_policy": (
            "homology_or_connected_group" if "split_group" in df else "gene_only"
        ),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_names_by_architecture": feature_names_by_architecture,
        "tabular_features": tabular_features,
        "folds": N_SPLITS,
        "tuned_parameters_by_architecture": tuned_parameter_sets,
        "model_roles": {name: _model_reporting_role(name) for name in model_names},
        "evaluation_status": "post_selection_deployment_oof_not_primary",
        "primary_internal_results_source": "stage14_nested_outer_oof",
        "stage14_final_partitions_reused": True,
        "stage14_final_training_seeds_reused_for_deep_models": True,
        "stage14_exact_oof_reproduction": False,
        "proposed_model": C.RELIABILITY_ARCHITECTURE,
        "reliability_residual_protocol": C.reliability_architecture_protocol(
            feature_names_by_architecture[C.RELIABILITY_ARCHITECTURE],
            tuned_parameter_sets.get(C.RELIABILITY_ARCHITECTURE, {}),
        ),
        "lightgbm_hyperparameter_protocol": (
            "stage14_same_persisted_inner_group_folds_compact_grid"
        ),
        "non_reproduction_reason": (
            "Stage11 refits deployment artifacts with one post-selection production "
            "configuration per architecture; Stage14 authoritative OOF uses "
            "fold-specific inner-CV winners and its prespecified reference routines."
        ),
    }
    for name in model_names:
        if not np.isfinite(probabilities[name]).all():
            raise RuntimeError(f"{name} OOF probabilities are incomplete")
        if (decisions[name] < 0).any():
            raise RuntimeError(f"{name} OOF decisions are incomplete")
        metrics = C.evaluate(
            labels,
            probabilities[name],
            thresholds[name],
            predictions=decisions[name],
        )
        metrics["fold_thresholds"] = [
            round(float(thresholds[name][fold_ids == fold][0]), 4)
            for fold in range(1, N_SPLITS + 1)
        ]
        metrics["fold_metrics"] = fold_metrics[name]
        metrics["model_role"] = _model_reporting_role(name)
        metrics["ci95"] = C.group_bootstrap_intervals(
            labels,
            probabilities[name],
            decisions[name],
            groups,
        )
        results[name] = metrics
    if any(
        not np.isfinite(values).all()
        for values in reliability_components.values()
    ):
        raise RuntimeError("Reliability OOF diagnostics are incomplete")
    reliability_anchor_decisions = (
        reliability_components["anchor_probability"]
        >= thresholds[C.RELIABILITY_ARCHITECTURE]
    ).astype(np.int8)
    reliability_diagnostics = C.summarize_reliability_diagnostics(
        reliability_components,
        labels,
        probabilities[C.RELIABILITY_ARCHITECTURE],
        thresholds[C.RELIABILITY_ARCHITECTURE],
        decisions[C.RELIABILITY_ARCHITECTURE],
        anchor_decisions=reliability_anchor_decisions,
    )
    reliability_diagnostics["evaluation_scope"] = (
        "stage11_deployment_oof_post_selection_not_primary"
    )
    reliability_diagnostics["fold_aggregation"] = (
        "one held-out deployment fold per internal row"
    )
    results["reliability_diagnostics"] = reliability_diagnostics
    primary_baseline = (
        "esm_conservation_logistic"
        if "esm_conservation_logistic" in model_names
        else "raw_esm_zero_shot"
    )
    paired_comparisons: dict[str, Any] = {}
    # CHANGELOG 2026-09 (novelty): iterate the same tuple that was trained so
    # the new reliability-family member also receives a cluster-aware paired
    # comparison against the primary baseline.  Previously this list was a
    # hard-coded duplicate of `deep_architectures` and would silently omit any
    # newly added architecture.
    for proposed in deep_architectures:
        key = f"{primary_baseline}_vs_{proposed}"
        paired_comparisons[key] = C.clustered_model_comparison(
            labels,
            probabilities[primary_baseline],
            probabilities[proposed],
            decisions[primary_baseline],
            decisions[proposed],
            groups,
        )
    for metric in ("mcc", "auroc", "auprc"):
        keys = [key for key in paired_comparisons if paired_comparisons[key][metric]]
        raw_values = [
            paired_comparisons[key][metric]["two_sided_probability"] for key in keys
        ]
        if raw_values:
            adjusted = multipletests(raw_values, method="holm")[1]
            for key, value in zip(keys, adjusted):
                paired_comparisons[key][metric]["holm_adjusted_probability"] = round(
                    float(value), 4
                )
    results["comparisons"] = {
        "primary_baseline": primary_baseline,
        "clustered_paired": paired_comparisons,
        "note": "Row-wise McNemar omitted because variants within groups are correlated.",
        "inference_scope": "conditional_on_one_nested_cv_oof_realization",
        "training_procedure_uncertainty_included": False,
    }
    results["contextual_established_predictors"] = _contextual_predictor_results(
        df, labels
    )
    C.save_oof_artifacts(
        OOF_STORE,
        labels,
        probabilities,
        MODEL_TAG,
        groups=groups,
        thresholds=thresholds,
        row_ids=df[C.ROW_ID_COL].astype(str).to_numpy(),
        fold_ids=fold_ids,
        extra_arrays={
            **{
                f"reliability_components__{name}": values
                for name, values in reliability_components.items()
            },
            "reliability_components__availability_stratum_code": (
                C.reliability_availability_codes(reliability_components)
            ),
            "reliability_components__anchor_decisions": (
                reliability_anchor_decisions
            ),
        },
    )
    C.save_shap_artifact(
        SHAP_STORE,
        shap_values,
        shap_inputs,
        tabular_features,
        shap_folds,
    )
    fold_table = pd.DataFrame(
        {
            C.ROW_ID_COL: df[C.ROW_ID_COL].astype(str),
            C.GENE_COL: df[C.GENE_COL].astype(str),
            "split_group": groups,
            C.LABEL_COL: labels,
            "outer_fold": fold_ids,
        }
    )
    fold_temporary = FOLD_ASSIGNMENTS.with_suffix(FOLD_ASSIGNMENTS.suffix + ".tmp")
    fold_table.to_csv(fold_temporary, index=False)
    fold_temporary.replace(FOLD_ASSIGNMENTS)
    _atomic_json(inner_records, INNER_PARTITIONS)
    rows = []
    for name in model_names:
        metrics = results[name]
        rows.append(
            {
                "model": name,
                "model_role": metrics["model_role"],
                "config": MODEL_TAG,
                "mcc": metrics["mcc"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "brier": metrics["brier"],
                "brier_skill": metrics["calibration"]["brier_skill"],
                "log_loss": metrics["calibration"]["log_loss"],
                "adaptive_ece": metrics["calibration"]["adaptive_ece"],
                "calibration_slope": metrics["calibration"]["calibration_slope"],
                "calibration_intercept": metrics["calibration"][
                    "calibration_intercept"
                ],
                "mean_threshold": metrics["threshold"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison_temporary = COMPARISON_CSV.with_suffix(COMPARISON_CSV.suffix + ".tmp")
    comparison.to_csv(comparison_temporary, index=False)
    comparison_temporary.replace(COMPARISON_CSV)
    reliability_table = pd.DataFrame(
        C.reliability_diagnostic_rows(
            reliability_diagnostics,
            {
                "scope": "internal_deployment_oof",
                "model": C.RELIABILITY_ARCHITECTURE,
                "config": MODEL_TAG,
            },
        )
    )
    reliability_temporary = RELIABILITY_DIAGNOSTICS_CSV.with_suffix(
        RELIABILITY_DIAGNOSTICS_CSV.suffix + ".tmp"
    )
    reliability_table.to_csv(reliability_temporary, index=False)
    reliability_temporary.replace(RELIABILITY_DIAGNOSTICS_CSV)
    _atomic_json({"primary": results}, RESULTS_JSON)
    model_outputs = sorted(path for path in MODEL_DIR.rglob("*") if path.is_file())
    stage_outputs = [
        RESULTS_JSON,
        COMPARISON_CSV,
        RELIABILITY_DIAGNOSTICS_CSV,
        FOLD_ASSIGNMENTS,
        INNER_PARTITIONS,
        OOF_STORE,
        *([SHAP_STORE] if SHAP_STORE.exists() else []),
        *model_outputs,
    ]
    write_run_manifest(
        MANIFEST_FILE,
        "11_train_and_evaluate",
        [
            INPUT_CSV,
            INPUT_NPY,
            STATUS_CSV,
            STAGE10_MANIFEST,
            TUNING_BEST_JSON,
            TUNING_SPLIT_PLAN,
            ARCHITECTURE_SELECTION,
            STAGE14_MANIFEST,
            *([AUDIT_SIDECAR] if AUDIT_SIDECAR.exists() else []),
        ],
        {
            "model_tag": MODEL_TAG,
            "models": model_names,
            "rows": len(df),
            "genes": int(df[C.GENE_COL].astype(str).nunique()),
            "split_groups": int(pd.Series(groups).nunique()),
            "split_policy": (
                "homology_or_connected_group" if "split_group" in df else "gene_only"
            ),
            "folds": N_SPLITS,
            "features": feature_names,
            "features_by_architecture": feature_names_by_architecture,
            "lora_enabled": ENABLE_LORA,
            "gpu_runtime": C.gpu_runtime_summary(),
            "threshold_policy": "dedicated_inner_threshold_partition",
            "calibration_policy": "dedicated_inner_affine_logit_partition",
            "evaluation_role": "deployment_oof_post_selection_not_primary",
            "stage14_final_partitions_reused": True,
            "stage14_final_training_seeds_reused_for_deep_models": True,
            "stage14_exact_oof_reproduction": False,
            "proposed_model": C.RELIABILITY_ARCHITECTURE,
            "reliability_residual_protocol": C.reliability_architecture_protocol(
                feature_names_by_architecture[C.RELIABILITY_ARCHITECTURE],
                tuned_parameter_sets.get(C.RELIABILITY_ARCHITECTURE, {}),
            ),
            "reliability_diagnostics": {
                "status": "descriptive_post_selection_not_primary",
                "strata_use_labels": False,
                "used_for_model_or_threshold_selection": False,
                "component_names": list(C.RELIABILITY_DIAGNOSTIC_COMPONENTS),
            },
            "lightgbm_hyperparameter_protocol": (
                "stage14_same_persisted_inner_group_folds_compact_grid"
            ),
        },
        outputs=stage_outputs,
    )
    logger.info("Saved nested OOF evaluation for %d models", len(model_names))


if __name__ == "__main__":
    run_evaluation()
