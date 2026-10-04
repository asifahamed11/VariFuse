from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import common as C
import publication_robustness as PR
from config import (
    MODEL_TAG,
    RANDOM_STATE,
    REQUIRE_EXTERNAL_CLINVAR,
    REQUIRE_EXTERNAL_DMS,
    STAGE10_OUT,
    STAGE07_OUT,
    STAGE09_OUT,
    STAGE11_OUT,
    STAGE12_OUT,
    STAGE13_OUT,
    STAGE14_OUT,
    artifact_record,
    ensure_directories,
    file_sha256,
    json_default,
    write_run_manifest,
)
from schema import GENE_COL, LABEL_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage13_figures")

INTERNAL_PREDICTIONS = STAGE14_OUT / "nested_tuning_oof.npz"
INTERNAL_RESULTS = STAGE14_OUT / "architecture_selection.json"
INTERNAL_SPLIT_PLAN = STAGE14_OUT / "nested_inner_splits.json"
INTERNAL_RUN_MANIFEST = STAGE14_OUT / "run_manifest.json"
CONFIRMATORY_SEED_PLAN = STAGE14_OUT / "confirmatory_seed_plan.json"
CONFIRMATORY_RESULTS = STAGE14_OUT / "confirmatory_repeated_cv_results.json"
CONFIRMATORY_PREDICTIONS = (
    STAGE14_OUT / "confirmatory_repeated_cv_predictions.npz"
)
CONFIRMATORY_FOLD_ASSIGNMENTS = (
    STAGE14_OUT / "confirmatory_repeated_cv_folds.csv"
)
CONFIRMATORY_CHECKPOINT_DIR = STAGE14_OUT / "confirmatory_repeats"
CONFIRMATORY_STABILITY_TABLE = STAGE13_OUT / "confirmatory_stability_table.csv"
MECHANISTIC_RELIABILITY_TABLE = STAGE13_OUT / "mechanistic_reliability_table.csv"
DEPLOYMENT_RESULTS = STAGE11_OUT / "results.json"
DEPLOYMENT_RUN_MANIFEST = STAGE11_OUT / "run_manifest.json"
SHAP_FILE = STAGE11_OUT / f"shap_values_{MODEL_TAG}.npz"
EXTERNAL_PREDICTIONS = STAGE12_OUT / "external_predictions.npz"
EXTERNAL_RESULTS = STAGE12_OUT / "external_validation.json"
EXTERNAL_SUMMARY_TABLE = STAGE12_OUT / "external_validation_table.csv"
EXTERNAL_PREDICTION_TABLE = STAGE12_OUT / "external_predictions.csv"
EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE = (
    STAGE12_OUT / "reliability_diagnostics.csv"
)
EXTERNAL_RUN_MANIFEST = STAGE12_OUT / "run_manifest.json"
EXTERNAL_PREP_MANIFEST = STAGE12_OUT.parent / "09_prepare_external_esm" / "run_manifest.json"
STAGE07_MANIFEST = STAGE07_OUT / "run_manifest.json"
INTERNAL_UNIVERSE = STAGE07_OUT / "Final_Dataset_Natural_Prevalence.parquet"
STAGE09_PREPARED_OUTPUTS = {
    source: STAGE09_OUT / f"{source}_esm_ready.csv" for source in ("clinvar", "dms")
}
DMS_SEQUENCE_OUTPUT = STAGE09_OUT / "dms_sequences.parquet"
STAGE10_MANIFESTS = {
    source: STAGE10_OUT / f"{source}_esm_manifest.json"
    for source in ("internal", "clinvar", "dms")
}
INTERNAL_CSV = STAGE10_OUT / "internal_with_esm.parquet"
INTERNAL_EMBEDDINGS = STAGE10_OUT / "internal_esm_embeddings.npy"
INTERNAL_STATUS = STAGE10_OUT / "internal_esm_extraction.parquet"
EXTERNAL_STAGE10_INPUTS = {
    source: (
        STAGE10_OUT / f"{source}_with_esm.parquet",
        STAGE10_OUT / f"{source}_esm_embeddings.npy",
        STAGE10_OUT / f"{source}_esm_extraction.parquet",
    )
    for source in ("clinvar", "dms")
}
FIGURE_MANIFEST = STAGE13_OUT / "figure_manifest.json"
RUN_MANIFEST = STAGE13_OUT / "run_manifest.json"
BOOTSTRAPS = int(os.environ.get("FIGURE_BOOTSTRAPS", "500"))
ALLOW_EXISTING_FIGURE_DIR = os.environ.get(
    "ALLOW_EXISTING_FIGURE_DIR", "0"
).strip().lower() in {"1", "true", "yes", "on"}
REQUIRED_STAGE14_PROTOCOL = "fixed_nested_group_cv_v4_reliability_residual"
REQUIRED_STAGE14_ARCHITECTURES = {
    "concatenation",
    "gated_fusion",
    "reliability_residual",
    C.RELIABILITY_ARCHITECTURE,
    "cross_attention",
}
OPTIONAL_STAGE14_ARCHITECTURES = (
    set(C.RELIABILITY_FAMILY_ARCHITECTURES) - REQUIRED_STAGE14_ARCHITECTURES
)
REQUIRED_STAGE14_REFERENCES = {
    "raw_esm_zero_shot",
    "esm_score_logistic",
    "conservation_logistic",
    "esm_conservation_logistic",
    "mutation_logistic",
    "availability_logistic",
    "esm_embedding_mutation",
    "lightgbm",
}
CONFIRMATORY_MODEL_NAMES = (
    C.RELIABILITY_ARCHITECTURE,
    "raw_esm_zero_shot",
    "esm_conservation_logistic",
    "lightgbm",
)
CONFIRMATORY_SCIENTIFIC_ROLE = (
    "separate_confirmatory_fixed_configuration_repeated_group_cv"
)
STAGE14_PROVENANCE_SOURCES = (
    "14_tune_cross_attention.py",
    "common.py",
    "gpu_runtime.py",
    "config.py",
    "schema.py",
)
STAGE12_PROVENANCE_SOURCES = (
    "12_external_validation.py",
    "publication_robustness.py",
    "common.py",
    "gpu_runtime.py",
    "config.py",
    "schema.py",
    "table_io.py",
)
STAGE11_PROVENANCE_SOURCES = (
    "11_train_and_evaluate.py",
    "common.py",
    "gpu_runtime.py",
    "config.py",
    "schema.py",
)
STAGE09_PROVENANCE_SOURCES = (
    "09_prepare_external_esm_dataset.py",
    "01_dbnsfp_processor.py",
    "04_feature_engineering.py",
    "08_prepare_esm_dataset.py",
    "config.py",
    "schema.py",
    "table_io.py",
)

MODEL_ORDER = (
    "lightgbm",
    "raw_esm_zero_shot",
    "esm_score_logistic",
    "conservation_logistic",
    "esm_conservation_logistic",
    "mutation_logistic",
    "availability_logistic",
    "esm_embedding_mutation",
    "esm_only",
    "concatenation",
    "gated_fusion",
    "reliability_residual",
    "evidential_residual",
    "cross_attention",
    "lora_esm",
    "esm_zero_shot",
)
PRIMARY_INTERNAL_PLOT_ORDER = (
    "raw_esm_zero_shot",
    "esm_conservation_logistic",
    "lightgbm",
    "concatenation",
    "gated_fusion",
    "reliability_residual",
    "evidential_residual",
    "cross_attention",
)
CONTEXTUAL_ORDER = (
    "context_AlphaMissense_score",
    "context_EVE_score",
    "context_PrimateAI_score",
    "context_PrimateAI-3D_score",
    "context_MetaRNN_score",
    "context_BayesDel_noAF_score",
    "context_BayesDel_addAF_score",
    "context_VEST4_score",
    "context_MutPred2_score",
    "context_MPC_score",
    "context_ClinPred_score",
    "context_DEOGEN2_score",
    "context_LIST-S2_score",
    "context_VARITY_R_score",
    "context_VARITY_ER_score",
    "context_VARITY_R_LOO_score",
    "context_VARITY_ER_LOO_score",
    "context_SIFT_score",
    "context_Polyphen2_HDIV_score",
    "context_CADD_phred",
    "context_REVEL_score",
    "context_CONSENSUS_SCORE",
)
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "raw_esm_zero_shot": "ESM score + calibration",
    "esm_score_logistic": "ESM-score logistic",
    "conservation_logistic": "Conservation logistic",
    "esm_conservation_logistic": "ESM + conservation",
    "mutation_logistic": "Mutation logistic",
    "availability_logistic": "Availability audit",
    "esm_embedding_mutation": "ESM embedding + mutation",
    "esm_only": "ESM-only",
    "concatenation": "Concatenation",
    "gated_fusion": "Gated fusion",
    "reliability_residual": "Reliability-residual fusion",
    "evidential_residual": "Availability-weighted expert fusion",
    "cross_attention": "Legacy pseudo-slot attention",
    "lora_esm": "LoRA ESM",
    "esm_zero_shot": "ESM2 zero-shot",
    "context_SIFT_score": "SIFT",
    "context_Polyphen2_HDIV_score": "PolyPhen-2 HDIV",
    "context_CADD_phred": "CADD",
    "context_REVEL_score": "REVEL",
    "context_CONSENSUS_SCORE": "Legacy consensus",
    "context_AlphaMissense_score": "AlphaMissense",
    "context_EVE_score": "EVE",
    "context_PrimateAI_score": "PrimateAI",
    "context_PrimateAI-3D_score": "PrimateAI-3D",
    "context_MetaRNN_score": "MetaRNN",
    "context_BayesDel_noAF_score": "BayesDel (no AF)",
    "context_BayesDel_addAF_score": "BayesDel (AF)",
    "context_VEST4_score": "VEST4",
    "context_MutPred2_score": "MutPred2",
    "context_MPC_score": "MPC",
    "context_ClinPred_score": "ClinPred",
    "context_DEOGEN2_score": "DEOGEN2",
    "context_LIST-S2_score": "LIST-S2",
    "context_VARITY_R_score": "VARITY-R",
    "context_VARITY_ER_score": "VARITY-ER",
    "context_VARITY_R_LOO_score": "VARITY-R LOO",
    "context_VARITY_ER_LOO_score": "VARITY-ER LOO",
}
MODEL_COLORS = {
    "lightgbm": "#0072B2",
    "raw_esm_zero_shot": "#4D4D4D",
    "esm_score_logistic": "#009E73",
    "conservation_logistic": "#56B4E9",
    "esm_conservation_logistic": "#1B7837",
    "mutation_logistic": "#A6D854",
    "availability_logistic": "#8C564B",
    "esm_embedding_mutation": "#E6AB02",
    "esm_only": "#009E73",
    "concatenation": "#CC79A7",
    "gated_fusion": "#7570B3",
    "reliability_residual": "#332288",
    "evidential_residual": "#AA4499",
    "cross_attention": "#D55E00",
    "lora_esm": "#E69F00",
    "esm_zero_shot": "#6A3D9A",
    "context_SIFT_score": "#B3B3B3",
    "context_Polyphen2_HDIV_score": "#969696",
    "context_CADD_phred": "#737373",
    "context_REVEL_score": "#525252",
    "context_CONSENSUS_SCORE": "#252525",
    "context_AlphaMissense_score": "#67001F",
    "context_EVE_score": "#8E0152",
    "context_PrimateAI_score": "#B2182B",
    "context_PrimateAI-3D_score": "#C51B7D",
    "context_MetaRNN_score": "#D6604D",
    "context_BayesDel_noAF_score": "#F4A582",
    "context_BayesDel_addAF_score": "#FDDBC7",
    "context_VEST4_score": "#92C5DE",
    "context_MutPred2_score": "#4393C3",
    "context_MPC_score": "#2166AC",
    "context_ClinPred_score": "#053061",
    "context_DEOGEN2_score": "#D1E5F0",
    "context_LIST-S2_score": "#67A9CF",
    "context_VARITY_R_score": "#762A83",
    "context_VARITY_ER_score": "#9970AB",
    "context_VARITY_R_LOO_score": "#C2A5CF",
    "context_VARITY_ER_LOO_score": "#E7D4E8",
}
FORMATS = ("png", "svg", "pdf")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass
class ExternalVectors:
    source: str
    policy: str
    prefix: str
    labels: np.ndarray
    groups: np.ndarray
    scores: dict[str, np.ndarray]
    decisions: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as stored:
        return {key: stored[key] for key in stored.files}


def _canonical_sha256(value: Any) -> str:
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


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _manifest_input_record(
    manifest: dict[str, Any], path: Path
) -> dict[str, Any]:
    suffix = f"/10_esm_features/{path.name}".casefold()
    matches = [
        record
        for recorded_path, record in manifest.get("inputs", {}).items()
        if str(recorded_path).replace("\\", "/").casefold().endswith(suffix)
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError(
            f"Stage 14 manifest does not identify exactly one {path.name} input"
        )
    return matches[0]


def _manifest_record_by_suffix(
    manifest: dict[str, Any],
    section: str,
    suffix: str,
    description: str,
) -> dict[str, Any]:
    normalized_suffix = "/" + suffix.replace("\\", "/").strip("/").casefold()
    records = manifest.get(section, {})
    if not isinstance(records, dict):
        raise RuntimeError(f"{description} manifest has no {section} records")
    matches = [
        record
        for recorded_path, record in records.items()
        if str(recorded_path)
        .replace("\\", "/")
        .casefold()
        .endswith(normalized_suffix)
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError(
            f"{description} manifest does not identify exactly one {suffix} record"
        )
    return matches[0]


def _verify_manifest_artifact(
    manifest: dict[str, Any],
    path: Path,
    suffix: str,
    description: str,
    section: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _manifest_record_by_suffix(
        manifest, section, suffix, description
    )
    observed = artifact_record(path)
    if expected.get("exists") is not True or observed.get("exists") is not True:
        raise RuntimeError(
            f"{description} {section[:-1]} is not recorded as present: {suffix}"
        )
    expected_full = expected.get("sha256")
    observed_full = observed.get("sha256")
    if expected_full is not None:
        matches = observed_full is not None and expected_full == observed_full
    else:
        matches = all(
            expected.get(field) is not None
            and expected.get(field) == observed.get(field)
            for field in ("size_bytes", "sample_sha256")
        )
    if not matches:
        raise RuntimeError(
            f"{description} {section[:-1]} fingerprint differs for {suffix}"
        )


def _verify_manifest_input(
    manifest: dict[str, Any],
    path: Path,
    suffix: str,
    description: str,
) -> None:
    _verify_manifest_artifact(
        manifest, path, suffix, description, section="inputs"
    )


def _verify_manifest_output(
    manifest: dict[str, Any],
    path: Path,
    suffix: str,
    description: str,
) -> None:
    artifact_id = suffix.replace("\\", "/").strip("/")
    outputs = manifest.get("outputs")
    record = outputs.get(artifact_id) if isinstance(outputs, dict) else None
    if (
        manifest.get("artifact_manifest_version") != 2
        or not isinstance(record, dict)
        or record.get("artifact_id") != artifact_id
    ):
        raise RuntimeError(
            f"{description} manifest does not bind portable output {artifact_id}"
        )
    observed = artifact_record(path)
    common_mismatch = (
        not path.exists()
        or record.get("exists") is not True
        or record.get("kind") != observed.get("kind")
    )
    if observed.get("kind") == "file":
        content_mismatch = (
            record.get("sha256") != observed.get("sha256")
            or record.get("size_bytes") != observed.get("size_bytes")
        )
    elif observed.get("kind") == "directory":
        content_mismatch = record.get("directory_fingerprint") != observed.get(
            "directory_fingerprint"
        )
    else:
        content_mismatch = True
    if common_mismatch or content_mismatch:
        raise RuntimeError(f"{description} output fingerprint differs for {artifact_id}")


def _validate_source_provenance(
    manifest: dict[str, Any],
    source_names: tuple[str, ...],
    description: str,
) -> None:
    source_dir = Path(__file__).resolve().parent
    recorded = manifest.get("source_files", {})
    if not isinstance(recorded, dict):
        raise RuntimeError(f"{description} manifest lacks source-file provenance")
    for source_name in source_names:
        current_hash = file_sha256(source_dir / source_name)
        if not current_hash or recorded.get(source_name) != current_hash:
            raise RuntimeError(
                f"{description} source provenance differs for {source_name}; "
                f"rerun {description}"
            )


def _assert_reported_value(
    reported: Any,
    expected: Any,
    path: str,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(reported, dict):
            raise RuntimeError(f"Stage 14 reported metric {path} is not an object")
        missing = set(expected) - set(reported)
        if missing:
            raise RuntimeError(
                f"Stage 14 reported metric {path} misses {sorted(missing)}"
            )
        for key, value in expected.items():
            _assert_reported_value(reported[key], value, f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(reported, list) or len(reported) != len(expected):
            raise RuntimeError(f"Stage 14 reported metric {path} has wrong length")
        for index, value in enumerate(expected):
            _assert_reported_value(reported[index], value, f"{path}[{index}]")
        return
    if expected is None:
        if reported is not None:
            raise RuntimeError(
                f"Stage 14 reported metric {path} differs: {reported!r} != None"
            )
        return
    if isinstance(expected, (float, np.floating)):
        # JSON metrics and their persisted prediction arrays are separate
        # artifacts.  Serializing predictions (especially older float32
        # artifacts) can create/remove an exact rank tie, and SciPy releases
        # can consequently differ by one unit in the final reported decimal.
        # Exact manifest hashes above still bind both files to the same run, so
        # accept no more than one unit of the metric's declared precision.
        metric_name = path.rsplit(".", 1)[-1]
        four_decimal_metrics = {
            "threshold",
            "mcc",
            "auroc",
            "auprc",
            "precision",
            "recall",
            "f1",
            "specificity",
            "npv",
            "functional_spearman",
            "coverage",
        }
        six_decimal_metrics = {
            "brier",
            "prevalence",
            "prevalence_brier",
            "log_loss",
            "adaptive_ece",
            "brier_skill",
            "calibration_slope",
            "calibration_intercept",
            "risk",
            "mean",
            "median",
            "p05",
            "p95",
            "active_rate",
            "exact_fallback_rate",
            "hard_unavailable_rate",
            "mean_absolute",
            "maximum_absolute_observed",
            "mean_gate",
        }
        if metric_name in four_decimal_metrics:
            absolute_tolerance = 1.0000001e-4
        elif metric_name in six_decimal_metrics:
            absolute_tolerance = 1.0000001e-6
        else:
            absolute_tolerance = 5e-7
        try:
            matches = bool(
                np.isclose(
                    float(reported),
                    float(expected),
                    rtol=0.0,
                    atol=absolute_tolerance,
                )
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise RuntimeError(
                f"Stage 14 reported metric {path} differs: "
                f"{reported!r} != {expected!r}"
            )
        return
    if reported != expected:
        raise RuntimeError(
            f"Stage 14 reported metric {path} differs: {reported!r} != {expected!r}"
        )


def _validate_stage14_model_output(
    data: dict[str, np.ndarray],
    prefix: str,
    reported: dict[str, Any],
    labels: np.ndarray,
    n_rows: int,
) -> None:
    keys = {
        role: f"{prefix}__{role}"
        for role in ("probabilities", "decisions", "thresholds")
    }
    missing = [key for key in keys.values() if key not in data]
    if missing:
        raise RuntimeError(f"Stage 14 OOF misses {prefix}: {missing}")
    probabilities = np.asarray(data[keys["probabilities"]], dtype=float)
    decisions = np.asarray(data[keys["decisions"]])
    thresholds = np.asarray(data[keys["thresholds"]], dtype=float)
    for role, values in (
        ("probabilities", probabilities),
        ("decisions", decisions),
        ("thresholds", thresholds),
    ):
        if values.ndim != 1 or len(values) != n_rows:
            raise RuntimeError(
                f"Stage 14 OOF {prefix} {role} has shape {values.shape}, "
                f"expected ({n_rows},)"
            )
    if not np.isfinite(probabilities).all() or not np.isfinite(thresholds).all():
        raise RuntimeError(f"Stage 14 OOF {prefix} contains non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise RuntimeError(f"Stage 14 OOF {prefix} probabilities are outside [0, 1]")
    if ((thresholds < 0.0) | (thresholds > 1.0)).any():
        raise RuntimeError(f"Stage 14 OOF {prefix} thresholds are outside [0, 1]")
    if not np.isin(decisions, [0, 1]).all():
        raise RuntimeError(f"Stage 14 OOF {prefix} decisions are not binary")
    decisions = decisions.astype(np.int8)
    expected_decisions = (probabilities >= thresholds).astype(np.int8)
    if not np.array_equal(decisions, expected_decisions):
        raise RuntimeError(
            f"Stage 14 OOF {prefix} decisions disagree with frozen thresholds"
        )
    recomputed = C.evaluate(
        labels,
        probabilities,
        thresholds,
        predictions=decisions,
    )
    _assert_reported_value(reported, recomputed, prefix)


def _confirmatory_prefix(model_name: str) -> str:
    if model_name == C.RELIABILITY_ARCHITECTURE:
        return model_name
    return f"reference__{model_name}"


def _confirmatory_scalar_metric_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recreate the Stage 14 across-repeat summary from authenticated arrays."""
    summary: dict[str, Any] = {}
    for name in (
        "mcc",
        "auroc",
        "auprc",
        "brier",
        "precision",
        "recall",
        "f1",
        "specificity",
        "npv",
    ):
        values = [
            float(record[name])
            for record in records
            if record.get(name) is not None
        ]
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


def _path_value_has_suffix(value: Any, suffix: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.replace("\\", "/").casefold()
    return normalized.endswith("/" + suffix.strip("/").casefold())


def _confirmatory_expected_fold_ids(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    outer_folds: int,
    split_seed: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    splits = C.make_group_splits(labels, groups, outer_folds, split_seed)
    fold_ids = np.full(len(labels), -1, dtype=np.int16)
    validation_indices: list[np.ndarray] = []
    for fold, (_, validation) in enumerate(splits, 1):
        validation = np.asarray(validation, dtype=np.int64)
        if (fold_ids[validation] != -1).any():
            raise RuntimeError("Confirmatory split generator produced overlapping folds")
        fold_ids[validation] = fold
        validation_indices.append(validation)
    if (fold_ids < 1).any():
        raise RuntimeError("Confirmatory split generator did not cover every row")
    return fold_ids, validation_indices


def _validate_confirmatory_repeat_split_plan(
    repeat_plan: Any,
    plan: dict[str, Any],
    repeat_record: dict[str, Any],
    labels: np.ndarray,
    groups: np.ndarray,
    row_ids: np.ndarray,
) -> np.ndarray:
    if not isinstance(repeat_plan, dict):
        raise RuntimeError("Confirmatory checkpoint has no repeat split plan")
    recorded_hash = repeat_plan.get("canonical_sha256")
    unhashed = dict(repeat_plan)
    unhashed.pop("canonical_sha256", None)
    if recorded_hash != _canonical_sha256(unhashed):
        raise RuntimeError("Confirmatory repeat split-plan canonical hash is invalid")
    context = plan["execution_context"]
    expected_identity = {
        "protocol_version": REQUIRED_STAGE14_PROTOCOL,
        "n_rows": len(labels),
        "row_order_sha256": _ordered_text_sha256(row_ids),
        "input_sha256": context["input_sha256"],
        "embedding_sha256": context["embedding_sha256"],
        "outer_folds": int(context["outer_folds"]),
        "inner_folds": int(context["inner_folds"]),
        "split_seed": int(repeat_record["split_seed"]),
        "training_seed": int(repeat_record["training_seed"]),
        "search_ensemble": 1,
        "final_ensemble": int(plan["production_parameters"]["N_ENSEMBLE"]),
        "final_epochs": int(plan["production_parameters"]["DEEP_MAX_EPOCHS"]),
        "final_patience": int(plan["production_parameters"]["DEEP_PATIENCE"]),
        "deterministic": True,
    }
    for key, expected in expected_identity.items():
        if repeat_plan.get(key) != expected:
            raise RuntimeError(
                f"Confirmatory repeat split plan differs for {key}: "
                f"{repeat_plan.get(key)!r} != {expected!r}"
            )
    fingerprint = repeat_plan.get("protocol_fingerprint")
    if fingerprint != {
        "scientific_role": plan["scientific_role"],
        "confirmatory_plan_sha256": plan["canonical_sha256"],
        "rehpo": False,
    }:
        raise RuntimeError("Confirmatory repeat split-plan role or selection flag differs")

    expected_folds, expected_validation = _confirmatory_expected_fold_ids(
        labels,
        groups,
        outer_folds=int(context["outer_folds"]),
        split_seed=int(repeat_record["split_seed"]),
    )
    records = repeat_plan.get("folds")
    if not isinstance(records, list) or len(records) != len(expected_validation):
        raise RuntimeError("Confirmatory repeat split-plan fold inventory differs")
    for fold, (record, expected_indices) in enumerate(
        zip(records, expected_validation), 1
    ):
        if not isinstance(record, dict) or record.get("outer_fold") != fold:
            raise RuntimeError("Confirmatory repeat split-plan fold identity differs")
        validation = record.get("outer_validation")
        if not isinstance(validation, dict):
            raise RuntimeError("Confirmatory repeat has no outer-validation record")
        observed_indices = np.asarray(validation.get("indices", []), dtype=np.int64)
        if not np.array_equal(observed_indices, expected_indices):
            raise RuntimeError(
                "Confirmatory repeat outer folds differ from the prespecified seed"
            )
        if validation.get("n") != len(expected_indices) or validation.get(
            "row_ids_sha256"
        ) != _ordered_text_sha256(row_ids[expected_indices]):
            raise RuntimeError("Confirmatory repeat outer-fold row identity differs")
    return expected_folds


def _validate_stage14_confirmatory_artifacts(
    payload: dict[str, Any],
    run_manifest: dict[str, Any],
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    row_ids: np.ndarray,
    primary_split_plan: dict[str, Any],
    actual_hashes: dict[Path, str],
) -> dict[str, Any]:
    """Authenticate optional fixed-configuration repeated-CV confirmation outputs."""
    declaration = payload.get("confirmatory_repeated_cv")
    if declaration is None:
        return {"requested": False, "status": "not_requested"}
    if not isinstance(declaration, dict) or not isinstance(
        declaration.get("requested"), bool
    ):
        raise RuntimeError("Stage 14 confirmatory declaration is malformed")
    if declaration["requested"] is False:
        return {"requested": False, "status": "not_requested"}

    required_files = (
        CONFIRMATORY_SEED_PLAN,
        CONFIRMATORY_RESULTS,
        CONFIRMATORY_PREDICTIONS,
        CONFIRMATORY_FOLD_ASSIGNMENTS,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing or not CONFIRMATORY_CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            "Requested Stage 14 confirmatory artifacts are incomplete: "
            f"{missing + ([str(CONFIRMATORY_CHECKPOINT_DIR)] if not CONFIRMATORY_CHECKPOINT_DIR.is_dir() else [])}"
        )
    if declaration.get("scientific_role") != "separate_fixed_configuration_confirmation":
        raise RuntimeError("Stage 14 confirmatory declaration has the wrong scientific role")
    if declaration.get("authoritative_primary_nested_oof_replaced") is not False:
        raise RuntimeError("Confirmatory repeats must not replace the primary nested OOF")
    for key, suffix in (
        ("seed_plan", "14_tuning/confirmatory_seed_plan.json"),
        ("results", "14_tuning/confirmatory_repeated_cv_results.json"),
    ):
        if not _path_value_has_suffix(declaration.get(key), suffix):
            raise RuntimeError(f"Stage 14 confirmatory declaration path differs for {key}")

    manifest_extra = run_manifest.get("extra", {})
    manifest_contract = {
        "confirmatory_seed_plan": "14_tuning/confirmatory_seed_plan.json",
        "confirmatory_repeated_cv_results": (
            "14_tuning/confirmatory_repeated_cv_results.json"
        ),
        "confirmatory_repeated_cv_predictions": (
            "14_tuning/confirmatory_repeated_cv_predictions.npz"
        ),
    }
    for key, suffix in manifest_contract.items():
        if not _path_value_has_suffix(manifest_extra.get(key), suffix):
            raise RuntimeError(f"Stage 14 manifest confirmatory path differs for {key}")
    if manifest_extra.get("confirmatory_scientific_role") != (
        "separate_fixed_configuration_split_and_training_instability"
    ):
        raise RuntimeError("Stage 14 manifest confirmatory scientific role differs")
    if manifest_extra.get("confirmatory_results_used_for_model_selection") is not False:
        raise RuntimeError("Stage 14 manifest allows confirmatory model selection")
    for path, suffix in (
        (CONFIRMATORY_SEED_PLAN, "14_tuning/confirmatory_seed_plan.json"),
        (
            CONFIRMATORY_RESULTS,
            "14_tuning/confirmatory_repeated_cv_results.json",
        ),
        (
            CONFIRMATORY_PREDICTIONS,
            "14_tuning/confirmatory_repeated_cv_predictions.npz",
        ),
        (
            CONFIRMATORY_FOLD_ASSIGNMENTS,
            "14_tuning/confirmatory_repeated_cv_folds.csv",
        ),
        (CONFIRMATORY_CHECKPOINT_DIR, "14_tuning/confirmatory_repeats"),
    ):
        _verify_manifest_output(run_manifest, path, suffix, "Stage 14")

    plan = _load_json_object(CONFIRMATORY_SEED_PLAN, "confirmatory seed plan")
    result = _load_json_object(CONFIRMATORY_RESULTS, "confirmatory repeated-CV result")
    if plan.get("schema_version") != 2:
        raise RuntimeError("Confirmatory seed-plan schema_version must be 2")
    canonical = plan.get("canonical_sha256")
    unhashed_plan = dict(plan)
    unhashed_plan.pop("canonical_sha256", None)
    if canonical != _canonical_sha256(unhashed_plan):
        raise RuntimeError("Confirmatory seed-plan canonical hash is invalid")
    if (
        plan.get("protocol_version") != REQUIRED_STAGE14_PROTOCOL
        or plan.get("status") != "scheduled_for_execution"
        or plan.get("scientific_role") != CONFIRMATORY_SCIENTIFIC_ROLE
        or plan.get("execution_policy")
        != "reuse_primary_nested_cv_selected_fixed_configuration_without_rehpo"
        or plan.get("architecture") != C.RELIABILITY_ARCHITECTURE
        or plan.get("authoritative_primary_artifacts_unchanged") is not True
    ):
        raise RuntimeError("Confirmatory seed plan has an invalid protocol or role")
    if plan.get("inference_requirement") != (
        "report every prespecified repeat and aggregate paired effects; "
        "never select seeds or configurations by confirmatory performance"
    ):
        raise RuntimeError("Confirmatory seed plan permits selective reporting")
    if not isinstance(plan.get("uncertainty_scope"), str) or not plan[
        "uncertainty_scope"
    ].startswith("joint_split_and_training_instability"):
        raise RuntimeError("Confirmatory seed plan has the wrong uncertainty scope")

    production_parameters = plan.get("production_parameters")
    fixed_references = plan.get("fixed_reference_parameters")
    if not isinstance(production_parameters, dict) or not isinstance(
        fixed_references, dict
    ):
        raise RuntimeError("Confirmatory fixed configurations are missing")
    if plan.get("production_parameters_sha256") != _canonical_sha256(
        production_parameters
    ) or plan.get("fixed_reference_parameters_sha256") != _canonical_sha256(
        fixed_references
    ):
        raise RuntimeError("Confirmatory fixed-configuration hash is invalid")
    if set(fixed_references) != set(CONFIRMATORY_MODEL_NAMES[1:]):
        raise RuntimeError("Confirmatory fixed-reference panel differs")
    if plan.get("strongest_prespecified_references") != list(
        CONFIRMATORY_MODEL_NAMES[1:]
    ):
        raise RuntimeError("Confirmatory prespecified-reference declaration differs")
    selected_parameters = (
        payload.get("architectures", {})
        .get(C.RELIABILITY_ARCHITECTURE, {})
        .get("production_params")
    )
    if selected_parameters != production_parameters or payload.get(
        "production_params_by_architecture", {}
    ).get(C.RELIABILITY_ARCHITECTURE) != production_parameters:
        raise RuntimeError(
            "Confirmatory configuration differs from the primary selected model"
        )

    context = plan.get("execution_context")
    if not isinstance(context, dict):
        raise RuntimeError("Confirmatory execution context is missing")
    required_context = {
        "outer_folds",
        "inner_folds",
        "deterministic",
        "feature_names",
        "feature_names_sha256",
        "n_rows",
        "row_order_sha256",
        "input_sha256",
        "embedding_sha256",
        "primary_split_plan_canonical_sha256",
        "primary_split_plan_file_sha256",
        "source_sha256",
    }
    if not required_context <= set(context):
        raise RuntimeError("Confirmatory execution context is incomplete")
    feature_names = context.get("feature_names")
    expected_feature_names = payload.get("feature_names_by_architecture", {}).get(
        C.RELIABILITY_ARCHITECTURE
    )
    if (
        not isinstance(feature_names, list)
        or feature_names != expected_feature_names
        or context.get("feature_names_sha256") != _canonical_sha256(feature_names)
    ):
        raise RuntimeError("Confirmatory reliability feature schema differs")
    expected_context = {
        "outer_folds": int(primary_split_plan["outer_folds"]),
        "inner_folds": int(primary_split_plan["inner_folds"]),
        "deterministic": True,
        "n_rows": len(frame),
        "row_order_sha256": _ordered_text_sha256(row_ids),
        "input_sha256": actual_hashes[INTERNAL_CSV],
        "embedding_sha256": actual_hashes[INTERNAL_EMBEDDINGS],
        "primary_split_plan_canonical_sha256": primary_split_plan[
            "canonical_sha256"
        ],
        "primary_split_plan_file_sha256": file_sha256(INTERNAL_SPLIT_PLAN),
    }
    for key, expected in expected_context.items():
        if context.get(key) != expected:
            raise RuntimeError(f"Confirmatory execution context differs for {key}")
    source_hashes = context.get("source_sha256")
    expected_sources = {
        "stage14": "14_tune_cross_attention.py",
        "common": "common.py",
        "gpu_runtime": "gpu_runtime.py",
        "schema": "schema.py",
        "config": "config.py",
    }
    source_dir = Path(__file__).resolve().parent
    if not isinstance(source_hashes, dict) or any(
        source_hashes.get(key) != file_sha256(source_dir / source_name)
        for key, source_name in expected_sources.items()
    ):
        raise RuntimeError("Confirmatory execution source provenance differs")

    repeats = plan.get("repeats")
    if not isinstance(repeats, list) or not repeats:
        raise RuntimeError("Confirmatory seed plan contains no repeats")
    primary_split_seed = plan.get("primary_split_seed")
    if not isinstance(primary_split_seed, int):
        raise RuntimeError("Confirmatory seed plan has no primary split seed")
    primary_reproducibility = payload.get("reproducibility", {})
    if (
        primary_split_seed != int(primary_split_plan.get("split_seed", -1))
        or primary_split_seed
        != int(primary_reproducibility.get("split_seed", -1))
        or not isinstance(primary_reproducibility.get("training_seed"), int)
    ):
        raise RuntimeError("Confirmatory plan and primary seed provenance differ")
    split_seeds: list[int] = []
    training_seeds: list[int] = []
    expected_checkpoint_names: list[str] = []
    for index, record in enumerate(repeats, 1):
        if not isinstance(record, dict):
            raise RuntimeError("Confirmatory repeat identity is malformed")
        split_seed = record.get("split_seed")
        training_seed = record.get("training_seed")
        namespace = f"repeat_{index:03d}_seed_{split_seed}"
        if (
            record.get("repeat") != index
            or not isinstance(split_seed, int)
            or not isinstance(training_seed, int)
            or record.get("sampler_seed") is not None
            or record.get("rehpo") is not False
            or record.get("output_namespace") != namespace
        ):
            raise RuntimeError("Confirmatory repeat identity or no-re-HPO flag differs")
        split_seeds.append(split_seed)
        training_seeds.append(training_seed)
        expected_checkpoint_names.append(f"{namespace}.json")
    if len(set(split_seeds)) != len(split_seeds) or primary_split_seed in split_seeds:
        raise RuntimeError("Confirmatory split seeds are duplicated or reuse primary CV")
    expected_training_seeds = [
        int(primary_reproducibility["training_seed"]) + index * 100_003
        for index in range(1, len(repeats) + 1)
    ]
    if training_seeds != expected_training_seeds:
        raise RuntimeError("Confirmatory training-seed derivation differs")

    if (
        result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("protocol_version") != REQUIRED_STAGE14_PROTOCOL
        or result.get("scientific_role") != CONFIRMATORY_SCIENTIFIC_ROLE
        or result.get("authoritative_primary_nested_oof_replaced") is not False
        or result.get("hyperparameter_selection_uses_confirmatory_results") is not False
        or result.get("external_validation_touched") is not False
    ):
        raise RuntimeError("Confirmatory results have an invalid status, role or flag")
    if result.get("confirmatory_plan_sha256") != canonical or result.get(
        "confirmatory_plan_file_sha256"
    ) != file_sha256(CONFIRMATORY_SEED_PLAN):
        raise RuntimeError("Confirmatory result and seed-plan hashes differ")
    if result.get("model_order") != list(CONFIRMATORY_MODEL_NAMES) or result.get(
        "repeat_count"
    ) != len(repeats):
        raise RuntimeError("Confirmatory result model or repeat inventory differs")
    if result.get("inference_scope") != plan.get("uncertainty_scope"):
        raise RuntimeError("Confirmatory result uncertainty scope differs")
    warning = result.get("pooled_metrics_warning")
    if not isinstance(warning, str) or "correlated copies" not in warning or (
        "Per-seed paired effects" not in warning
    ):
        raise RuntimeError("Confirmatory pooled-metric non-independence warning is absent")
    result_artifacts = result.get("artifacts")
    if not isinstance(result_artifacts, dict):
        raise RuntimeError("Confirmatory result artifact bindings are absent")
    artifact_contract = {
        "predictions_npz": (
            "14_tuning/confirmatory_repeated_cv_predictions.npz",
            CONFIRMATORY_PREDICTIONS,
            "predictions_npz_sha256",
        ),
        "fold_assignments_csv": (
            "14_tuning/confirmatory_repeated_cv_folds.csv",
            CONFIRMATORY_FOLD_ASSIGNMENTS,
            "fold_assignments_csv_sha256",
        ),
    }
    for key, (suffix, path, hash_key) in artifact_contract.items():
        if not _path_value_has_suffix(result_artifacts.get(key), suffix) or (
            result_artifacts.get(hash_key) != file_sha256(path)
        ):
            raise RuntimeError(f"Confirmatory result artifact binding differs for {key}")
    if not _path_value_has_suffix(
        result_artifacts.get("checkpoint_directory"),
        "14_tuning/confirmatory_repeats",
    ):
        raise RuntimeError("Confirmatory checkpoint-directory binding differs")

    arrays = _load_npz(CONFIRMATORY_PREDICTIONS)
    expected_array_keys = {
        "y",
        "groups",
        "row_ids",
        "repeat_ids",
        "split_seeds",
        "training_seeds",
        "fold_ids",
        *(
            f"{_confirmatory_prefix(model)}__{role}"
            for model in CONFIRMATORY_MODEL_NAMES
            for role in ("probabilities", "decisions", "thresholds")
        ),
    }
    if set(arrays) != expected_array_keys:
        raise RuntimeError("Confirmatory prediction-array schema or model panel differs")
    repeat_count = len(repeats)
    n_rows = len(frame)
    observed_row_ids = np.asarray(arrays["row_ids"]).astype(str)
    observed_groups = np.asarray(arrays["groups"]).astype(str)
    observed_labels = np.asarray(arrays["y"])
    if (
        observed_row_ids.shape != (n_rows,)
        or observed_groups.shape != (n_rows,)
        or observed_labels.shape != (n_rows,)
        or not np.array_equal(observed_row_ids, row_ids)
        or not np.array_equal(observed_groups, groups.astype(str))
        or not np.array_equal(observed_labels.astype(np.int8), labels)
        or not np.isin(observed_labels, (0, 1)).all()
    ):
        raise RuntimeError("Confirmatory predictions use different rows, labels or groups")
    expected_repeat_ids = np.arange(1, repeat_count + 1, dtype=np.int16)
    if (
        np.asarray(arrays["repeat_ids"]).shape != (repeat_count,)
        or not np.array_equal(arrays["repeat_ids"], expected_repeat_ids)
        or not np.array_equal(arrays["split_seeds"], np.asarray(split_seeds))
        or not np.array_equal(arrays["training_seeds"], np.asarray(training_seeds))
    ):
        raise RuntimeError("Confirmatory prediction repeat IDs or seeds differ")
    fold_matrix = np.asarray(arrays["fold_ids"])
    if fold_matrix.shape != (repeat_count, n_rows):
        raise RuntimeError("Confirmatory fold matrix has the wrong shape")

    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    for model in CONFIRMATORY_MODEL_NAMES:
        prefix = _confirmatory_prefix(model)
        model_arrays[model] = {
            role: np.asarray(arrays[f"{prefix}__{role}"])
            for role in ("probabilities", "decisions", "thresholds")
        }
        probabilities = model_arrays[model]["probabilities"].astype(np.float64)
        thresholds = model_arrays[model]["thresholds"].astype(np.float64)
        decisions = model_arrays[model]["decisions"]
        if any(
            values.shape != (repeat_count, n_rows)
            for values in (probabilities, thresholds, decisions)
        ):
            raise RuntimeError(f"Confirmatory {model} prediction shape differs")
        if (
            not np.isfinite(probabilities).all()
            or not np.isfinite(thresholds).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
            or ((thresholds < 0.0) | (thresholds > 1.0)).any()
            or not np.isin(decisions, (0, 1)).all()
            or not np.array_equal(
                decisions.astype(np.int8),
                (probabilities >= thresholds).astype(np.int8),
            )
        ):
            raise RuntimeError(f"Confirmatory {model} predictions are invalid")

    expected_files = set(expected_checkpoint_names)
    observed_entries = {path.name for path in CONFIRMATORY_CHECKPOINT_DIR.iterdir()}
    if observed_entries != expected_files or any(
        not (CONFIRMATORY_CHECKPOINT_DIR / name).is_file()
        for name in expected_files
    ):
        raise RuntimeError("Confirmatory repeat checkpoint inventory differs")
    per_seed = result.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != repeat_count:
        raise RuntimeError("Confirmatory per-seed result inventory differs")

    recomputed_metrics: dict[str, list[dict[str, Any]]] = {
        model: [] for model in CONFIRMATORY_MODEL_NAMES
    }
    checkpoint_hashes: dict[str, str] = {}
    for repeat_index, (repeat_record, aggregate_record, checkpoint_name) in enumerate(
        zip(repeats, per_seed, expected_checkpoint_names)
    ):
        checkpoint_path = CONFIRMATORY_CHECKPOINT_DIR / checkpoint_name
        checkpoint = _load_json_object(checkpoint_path, "confirmatory checkpoint")
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash is None:
            raise RuntimeError("Confirmatory checkpoint could not be hashed")
        checkpoint_hashes[checkpoint_name] = checkpoint_hash
        identity = {
            "repeat": int(repeat_record["repeat"]),
            "split_seed": int(repeat_record["split_seed"]),
            "training_seed": int(repeat_record["training_seed"]),
        }
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("status") != "complete"
            or checkpoint.get("confirmatory_plan_sha256") != canonical
            or checkpoint.get("identity") != identity
            or checkpoint.get("row_order_sha256") != _ordered_text_sha256(row_ids)
            or checkpoint.get("external_validation_touched") is not False
        ):
            raise RuntimeError("Confirmatory checkpoint status or identity differs")
        expected_fold_ids = _validate_confirmatory_repeat_split_plan(
            checkpoint.get("repeat_split_plan"),
            plan,
            repeat_record,
            labels,
            groups,
            row_ids,
        )
        observed_fold_ids = np.asarray(checkpoint.get("fold_ids", []))
        if (
            observed_fold_ids.shape != (n_rows,)
            or not np.array_equal(observed_fold_ids, expected_fold_ids)
            or not np.array_equal(fold_matrix[repeat_index], expected_fold_ids)
        ):
            raise RuntimeError("Confirmatory checkpoint and NPZ fold assignments differ")
        group_fold_counts = pd.DataFrame(
            {"group": groups.astype(str), "fold": expected_fold_ids}
        ).groupby("group", sort=False)["fold"].nunique()
        if group_fold_counts.ne(1).any():
            raise RuntimeError("Confirmatory split groups cross outer folds")

        checkpoint_predictions = checkpoint.get("predictions")
        checkpoint_results = checkpoint.get("model_results")
        comparisons = checkpoint.get("comparisons")
        expected_comparisons = {
            f"{C.RELIABILITY_ARCHITECTURE}_minus_{reference}"
            for reference in CONFIRMATORY_MODEL_NAMES[1:]
        }
        if (
            not isinstance(checkpoint_predictions, dict)
            or set(checkpoint_predictions) != set(CONFIRMATORY_MODEL_NAMES)
            or not isinstance(checkpoint_results, dict)
            or set(checkpoint_results) != set(CONFIRMATORY_MODEL_NAMES)
            or not isinstance(comparisons, dict)
            or set(comparisons) != expected_comparisons
        ):
            raise RuntimeError("Confirmatory checkpoint model/comparison panel differs")
        if not isinstance(aggregate_record, dict) or aggregate_record.get(
            "identity"
        ) != identity:
            raise RuntimeError("Confirmatory aggregate per-seed identity differs")
        if aggregate_record.get("repeat_split_plan_canonical_sha256") != checkpoint[
            "repeat_split_plan"
        ]["canonical_sha256"]:
            raise RuntimeError("Confirmatory aggregate repeat split hash differs")
        if aggregate_record.get("models") != checkpoint_results or aggregate_record.get(
            "paired_comparisons"
        ) != comparisons:
            raise RuntimeError("Confirmatory aggregate and checkpoint contents differ")

        for model in CONFIRMATORY_MODEL_NAMES:
            checkpoint_arrays = checkpoint_predictions[model]
            if not isinstance(checkpoint_arrays, dict) or set(checkpoint_arrays) != {
                "probabilities",
                "decisions",
                "thresholds",
            }:
                raise RuntimeError(f"Confirmatory checkpoint {model} arrays differ")
            for role in ("probabilities", "decisions", "thresholds"):
                observed = np.asarray(checkpoint_arrays[role])
                expected = model_arrays[model][role][repeat_index]
                if not np.array_equal(observed, expected):
                    raise RuntimeError(
                        f"Confirmatory checkpoint and NPZ differ for {model} {role}"
                    )
            probabilities = model_arrays[model]["probabilities"][repeat_index].astype(
                np.float64
            )
            decisions = model_arrays[model]["decisions"][repeat_index].astype(
                np.int8
            )
            thresholds = model_arrays[model]["thresholds"][repeat_index].astype(
                np.float64
            )
            metrics = C.evaluate(
                labels, probabilities, thresholds, predictions=decisions
            )
            recomputed_metrics[model].append(metrics)
            model_result = checkpoint_results[model]
            expected_role = (
                "confirmatory_fixed_primary_selected_configuration"
                if model == C.RELIABILITY_ARCHITECTURE
                else "fixed_configuration_confirmatory_reference"
            )
            if (
                not isinstance(model_result, dict)
                or model_result.get("evaluation_role") != expected_role
                or model_result.get("rehpo") is not False
                or not isinstance(model_result.get("fold_results"), list)
                or len(model_result["fold_results"]) != int(context["outer_folds"])
            ):
                raise RuntimeError(
                    f"Confirmatory {model} scientific role or fold inventory differs"
                )
            _assert_reported_value(
                model_result.get("metrics"),
                metrics,
                f"confirmatory.repeat_{repeat_index + 1}.{model}",
            )

        for reference in CONFIRMATORY_MODEL_NAMES[1:]:
            comparison = comparisons[
                f"{C.RELIABILITY_ARCHITECTURE}_minus_{reference}"
            ]
            if not isinstance(comparison, dict) or set(comparison) != {
                "mcc",
                "auroc",
                "auprc",
            }:
                raise RuntimeError("Confirmatory paired comparison schema differs")
            proposed = model_arrays[C.RELIABILITY_ARCHITECTURE]
            comparator = model_arrays[reference]
            direct_differences = {
                "mcc": float(
                    matthews_corrcoef(
                        labels, proposed["decisions"][repeat_index]
                    )
                )
                - float(
                    matthews_corrcoef(
                        labels, comparator["decisions"][repeat_index]
                    )
                ),
                "auroc": float(
                    roc_auc_score(labels, proposed["probabilities"][repeat_index])
                )
                - float(
                    roc_auc_score(labels, comparator["probabilities"][repeat_index])
                ),
                "auprc": float(
                    average_precision_score(
                        labels, proposed["probabilities"][repeat_index]
                    )
                )
                - float(
                    average_precision_score(
                        labels, comparator["probabilities"][repeat_index]
                    )
                ),
            }
            for metric, expected_difference in direct_differences.items():
                record = comparison.get(metric)
                if (
                    not isinstance(record, dict)
                    or record.get("inference_method")
                    != "paired_split_group_randomization"
                    or record.get("exchangeability_unit") != "split_group"
                    or int(record.get("randomization_iterations", 0)) < 1000
                ):
                    raise RuntimeError(
                        "Confirmatory paired comparison inference contract differs"
                    )
                _assert_reported_value(
                    record.get("mean_difference_second_minus_first"),
                    round(expected_difference, 4),
                    (
                        f"confirmatory.repeat_{repeat_index + 1}."
                        f"{reference}.{metric}.observed_difference"
                    ),
                )

    fold_table = pd.read_csv(CONFIRMATORY_FOLD_ASSIGNMENTS)
    expected_columns = [
        "repeat",
        "split_seed",
        "training_seed",
        "row_index",
        C.ROW_ID_COL,
        C.GENE_COL,
        "split_group",
        C.LABEL_COL,
        "outer_fold",
    ]
    if list(fold_table.columns) != expected_columns or len(fold_table) != (
        repeat_count * n_rows
    ):
        raise RuntimeError("Confirmatory long fold-table schema or row count differs")
    expected_genes = frame[C.GENE_COL].astype(str).to_numpy()
    for repeat_index, repeat_record in enumerate(repeats):
        selected = fold_table.iloc[repeat_index * n_rows : (repeat_index + 1) * n_rows]
        if (
            not np.array_equal(selected["row_index"].to_numpy(), np.arange(n_rows))
            or not np.array_equal(selected[C.ROW_ID_COL].astype(str).to_numpy(), row_ids)
            or not np.array_equal(selected[C.GENE_COL].astype(str).to_numpy(), expected_genes)
            or not np.array_equal(selected["split_group"].astype(str).to_numpy(), groups.astype(str))
            or not np.array_equal(selected[C.LABEL_COL].to_numpy(dtype=np.int8), labels)
            or not np.array_equal(selected["outer_fold"].to_numpy(), fold_matrix[repeat_index])
            or selected["repeat"].nunique() != 1
            or int(selected["repeat"].iloc[0]) != int(repeat_record["repeat"])
            or selected["split_seed"].nunique() != 1
            or int(selected["split_seed"].iloc[0]) != int(repeat_record["split_seed"])
            or selected["training_seed"].nunique() != 1
            or int(selected["training_seed"].iloc[0])
            != int(repeat_record["training_seed"])
        ):
            raise RuntimeError("Confirmatory long fold table differs from NPZ or seed plan")

    across_seed = {
        model: _confirmatory_scalar_metric_summary(recomputed_metrics[model])
        for model in CONFIRMATORY_MODEL_NAMES
    }
    _assert_reported_value(
        result.get("across_seed_metric_summary"),
        across_seed,
        "confirmatory.across_seed_metric_summary",
    )
    paired_summary: dict[str, Any] = {}
    for reference in CONFIRMATORY_MODEL_NAMES[1:]:
        differences = [
            {
                metric: (
                    None
                    if proposed.get(metric) is None or comparator.get(metric) is None
                    else float(proposed[metric]) - float(comparator[metric])
                )
                for metric in ("mcc", "auroc", "auprc", "brier")
            }
            for proposed, comparator in zip(
                recomputed_metrics[C.RELIABILITY_ARCHITECTURE],
                recomputed_metrics[reference],
            )
        ]
        paired_summary[
            f"{C.RELIABILITY_ARCHITECTURE}_minus_{reference}"
        ] = _confirmatory_scalar_metric_summary(differences)
    _assert_reported_value(
        result.get("paired_metric_difference_summary"),
        paired_summary,
        "confirmatory.paired_metric_difference_summary",
    )

    pooled: dict[str, Any] = {}
    instability: dict[str, Any] = {}
    repeated_labels = np.tile(labels, repeat_count)
    for model in CONFIRMATORY_MODEL_NAMES:
        values = model_arrays[model]
        probabilities = values["probabilities"].astype(np.float64)
        decisions = values["decisions"].astype(np.int8)
        thresholds = values["thresholds"].astype(np.float64)
        pooled[model] = {
            "stacked_repeated_oof_descriptive": C.evaluate(
                repeated_labels,
                probabilities.reshape(-1),
                thresholds.reshape(-1),
                predictions=decisions.reshape(-1),
            ),
            "mean_probability_repeated_oof_descriptive": C.evaluate(
                labels,
                probabilities.mean(axis=0),
                thresholds.mean(axis=0),
                predictions=(decisions.mean(axis=0) >= 0.5).astype(np.int8),
            ),
        }
        row_sd = probabilities.std(axis=0, ddof=0)
        instability[model] = {
            "per_row_probability_standard_deviation": {
                "mean": round(float(row_sd.mean()), 6),
                "median": round(float(np.median(row_sd)), 6),
                "p90": round(float(np.quantile(row_sd, 0.90)), 6),
                "maximum": round(float(row_sd.max()), 6),
            },
            "metric_variation": across_seed[model],
        }
    _assert_reported_value(
        result.get("pooled_descriptive_metrics"),
        pooled,
        "confirmatory.pooled_descriptive_metrics",
    )
    _assert_reported_value(
        result.get("split_and_training_instability"),
        instability,
        "confirmatory.split_and_training_instability",
    )

    compact_metrics = {
        model: {
            metric: across_seed[model][metric]
            for metric in ("mcc", "auroc", "auprc", "brier")
        }
        for model in CONFIRMATORY_MODEL_NAMES
    }
    compact_paired = {
        comparison: {
            metric: summary[metric]
            for metric in ("mcc", "auroc", "auprc", "brier")
        }
        for comparison, summary in paired_summary.items()
    }
    return {
        "requested": True,
        "status": "validated",
        "scientific_role": CONFIRMATORY_SCIENTIFIC_ROLE,
        "results_used_for_model_selection": False,
        "authoritative_primary_nested_oof_replaced": False,
        "repeat_count": repeat_count,
        "split_seeds": split_seeds,
        "training_seeds": training_seeds,
        "model_order": list(CONFIRMATORY_MODEL_NAMES),
        "across_seed_metric_summary": compact_metrics,
        "paired_metric_difference_summary": compact_paired,
        "seed_plan_sha256": file_sha256(CONFIRMATORY_SEED_PLAN),
        "results_sha256": file_sha256(CONFIRMATORY_RESULTS),
        "predictions_sha256": file_sha256(CONFIRMATORY_PREDICTIONS),
        "fold_assignments_sha256": file_sha256(CONFIRMATORY_FOLD_ASSIGNMENTS),
        "checkpoint_sha256": checkpoint_hashes,
    }


def _validate_stage14_artifacts() -> dict[str, Any]:
    """Fail closed when Stage 14 JSON, OOF, split plan or provenance diverge."""
    required_files = (
        INTERNAL_RESULTS,
        INTERNAL_PREDICTIONS,
        INTERNAL_SPLIT_PLAN,
        INTERNAL_RUN_MANIFEST,
        INTERNAL_CSV,
        INTERNAL_EMBEDDINGS,
        INTERNAL_STATUS,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 13 inputs are missing: {missing}")

    payload = _load_json_object(INTERNAL_RESULTS, "Stage 14 result")
    split_plan = _load_json_object(INTERNAL_SPLIT_PLAN, "Stage 14 split plan")
    run_manifest = _load_json_object(
        INTERNAL_RUN_MANIFEST, "Stage 14 run manifest"
    )
    if payload.get("schema_version") != 2:
        raise RuntimeError("Stage 14 result schema_version must be 2")
    if payload.get("model_tag") != MODEL_TAG:
        raise RuntimeError("Stage 14 result model tag differs from Stage 13")
    if payload.get("protocol_version") != REQUIRED_STAGE14_PROTOCOL:
        raise RuntimeError("Stage 14 result is not the publication protocol")
    if payload.get("selection_protocol") != (
        "fixed_nested_split_group_disjoint_cross_validation"
    ):
        raise RuntimeError("Stage 14 selection protocol is not fixed nested group CV")
    if payload.get("external_validation_touched") is not False:
        raise RuntimeError("Stage 14 must not touch external validation data")
    if payload.get("reproducibility", {}).get("deterministic") is not True:
        raise RuntimeError("Stage 14 publication artifacts must be deterministic")

    architectures = payload.get("architectures")
    if not isinstance(architectures, dict):
        raise RuntimeError("Stage 14 result has no architecture payloads")
    architecture_names = set(architectures)
    if (
        not REQUIRED_STAGE14_ARCHITECTURES.issubset(architecture_names)
        or architecture_names
        - REQUIRED_STAGE14_ARCHITECTURES
        - OPTIONAL_STAGE14_ARCHITECTURES
    ):
        raise RuntimeError(
            "Stage 14 publication result must contain concatenation, gated_fusion, "
            "cross_attention and reliability_residual, with only declared optional "
            "architectures"
        )
    if set(payload.get("architectures_evaluated", [])) != architecture_names:
        raise RuntimeError("Stage 14 evaluated-architecture declaration is inconsistent")
    if set(payload.get("production_params_by_architecture", {})) != architecture_names:
        raise RuntimeError("Stage 14 production parameters are architecture-incomplete")

    for name, details in architectures.items():
        if details.get("schema_version") != 2:
            raise RuntimeError(f"Stage 14 {name} schema_version must be 2")
        if details.get("architecture") != name:
            raise RuntimeError(f"Stage 14 {name} architecture identity differs")
        if details.get("model_tag") != MODEL_TAG:
            raise RuntimeError(f"Stage 14 {name} model tag differs")
        if details.get("protocol_version") != REQUIRED_STAGE14_PROTOCOL:
            raise RuntimeError(f"Stage 14 {name} protocol differs")
        if details.get("external_validation_touched") is not False:
            raise RuntimeError(f"Stage 14 {name} touched external validation")
        search = details.get("search", {})
        minimums = {
            "trials_target_per_outer_fold": 40,
            "outer_folds": 5,
            "inner_folds": 3,
            "search_epochs": 25,
            "final_ensemble": 3,
            "final_epochs": 60,
        }
        for field, minimum in minimums.items():
            if int(search.get(field, 0)) < minimum:
                raise RuntimeError(
                    f"Stage 14 {name} {field}={search.get(field)!r} is below "
                    f"the publication profile minimum {minimum}"
                )

    references = payload.get("reference_baselines")
    if not isinstance(references, dict) or not references:
        raise RuntimeError("Stage 14 result has no nested reference baselines")
    missing_references = REQUIRED_STAGE14_REFERENCES - set(references)
    if missing_references:
        raise RuntimeError(
            "Stage 14 publication result misses mandatory nested baselines: "
            f"{sorted(missing_references)}"
        )
    primary_reference = payload.get("primary_reference_baseline")
    if primary_reference not in references:
        raise RuntimeError("Stage 14 primary reference baseline is unavailable")

    if run_manifest.get("stage") != "14_tune_cross_attention":
        raise RuntimeError("Stage 14 run manifest stage identity differs")
    if run_manifest.get("label_task") != "clinical":
        raise RuntimeError("Stage 14 run manifest is not a clinical-label run")
    if run_manifest.get("model_tag") != MODEL_TAG:
        raise RuntimeError("Stage 14 run manifest model tag differs")
    manifest_extra = run_manifest.get("extra", {})
    if manifest_extra.get("protocol_version") != REQUIRED_STAGE14_PROTOCOL:
        raise RuntimeError("Stage 14 run manifest protocol differs")
    if manifest_extra.get("external_validation_touched") is not False:
        raise RuntimeError("Stage 14 run manifest indicates external data access")
    if set(manifest_extra.get("architectures", [])) != architecture_names:
        raise RuntimeError("Stage 14 manifest architecture inventory differs")
    if set(manifest_extra.get("reference_baselines", [])) != set(references):
        raise RuntimeError("Stage 14 manifest reference-baseline inventory differs")
    for path, suffix in (
        (INTERNAL_RESULTS, "14_tuning/architecture_selection.json"),
        (INTERNAL_PREDICTIONS, "14_tuning/nested_tuning_oof.npz"),
        (INTERNAL_SPLIT_PLAN, "14_tuning/nested_inner_splits.json"),
    ):
        _verify_manifest_output(run_manifest, path, suffix, "Stage 14")

    actual_hashes: dict[Path, str] = {}
    for path in (INTERNAL_CSV, INTERNAL_EMBEDDINGS, INTERNAL_STATUS):
        digest = file_sha256(path)
        if digest is None:
            raise RuntimeError(
                f"A full SHA256 is required for Stage 14 provenance: {path}"
            )
        actual_hashes[path] = digest
        record = _manifest_input_record(run_manifest, path)
        if record.get("exists") is not True or record.get("sha256") != digest:
            raise RuntimeError(
                f"Stage 14 manifest input hash differs for {path.name}"
            )
    reproducibility = payload.get("reproducibility", {})
    if reproducibility.get("input_sha256") != actual_hashes[INTERNAL_CSV]:
        raise RuntimeError("Stage 14 JSON input table hash differs")
    if reproducibility.get("embedding_sha256") != actual_hashes[INTERNAL_EMBEDDINGS]:
        raise RuntimeError("Stage 14 JSON embedding hash differs")

    _validate_source_provenance(
        run_manifest, STAGE14_PROVENANCE_SOURCES, "Stage 14"
    )

    if split_plan.get("protocol_version") != REQUIRED_STAGE14_PROTOCOL:
        raise RuntimeError("Stage 14 split-plan protocol differs")
    recorded_canonical = split_plan.get("canonical_sha256")
    unhashed_plan = dict(split_plan)
    unhashed_plan.pop("canonical_sha256", None)
    if recorded_canonical != _canonical_sha256(unhashed_plan):
        raise RuntimeError("Stage 14 split-plan canonical hash is invalid")
    split_file_hash = file_sha256(INTERNAL_SPLIT_PLAN)
    if split_file_hash is None:
        raise RuntimeError("Stage 14 split-plan file could not be hashed")
    if reproducibility.get("split_plan_canonical_sha256") != recorded_canonical:
        raise RuntimeError("Stage 14 JSON and split-plan canonical hashes differ")
    if reproducibility.get("split_plan_file_sha256") != split_file_hash:
        raise RuntimeError("Stage 14 JSON and split-plan file hashes differ")
    if manifest_extra.get("split_plan_canonical_sha256") != recorded_canonical:
        raise RuntimeError("Stage 14 manifest and split-plan canonical hashes differ")
    if manifest_extra.get("split_plan_file_sha256") != split_file_hash:
        raise RuntimeError("Stage 14 manifest and split-plan file hashes differ")
    if split_plan.get("input_sha256") != actual_hashes[INTERNAL_CSV]:
        raise RuntimeError("Stage 14 split-plan input table hash differs")
    if split_plan.get("embedding_sha256") != actual_hashes[INTERNAL_EMBEDDINGS]:
        raise RuntimeError("Stage 14 split-plan embedding hash differs")

    frame = pd.read_parquet(INTERNAL_CSV).reset_index(drop=True)
    status = pd.read_parquet(INTERNAL_STATUS)
    embeddings = np.load(INTERNAL_EMBEDDINGS, mmap_mode="r")
    data = _load_npz(INTERNAL_PREDICTIONS)
    base_keys = {"y", "groups", "row_ids", "fold_ids"}
    if not base_keys.issubset(data):
        raise RuntimeError(
            f"Stage 14 OOF misses base arrays: {sorted(base_keys - set(data))}"
        )
    n_rows = len(frame)
    if len(status) != n_rows or len(embeddings) != n_rows:
        raise RuntimeError("Stage 14 current table, status and embeddings are misaligned")
    if int(split_plan.get("n_rows", -1)) != n_rows:
        raise RuntimeError("Stage 14 split-plan row count differs")

    row_ids = np.asarray(data["row_ids"]).astype(str)
    labels = np.asarray(data["y"])
    groups = np.asarray(data["groups"]).astype(str)
    fold_values = np.asarray(data["fold_ids"])
    for name, values in (
        ("row_ids", row_ids),
        ("y", labels),
        ("groups", groups),
        ("fold_ids", fold_values),
    ):
        if values.ndim != 1 or len(values) != n_rows:
            raise RuntimeError(
                f"Stage 14 OOF {name} has shape {values.shape}, expected ({n_rows},)"
            )
    if len(set(row_ids)) != n_rows or np.any(np.char.strip(row_ids) == ""):
        raise RuntimeError("Stage 14 OOF row identifiers are blank or duplicated")
    frame_row_ids = frame[C.ROW_ID_COL].astype(str).to_numpy()
    status_row_ids = status[C.ROW_ID_COL].astype(str).to_numpy()
    if not np.array_equal(row_ids, frame_row_ids):
        raise RuntimeError("Stage 14 OOF row order differs from the current table")
    if not np.array_equal(row_ids, status_row_ids):
        raise RuntimeError("Stage 14 OOF row order differs from the ESM status")
    if split_plan.get("row_order_sha256") != _ordered_text_sha256(row_ids):
        raise RuntimeError("Stage 14 OOF row order differs from the split plan")

    numeric_labels = pd.to_numeric(frame[LABEL_COL], errors="coerce").to_numpy()
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, [0, 1]).all():
        raise RuntimeError("Current internal labels are invalid")
    numeric_labels = numeric_labels.astype(np.int8)
    if not np.array_equal(labels.astype(np.int8), numeric_labels) or not np.isin(
        labels, [0, 1]
    ).all():
        raise RuntimeError("Stage 14 OOF labels differ from the current table")
    split_column = split_plan.get("split_group_column")
    if split_column not in frame:
        raise RuntimeError(f"Stage 14 split group column is absent: {split_column!r}")
    expected_groups = frame[split_column].astype(str).to_numpy()
    if not np.array_equal(groups, expected_groups):
        raise RuntimeError("Stage 14 OOF groups differ from the current table")

    try:
        fold_ids = fold_values.astype(np.int16)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Stage 14 OOF fold IDs are not integers") from error
    if not np.array_equal(fold_values, fold_ids):
        raise RuntimeError("Stage 14 OOF fold IDs are not exact integers")
    outer_folds = int(split_plan.get("outer_folds", 0))
    if outer_folds < 5 or set(np.unique(fold_ids)) != set(range(1, outer_folds + 1)):
        raise RuntimeError("Stage 14 OOF fold IDs are incomplete for publication")
    expected_folds = np.full(n_rows, -1, dtype=np.int16)
    fold_records = split_plan.get("folds", [])
    if len(fold_records) != outer_folds:
        raise RuntimeError("Stage 14 split plan has the wrong outer-fold count")
    for record in fold_records:
        fold = int(record.get("outer_fold", 0))
        validation = record.get("outer_validation", {})
        indices = np.asarray(validation.get("indices", []), dtype=np.int64)
        if (
            fold not in range(1, outer_folds + 1)
            or indices.ndim != 1
            or not len(indices)
            or (indices < 0).any()
            or (indices >= n_rows).any()
            or (expected_folds[indices] != -1).any()
        ):
            raise RuntimeError(f"Stage 14 split-plan outer fold {fold} is invalid")
        expected_folds[indices] = fold
        if validation.get("n") != int(len(indices)):
            raise RuntimeError(f"Stage 14 split-plan outer fold {fold} n differs")
        if validation.get("row_ids_sha256") != _ordered_text_sha256(row_ids[indices]):
            raise RuntimeError(
                f"Stage 14 split-plan outer fold {fold} row hash differs"
            )
    if not np.array_equal(fold_ids, expected_folds):
        raise RuntimeError("Stage 14 OOF fold IDs differ from the split plan")
    fold_counts_by_group = (
        pd.DataFrame({"group": groups, "fold": fold_ids})
        .groupby("group", sort=False)["fold"]
        .nunique()
    )
    if fold_counts_by_group.gt(1).any():
        raise RuntimeError("Stage 14 OOF split groups cross outer folds")

    expected_prefixes: set[str] = set()
    for name, details in architectures.items():
        expected_prefixes.add(name)
        reported = details.get("nested_outer_metrics")
        if not isinstance(reported, dict):
            raise RuntimeError(f"Stage 14 {name} has no nested outer metrics")
        _validate_stage14_model_output(data, name, reported, numeric_labels, n_rows)
    for name, reported in references.items():
        prefix = f"reference__{name}"
        expected_prefixes.add(prefix)
        if not isinstance(reported, dict):
            raise RuntimeError(f"Stage 14 reference {name} metrics are invalid")
        _validate_stage14_model_output(
            data, prefix, reported, numeric_labels, n_rows
        )
    observed_prefixes = {
        key[: -len("__probabilities")]
        for key in data
        if key.endswith("__probabilities")
    }
    if observed_prefixes != expected_prefixes:
        raise RuntimeError(
            "Stage 14 JSON and OOF model inventories differ: "
            f"JSON={sorted(expected_prefixes)}, OOF={sorted(observed_prefixes)}"
        )

    confirmatory_validation = _validate_stage14_confirmatory_artifacts(
        payload,
        run_manifest,
        frame,
        numeric_labels,
        groups,
        row_ids,
        split_plan,
        actual_hashes,
    )

    return {
        "status": "validated",
        "protocol_version": REQUIRED_STAGE14_PROTOCOL,
        "n_rows": n_rows,
        "outer_folds": outer_folds,
        "architectures": sorted(architecture_names),
        "reference_baselines": sorted(references),
        "input_sha256": actual_hashes[INTERNAL_CSV],
        "embedding_sha256": actual_hashes[INTERNAL_EMBEDDINGS],
        "split_plan_file_sha256": split_file_hash,
        "architecture_results_sha256": file_sha256(INTERNAL_RESULTS),
        "nested_oof_sha256": file_sha256(INTERNAL_PREDICTIONS),
        "confirmatory_repeated_cv": confirmatory_validation,
    }


def _one_dimensional_external_array(
    data: dict[str, np.ndarray], key: str, n_rows: int
) -> np.ndarray:
    if key not in data:
        raise RuntimeError(f"Stage 12 prediction artifact misses {key}")
    values = np.asarray(data[key])
    if values.ndim != 1 or len(values) != n_rows:
        raise RuntimeError(
            f"Stage 12 {key} has shape {values.shape}, expected ({n_rows},)"
        )
    return values


def _assert_metric_value(reported: Any, expected: Any, path: str) -> None:
    _assert_reported_value(reported, expected, path)


def _validate_interval(
    interval: Any, estimate: float | None, path: str
) -> None:
    if estimate is None:
        return
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(np.isfinite(value) for value in interval)
        or float(interval[0]) > float(interval[1])
        or float(estimate) < float(interval[0]) - 5e-7
        or float(estimate) > float(interval[1]) + 5e-7
    ):
        raise RuntimeError(f"Stage 12 interval is invalid for {path}: {interval!r}")


def _external_model_names(
    data: dict[str, np.ndarray], set_name: str
) -> set[str]:
    prefix = f"{set_name}__"
    reserved = {
        "y",
        "groups",
        "row_ids",
        "variant_ids",
        "annotation_row_counts",
    }
    return {
        remainder
        for key in data
        if key.startswith(prefix)
        and "__" not in (remainder := key[len(prefix) :])
        and remainder not in reserved
    }


def _validate_prediction_table_set(
    table: pd.DataFrame,
    set_name: str,
    row_ids: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    source: str,
) -> pd.DataFrame:
    required = {"evaluation_set", C.ROW_ID_COL, C.LABEL_COL}
    if not required.issubset(table):
        raise RuntimeError(
            f"Stage 12 prediction CSV misses columns: {sorted(required - set(table))}"
        )
    selected = table.loc[table["evaluation_set"].astype(str).eq(set_name)].reset_index(
        drop=True
    )
    if len(selected) != len(row_ids):
        raise RuntimeError(f"Stage 12 prediction CSV row count differs for {set_name}")
    if not np.array_equal(selected[C.ROW_ID_COL].astype(str).to_numpy(), row_ids):
        raise RuntimeError(f"Stage 12 prediction CSV row order differs for {set_name}")
    csv_labels = pd.to_numeric(selected[C.LABEL_COL], errors="coerce").to_numpy()
    if not np.array_equal(csv_labels, labels):
        raise RuntimeError(f"Stage 12 prediction CSV labels differ for {set_name}")
    group_column = C.GENE_COL if source == "clinvar" else "ASSAY_ID"
    if group_column not in selected or not np.array_equal(
        selected[group_column].astype(str).to_numpy(), groups
    ):
        raise RuntimeError(f"Stage 12 prediction CSV groups differ for {set_name}")
    return selected


def _validate_dms_assay_results(
    result: dict[str, Any],
    labels: np.ndarray,
    groups: np.ndarray,
    scores: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    decision_confidences: dict[str, np.ndarray],
    thresholds: dict[str, float],
    dms_scores: np.ndarray,
    set_name: str,
) -> None:
    per_assay = result.get("per_assay")
    if not isinstance(per_assay, dict) or per_assay.get("available") is not True:
        raise RuntimeError(f"Stage 12 {set_name} has no assay-level result")
    if per_assay.get("analysis_unit") != "assay":
        raise RuntimeError(f"Stage 12 {set_name} assay analysis unit differs")
    assays = per_assay.get("assays")
    expected_assays = sorted(np.unique(groups).tolist())
    if not isinstance(assays, dict) or set(assays) != set(expected_assays):
        raise RuntimeError(f"Stage 12 {set_name} assay inventory differs")
    if int(per_assay.get("assay_count", -1)) != len(expected_assays):
        raise RuntimeError(f"Stage 12 {set_name} assay count differs")

    recomputed_by_assay: dict[str, dict[str, dict[str, Any]]] = {}
    for assay in expected_assays:
        index = np.flatnonzero(groups == assay)
        record = assays[assay]
        if record.get("n") != int(len(index)) or record.get("positives") != int(
            labels[index].sum()
        ):
            raise RuntimeError(f"Stage 12 {set_name}/{assay} counts differ")
        reported_models = record.get("models", {})
        if set(reported_models) != set(scores):
            raise RuntimeError(f"Stage 12 {set_name}/{assay} model inventory differs")
        recomputed_by_assay[assay] = {}
        for name, values in scores.items():
            finite_dms = np.isfinite(dms_scores[index])
            if name in decisions:
                recomputed = C.evaluate(
                    labels[index],
                    values[index],
                    thresholds[name],
                    predictions=decisions[name][index],
                    decision_confidence=decision_confidences[name][index],
                )
                correlation_values = 1.0 - values[index][finite_dms]
            else:
                recomputed = {
                    "auroc": round(float(roc_auc_score(labels[index], values[index])), 4)
                    if len(np.unique(labels[index])) == 2
                    else None,
                    "auprc": round(
                        float(average_precision_score(labels[index], values[index])), 4
                    )
                    if len(np.unique(labels[index])) == 2
                    else None,
                }
                correlation_values = -values[index][finite_dms]
            if finite_dms.sum() >= 3:
                correlation = spearmanr(
                    dms_scores[index][finite_dms], correlation_values
                ).statistic
                recomputed["functional_spearman"] = (
                    None
                    if not np.isfinite(correlation)
                    else round(float(correlation), 4)
                )
            else:
                recomputed["functional_spearman"] = None
            _assert_reported_value(
                reported_models[name], recomputed, f"{set_name}.{assay}.{name}"
            )
            recomputed_by_assay[assay][name] = recomputed

    macro = per_assay.get("macro", {})
    intervals = per_assay.get("macro_ci95", {})
    for name in scores:
        if name not in macro:
            raise RuntimeError(f"Stage 12 {set_name} macro result misses {name}")
        for metric in ("mcc", "auroc", "auprc", "f1", "functional_spearman"):
            values = [
                recomputed_by_assay[assay][name].get(metric)
                for assay in expected_assays
                if recomputed_by_assay[assay][name].get(metric) is not None
            ]
            expected = round(float(np.mean(values)), 4) if values else None
            _assert_metric_value(
                macro[name].get(metric), expected, f"{set_name}.macro.{name}.{metric}"
            )
        for metric in ("functional_spearman", "auroc"):
            estimate = macro[name].get(metric)
            if estimate is not None:
                _validate_interval(
                    intervals.get(name, {}).get(metric),
                    estimate,
                    f"{set_name}.macro_ci95.{name}.{metric}",
                )


def _validate_external_primary_set(
    source: str,
    result: dict[str, Any],
    data: dict[str, np.ndarray],
    prediction_table: pd.DataFrame,
    dms_source: pd.DataFrame | None,
) -> dict[str, Any]:
    set_name = f"{source}_exact_variant_disjoint"
    if result.get("status") != "evaluated":
        raise RuntimeError(f"Stage 12 primary set was not evaluated: {set_name}")
    expected_unit = "unique_genomic_variant" if source == "clinvar" else "assay_variant_row"
    if (
        result.get("source") != source
        or result.get("policy") != "exact_variant_disjoint"
        or result.get("role") != "primary"
        or result.get("analysis_unit") != expected_unit
    ):
        raise RuntimeError(f"Stage 12 primary-set contract differs for {set_name}")
    n_rows = int(result.get("n", -1))
    if n_rows <= 0:
        raise RuntimeError(f"Stage 12 primary set is empty: {set_name}")
    labels = _one_dimensional_external_array(data, f"{set_name}__y", n_rows)
    groups = _one_dimensional_external_array(data, f"{set_name}__groups", n_rows).astype(
        str
    )
    row_ids = _one_dimensional_external_array(
        data, f"{set_name}__row_ids", n_rows
    ).astype(str)
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError(f"Stage 12 labels are not binary for {set_name}")
    labels = labels.astype(np.int8)
    if (
        len(set(row_ids)) != n_rows
        or np.any(np.char.strip(row_ids) == "")
        or np.any(np.char.strip(groups) == "")
    ):
        raise RuntimeError(f"Stage 12 identifiers are blank or duplicated for {set_name}")
    if result.get("positives") != int(labels.sum()):
        raise RuntimeError(f"Stage 12 positive count differs for {set_name}")
    group_count_key = "genes" if source == "clinvar" else "assays"
    if result.get(group_count_key) != int(len(np.unique(groups))):
        raise RuntimeError(f"Stage 12 {group_count_key} count differs for {set_name}")

    csv_rows = _validate_prediction_table_set(
        prediction_table, set_name, row_ids, labels, groups, source
    )
    model_results = result.get("models")
    if not isinstance(model_results, dict) or not model_results:
        raise RuntimeError(f"Stage 12 {set_name} has no model results")
    model_names = _external_model_names(data, set_name)
    if model_names != set(model_results):
        raise RuntimeError(
            f"Stage 12 JSON/NPZ model inventories differ for {set_name}: "
            f"JSON={sorted(model_results)}, NPZ={sorted(model_names)}"
        )
    scores: dict[str, np.ndarray] = {}
    decisions: dict[str, np.ndarray] = {}
    decision_confidences: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name in sorted(model_names):
        values = _one_dimensional_external_array(data, f"{set_name}__{name}", n_rows).astype(
            float
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Stage 12 {set_name}/{name} has non-finite scores")
        scores[name] = values
        decision_key = f"{set_name}__{name}__decisions"
        if decision_key in data:
            model_decisions = _one_dimensional_external_array(
                data, decision_key, n_rows
            )
            if not np.isin(model_decisions, [0, 1]).all():
                raise RuntimeError(f"Stage 12 {set_name}/{name} decisions are not binary")
            threshold_key = f"{set_name}__{name}__threshold"
            threshold_array = np.asarray(data.get(threshold_key, []), dtype=float)
            if threshold_array.shape != (1,) or not np.isfinite(threshold_array[0]):
                raise RuntimeError(f"Stage 12 {set_name}/{name} threshold is invalid")
            threshold = float(threshold_array[0])
            if not 0.0 <= threshold <= 1.0 or ((values < 0.0) | (values > 1.0)).any():
                raise RuntimeError(f"Stage 12 {set_name}/{name} probability scale is invalid")
            model_decisions = model_decisions.astype(np.int8)
            decisions[name] = model_decisions
            confidence_key = f"{set_name}__{name}__decision_confidence"
            if confidence_key not in data:
                raise RuntimeError(
                    f"Stage 12 {set_name}/{name} misses fold-vote decision confidence"
                )
            confidence = _one_dimensional_external_array(
                data, confidence_key, n_rows
            ).astype(float)
            if (
                not np.isfinite(confidence).all()
                or (confidence < 0.0).any()
                or (confidence > 1.0).any()
            ):
                raise RuntimeError(
                    f"Stage 12 {set_name}/{name} decision confidence is invalid"
                )
            decision_confidences[name] = confidence
            thresholds[name] = threshold
            recomputed = C.evaluate(
                labels,
                values,
                threshold,
                predictions=model_decisions,
                decision_confidence=confidence,
            )
            _assert_reported_value(
                model_results[name], recomputed, f"{set_name}.{name}"
            )
            if model_results[name].get("output_scale") != "calibrated_probability":
                raise RuntimeError(f"Stage 12 {set_name}/{name} output scale differs")
            probability_column = f"{name}_probability"
            decision_column = f"{name}_decision"
            if probability_column not in csv_rows or decision_column not in csv_rows:
                raise RuntimeError(f"Stage 12 prediction CSV misses {name} outputs")
            if not np.allclose(
                pd.to_numeric(csv_rows[probability_column], errors="coerce"),
                values,
                rtol=0.0,
                atol=5e-12,
                equal_nan=False,
            ) or not np.array_equal(
                pd.to_numeric(csv_rows[decision_column], errors="coerce").to_numpy(),
                model_decisions,
            ):
                raise RuntimeError(f"Stage 12 prediction CSV differs for {set_name}/{name}")
        else:
            expected_rank = {
                "auroc": round(float(roc_auc_score(labels, values)), 4),
                "auprc": round(float(average_precision_score(labels, values)), 4),
            }
            _assert_reported_value(
                model_results[name], expected_rank, f"{set_name}.{name}"
            )
            if model_results[name].get("output_scale") != "raw_ranking_score":
                raise RuntimeError(f"Stage 12 {set_name}/{name} rank scale differs")
            rank_column = f"{name}_raw_deleteriousness_score"
            if rank_column not in csv_rows or not np.allclose(
                pd.to_numeric(csv_rows[rank_column], errors="coerce"),
                values,
                rtol=0.0,
                atol=5e-12,
                equal_nan=False,
            ):
                raise RuntimeError(f"Stage 12 rank CSV differs for {set_name}/{name}")

    if source == "clinvar":
        variant_ids = _one_dimensional_external_array(
            data, f"{set_name}__variant_ids", n_rows
        ).astype(str)
        counts = _one_dimensional_external_array(
            data, f"{set_name}__annotation_row_counts", n_rows
        )
        if len(set(variant_ids)) != n_rows or np.any(np.char.strip(variant_ids) == ""):
            raise RuntimeError("Stage 12 ClinVar primary variants are duplicated or blank")
        if not np.isfinite(counts).all() or (counts < 1).any():
            raise RuntimeError("Stage 12 ClinVar annotation counts are invalid")
        audit = result.get("aggregation_audit", {})
        if audit.get("unique_genomic_variants") != n_rows:
            raise RuntimeError("Stage 12 ClinVar aggregation audit count differs")
    else:
        if dms_source is None:
            raise RuntimeError("Stage 12 DMS source table was not loaded")
        if C.ROW_ID_COL not in dms_source or dms_source[C.ROW_ID_COL].duplicated().any():
            raise RuntimeError("Stage 10 DMS source row identifiers are invalid")
        indexed = dms_source.set_index(C.ROW_ID_COL, verify_integrity=True)
        if not set(row_ids).issubset(indexed.index.astype(str)):
            raise RuntimeError("Stage 12 DMS rows are absent from the Stage 10 source")
        selected_source = indexed.loc[row_ids]
        source_groups = selected_source["ASSAY_ID"].astype(str).to_numpy()
        if not np.array_equal(source_groups, groups):
            raise RuntimeError("Stage 12 DMS assay identifiers differ from Stage 10")
        pairs = pd.MultiIndex.from_arrays(
            [source_groups, selected_source["variant_id"].astype(str).to_numpy()]
        )
        if pairs.duplicated().any():
            raise RuntimeError("Stage 12 DMS primary assay/variant pairs are duplicated")
        dms_scores = pd.to_numeric(
            selected_source["DMS_SCORE"], errors="coerce"
        ).to_numpy(dtype=float)
        _validate_dms_assay_results(
            result,
            labels,
            groups,
            scores,
            decisions,
            decision_confidences,
            thresholds,
            dms_scores,
            set_name,
        )
        if result.get("pooled_row_metrics_are_primary") is not False:
            raise RuntimeError("Stage 12 DMS pooled metrics are incorrectly marked primary")
    return {
        "n": n_rows,
        "positives": int(labels.sum()),
        group_count_key: int(len(np.unique(groups))),
        "models": sorted(model_names),
    }


def _validate_contextual_common_coverage(
    results: dict[str, Any], data: dict[str, np.ndarray]
) -> None:
    primary = results.get("sets", {}).get("clinvar_exact_variant_disjoint", {})
    benchmark = primary.get("contextual_predictor_benchmark", {})
    if not benchmark.get("available"):
        return
    common = benchmark.get("common_coverage", {})
    prefix = common.get("artifact_prefix")
    models = common.get("models", {})
    n_rows = int(common.get("n", -1))
    if not prefix or n_rows <= 0 or not isinstance(models, dict) or not models:
        raise RuntimeError("Stage 12 contextual common-coverage declaration is invalid")
    labels = _one_dimensional_external_array(data, f"{prefix}__y", n_rows).astype(int)
    groups = _one_dimensional_external_array(data, f"{prefix}__groups", n_rows)
    row_ids = _one_dimensional_external_array(data, f"{prefix}__row_ids", n_rows).astype(
        str
    )
    if len(set(row_ids)) != n_rows or len(groups) != n_rows:
        raise RuntimeError("Stage 12 contextual common-coverage identifiers differ")
    if common.get("positives") != int(labels.sum()):
        raise RuntimeError("Stage 12 contextual common-coverage positive count differs")
    for name, metrics in models.items():
        values = _one_dimensional_external_array(data, f"{prefix}__{name}", n_rows).astype(
            float
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Stage 12 contextual common coverage is non-finite: {name}")
        expected = {
            "auroc": round(float(roc_auc_score(labels, values)), 4),
            "auprc": round(float(average_precision_score(labels, values)), 4),
        }
        _assert_reported_value(metrics, expected, f"{prefix}.{name}")


def _validate_clinical_reporting_guard(
    result: dict[str, Any],
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Verify fail-closed reporting for a small external ClinVar cohort."""
    set_name = "clinvar_exact_variant_disjoint"
    assessment = result.get("clinical_evidence_assessment")
    if assessment is None:
        return {"status": "legacy_not_declared"}
    n_rows = int(result.get("n", -1))
    labels = _one_dimensional_external_array(data, f"{set_name}__y", n_rows).astype(int)
    groups = _one_dimensional_external_array(data, f"{set_name}__groups", n_rows)
    expected = PR.clinical_evidence_assessment(labels, groups)
    _assert_reported_value(assessment, expected, f"{set_name}.clinical_evidence")
    if result.get("publication_reporting_guard_enforced") is not True:
        raise RuntimeError("Stage 12 did not enforce the clinical reporting guard")
    utility = result.get("clinical_utility_claim", {})
    if utility.get("allowed") is not False:
        raise RuntimeError("Stage 12 must not claim external clinical utility")
    conformal = result.get("conformal_prediction", {})
    if conformal.get("coverage_guarantee_claimed") is not False:
        raise RuntimeError(
            "Stage 12 claimed conformal coverage without external calibration evidence"
        )
    if assessment["status"] == "underpowered":
        comparisons = result.get("comparisons", {})
        if comparisons.get("status") != "skipped" or comparisons.get("reason") != (
            "underpowered_clinical_cohort_reporting_guard"
        ):
            raise RuntimeError(
                "Stage 12 emitted inferential comparison for underpowered ClinVar"
            )
        for name, metrics in result.get("models", {}).items():
            if metrics.get("ci95") is not None:
                raise RuntimeError(
                    f"Stage 12 emitted {name} interval for underpowered ClinVar"
                )
            if metrics.get("selective_prediction_interpretation", {}).get(
                "clinical_utility_claimed"
            ) is not False:
                raise RuntimeError(
                    f"Stage 12 selective reporting is unsafe for {name}"
                )
        contextual = result.get("contextual_predictor_benchmark", {})
        for record in contextual.get("individual_coverage", {}).values():
            if record.get("metrics", {}).get("ci95") is not None:
                raise RuntimeError(
                    "Stage 12 emitted contextual interval for underpowered ClinVar"
                )
        for metrics in contextual.get("common_coverage", {}).get("models", {}).values():
            if metrics.get("ci95") is not None:
                raise RuntimeError(
                    "Stage 12 emitted common-coverage interval for underpowered ClinVar"
                )
    return {
        "status": "validated",
        "reporting_mode": assessment["reporting_mode"],
        "underpowered": assessment["status"] == "underpowered",
        "clinical_utility_claim_allowed": False,
        "conformal_coverage_claimed": False,
    }


def _stress_level_key(level: float) -> str:
    return f"mask_{int(round(100.0 * float(level))):03d}pct"


def _validate_robustness_reporting(
    results: dict[str, Any],
    summary_table: pd.DataFrame,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate JSON/table coherence for deterministic modality stress."""
    robustness = results.get("robustness")
    declared = run_manifest.get("extra", {}).get("robustness_protocol_version")
    if robustness is None:
        if declared is not None:
            raise RuntimeError("Stage 12 manifest declares absent robustness results")
        return {"status": "legacy_not_declared"}
    if robustness.get("protocol_version") != PR.STRESS_PROTOCOL_VERSION:
        raise RuntimeError("Stage 12 robustness protocol version differs")
    if declared != PR.STRESS_PROTOCOL_VERSION:
        raise RuntimeError("Stage 12 manifest robustness protocol differs")
    sources = robustness.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("Stage 12 robustness source inventory is empty")
    if set(run_manifest.get("extra", {}).get("robustness_sources", [])) != set(
        sources
    ):
        raise RuntimeError("Stage 12 robustness source inventory differs")
    expected_models = set(results.get("models", []))
    validated: dict[str, Any] = {}
    for source, source_report in sources.items():
        if source_report.get("status") != "evaluated":
            raise RuntimeError(f"Stage 12 robustness was not evaluated for {source}")
        if source_report.get("protocol_version") != PR.STRESS_PROTOCOL_VERSION:
            raise RuntimeError(f"Stage 12 {source} robustness protocol differs")
        scenarios = source_report.get("scenarios", {})
        if set(scenarios) != set(PR.STRESS_SCENARIOS):
            raise RuntimeError(f"Stage 12 {source} stress scenarios differ")
        anchor = source_report.get("anchor_model")
        candidates = set(source_report.get("candidate_models", []))
        if anchor is not None and anchor not in expected_models:
            raise RuntimeError(f"Stage 12 {source} stress anchor is unavailable")
        if not candidates.issubset(expected_models):
            raise RuntimeError(f"Stage 12 {source} stress candidates differ")
        for scenario, scenario_report in scenarios.items():
            levels = scenario_report.get("levels", {})
            expected_keys = {_stress_level_key(level) for level in PR.STRESS_LEVELS}
            if set(levels) != expected_keys:
                raise RuntimeError(
                    f"Stage 12 {source}/{scenario} stress levels differ"
                )
            selected_counts = []
            for level in PR.STRESS_LEVELS:
                level_key = _stress_level_key(level)
                record = levels[level_key]
                plan = record.get("mask_plan", {})
                if not np.isclose(
                    float(plan.get("target_fraction", -1.0)),
                    level,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"Stage 12 {source}/{scenario}/{level_key} target differs"
                    )
                selected_counts.append(int(plan.get("selected_groups", -1)))
                digest = str(plan.get("selected_group_ids_sha256", ""))
                if len(digest) != 64:
                    raise RuntimeError(
                        f"Stage 12 {source}/{scenario}/{level_key} mask hash is invalid"
                    )
                model_metrics = record.get("models", {})
                if set(model_metrics) != expected_models:
                    raise RuntimeError(
                        f"Stage 12 {source}/{scenario}/{level_key} models differ"
                    )
                fallback = record.get("safe_fallback_models", {})
                expected_fallback = candidates if anchor is not None else set()
                if set(fallback) != expected_fallback:
                    raise RuntimeError(
                        f"Stage 12 {source}/{scenario}/{level_key} fallback differs"
                    )
                for candidate, metrics in fallback.items():
                    identity = metrics.get("anchor_identity_on_masked_units", {})
                    if identity.get("satisfied") is not True:
                        raise RuntimeError(
                            f"Stage 12 {source}/{scenario}/{level_key}/{candidate} "
                            "violates safe fallback"
                        )
                    if level == 1.0:
                        for metric in (
                            "auroc",
                            "auprc",
                            "mcc",
                            "brier",
                            "group_macro_auroc",
                            "group_macro_functional_spearman",
                        ):
                            _assert_reported_value(
                                metrics.get(metric),
                                model_metrics[anchor].get(metric),
                                f"{source}.{scenario}.{level_key}.{candidate}.{metric}",
                            )
                policy = f"missing_modality_{scenario}_{level_key}"
                rows = summary_table.loc[
                    summary_table["source"].astype(str).eq(source)
                    & summary_table["policy"].astype(str).eq(policy)
                ]
                expected_table_models = expected_models | {
                    f"{candidate}__safe_fallback_to__{anchor}"
                    for candidate in expected_fallback
                }
                if set(rows["model"].astype(str)) != expected_table_models:
                    raise RuntimeError(
                        f"Stage 12 {source}/{scenario}/{level_key} table differs"
                    )
                expected_records = dict(model_metrics)
                expected_records.update(
                    {
                        f"{candidate}__safe_fallback_to__{anchor}": metrics
                        for candidate, metrics in fallback.items()
                    }
                )
                indexed_rows = rows.set_index(rows["model"].astype(str))
                for model_name, metrics in expected_records.items():
                    row = indexed_rows.loc[model_name]
                    if isinstance(row, pd.DataFrame):
                        raise RuntimeError(
                            f"Stage 12 robustness table duplicates {model_name}"
                        )
                    for table_metric, json_metric in (
                        ("auroc", "auroc"),
                        ("auprc", "auprc"),
                        ("mcc", "mcc"),
                        ("brier", "brier"),
                        ("assay_macro_auroc", "group_macro_auroc"),
                        (
                            "assay_macro_functional_spearman",
                            "group_macro_functional_spearman",
                        ),
                    ):
                        reported_value = pd.to_numeric(
                            pd.Series([row.get(table_metric)]), errors="coerce"
                        ).iloc[0]
                        reported = (
                            None if pd.isna(reported_value) else float(reported_value)
                        )
                        _assert_reported_value(
                            reported,
                            metrics.get(json_metric),
                            f"{source}.{scenario}.{level_key}.{model_name}.table.{table_metric}",
                        )
                if "inferential_claim_allowed" not in rows or not rows[
                    "inferential_claim_allowed"
                ].astype(str).str.lower().eq("false").all():
                    raise RuntimeError("Stage 12 robustness table permits inference")
            if selected_counts != sorted(selected_counts):
                raise RuntimeError(
                    f"Stage 12 {source}/{scenario} masks are not nested by count"
                )
        validated[source] = {
            "scenarios": sorted(scenarios),
            "anchor_model": anchor,
            "candidate_models": sorted(candidates),
        }
    return {
        "status": "validated",
        "protocol_version": PR.STRESS_PROTOCOL_VERSION,
        "sources": validated,
        "inferential_claims": False,
    }


def _validate_reliability_diagnostic_reporting(
    results: dict[str, Any],
    data: dict[str, np.ndarray],
    run_manifest: dict[str, Any],
    present_sources: set[str],
) -> dict[str, Any]:
    """Validate descriptive reliability diagnostics without upgrading their role."""
    expected_status = "descriptive_post_selection_not_primary"
    expected_components = list(C.RELIABILITY_DIAGNOSTIC_COMPONENTS)
    policy = results.get("reliability_diagnostic_policy")
    if not isinstance(policy, dict):
        raise RuntimeError("Stage 12 reliability diagnostic policy is missing")
    if (
        policy.get("status") != expected_status
        or policy.get("strata_use_labels") is not False
        or policy.get("used_for_model_or_threshold_selection") is not False
        or policy.get("external_label_refitting") is not False
        or policy.get("fold_aggregation")
        != "mean_components_across_frozen_internal_deployment_folds"
        or policy.get("component_names") != expected_components
    ):
        raise RuntimeError(
            "Stage 12 reliability diagnostics are not publication-safe descriptive outputs"
        )
    manifest_policy = run_manifest.get("extra", {}).get(
        "reliability_diagnostics"
    )
    if (
        not isinstance(manifest_policy, dict)
        or manifest_policy.get("status") != expected_status
        or manifest_policy.get("strata_use_labels") is not False
        or manifest_policy.get("used_for_model_or_threshold_selection") is not False
        or manifest_policy.get("component_names") != expected_components
    ):
        raise RuntimeError("Stage 12 manifest reliability diagnostic policy differs")

    sets = results.get("sets", {})
    diagnostic_sets = {
        name
        for name, details in sets.items()
        if isinstance(details, dict)
        and isinstance(details.get("reliability_diagnostics"), dict)
    }
    required_primary_sets = {
        f"{source}_exact_variant_disjoint" for source in present_sources
    }
    if not required_primary_sets.issubset(diagnostic_sets):
        raise RuntimeError(
            "Stage 12 primary sets lack reliability diagnostics: "
            f"{sorted(required_primary_sets - diagnostic_sets)}"
        )
    marker = "__reliability_components__"
    npz_diagnostic_sets = {
        key.split(marker, 1)[0] for key in data if marker in key
    }
    if npz_diagnostic_sets != diagnostic_sets:
        raise RuntimeError(
            "Stage 12 JSON/NPZ reliability diagnostic set inventories differ"
        )

    expected_csv_rows: list[dict[str, Any]] = []
    primary_mechanistic_rows: list[dict[str, Any]] = []
    validated_sets: dict[str, Any] = {}
    expected_component_suffixes = {
        *expected_components,
        "availability_stratum_code",
        "anchor_decisions",
    }
    for set_name in sorted(diagnostic_sets):
        set_result = sets[set_name]
        diagnostic = set_result["reliability_diagnostics"]
        n_rows = int(set_result.get("n", -1))
        if n_rows <= 0 or diagnostic.get("n") != n_rows:
            raise RuntimeError(f"Stage 12 reliability row count differs for {set_name}")
        labels = _one_dimensional_external_array(
            data, f"{set_name}__y", n_rows
        ).astype(np.int8)
        row_ids = _one_dimensional_external_array(
            data, f"{set_name}__row_ids", n_rows
        ).astype(str)
        if (
            not np.isin(labels, (0, 1)).all()
            or len(set(row_ids)) != n_rows
            or np.any(np.char.strip(row_ids) == "")
        ):
            raise RuntimeError(
                f"Stage 12 reliability labels or row identities differ for {set_name}"
            )
        prefix = f"{set_name}{marker}"
        observed_suffixes = {
            key[len(prefix) :] for key in data if key.startswith(prefix)
        }
        if observed_suffixes != expected_component_suffixes:
            raise RuntimeError(
                f"Stage 12 reliability component inventory differs for {set_name}"
            )
        components: dict[str, np.ndarray] = {}
        for component in expected_components:
            values = _one_dimensional_external_array(
                data, f"{prefix}{component}", n_rows
            )
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise RuntimeError(
                    f"Stage 12 reliability component {set_name}/{component} is invalid"
                )
            components[component] = values.astype(np.float64)
        for component in (
            "anchor_probability",
            "gate",
            "hard_availability",
            "structure_reliability",
            "conservation_reliability",
        ):
            values = components[component]
            if ((values < -1e-6) | (values > 1.0 + 1e-6)).any():
                raise RuntimeError(
                    f"Stage 12 reliability component {set_name}/{component} "
                    "is outside [0, 1]"
                )
        codes = _one_dimensional_external_array(
            data, f"{prefix}availability_stratum_code", n_rows
        )
        anchor_decisions = _one_dimensional_external_array(
            data, f"{prefix}anchor_decisions", n_rows
        )
        if (
            codes.dtype != np.int8
            or anchor_decisions.dtype != np.int8
            or not np.isin(codes, tuple(C.RELIABILITY_AVAILABILITY_STRATA)).all()
            or not np.isin(anchor_decisions, (0, 1)).all()
        ):
            raise RuntimeError(
                f"Stage 12 reliability stratum codes or anchor decisions differ for {set_name}"
            )
        expected_codes = C.reliability_availability_codes(components)
        if not np.array_equal(codes, expected_codes):
            raise RuntimeError(
                f"Stage 12 reliability strata are not label-independent for {set_name}"
            )
        expected_unavailable = (
            (components["structure_reliability"] <= 1e-7)
            & (components["conservation_reliability"] <= 1e-7)
        )
        if not np.array_equal(
            components["hard_availability"] <= 1e-7,
            expected_unavailable,
        ):
            raise RuntimeError(
                f"Stage 12 reliability hard-availability semantics differ for {set_name}"
            )
        if (
            components["gate"]
            > components["hard_availability"] + 1e-6
        ).any():
            raise RuntimeError(
                f"Stage 12 reliability gate exceeds hard availability for {set_name}"
            )

        model_key = f"{set_name}__{C.RELIABILITY_ARCHITECTURE}"
        decision_key = f"{model_key}__decisions"
        threshold_key = f"{model_key}__threshold"
        probabilities = _one_dimensional_external_array(
            data, model_key, n_rows
        ).astype(np.float64)
        decisions = _one_dimensional_external_array(
            data, decision_key, n_rows
        ).astype(np.int8)
        threshold_values = np.asarray(data.get(threshold_key, []), dtype=np.float64)
        if (
            threshold_values.shape != (1,)
            or not np.isfinite(threshold_values[0])
            or not 0.0 <= threshold_values[0] <= 1.0
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
            or not np.isin(decisions, (0, 1)).all()
        ):
            raise RuntimeError(
                f"Stage 12 proposed reliability predictions are invalid for {set_name}"
            )
        hard_fallback = components["hard_availability"] <= 1e-7
        if not np.array_equal(
            decisions[hard_fallback], anchor_decisions[hard_fallback]
        ):
            raise RuntimeError(
                f"Stage 12 hard-fallback anchor votes differ for {set_name}"
            )
        recomputed = C.summarize_reliability_diagnostics(
            components,
            labels,
            probabilities,
            float(threshold_values[0]),
            decisions,
            anchor_decisions=anchor_decisions,
        )
        _assert_reported_value(
            diagnostic,
            recomputed,
            f"{set_name}.reliability_diagnostics",
        )
        expected_selection = {
            "used_for_model_selection": False,
            "used_for_threshold_selection": False,
            "external_labels_used_for_refitting": False,
            "inference_claim_allowed": False,
        }
        if diagnostic.get("selection_or_refitting") != expected_selection:
            raise RuntimeError(
                f"Stage 12 reliability selection/refitting guard differs for {set_name}"
            )
        if diagnostic.get("stratification", {}).get("uses_labels") is not False:
            raise RuntimeError(
                f"Stage 12 reliability strata use labels for {set_name}"
            )
        if diagnostic.get("evaluation_scope") != (
            "external_frozen_deployment_folds_descriptive_only"
        ) or diagnostic.get("fold_aggregation") != (
            "arithmetic_mean_component_and_calibrated_probability_across_five_"
            "frozen_internal_deployment_folds"
        ):
            raise RuntimeError(
                f"Stage 12 reliability external evaluation scope differs for {set_name}"
            )
        source = str(set_result.get("source"))
        policy_name = str(set_result.get("policy"))
        expected_role = (
            "primary"
            if policy_name == "exact_variant_disjoint"
            else "secondary_stress_test"
        )
        if set_name != f"{source}_{policy_name}" or set_result.get(
            "role"
        ) != expected_role:
            raise RuntimeError(
                f"Stage 12 reliability set identity or role differs for {set_name}"
            )
        expected_analysis_aggregation = (
            "mean_across_retained_annotation_rows_after_fold_prediction"
            if source == "clinvar"
            else "none_assay_variant_row"
        )
        if diagnostic.get("analysis_unit_aggregation") != expected_analysis_aggregation:
            raise RuntimeError(
                f"Stage 12 reliability analysis-unit aggregation differs for {set_name}"
            )
        context = {
            "scope": "external_validation",
            "source": source,
            "policy": policy_name,
            "role": set_result.get("role"),
            "analysis_unit": set_result.get("analysis_unit"),
            "model": C.RELIABILITY_ARCHITECTURE,
        }
        diagnostic_rows = C.reliability_diagnostic_rows(recomputed, context)
        expected_csv_rows.extend(diagnostic_rows)
        if set_name in required_primary_sets:
            for row in diagnostic_rows:
                primary_mechanistic_rows.append(
                    {
                        **row,
                        "reporting_scope": expected_status,
                        "not_primary_or_confirmatory": True,
                        "primary_endpoint": False,
                        "confirmatory_evidence": False,
                        "external_labels_used_for_refitting": False,
                        "interpretation": (
                            "mechanistic_model_diagnostics_not_causal_attributions"
                        ),
                    }
                )
        validated_sets[set_name] = {
            "n": n_rows,
            "source": source,
            "policy": policy_name,
            "role": set_result.get("role"),
            "gate_behavior": recomputed["gate_behavior"],
            "bounded_residual_behavior": recomputed[
                "bounded_residual_behavior"
            ],
            "strata": {
                name: {
                    "code": record["code"],
                    "n": record["n"],
                    "coverage": record["coverage"],
                    "mean_gate": record["mean_gate"],
                    "exact_fallback_rate": record["exact_fallback_rate"],
                    "proposed_minus_anchor": record["proposed_minus_anchor"],
                }
                for name, record in recomputed["strata"].items()
            },
        }

    observed_table = pd.read_csv(
        EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE, low_memory=False
    )
    expected_table = pd.DataFrame(expected_csv_rows)
    expected_table = pd.read_csv(StringIO(expected_table.to_csv(index=False)))
    sort_columns = ["source", "policy", "availability_code"]
    if list(observed_table.columns) != list(expected_table.columns):
        raise RuntimeError("Stage 12 reliability diagnostics CSV schema differs")
    observed_table = observed_table.sort_values(sort_columns).reset_index(drop=True)
    expected_table = expected_table.sort_values(sort_columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_table,
            expected_table,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1.0000001e-6,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Stage 12 reliability diagnostics CSV differs from JSON/NPZ"
        ) from error
    return {
        "status": "validated_descriptive_post_selection_not_primary",
        "scientific_role": expected_status,
        "strata_use_labels": False,
        "used_for_model_or_threshold_selection": False,
        "external_label_refitting": False,
        "inferential_claim_allowed": False,
        "not_primary_or_confirmatory": True,
        "sets": validated_sets,
        "primary_mechanistic_rows": primary_mechanistic_rows,
        "source_table_sha256": file_sha256(
            EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE
        ),
    }


def _validate_stage12_artifacts(
    stage14_validation: dict[str, Any],
) -> dict[str, Any]:
    """Bind Stage 12 results, arrays, tables, inputs and source provenance."""
    required_files = (
        EXTERNAL_RESULTS,
        EXTERNAL_PREDICTIONS,
        EXTERNAL_SUMMARY_TABLE,
        EXTERNAL_PREDICTION_TABLE,
        EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE,
        EXTERNAL_RUN_MANIFEST,
        EXTERNAL_PREP_MANIFEST,
        DEPLOYMENT_RESULTS,
        DEPLOYMENT_RUN_MANIFEST,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 12 reporting inputs are missing: {missing}")
    results = _load_json_object(EXTERNAL_RESULTS, "Stage 12 result")
    run_manifest = _load_json_object(EXTERNAL_RUN_MANIFEST, "Stage 12 run manifest")
    prep_manifest = _load_json_object(EXTERNAL_PREP_MANIFEST, "Stage 09 run manifest")
    deployment_manifest = _load_json_object(
        DEPLOYMENT_RUN_MANIFEST, "Stage 11 run manifest"
    )
    if results.get("model_tag") != MODEL_TAG:
        raise RuntimeError("Stage 12 result model tag differs")
    if results.get("prediction_aggregation") != (
        "mean_calibrated_probability_with_majority_vote_of_frozen_fold_thresholds"
    ):
        raise RuntimeError("Stage 12 prediction aggregation policy differs")
    if results.get("threshold_source") != "dedicated_internal_fold_partitions":
        raise RuntimeError("Stage 12 threshold source differs")
    primary_policy = results.get("primary_evaluation_policy", {})
    if primary_policy.get("clinvar", {}).get("set") != (
        "clinvar_exact_variant_disjoint"
    ) or primary_policy.get("dms", {}).get("set") != "dms_exact_variant_disjoint":
        raise RuntimeError("Stage 12 predeclared primary-set policy differs")

    if (
        run_manifest.get("stage") != "12_external_validation"
        or run_manifest.get("label_task") != "clinical"
        or run_manifest.get("model_tag") != MODEL_TAG
    ):
        raise RuntimeError("Stage 12 run manifest contract differs")
    external_contract = run_manifest.get("external_contract", {})
    if external_contract.get("require_clinvar") is not REQUIRE_EXTERNAL_CLINVAR or (
        external_contract.get("require_dms") is not REQUIRE_EXTERNAL_DMS
    ):
        raise RuntimeError("Stage 12 external-requirement contract differs")
    extra = run_manifest.get("extra", {})
    if (
        extra.get("clinvar_and_dms_pooled") is not False
        or extra.get("exact_variant_deoverlap") is not True
        or set(extra.get("primary_sets", []))
        != {"clinvar_exact_variant_disjoint", "dms_exact_variant_disjoint"}
    ):
        raise RuntimeError("Stage 12 manifest primary evaluation contract differs")
    model_inventory = results.get("models")
    if not isinstance(model_inventory, list) or not model_inventory:
        raise RuntimeError("Stage 12 trained-model inventory is empty")
    if set(model_inventory) != set(extra.get("models", [])):
        raise RuntimeError("Stage 12 result and manifest model inventories differ")
    _validate_source_provenance(
        run_manifest, STAGE12_PROVENANCE_SOURCES, "Stage 12"
    )
    for path, suffix in (
        (EXTERNAL_RESULTS, "12_external_validation/external_validation.json"),
        (EXTERNAL_PREDICTIONS, "12_external_validation/external_predictions.npz"),
        (EXTERNAL_SUMMARY_TABLE, "12_external_validation/external_validation_table.csv"),
        (EXTERNAL_PREDICTION_TABLE, "12_external_validation/external_predictions.csv"),
        (
            EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE,
            "12_external_validation/reliability_diagnostics.csv",
        ),
    ):
        _verify_manifest_output(run_manifest, path, suffix, "Stage 12")

    if (
        prep_manifest.get("stage") != "09_prepare_external_esm_dataset"
        or prep_manifest.get("label_task") != "clinical"
    ):
        raise RuntimeError("Stage 09 external-preparation manifest contract differs")
    prep_extra = prep_manifest.get("extra", {})
    sampling_policy = prep_extra.get("dms_sampling_policy")
    if (
        sampling_policy not in {"all", "hash_uniform"}
        or prep_extra.get("dms_sampling_uses_label") is not False
    ):
        raise RuntimeError("Stage 09 DMS sampling is not publication-safe")
    _validate_source_provenance(
        prep_manifest, STAGE09_PROVENANCE_SOURCES, "Stage 09"
    )

    if (
        deployment_manifest.get("stage") != "11_train_and_evaluate"
        or deployment_manifest.get("label_task") != "clinical"
        or deployment_manifest.get("model_tag") != MODEL_TAG
        or set(deployment_manifest.get("extra", {}).get("models", []))
        != set(model_inventory)
    ):
        raise RuntimeError("Stage 11 deployment manifest differs from Stage 12")
    _validate_source_provenance(
        deployment_manifest, STAGE11_PROVENANCE_SOURCES, "Stage 11"
    )

    stage12_inputs = (
        (STAGE07_MANIFEST, "07_natural_prevalence/run_manifest.json"),
        (
            INTERNAL_UNIVERSE,
            "07_natural_prevalence/Final_Dataset_Natural_Prevalence.parquet",
        ),
        (EXTERNAL_PREP_MANIFEST, "09_prepare_external_esm/run_manifest.json"),
        (DEPLOYMENT_RESULTS, "11_train_and_evaluate/results.json"),
        (DEPLOYMENT_RUN_MANIFEST, "11_train_and_evaluate/run_manifest.json"),
        (INTERNAL_RESULTS, "14_tuning/architecture_selection.json"),
        (INTERNAL_SPLIT_PLAN, "14_tuning/nested_inner_splits.json"),
        (INTERNAL_RUN_MANIFEST, "14_tuning/run_manifest.json"),
    )
    for path, suffix in stage12_inputs:
        _verify_manifest_input(run_manifest, path, suffix, "Stage 12")
    if file_sha256(INTERNAL_RESULTS) != stage14_validation.get(
        "architecture_results_sha256"
    ) or file_sha256(INTERNAL_SPLIT_PLAN) != stage14_validation.get(
        "split_plan_file_sha256"
    ):
        raise RuntimeError("Stage 12 and validated Stage 14 linkage differs")

    sets = results.get("sets")
    if not isinstance(sets, dict):
        raise RuntimeError("Stage 12 result has no evaluation sets")
    required_sources = {
        source
        for source, required in (
            ("clinvar", REQUIRE_EXTERNAL_CLINVAR),
            ("dms", REQUIRE_EXTERNAL_DMS),
        )
        if required
    }
    present_sources = {
        source
        for source in ("clinvar", "dms")
        if isinstance(sets.get(f"{source}_exact_variant_disjoint"), dict)
        and sets[f"{source}_exact_variant_disjoint"].get("status") == "evaluated"
    }
    missing_sources = required_sources - present_sources
    if missing_sources:
        raise RuntimeError(
            f"Stage 12 misses required primary external sources: {sorted(missing_sources)}"
        )
    for source in present_sources:
        for path, suffix in (
            (
                STAGE09_PREPARED_OUTPUTS[source],
                f"09_prepare_external_esm/{source}_esm_ready.csv",
            ),
            (
                STAGE10_MANIFESTS[source],
                f"10_esm_features/{source}_esm_manifest.json",
            ),
        ):
            _verify_manifest_input(run_manifest, path, suffix, "Stage 12")
        if source == "dms":
            sequence_records = [
                record
                for key, record in run_manifest.get("inputs", {}).items()
                if str(key).replace("\\", "/").endswith(
                    "/09_prepare_external_esm/dms_sequences.parquet"
                )
            ]
            if sequence_records:
                _verify_manifest_input(
                    run_manifest,
                    DMS_SEQUENCE_OUTPUT,
                    "09_prepare_external_esm/dms_sequences.parquet",
                    "Stage 12",
                )
                _verify_manifest_output(
                    prep_manifest,
                    DMS_SEQUENCE_OUTPUT,
                    "09_prepare_external_esm/dms_sequences.parquet",
                    "Stage 09",
                )
    sources_to_validate = sorted(present_sources)
    for source in sources_to_validate:
        for path in EXTERNAL_STAGE10_INPUTS[source]:
            _verify_manifest_input(
                run_manifest,
                path,
                f"10_esm_features/{path.name}",
                "Stage 12",
            )

    data = _load_npz(EXTERNAL_PREDICTIONS)
    prediction_table = pd.read_csv(EXTERNAL_PREDICTION_TABLE, low_memory=False)
    summary_table = pd.read_csv(EXTERNAL_SUMMARY_TABLE, low_memory=False)
    required_summary_columns = {
        "source",
        "policy",
        "model",
        "n",
        "positives",
    }
    if not required_summary_columns.issubset(summary_table):
        raise RuntimeError(
            "Stage 12 summary table misses columns: "
            f"{sorted(required_summary_columns - set(summary_table))}"
        )
    dms_source = None
    if "dms" in sources_to_validate:
        dms_path = EXTERNAL_STAGE10_INPUTS["dms"][0]
        dms_source = pd.read_parquet(
            dms_path,
            columns=[C.ROW_ID_COL, "ASSAY_ID", "variant_id", "DMS_SCORE"],
        )
        dms_source[C.ROW_ID_COL] = dms_source[C.ROW_ID_COL].astype(str)
    set_reports = {
        source: _validate_external_primary_set(
            source,
            sets[f"{source}_exact_variant_disjoint"],
            data,
            prediction_table,
            dms_source if source == "dms" else None,
        )
        for source in sources_to_validate
    }
    for source, report in set_reports.items():
        selected_summary = summary_table.loc[
            summary_table["source"].astype(str).eq(source)
            & summary_table["policy"].astype(str).eq("exact_variant_disjoint")
        ]
        if set(selected_summary["model"].astype(str)) != set(report["models"]):
            raise RuntimeError(f"Stage 12 summary model inventory differs for {source}")
        if (
            not pd.to_numeric(selected_summary["n"], errors="coerce")
            .eq(report["n"])
            .all()
            or not pd.to_numeric(selected_summary["positives"], errors="coerce")
            .eq(report["positives"])
            .all()
        ):
            raise RuntimeError(f"Stage 12 summary counts differ for {source}")
    reliability_reporting = _validate_reliability_diagnostic_reporting(
        results,
        data,
        run_manifest,
        present_sources,
    )
    _validate_contextual_common_coverage(results, data)
    clinical_reporting = (
        _validate_clinical_reporting_guard(
            sets["clinvar_exact_variant_disjoint"], data
        )
        if "clinvar" in present_sources
        else {"status": "not_available"}
    )
    if clinical_reporting.get("underpowered") is True:
        clinical_summary = summary_table.loc[
            summary_table["source"].astype(str).eq("clinvar")
            & summary_table["policy"].astype(str).eq("exact_variant_disjoint")
        ]
        for field in (
            "inferential_claim_allowed",
            "clinical_calibration_claim_allowed",
            "clinical_utility_claim_allowed",
        ):
            if field not in clinical_summary or not clinical_summary[field].astype(
                str
            ).str.lower().eq("false").all():
                raise RuntimeError(
                    f"Stage 12 ClinVar summary does not enforce {field}=False"
                )
    robustness_reporting = _validate_robustness_reporting(
        results, summary_table, run_manifest
    )
    if robustness_reporting.get("status") == "validated":
        uncertainty_policy = results.get("uncertainty_reporting_policy", {})
        if (
            uncertainty_policy.get("external_label_tuning") is not False
            or "withheld_without" not in str(uncertainty_policy.get("conformal", ""))
            or "never_inferred" not in str(
                uncertainty_policy.get("clinical_utility", "")
            )
        ):
            raise RuntimeError("Stage 12 uncertainty reporting policy is unsafe")
    manifest_counts = extra.get("sets", {})
    for set_name, result in sets.items():
        if isinstance(result, dict) and manifest_counts.get(set_name) != result.get("n", 0):
            raise RuntimeError(f"Stage 12 manifest count differs for {set_name}")
    if "dms" in present_sources:
        caveats = sets["dms_exact_variant_disjoint"].get("per_assay", {}).get(
            "sampling_caveats", {}
        )
        if (
            caveats.get("manifest_available") is not True
            or caveats.get("legacy_fallback") is not False
            or caveats.get("sampling_uses_label") is not False
            or caveats.get("sampling_policy") != sampling_policy
        ):
            raise RuntimeError("Stage 12 DMS sampling audit differs from Stage 09")

    return {
        "status": "validated",
        "publication_complete": bool(
            REQUIRE_EXTERNAL_CLINVAR
            and REQUIRE_EXTERNAL_DMS
            and {"clinvar", "dms"}.issubset(present_sources)
        ),
        "required_sources": sorted(required_sources),
        "validated_sources": sources_to_validate,
        "sets": set_reports,
        "dms_sampling_policy": sampling_policy,
        "external_results_sha256": file_sha256(EXTERNAL_RESULTS),
        "external_predictions_sha256": file_sha256(EXTERNAL_PREDICTIONS),
        "external_prediction_table_sha256": file_sha256(EXTERNAL_PREDICTION_TABLE),
        "external_summary_table_sha256": file_sha256(EXTERNAL_SUMMARY_TABLE),
        "external_reliability_diagnostics_table_sha256": file_sha256(
            EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE
        ),
        "stage12_manifest_sha256": file_sha256(EXTERNAL_RUN_MANIFEST),
        "stage09_manifest_sha256": file_sha256(EXTERNAL_PREP_MANIFEST),
        "clinical_reporting_guard": clinical_reporting,
        "robustness_reporting": robustness_reporting,
        "reliability_diagnostic_reporting": reliability_reporting,
    }


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def _save_figure(figure: plt.Figure, name: str) -> list[str]:
    paths: list[str] = []
    for extension in FORMATS:
        output = STAGE13_OUT / f"{name}.{extension}"
        temporary = output.with_suffix(output.suffix + ".tmp")
        figure.savefig(
            temporary,
            format=extension,
            bbox_inches="tight",
            facecolor="white",
        )
        temporary.replace(output)
        # Store a portable path; the manifest may be copied from Kaggle to a
        # different local project root.
        paths.append(output.name)
    plt.close(figure)
    return paths


def _model_names(data: dict[str, np.ndarray], prefix: str) -> list[str]:
    return [name for name in MODEL_ORDER if f"{prefix}__{name}" in data]


def _group_indices(groups: np.ndarray) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    groups = np.asarray(groups, dtype=object)
    codes, unique = pd.factorize(groups, sort=False)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(unique))
    boundaries = np.cumsum(counts)
    pieces = np.split(order, boundaries[:-1])
    return np.asarray(unique, dtype=object), dict(zip(unique, pieces))


def _bootstrap_indices(
    groups: np.ndarray,
    iterations: int,
    seed: int,
    hierarchical_within_group: bool = False,
) -> Iterator[np.ndarray]:
    unique, mapping = _group_indices(groups)
    random_state = np.random.RandomState(seed)
    for _ in range(iterations):
        selected = random_state.choice(unique, len(unique), replace=True)
        pieces = []
        for group in selected:
            indices = mapping[group]
            if hierarchical_within_group:
                indices = random_state.choice(indices, len(indices), replace=True)
            pieces.append(indices)
        yield np.concatenate(pieces)


def _roc_band(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
    hierarchical_within_group: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for index in _bootstrap_indices(
        groups, BOOTSTRAPS, seed, hierarchical_within_group
    ):
        if len(np.unique(labels[index])) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(
            labels[index], probabilities[index]
        )
        curves.append(np.interp(grid, false_positive, true_positive))
    if not curves:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    return grid, *np.percentile(np.asarray(curves), [2.5, 97.5], axis=0)


def _pr_band(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
    hierarchical_within_group: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for index in _bootstrap_indices(
        groups, BOOTSTRAPS, seed, hierarchical_within_group
    ):
        if len(np.unique(labels[index])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(
            labels[index], probabilities[index]
        )
        curves.append(np.interp(grid, recall[::-1], precision[::-1]))
    if not curves:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    return grid, *np.percentile(np.asarray(curves), [2.5, 97.5], axis=0)


def _reliability(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    predicted = np.full(bins, np.nan)
    observed = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for index in range(bins):
        selected = assignments == index
        counts[index] = int(selected.sum())
        if selected.any():
            predicted[index] = float(probabilities[selected].mean())
            observed[index] = float(labels[selected].mean())
    return predicted, observed, counts


def _reliability_band(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
    bins: int = 10,
    hierarchical_within_group: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    observed = []
    for index in _bootstrap_indices(
        groups, BOOTSTRAPS, seed, hierarchical_within_group
    ):
        _, values, _ = _reliability(labels[index], probabilities[index], bins)
        observed.append(values)
    matrix = np.asarray(observed)
    low = np.full(bins, np.nan)
    high = np.full(bins, np.nan)
    for index in range(bins):
        finite = matrix[:, index][np.isfinite(matrix[:, index])]
        if finite.size:
            low[index], high[index] = np.percentile(finite, [2.5, 97.5])
    return low, high


def _plot_roc(
    axis: plt.Axes,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    title: str,
    hierarchical_within_group: bool = False,
    show_intervals: bool = True,
) -> None:
    if len(np.unique(labels)) < 2:
        axis.text(0.5, 0.5, "Only one class remains", ha="center", va="center")
        axis.set_title(title)
        return
    for offset, (name, values) in enumerate(probabilities.items()):
        false_positive, true_positive, _ = roc_curve(labels, values)
        area = roc_auc_score(labels, values)
        color = MODEL_COLORS[name]
        axis.plot(
            false_positive,
            true_positive,
            color=color,
            linewidth=1.8,
            label=f"{MODEL_LABELS[name]} ({area:.3f})",
        )
        if show_intervals:
            grid, low, high = _roc_band(
                labels,
                values,
                groups,
                RANDOM_STATE + offset,
                hierarchical_within_group,
            )
            axis.fill_between(grid, low, high, color=color, alpha=0.12, linewidth=0)
    axis.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="lower right")


def _plot_pr(
    axis: plt.Axes,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    groups: np.ndarray,
    title: str,
    hierarchical_within_group: bool = False,
    show_intervals: bool = True,
) -> None:
    if len(np.unique(labels)) < 2:
        axis.text(0.5, 0.5, "Only one class remains", ha="center", va="center")
        axis.set_title(title)
        return
    for offset, (name, values) in enumerate(probabilities.items()):
        precision, recall, _ = precision_recall_curve(labels, values)
        area = average_precision_score(labels, values)
        color = MODEL_COLORS[name]
        axis.plot(
            recall,
            precision,
            color=color,
            linewidth=1.8,
            label=f"{MODEL_LABELS[name]} ({area:.3f})",
        )
        if show_intervals:
            grid, low, high = _pr_band(
                labels,
                values,
                groups,
                RANDOM_STATE + 100 + offset,
                hierarchical_within_group,
            )
            axis.fill_between(grid, low, high, color=color, alpha=0.12, linewidth=0)
        if name in decisions:
            operating_precision = precision_score(
                labels, decisions[name], zero_division=0
            )
            operating_recall = recall_score(
                labels, decisions[name], zero_division=0
            )
            axis.scatter(
                operating_recall,
                operating_precision,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=30,
                zorder=5,
            )
    axis.axhline(labels.mean(), color="0.5", linestyle="--", linewidth=1)
    axis.set(xlabel="Recall", ylabel="Precision", title=title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="lower left")


def _internal_vectors() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    data = _load_npz(INTERNAL_PREDICTIONS)
    labels = data["y"].astype(int)
    groups = data["groups"].astype(object)
    probabilities: dict[str, np.ndarray] = {}
    decisions: dict[str, np.ndarray] = {}
    for name in PRIMARY_INTERNAL_PLOT_ORDER:
        architecture_key = f"{name}__probabilities"
        reference_key = f"reference__{name}__probabilities"
        if architecture_key in data:
            key_prefix = name
        elif reference_key in data:
            key_prefix = f"reference__{name}"
        else:
            continue
        probability_key = f"{key_prefix}__probabilities"
        decision_key = f"{key_prefix}__decisions"
        threshold_key = f"{key_prefix}__thresholds"
        if decision_key not in data or threshold_key not in data:
            raise KeyError(f"Stage 14 OOF output is incomplete for {name}")
        probabilities[name] = data[probability_key].astype(float)
        decisions[name] = data[decision_key].astype(int)
    if not probabilities:
        raise ValueError("No Stage 14 nested OOF models are available for plotting")
    return labels, groups, probabilities, decisions


def figure_internal_performance() -> plt.Figure:
    labels, groups, probabilities, decisions = _internal_vectors()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_roc(
        axes[0],
        labels,
        probabilities,
        groups,
        "Internal nested split-group-disjoint ROC",
    )
    _plot_pr(
        axes[1],
        labels,
        probabilities,
        decisions,
        groups,
        "Internal nested split-group-disjoint precision-recall",
    )
    figure.suptitle(
        "Authoritative Stage 14 outer-fold predictions with 95% group-bootstrap bands"
    )
    figure.tight_layout()
    return figure


def _plot_calibration_panel(
    reliability_axis: plt.Axes,
    histogram_axis: plt.Axes,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    title: str,
    hierarchical_within_group: bool = False,
    show_intervals: bool = True,
) -> None:
    reliability_axis.plot([0, 1], [0, 1], color="0.4", linestyle="--")
    bins = np.linspace(0.0, 1.0, 11)
    for offset, (name, values) in enumerate(probabilities.items()):
        predicted, observed, counts = _reliability(labels, values)
        valid = counts > 0
        color = MODEL_COLORS[name]
        reliability_axis.plot(
            predicted[valid],
            observed[valid],
            marker="o",
            markersize=3,
            color=color,
            label=MODEL_LABELS[name],
        )
        if show_intervals:
            low, high = _reliability_band(
                labels,
                values,
                groups,
                RANDOM_STATE + 200 + offset,
                hierarchical_within_group=hierarchical_within_group,
            )
            reliability_axis.fill_between(
                predicted[valid],
                low[valid],
                high[valid],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        histogram_axis.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.4,
            color=color,
            label=MODEL_LABELS[name],
        )
    reliability_axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    reliability_axis.legend(loc="upper left")
    histogram_axis.set(
        xlabel="Predicted probability",
        ylabel="Sample count",
        title="Probability distribution",
        xlim=(0, 1),
    )
    histogram_axis.legend(loc="upper center", ncol=2)


def figure_internal_calibration() -> plt.Figure:
    labels, groups, probabilities, _ = _internal_vectors()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_calibration_panel(
        axes[0],
        axes[1],
        labels,
        probabilities,
        groups,
        "Internal nested-OOF reliability with 95% group-bootstrap bands",
    )
    figure.tight_layout()
    return figure


def figure_model_comparison() -> plt.Figure:
    if not INTERNAL_RESULTS.exists():
        raise FileNotFoundError(INTERNAL_RESULTS)
    payload = json.loads(INTERNAL_RESULTS.read_text(encoding="utf-8"))
    metric_payload: dict[str, dict[str, Any]] = {
        name: details["nested_outer_metrics"]
        for name, details in payload.get("architectures", {}).items()
    }
    metric_payload.update(payload.get("reference_baselines", {}))
    models = [name for name in MODEL_ORDER if name in metric_payload]
    if not models:
        raise ValueError("Stage 14 contains no nested model metrics")
    metrics = ("mcc", "auroc", "auprc")
    # Long publication labels cannot fit legibly beneath 13 side-by-side bars.
    # Horizontal panels keep every model name readable at normal page width.
    figure, axes = plt.subplots(1, 3, figsize=(17, 7.5), sharey=True)
    positions = np.arange(len(models))
    for axis, metric in zip(axes, metrics):
        values = [metric_payload[name][metric] for name in models]
        intervals = [
            metric_payload[name].get("ci95", {}).get(metric) for name in models
        ]
        errors = np.asarray(
            [
                [
                    value - interval[0] if interval else 0.0,
                    interval[1] - value if interval else 0.0,
                ]
                for value, interval in zip(values, intervals)
            ]
        ).T
        axis.barh(
            positions,
            values,
            color=[MODEL_COLORS[name] for name in models],
            xerr=errors,
            capsize=3,
        )
        axis.set_yticks(positions, [MODEL_LABELS[name] for name in models])
        axis.set_title(metric.upper())
        axis.set_xlim(-1, 1) if metric == "mcc" else axis.set_xlim(0, 1)
        axis.axvline(0, color="0.4", linewidth=0.8)
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    figure.suptitle(
        "Primary internal nested outer-fold metrics and group-bootstrap intervals"
    )
    figure.tight_layout()
    return figure


def _external_vectors(
    source: str,
    policy: str = "exact_variant_disjoint",
) -> ExternalVectors | None:
    data = _load_npz(EXTERNAL_PREDICTIONS)
    prefix = f"{source}_{policy}"
    if f"{prefix}__y" not in data:
        return None
    labels = data[f"{prefix}__y"].astype(int)
    groups = data[f"{prefix}__groups"].astype(object)
    scores: dict[str, np.ndarray] = {}
    decisions: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        key = f"{prefix}__{name}"
        decision_key = f"{prefix}__{name}__decisions"
        if key in data:
            scores[name] = data[key].astype(float)
        if decision_key in data:
            decisions[name] = data[decision_key].astype(int)
    metadata: dict[str, Any] = {}
    if EXTERNAL_RESULTS.exists():
        payload = json.loads(EXTERNAL_RESULTS.read_text(encoding="utf-8"))
        metadata = payload.get("sets", {}).get(prefix, {})
    if not metadata:
        metadata = {
            "source": source,
            "policy": policy,
            "n": int(len(labels)),
            "positives": int(labels.sum()),
            "genes": int(len(np.unique(groups))) if source == "clinvar" else None,
            "assays": int(len(np.unique(groups))) if source == "dms" else None,
        }
    return ExternalVectors(
        source=source,
        policy=policy,
        prefix=prefix,
        labels=labels,
        groups=groups,
        scores=scores,
        decisions=decisions,
        metadata=metadata,
    )


def _external_panel_title(vectors: ExternalVectors) -> str:
    metadata = vectors.metadata
    policy = vectors.policy.replace("_", "-")
    if vectors.source == "clinvar":
        evidence = metadata.get("clinical_evidence_assessment", {})
        guard = (
            " | UNDERPOWERED - DESCRIPTIVE ONLY"
            if evidence.get("status") == "underpowered"
            else ""
        )
        return (
            f"ClinVar later-snapshot | {policy}{guard}\n"
            f"n={metadata.get('n', len(vectors.labels))} unique variants, "
            f"pos={metadata.get('positives', int(vectors.labels.sum()))}, "
            f"genes={metadata.get('genes', len(np.unique(vectors.groups)))}"
        )
    return (
        f"DMS | {policy} | assay-macro\n"
        f"n={metadata.get('n', len(vectors.labels))} prepared rows, "
        f"pos={metadata.get('positives', int(vectors.labels.sum()))}, "
        f"assays={metadata.get('assays', len(np.unique(vectors.groups)))}"
    )


def _plot_dms_macro_metric(
    axis: plt.Axes,
    vectors: ExternalVectors,
    metric: str,
    title_suffix: str,
) -> None:
    per_assay = vectors.metadata.get("per_assay", {})
    macro = per_assay.get("macro", {})
    intervals = per_assay.get("macro_ci95", {})
    models = [
        name
        for name in MODEL_ORDER
        if name in macro and macro[name].get(metric) is not None
    ]
    if not models:
        axis.text(0.5, 0.5, "No assay-macro result", ha="center", va="center")
        axis.set_title(f"{_external_panel_title(vectors)}\n{title_suffix}")
        return
    values = np.asarray([macro[name][metric] for name in models], dtype=float)
    errors = []
    for name, value in zip(models, values):
        interval = intervals.get(name, {}).get(metric)
        errors.append(
            [
                max(0.0, value - interval[0]),
                max(0.0, interval[1] - value),
            ]
            if interval
            else [0.0, 0.0]
        )
    positions = np.arange(len(models))
    axis.bar(
        positions,
        values,
        color=[MODEL_COLORS[name] for name in models],
        yerr=np.asarray(errors).T,
        capsize=3,
    )
    axis.set_xticks(
        positions, [MODEL_LABELS[name] for name in models], rotation=30, ha="right"
    )
    axis.axhline(0.5 if metric == "auroc" else 0.0, color="0.5", linestyle="--")
    axis.set_ylabel("Assay-macro AUROC" if metric == "auroc" else "Assay-macro Spearman")
    axis.set_title(f"{_external_panel_title(vectors)}\n{title_suffix}")
    if metric == "auroc":
        axis.set_ylim(0, 1)
    else:
        lower = min(-0.1, float(values.min()) - 0.1)
        upper = max(0.1, float(values.max()) + 0.1)
        axis.set_ylim(max(-1.0, lower), min(1.0, upper))


def figure_external_performance() -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    clinvar = _external_vectors("clinvar", "exact_variant_disjoint")
    if clinvar is None:
        for axis in axes[0]:
            axis.text(0.5, 0.5, "No primary ClinVar set", ha="center", va="center")
            axis.set_title("ClinVar exact-variant-disjoint")
    else:
        title = _external_panel_title(clinvar)
        inference_allowed = bool(
            clinvar.metadata.get("clinical_evidence_assessment", {}).get(
                "inferential_model_comparison_allowed", True
            )
        )
        uncertainty_title = (
            "95% gene-to-variant bootstrap"
            if inference_allowed
            else "descriptive curve; uncertainty interval withheld"
        )
        _plot_roc(
            axes[0, 0],
            clinvar.labels,
            clinvar.scores,
            clinvar.groups,
            f"{title}\nROC ({uncertainty_title})",
            hierarchical_within_group=True,
            show_intervals=inference_allowed,
        )
        _plot_pr(
            axes[0, 1],
            clinvar.labels,
            clinvar.scores,
            clinvar.decisions,
            clinvar.groups,
            f"{title}\nPrecision-recall ({uncertainty_title})",
            hierarchical_within_group=True,
            show_intervals=inference_allowed,
        )

    dms = _external_vectors("dms", "exact_variant_disjoint")
    if dms is None:
        for axis in axes[1]:
            axis.text(0.5, 0.5, "No primary DMS set", ha="center", va="center")
            axis.set_title("DMS exact-variant-disjoint")
        dms_scope = ""
    else:
        _plot_dms_macro_metric(
            axes[1, 0], dms, "functional_spearman", "Functional ranking (95% assay CI)"
        )
        _plot_dms_macro_metric(
            axes[1, 1], dms, "auroc", "Damaging-label ranking (95% assay CI)"
        )
        coverage = dms.metadata.get("modality_coverage", {})
        structure_coverage = coverage.get("structure", {}).get("row_coverage", 0.0)
        conservation_coverage = coverage.get("conservation", {}).get(
            "row_coverage", 0.0
        )
        dms_scope = (
            " DMS is not full multimodal validation: "
            f"structure={structure_coverage:.0%}, "
            f"conservation={conservation_coverage:.0%}."
            if not dms.metadata.get("full_multimodal_validation", False)
            else ""
        )
    figure.suptitle(
        "Predeclared primary external sets: ClinVar unique variants and DMS assay-macro."
        + dms_scope
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def figure_external_clinvar_calibration() -> plt.Figure:
    vectors = _external_vectors("clinvar", "exact_variant_disjoint")
    if vectors is None:
        raise FileNotFoundError("No eligible ClinVar predictions exist")
    probabilities = {
        name: values
        for name, values in vectors.scores.items()
        if name in vectors.decisions
    }
    if not probabilities:
        raise ValueError("ClinVar has no calibrated probability outputs")
    calibration_claim_allowed = bool(
        vectors.metadata.get("clinical_evidence_assessment", {}).get(
            "clinical_calibration_claim_allowed", True
        )
    )
    title_suffix = (
        "Reliability (gene-to-variant bands)"
        if calibration_claim_allowed
        else "Descriptive reliability; calibration claim and bands withheld"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_calibration_panel(
        axes[0],
        axes[1],
        vectors.labels,
        probabilities,
        vectors.groups,
        f"{_external_panel_title(vectors)}\n{title_suffix}",
        hierarchical_within_group=True,
        show_intervals=calibration_claim_allowed,
    )
    figure.tight_layout()
    return figure


def figure_missing_modality_robustness() -> plt.Figure:
    """Plot the prespecified, descriptive missing-modality sensitivity curves."""
    payload = _load_json_object(EXTERNAL_RESULTS, "Stage 12 result")
    robustness = payload.get("robustness", {})
    if robustness.get("protocol_version") != PR.STRESS_PROTOCOL_VERSION:
        raise FileNotFoundError("No current missing-modality robustness result")
    sources = robustness.get("sources", {})
    available_sources = [name for name in ("clinvar", "dms") if name in sources]
    if not available_sources:
        raise FileNotFoundError("No missing-modality robustness source")
    scenarios = list(PR.STRESS_SCENARIOS)
    figure, axes = plt.subplots(
        len(available_sources),
        len(scenarios),
        figsize=(5.2 * len(scenarios), 4.2 * len(available_sources)),
        squeeze=False,
        sharex=True,
    )
    for row, source in enumerate(available_sources):
        source_report = sources[source]
        anchor = source_report.get("anchor_model")
        candidates = [
            name
            for name in MODEL_ORDER
            if name in source_report.get("candidate_models", [])
        ]
        metric = "auprc" if source == "clinvar" else "group_macro_functional_spearman"
        metric_label = "AUPRC" if source == "clinvar" else "Assay-macro Spearman"
        for column, scenario in enumerate(scenarios):
            axis = axes[row, column]
            scenario_report = source_report["scenarios"][scenario]
            levels = [
                scenario_report["levels"][_stress_level_key(level)]
                for level in PR.STRESS_LEVELS
            ]
            x = np.asarray(PR.STRESS_LEVELS, dtype=float) * 100.0
            names = ([anchor] if anchor else []) + candidates
            for name in names:
                values = [record["models"][name].get(metric) for record in levels]
                if any(value is None for value in values):
                    continue
                color = MODEL_COLORS.get(name, "0.4")
                axis.plot(
                    x,
                    values,
                    color=color,
                    marker="o",
                    linewidth=2.0 if name == C.RELIABILITY_ARCHITECTURE else 1.4,
                    label=MODEL_LABELS.get(name, name),
                )
            for candidate in candidates:
                values = [
                    record.get("safe_fallback_models", {})
                    .get(candidate, {})
                    .get(metric)
                    for record in levels
                ]
                if any(value is None for value in values):
                    continue
                axis.plot(
                    x,
                    values,
                    color=MODEL_COLORS.get(candidate, "0.4"),
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                    label=f"{MODEL_LABELS.get(candidate, candidate)} + safe fallback",
                )
            axis.set_title(scenario.replace("_", " ").title())
            axis.set_xlabel("Prespecified masked groups (%)")
            axis.set_ylabel(metric_label)
            axis.grid(axis="y", alpha=0.2)
            if scenario_report.get("status") == "not_applicable_no_source_values":
                axis.text(
                    0.5,
                    0.06,
                    "No source values: stress is non-informative",
                    transform=axis.transAxes,
                    ha="center",
                    fontsize=8,
                    color="0.35",
                )
            if row == 0 and column == len(scenarios) - 1:
                axis.legend(loc="best", fontsize=7)
    figure.suptitle(
        "Prespecified group-masked modality sensitivity (descriptive only; no "
        "external-label tuning or clinical-utility claim)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def figure_external_contextual_predictors() -> plt.Figure:
    if not EXTERNAL_RESULTS.exists():
        raise FileNotFoundError(EXTERNAL_RESULTS)
    payload = json.loads(EXTERNAL_RESULTS.read_text(encoding="utf-8"))
    set_name = "clinvar_exact_variant_disjoint"
    primary = payload.get("sets", {}).get(set_name, {})
    benchmark = primary.get("contextual_predictor_benchmark", {})
    common = benchmark.get("common_coverage", {})
    metrics_by_model = common.get("models", {})
    if not benchmark.get("available") or not metrics_by_model:
        raise FileNotFoundError("No common-coverage contextual predictor benchmark")
    preferred_order = [
        name
        for name in (*CONTEXTUAL_ORDER, *MODEL_ORDER)
        if name in metrics_by_model
    ]
    order = [
        *preferred_order,
        *sorted(set(metrics_by_model) - set(preferred_order)),
    ]
    fallback_colors = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, max(len(order), 1))
    )
    colors = {
        name: MODEL_COLORS.get(name, fallback_colors[index])
        for index, name in enumerate(order)
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    positions = np.arange(len(order))
    for axis, metric in zip(axes, ("auroc", "auprc")):
        values = np.asarray(
            [metrics_by_model[name].get(metric) for name in order], dtype=float
        )
        errors = []
        for name, value in zip(order, values):
            interval_payload = metrics_by_model[name].get("ci95") or {}
            interval = interval_payload.get(metric)
            errors.append(
                [
                    max(0.0, value - interval[0]),
                    max(0.0, interval[1] - value),
                ]
                if interval
                else [0.0, 0.0]
            )
        axis.bar(
            positions,
            values,
            yerr=np.asarray(errors).T,
            color=[colors[name] for name in order],
            capsize=2,
        )
        axis.set_xticks(
            positions,
            [MODEL_LABELS.get(name, name.removeprefix("context_")) for name in order],
            rotation=40,
            ha="right",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel(metric.upper())
        inference_allowed = bool(
            primary.get("clinical_evidence_assessment", {}).get(
                "inferential_model_comparison_allowed", True
            )
        )
        axis.set_title(
            f"Coverage-matched {metric.upper()} "
            + (
                "(gene-to-variant 95% CI)"
                if inference_allowed
                else "(descriptive only; interval withheld)"
            )
        )
        if metric == "auroc":
            axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
        else:
            prevalence = common.get("positives", 0) / max(common.get("n", 0), 1)
            axis.axhline(prevalence, color="0.5", linestyle="--", linewidth=1)
    figure.suptitle(
        "ClinVar contextual comparison | exact-variant-disjoint | "
        f"n={common.get('n')} unique variants, pos={common.get('positives')}, "
        f"genes={common.get('genes')}\n"
        + (
            "UNDERPOWERED - NO INFERENTIAL SUPERIORITY CLAIM.\n"
            if primary.get("clinical_evidence_assessment", {}).get("status")
            == "underpowered"
            else ""
        )
        + "All tools use identical coverage; established predictors are contextual "
        "comparators, not independent ACMG evidence or VariFuse inputs."
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def figure_shap_summary() -> plt.Figure:
    if not SHAP_FILE.exists():
        raise FileNotFoundError(SHAP_FILE)
    data = _load_npz(SHAP_FILE)
    shap_values = np.asarray(data["shap"], dtype=float)
    feature_values = np.asarray(data["X"], dtype=float)
    feature_names = np.asarray(data["features"], dtype=str)
    if shap_values.shape != feature_values.shape:
        raise ValueError("SHAP and feature matrices differ")
    importance = np.mean(np.abs(shap_values), axis=0)
    selected = np.argsort(importance)[-15:]
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].barh(
        np.arange(len(selected)),
        importance[selected],
        color="#0072B2",
    )
    axes[0].set_yticks(np.arange(len(selected)), feature_names[selected])
    axes[0].set_xlabel("Mean absolute held-out SHAP")
    axes[0].set_title("Fold-aggregated importance")
    random_state = np.random.RandomState(RANDOM_STATE)
    sample = np.arange(len(shap_values))
    if len(sample) > 1500:
        sample = random_state.choice(sample, 1500, replace=False)
    normalization = Normalize(vmin=-2.5, vmax=2.5)
    for row, feature_index in enumerate(selected):
        values = feature_values[sample, feature_index]
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            center = np.nanmedian(finite_values)
            spread = np.nanstd(finite_values)
            scaled = (values - center) / (spread if spread > 0 else 1.0)
        else:
            scaled = np.zeros_like(values)
        jitter = random_state.normal(0, 0.08, len(sample))
        axes[1].scatter(
            shap_values[sample, feature_index],
            row + jitter,
            c=scaled,
            cmap="coolwarm",
            norm=normalization,
            s=7,
            alpha=0.5,
            linewidths=0,
        )
    axes[1].axvline(0, color="0.4", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(selected)), feature_names[selected])
    axes[1].set_xlabel("Held-out SHAP value")
    axes[1].set_title("Direction and feature value")
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap="coolwarm"),
        ax=axes[1],
        fraction=0.045,
    )
    colorbar.set_label("Standardized feature value")
    figure.tight_layout()
    return figure


def _gene_aware_sample(frame: pd.DataFrame, valid: np.ndarray, limit: int) -> np.ndarray:
    candidates = frame.loc[valid, [GENE_COL, LABEL_COL]].copy()
    candidates["source_index"] = np.flatnonzero(valid)
    candidates = candidates.sample(frac=1.0, random_state=RANDOM_STATE)
    candidates = candidates.groupby([LABEL_COL, GENE_COL], sort=False).head(3)
    if len(candidates) <= limit:
        return candidates["source_index"].to_numpy(dtype=int)
    per_class = max(1, limit // 2)
    selected = []
    for label, group in candidates.groupby(LABEL_COL):
        count = min(per_class, len(group))
        selected.append(group.sample(count, random_state=RANDOM_STATE + int(label)))
    sampled = pd.concat(selected)
    if len(sampled) < limit:
        remaining = candidates.loc[~candidates.index.isin(sampled.index)]
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(
                    min(limit - len(sampled), len(remaining)),
                    random_state=RANDOM_STATE + 2,
                ),
            ]
        )
    return sampled["source_index"].to_numpy(dtype=int)


def figure_exploratory_embedding() -> plt.Figure:
    for path in (INTERNAL_CSV, INTERNAL_EMBEDDINGS):
        if not path.exists():
            raise FileNotFoundError(path)
    frame = pd.read_parquet(INTERNAL_CSV)
    embeddings = np.load(INTERNAL_EMBEDDINGS, mmap_mode="r")
    if len(frame) != len(embeddings):
        raise ValueError("Embedding rows differ from internal data")
    valid = (
        frame["ESM_EXTRACTION_SUCCESS"].eq(1).to_numpy()
        if "ESM_EXTRACTION_SUCCESS" in frame
        else np.ones(len(frame), dtype=bool)
    )
    sample = _gene_aware_sample(frame, valid, 2500)
    if len(sample) < 10:
        raise ValueError("Too few valid embeddings for projection")
    matrix = embeddings[sample].astype(np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    sample = sample[finite]
    if len(sample) < 10:
        raise ValueError("Too few finite sampled embeddings for projection")
    matrix = (matrix - matrix.mean(axis=0)) / np.where(
        matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0
    )
    components = min(50, matrix.shape[0] - 1, matrix.shape[1])
    reduced = PCA(n_components=components, random_state=RANDOM_STATE).fit_transform(
        matrix
    )
    method = "t-SNE"
    try:
        import umap

        projection = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.2,
            metric="cosine",
            random_state=RANDOM_STATE,
        ).fit_transform(reduced)
        method = "UMAP"
    except ImportError:
        perplexity = min(30, max(5, (len(sample) - 1) // 3))
        projection = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=RANDOM_STATE,
        ).fit_transform(reduced)
    labels = frame.iloc[sample][LABEL_COL].to_numpy(dtype=int)
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, color, text in ((0, "#0072B2", "Benign"), (1, "#D55E00", "Pathogenic")):
        selected = labels == label
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=9,
            alpha=0.45,
            color=color,
            label=text,
            linewidths=0,
        )
    axis.set(
        xlabel=f"{method} 1",
        ylabel=f"{method} 2",
        title=f"Exploratory {method} of gene-aware ESM sample",
    )
    axis.legend()
    figure.tight_layout()
    return figure


def _run_figure(
    manifest: dict[str, Any],
    name: str,
    required: bool,
    builder: Callable[[], plt.Figure],
) -> None:
    try:
        figure = builder()
        paths = _save_figure(figure, name)
        manifest[name] = {"status": "success", "required": required, "paths": paths}
        logger.info("Generated %s", name)
    except FileNotFoundError as error:
        manifest[name] = {
            "status": "failure" if required else "skipped",
            "required": required,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    except (KeyError, ValueError, RuntimeError, ImportError) as error:
        manifest[name] = {
            "status": "failure",
            "required": required,
            "error_type": type(error).__name__,
            "message": str(error),
        }


def _write_confirmatory_stability_table(
    stage14_validation: dict[str, Any],
) -> Path | None:
    """Write a compact descriptive table for prespecified confirmation repeats."""
    confirmation = stage14_validation.get("confirmatory_repeated_cv", {})
    if confirmation.get("requested") is not True:
        return None
    if confirmation.get("status") != "validated":
        raise RuntimeError("Cannot report unvalidated confirmatory repeated CV")
    rows: list[dict[str, Any]] = []
    for model, metrics in confirmation.get(
        "across_seed_metric_summary", {}
    ).items():
        for metric, summary in metrics.items():
            if not isinstance(summary, dict):
                continue
            rows.append(
                {
                    "record_type": "model_metric_across_prespecified_seeds",
                    "model_or_comparison": model,
                    "metric": metric,
                    "repeat_count": confirmation["repeat_count"],
                    "mean": summary["mean"],
                    "standard_deviation": summary["standard_deviation"],
                    "minimum": summary["minimum"],
                    "maximum": summary["maximum"],
                    "values": json.dumps(summary["values"]),
                    "scientific_role": confirmation["scientific_role"],
                    "used_for_model_selection": False,
                    "independence_note": (
                        "same variants across repeats; per-seed descriptive stability"
                    ),
                }
            )
    for comparison, metrics in confirmation.get(
        "paired_metric_difference_summary", {}
    ).items():
        for metric, summary in metrics.items():
            if not isinstance(summary, dict):
                continue
            rows.append(
                {
                    "record_type": "paired_metric_difference_across_seeds",
                    "model_or_comparison": comparison,
                    "metric": metric,
                    "repeat_count": confirmation["repeat_count"],
                    "mean": summary["mean"],
                    "standard_deviation": summary["standard_deviation"],
                    "minimum": summary["minimum"],
                    "maximum": summary["maximum"],
                    "values": json.dumps(summary["values"]),
                    "scientific_role": confirmation["scientific_role"],
                    "used_for_model_selection": False,
                    "independence_note": (
                        "same variants across repeats; per-seed paired effects"
                    ),
                }
            )
    if not rows:
        raise RuntimeError("Validated confirmatory summary contains no table rows")
    path = STAGE13_OUT / CONFIRMATORY_STABILITY_TABLE.name
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _write_mechanistic_reliability_table(
    stage12_validation: dict[str, Any],
) -> Path | None:
    """Write primary-set mechanism diagnostics with an explicit non-primary role."""
    reporting = stage12_validation.get("reliability_diagnostic_reporting", {})
    if reporting.get("status") is None:
        return None
    if (
        reporting.get("status")
        != "validated_descriptive_post_selection_not_primary"
        or reporting.get("strata_use_labels") is not False
        or reporting.get("used_for_model_or_threshold_selection") is not False
        or reporting.get("external_label_refitting") is not False
        or reporting.get("inferential_claim_allowed") is not False
        or reporting.get("not_primary_or_confirmatory") is not True
    ):
        raise RuntimeError("Cannot report unsafe reliability diagnostics")
    rows = reporting.get("primary_mechanistic_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Validated reliability diagnostics contain no primary rows")
    table = pd.DataFrame(rows)
    required_guard_columns = {
        "reporting_scope",
        "not_primary_or_confirmatory",
        "primary_endpoint",
        "confirmatory_evidence",
        "strata_use_labels",
        "used_for_model_selection",
        "used_for_threshold_selection",
        "external_labels_used_for_refitting",
        "inferential_claim_allowed",
        "interpretation",
    }
    if not required_guard_columns.issubset(table):
        raise RuntimeError("Mechanistic reliability table lacks reporting guards")
    if (
        not table["reporting_scope"]
        .eq("descriptive_post_selection_not_primary")
        .all()
        or not table["not_primary_or_confirmatory"].eq(True).all()
        or not table["primary_endpoint"].eq(False).all()
        or not table["confirmatory_evidence"].eq(False).all()
        or not table["strata_use_labels"].eq(False).all()
        or not table["used_for_model_selection"].eq(False).all()
        or not table["used_for_threshold_selection"].eq(False).all()
        or not table["external_labels_used_for_refitting"].eq(False).all()
        or not table["inferential_claim_allowed"].eq(False).all()
    ):
        raise RuntimeError("Mechanistic reliability table permits an unsafe claim")
    path = STAGE13_OUT / MECHANISTIC_RELIABILITY_TABLE.name
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def main() -> None:
    """Generate verified publication figures."""
    if (
        STAGE13_OUT.exists()
        and any(STAGE13_OUT.iterdir())
        and not ALLOW_EXISTING_FIGURE_DIR
    ):
        raise RuntimeError(
            f"Refusing to write Stage 13 into non-empty {STAGE13_OUT}. Use a new "
            "VARIANT_FIGURE_DIR, or set ALLOW_EXISTING_FIGURE_DIR=1 only when "
            "replacing that figure run is intentional."
        )
    ensure_directories(STAGE13_OUT)
    _atomic_json(
        {
            "stage": "13_generate_figures",
            "model_tag": MODEL_TAG,
            "extra": {"completed": False, "run_state": "validating_inputs"},
        },
        RUN_MANIFEST,
    )
    stage14_validation = _validate_stage14_artifacts()
    stage12_validation = _validate_stage12_artifacts(stage14_validation)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "model_tag": MODEL_TAG,
        "bootstrap_iterations": BOOTSTRAPS,
        "stage14_validation": stage14_validation,
        "stage12_validation": stage12_validation,
        "completed": False,
        "publication_complete": False,
        "figures": {},
        "tables": {},
    }
    confirmatory_table = _write_confirmatory_stability_table(stage14_validation)
    if confirmatory_table is not None:
        manifest["tables"]["confirmatory_stability"] = {
            "status": "success",
            "required": True,
            "path": confirmatory_table.name,
            "scientific_role": CONFIRMATORY_SCIENTIFIC_ROLE,
            "used_for_model_selection": False,
        }
    mechanistic_table = _write_mechanistic_reliability_table(stage12_validation)
    if mechanistic_table is not None:
        manifest["tables"]["mechanistic_reliability"] = {
            "status": "success",
            "required": True,
            "path": mechanistic_table.name,
            "scientific_role": "descriptive_post_selection_not_primary",
            "not_primary_or_confirmatory": True,
            "primary_endpoint": False,
            "confirmatory_evidence": False,
            "strata_use_labels": False,
            "used_for_model_or_threshold_selection": False,
            "external_label_refitting": False,
            "inferential_claim_allowed": False,
            "caption": (
                "Descriptive post-selection mechanism diagnostics on external "
                "sets; availability strata are label-independent. This table is "
                "neither a primary endpoint nor confirmatory evidence."
            ),
        }
    external_data = _load_npz(EXTERNAL_PREDICTIONS)
    clinvar_present = "clinvar_exact_variant_disjoint__y" in external_data
    clinvar_required = REQUIRE_EXTERNAL_CLINVAR or clinvar_present
    external_results = _load_json_object(EXTERNAL_RESULTS, "Stage 12 result")
    contextual_required = bool(
        external_results.get("sets", {})
        .get("clinvar_exact_variant_disjoint", {})
        .get("contextual_predictor_benchmark", {})
        .get("common_coverage", {})
        .get("models", {})
    )
    robustness_required = (
        external_results.get("robustness", {}).get("protocol_version")
        == PR.STRESS_PROTOCOL_VERSION
    )
    definitions = (
        ("internal_roc_pr", True, figure_internal_performance),
        ("internal_calibration", True, figure_internal_calibration),
        ("internal_model_comparison", True, figure_model_comparison),
        ("external_roc_pr", True, figure_external_performance),
        (
            "external_clinvar_calibration",
            clinvar_required,
            figure_external_clinvar_calibration,
        ),
        (
            "external_clinvar_contextual_predictors",
            contextual_required,
            figure_external_contextual_predictors,
        ),
        (
            "missing_modality_robustness",
            robustness_required,
            figure_missing_modality_robustness,
        ),
        ("heldout_shap_summary", False, figure_shap_summary),
        ("exploratory_esm_projection", False, figure_exploratory_embedding),
    )
    for name, required, builder in definitions:
        _run_figure(manifest["figures"], name, required, builder)
    failed = [
        name
        for name, record in manifest["figures"].items()
        if record["required"] and record["status"] != "success"
    ]
    completed = not failed
    publication_complete = bool(
        completed and stage12_validation.get("publication_complete") is True
    )
    manifest["completed"] = completed
    manifest["publication_complete"] = publication_complete
    manifest["failed_required_figures"] = failed
    _atomic_json(manifest, FIGURE_MANIFEST)
    output_paths = [FIGURE_MANIFEST]
    if confirmatory_table is not None:
        output_paths.append(confirmatory_table)
    if mechanistic_table is not None:
        output_paths.append(mechanistic_table)
    output_paths.extend(
        STAGE13_OUT / path
        for record in manifest["figures"].values()
        if record["status"] == "success"
        for path in record["paths"]
    )
    stage13_inputs = [
        INTERNAL_PREDICTIONS,
        INTERNAL_RESULTS,
        INTERNAL_SPLIT_PLAN,
        INTERNAL_RUN_MANIFEST,
        DEPLOYMENT_RESULTS,
        DEPLOYMENT_RUN_MANIFEST,
        EXTERNAL_PREDICTIONS,
        EXTERNAL_RESULTS,
        EXTERNAL_SUMMARY_TABLE,
        EXTERNAL_PREDICTION_TABLE,
        EXTERNAL_RELIABILITY_DIAGNOSTICS_TABLE,
        EXTERNAL_RUN_MANIFEST,
        EXTERNAL_PREP_MANIFEST,
        SHAP_FILE,
        INTERNAL_CSV,
        INTERNAL_EMBEDDINGS,
        INTERNAL_STATUS,
        *[
            path
            for paths in EXTERNAL_STAGE10_INPUTS.values()
            for path in paths
        ],
    ]
    if confirmatory_table is not None:
        stage13_inputs.extend(
            [
                CONFIRMATORY_SEED_PLAN,
                CONFIRMATORY_RESULTS,
                CONFIRMATORY_PREDICTIONS,
                CONFIRMATORY_FOLD_ASSIGNMENTS,
                CONFIRMATORY_CHECKPOINT_DIR,
            ]
        )
    write_run_manifest(
        RUN_MANIFEST,
        "13_generate_figures",
        stage13_inputs,
        {
            "model_tag": MODEL_TAG,
            "formats": FORMATS,
            "bootstrap_iterations": BOOTSTRAPS,
            "external_primary_policy": "exact_variant_disjoint",
            "clinvar_analysis_unit": "unique_genomic_variant",
            "dms_primary_endpoints": ["assay_macro_spearman", "assay_macro_auroc"],
            "gene_disjoint_figure_selection": "never_automatic",
            "contextual_predictor_policy": (
                "common-coverage rank comparison; never used as model inputs"
            ),
            "internal_primary_results_source": "stage14_nested_outer_oof",
            "stage14_validation": stage14_validation,
            "confirmatory_stability_table": (
                confirmatory_table.name if confirmatory_table is not None else None
            ),
            "mechanistic_reliability_table": (
                mechanistic_table.name if mechanistic_table is not None else None
            ),
            "mechanistic_reliability_reporting_role": (
                "descriptive_post_selection_not_primary_not_confirmatory"
                if mechanistic_table is not None
                else None
            ),
            "mechanistic_reliability_claim_guard": (
                {
                    "primary_endpoint": False,
                    "confirmatory_evidence": False,
                    "strata_use_labels": False,
                    "used_for_model_or_threshold_selection": False,
                    "external_label_refitting": False,
                    "inferential_claim_allowed": False,
                    "causal_attribution_allowed": False,
                }
                if mechanistic_table is not None
                else None
            ),
            "stage12_validation": stage12_validation,
            "stage11_role": "deployment_and_explanation_only_post_selection",
            "robustness_protocol_version": (
                PR.STRESS_PROTOCOL_VERSION if robustness_required else None
            ),
            "completed": completed,
            "publication_complete": publication_complete,
            "failed_required_figures": failed,
            "figure_status": {
                name: record["status"]
                for name, record in manifest["figures"].items()
            },
            "table_status": {
                name: record["status"]
                for name, record in manifest["tables"].items()
            },
        },
        outputs=output_paths,
    )
    if failed:
        raise RuntimeError(f"Required figures failed: {failed}")
    logger.info("All required figures were generated")


if __name__ == "__main__":
    main()
