from __future__ import annotations

import json
import gc
import itertools
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightgbm import Booster
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests

import common as C
import publication_robustness as PR
from clinvar_identity import normalize_variation_ids
from config import (
    ENABLE_LORA,
    LABEL_TASK,
    MODEL_TAG,
    RANDOM_STATE,
    STAGE09_OUT,
    STAGE10_OUT,
    STAGE11_OUT,
    STAGE12_OUT,
    STAGE14_OUT,
    STAGE07_OUT,
    TUNING_BEST_JSON,
    ensure_directories,
    json_default,
    validate_manifest_input_bindings,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import MUTATION_FEATURE_COLS, select_tabular_features
from table_io import iter_table, table_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage12_external")

INTERNAL_CSV = STAGE10_OUT / "internal_with_esm.parquet"
INTERNAL_NPY = STAGE10_OUT / "internal_esm_embeddings.npy"
INTERNAL_STATUS = STAGE10_OUT / "internal_esm_extraction.parquet"
INTERNAL_UNIVERSE = STAGE07_OUT / "Final_Dataset_Natural_Prevalence.parquet"
INTERNAL_OOF = STAGE11_OUT / "oof_predictions.npz"
MODEL_DIR = STAGE11_OUT / "models"
RESULTS_FILE = STAGE12_OUT / "external_validation.json"
TABLE_FILE = STAGE12_OUT / "external_validation_table.csv"
PREDICTION_FILE = STAGE12_OUT / "external_predictions.npz"
PREDICTION_TABLE = STAGE12_OUT / "external_predictions.csv"
RELIABILITY_DIAGNOSTICS_TABLE = STAGE12_OUT / "reliability_diagnostics.csv"
MANIFEST_FILE = STAGE12_OUT / "run_manifest.json"
N_FOLDS = 5
BOOTSTRAP_ITERATIONS = 1000
ZERO_SHOT_MODEL = "esm_zero_shot"
MASKED_MARGINAL_COL = "esm_variant_score_masked_marginal"
EXTERNAL_PREP_MANIFEST = STAGE09_OUT / "run_manifest.json"
STAGE09_PREPARED_OUTPUTS = {
    "clinvar": STAGE09_OUT / "clinvar_esm_ready.csv",
    "dms": STAGE09_OUT / "dms_esm_ready.csv",
}
DMS_SEQUENCE_OUTPUT = STAGE09_OUT / "dms_sequences.parquet"
STAGE10_MANIFESTS = {
    "internal": STAGE10_OUT / "internal_esm_manifest.json",
    "clinvar": STAGE10_OUT / "clinvar_esm_manifest.json",
    "dms": STAGE10_OUT / "dms_esm_manifest.json",
}
STAGE07_MANIFEST = STAGE07_OUT / "run_manifest.json"
STAGE11_MANIFEST = STAGE11_OUT / "run_manifest.json"
STAGE14_MANIFEST = STAGE14_OUT / "run_manifest.json"
MODEL_DISCOVERY_ORDER = (
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
)
DEEP_MODEL_NAMES = (
    "concatenation",
    "gated_fusion",
    *C.RELIABILITY_FAMILY_ARCHITECTURES,
    "cross_attention",
)
PUBLICATION_BASELINE_NAMES = {
    "esm_score_logistic",
    "conservation_logistic",
    "esm_conservation_logistic",
    "mutation_logistic",
    "availability_logistic",
}
CONTEXTUAL_PREDICTOR_DIRECTIONS = {
    "SIFT_score": {
        "multiplier": -1.0,
        "source_orientation": "lower_is_more_deleterious",
    },
    "Polyphen2_HDIV_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "CADD_phred": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "REVEL_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "CONSENSUS_SCORE": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    # Modern dbNSFP predictors are evaluation-only contextual comparators.  The
    # aliases cover column names used by recent dbNSFP releases without making
    # any of them a VariFuse training feature.
    "AlphaMissense_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "EVE_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "PrimateAI_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "PrimateAI-3D_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "MetaRNN_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "ClinPred_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "BayesDel_addAF_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "BayesDel_noAF_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "MPC_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "VEST4_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "MutPred2_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "DEOGEN2_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "LIST-S2_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "VARITY_R_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "VARITY_ER_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "VARITY_R_LOO_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
    "VARITY_ER_LOO_score": {
        "multiplier": 1.0,
        "source_orientation": "higher_is_more_deleterious",
    },
}


def _model_reporting_role(name: str) -> str:
    if name == "availability_logistic":
        return "diagnostic_missingness_availability_negative_control"
    if name == "cross_attention":
        return "legacy_pooled_vector_pseudo_slot_comparator"
    if name == "lora_esm":
        return "optional_exploratory_model"
    if name == "reliability_residual":
        return "candidate_reliability_conditioned_residual_fusion_model"
    if name == "evidential_residual":
        return "candidate_availability_weighted_expert_residual_fusion_model"
    if name in {
        ZERO_SHOT_MODEL,
        "raw_esm_zero_shot",
        "esm_score_logistic",
        "conservation_logistic",
        "esm_conservation_logistic",
        "mutation_logistic",
        "esm_embedding_mutation",
        "esm_only",
        "lightgbm",
    }:
        return "prespecified_reference_baseline"
    return "candidate_fusion_model"


@dataclass
class DatasetPair:
    name: str
    frame: pd.DataFrame
    embeddings: np.ndarray
    extraction_coverage: float


@dataclass
class EvaluationUnit:
    frame: pd.DataFrame
    probabilities: dict[str, np.ndarray]
    decisions: dict[str, np.ndarray]
    ranking_scores: dict[str, np.ndarray]
    audit: dict[str, Any]
    decision_confidence: dict[str, np.ndarray] | None = None
    reliability_components: dict[str, np.ndarray] | None = None
    reliability_anchor_decisions: np.ndarray | None = None


EXTERNAL_INPUTS = {
    "clinvar": (
        STAGE10_OUT / "clinvar_with_esm.parquet",
        STAGE10_OUT / "clinvar_esm_embeddings.npy",
        STAGE10_OUT / "clinvar_esm_extraction.parquet",
    ),
    "dms": (
        STAGE10_OUT / "dms_with_esm.parquet",
        STAGE10_OUT / "dms_esm_embeddings.npy",
        STAGE10_OUT / "dms_esm_extraction.parquet",
    ),
}


def _validate_upstream_chain(external_sources: list[str]) -> list[Path]:
    """Authenticate every model/data byte consumed by external validation."""
    validate_upstream_manifest(
        STAGE07_MANIFEST,
        "07_dataset_balancing",
        [INTERNAL_UNIVERSE],
    )
    validate_upstream_manifest(
        STAGE10_MANIFESTS["internal"],
        "10_extract_esm_features:internal",
        [INTERNAL_CSV, INTERNAL_NPY, INTERNAL_STATUS],
        required_source_files=("gpu_runtime.py",),
    )
    model_artifacts = sorted(path for path in MODEL_DIR.rglob("*") if path.is_file())
    if not model_artifacts:
        raise FileNotFoundError(f"No Stage 11 model artifacts found in {MODEL_DIR}")
    validate_upstream_manifest(
        STAGE11_MANIFEST,
        "11_train_and_evaluate",
        [INTERNAL_OOF, STAGE11_OUT / "results.json", *model_artifacts],
        required_source_files=("common.py", "gpu_runtime.py"),
    )
    stage14_artifacts = [
        TUNING_BEST_JSON,
        STAGE14_OUT / "best_concatenation_params.json",
        STAGE14_OUT / "best_gated_fusion_params.json",
        STAGE14_OUT / "architecture_selection.json",
        STAGE14_OUT / "nested_inner_splits.json",
    ]
    for architecture in C.RELIABILITY_FAMILY_ARCHITECTURES:
        tuning_path = STAGE14_OUT / f"best_{architecture}_params.json"
        if tuning_path.is_file():
            stage14_artifacts.append(tuning_path)
    validate_upstream_manifest(
        STAGE14_MANIFEST,
        "14_tune_cross_attention",
        stage14_artifacts,
        required_source_files=("common.py", "gpu_runtime.py"),
    )
    stage09_artifacts = [STAGE09_PREPARED_OUTPUTS[source] for source in external_sources]
    if "dms" in external_sources and DMS_SEQUENCE_OUTPUT.is_file():
        stage09_artifacts.append(DMS_SEQUENCE_OUTPUT)
    validate_upstream_manifest(
        EXTERNAL_PREP_MANIFEST,
        "09_prepare_external_esm_dataset",
        stage09_artifacts,
        required_source_files=(
            "01_dbnsfp_processor.py",
            "04_feature_engineering.py",
            "08_prepare_esm_dataset.py",
        ),
    )
    for source in external_sources:
        validate_upstream_manifest(
            STAGE10_MANIFESTS[source],
            f"10_extract_esm_features:{source}",
            EXTERNAL_INPUTS[source],
            required_source_files=("gpu_runtime.py",),
        )
        prepared_inputs = [EXTERNAL_PREP_MANIFEST, STAGE09_PREPARED_OUTPUTS[source]]
        if source == "dms":
            prepared_inputs.append(DMS_SEQUENCE_OUTPUT)
        validate_manifest_input_bindings(
            STAGE10_MANIFESTS[source],
            f"10_extract_esm_features:{source}",
            prepared_inputs,
        )
    return model_artifacts


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")
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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_pair(
    name: str,
    csv_path: Path,
    npy_path: Path,
    status_path: Path,
    require_both_classes: bool = True,
) -> DatasetPair:
    for path in (csv_path, npy_path, status_path):
        if not path.exists():
            raise FileNotFoundError(path)
    frame = (
        pd.read_parquet(csv_path)
        if csv_path.suffix.lower() == ".parquet"
        else pd.read_csv(csv_path, low_memory=False)
    ).reset_index(drop=True)
    embeddings = np.load(npy_path, mmap_mode="r")
    statuses = (
        pd.read_parquet(status_path)
        if status_path.suffix.lower() == ".parquet"
        else pd.read_csv(status_path, low_memory=False)
    )
    if len(frame) != len(embeddings) or len(frame) != len(statuses):
        raise ValueError(
            f"{name} row mismatch: CSV={len(frame)}, NPY={len(embeddings)}, status={len(statuses)}"
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != C.ESM_DIM:
        raise ValueError(f"{name} has unexpected embedding shape {embeddings.shape}")
    required_columns = {
        C.ROW_ID_COL,
        C.LABEL_COL,
        C.GENE_COL,
        "esm_variant_score",
        "ESM_EXTRACTION_SUCCESS",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(f"{name} misses columns: {sorted(missing_columns)}")
    if C.ROW_ID_COL not in frame or C.ROW_ID_COL not in statuses:
        raise KeyError(f"{name} is missing row identifiers")
    frame_ids = frame[C.ROW_ID_COL].astype(str)
    status_ids = statuses[C.ROW_ID_COL].astype(str)
    if not frame_ids.equals(status_ids):
        raise ValueError(f"{name} extraction status order differs from its CSV")
    if frame_ids.duplicated().any() or frame_ids.str.strip().eq("").any():
        raise ValueError(f"{name} row identifiers are invalid")
    labels = pd.to_numeric(frame[C.LABEL_COL], errors="coerce")
    if labels.isna().any() or frame.empty:
        raise ValueError(f"{name} labels are empty or missing")
    frame[C.LABEL_COL] = C.validate_binary_labels(
        labels.to_numpy(), f"{name} labels", require_both_classes=require_both_classes
    )
    if C.GENE_COL not in frame or frame[C.GENE_COL].isna().any():
        raise ValueError(f"{name} requires complete gene identifiers")
    scores = pd.to_numeric(frame.get("esm_variant_score"), errors="coerce")
    success = np.isfinite(embeddings).all(axis=1) & np.isfinite(scores)
    if "ESM_EXTRACTION_SUCCESS" in frame:
        declared = frame["ESM_EXTRACTION_SUCCESS"].fillna(0).astype(bool).to_numpy()
        if not np.array_equal(success, declared):
            raise ValueError(f"{name} ESM success flags disagree with embeddings")
    return DatasetPair(
        name=name,
        frame=frame,
        embeddings=embeddings,
        extraction_coverage=float(success.mean()),
    )


def _normal_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.upper()


def _integer_text(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string").fillna("")


def _genomic_keys(frame: pd.DataFrame) -> pd.Series:
    required = {"chr", "pos", "ref", "alt"}
    if not required.issubset(frame.columns):
        return pd.Series("", index=frame.index, dtype="string")
    chromosome = _normal_text(frame["chr"]).str.removeprefix("CHR")
    position = _integer_text(frame["pos"])
    reference = _normal_text(frame["ref"])
    alternate = _normal_text(frame["alt"])
    valid = chromosome.ne("") & position.ne("") & reference.ne("") & alternate.ne("")
    keys = chromosome + ":" + position + ":" + reference + ":" + alternate
    return keys.where(valid, "")


def _protein_keys(frame: pd.DataFrame) -> pd.Series:
    required = {"aa_pos", "aa_ref", "aa_alt", C.GENE_COL}
    if not required.issubset(frame.columns):
        return pd.Series("", index=frame.index, dtype="string")
    accession = (
        _normal_text(frame["uniprot_id"])
        if "uniprot_id" in frame
        else _normal_text(frame[C.GENE_COL])
    )
    position = _integer_text(frame["aa_pos"])
    reference = _normal_text(frame["aa_ref"])
    alternate = _normal_text(frame["aa_alt"])
    valid = accession.ne("") & position.ne("") & reference.ne("") & alternate.ne("")
    keys = accession + ":" + reference + position + alternate
    return keys.where(valid, "")


def _deoverlap_masks(internal_universe: Path, external: pd.DataFrame) -> dict[str, np.ndarray]:
    """Remove overlap against the full labelled universe, not the ESM sample."""
    if not internal_universe.exists():
        raise FileNotFoundError(internal_universe)
    external_genomic = _genomic_keys(external)
    external_protein = _protein_keys(external)
    external_stable_ids = normalize_variation_ids(
        external.get(
            "CLINVAR_VARIATION_ID",
            pd.Series(pd.NA, index=external.index, dtype="string"),
        )
    )
    if "EXT_SOURCE" in external:
        clinvar_rows = _normal_text(external["EXT_SOURCE"]).eq("CLINVAR")
        if clinvar_rows.any() and external_stable_ids.loc[clinvar_rows].isna().any():
            raise RuntimeError(
                "External ClinVar rows require valid stable VariationIDs for publication de-overlap"
            )
    stable_external = external_stable_ids.notna()
    if stable_external.any():
        stable_identity = pd.DataFrame(
            {
                "stable_id": external_stable_ids.loc[stable_external],
                "genomic_key": external_genomic.loc[stable_external],
            }
        )
        if stable_identity["genomic_key"].eq("").any():
            raise RuntimeError("External ClinVar stable identities require genomic coordinates")
        if (
            stable_identity.groupby("stable_id", sort=False)["genomic_key"].nunique().gt(1).any()
            or stable_identity.groupby("genomic_key", sort=False)["stable_id"].nunique().gt(1).any()
        ):
            raise RuntimeError(
                "External ClinVar contains an ambiguous stable-ID/genomic-key mapping"
            )
    genomic_targets = set(external_genomic) - {""}
    protein_targets = set(external_protein) - {""}
    stable_id_targets = set(external_stable_ids.dropna().astype(str))
    gene_targets = set(_normal_text(external[C.GENE_COL])) - {""}
    genomic_overlap: set[str] = set()
    protein_overlap: set[str] = set()
    stable_id_overlap: set[str] = set()
    internal_stable_ids_seen: set[str] = set()
    shared_genes: set[str] = set()
    header = table_columns(internal_universe)
    wanted = {
        "chr",
        "pos",
        "ref",
        "alt",
        "CLINVAR_VARIATION_ID",
        C.GENE_COL,
        "uniprot_id",
        "aapos",
        "aaref",
        "aaalt",
    }
    if LABEL_TASK == "clinical" and "CLINVAR_VARIATION_ID" not in header:
        raise RuntimeError(
            "Internal clinical universe lacks stable ClinVar VariationID provenance; "
            "rerun from Stage 01"
        )
    usecols = [column for column in header if column in wanted]
    for chunk in iter_table(internal_universe, 100_000, columns=usecols):
        normalized = chunk.rename(columns={"aapos": "aa_pos", "aaref": "aa_ref", "aaalt": "aa_alt"})
        genomic_overlap.update(set(_genomic_keys(normalized)) & genomic_targets)
        protein_overlap.update(set(_protein_keys(normalized)) & protein_targets)
        internal_stable_ids = normalize_variation_ids(
            normalized.get(
                "CLINVAR_VARIATION_ID",
                pd.Series(pd.NA, index=normalized.index, dtype="string"),
            )
        )
        if LABEL_TASK == "clinical" and internal_stable_ids.isna().any():
            raise RuntimeError(
                "Internal clinical universe contains missing or invalid stable ClinVar VariationIDs"
            )
        observed_internal_ids = set(internal_stable_ids.dropna().astype(str))
        if len(observed_internal_ids) != int(internal_stable_ids.notna().sum()):
            raise RuntimeError("Internal clinical universe repeats a stable ClinVar VariationID")
        repeated_across_chunks = observed_internal_ids & internal_stable_ids_seen
        if repeated_across_chunks:
            raise RuntimeError(
                "Internal clinical universe repeats stable ClinVar VariationIDs across chunks"
            )
        internal_stable_ids_seen.update(observed_internal_ids)
        stable_id_overlap.update(set(internal_stable_ids.dropna().astype(str)) & stable_id_targets)
        shared_genes.update(set(_normal_text(normalized[C.GENE_COL])) & gene_targets)
    exact_overlap = (
        external_genomic.isin(genomic_overlap)
        | external_protein.isin(protein_overlap)
        | external_stable_ids.astype("string").isin(stable_id_overlap)
    )
    exact_disjoint = ~exact_overlap.to_numpy()
    shared_gene = _normal_text(external[C.GENE_COL]).isin(shared_genes).to_numpy()
    gene_disjoint = exact_disjoint & ~shared_gene

    # ClinVar may have several transcript/protein annotation rows for one genomic
    # variant. A variant is disjoint only when every one of its mappings is
    # disjoint; otherwise filtering first could retain a convenient mapping and
    # hide overlap in another mapping.
    valid_genomic = external_genomic.ne("")
    if valid_genomic.any():
        for values in (exact_disjoint, gene_disjoint):
            series = pd.Series(values, index=external.index)
            propagated = (
                series.loc[valid_genomic]
                .groupby(external_genomic.loc[valid_genomic], sort=False)
                .transform("all")
            )
            values[np.flatnonzero(valid_genomic.to_numpy())] = propagated.to_numpy(dtype=bool)
    return {
        "exact_variant_disjoint": exact_disjoint,
        "gene_disjoint": gene_disjoint,
    }


def _esm_zero_shot_scores(frame: pd.DataFrame) -> np.ndarray | None:
    """Return the raw masked-marginal score oriented toward pathogenicity.

    ESM produces log P(alt)-log P(reference); more negative values indicate a
    less sequence-compatible alternate residue. The sign is reversed solely so
    that higher values consistently mean greater predicted deleteriousness.
    No sigmoid or post-hoc calibration is applied.
    """
    if MASKED_MARGINAL_COL not in frame:
        return None
    masked_marginal = pd.to_numeric(frame[MASKED_MARGINAL_COL], errors="coerce").to_numpy(
        dtype=float
    )
    return -masked_marginal


def _contextual_predictor_scores(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Discover evaluation-only comparator scores without using them as inputs."""
    scores: dict[str, np.ndarray] = {}
    status: dict[str, Any] = {}
    for column, specification in CONTEXTUAL_PREDICTOR_DIRECTIONS.items():
        if column not in frame:
            status[column] = {"available": False, "reason": "column_absent"}
            continue
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        oriented = raw * float(specification["multiplier"])
        finite = np.isfinite(oriented)
        status[column] = {
            "available": bool(finite.any()),
            "finite_rows": int(finite.sum()),
            "total_rows": int(len(frame)),
            "row_coverage": round(float(finite.mean()), 6) if len(frame) else 0.0,
            "source_orientation": specification["source_orientation"],
            "evaluation_orientation": "higher_is_more_deleterious",
            "model_input": False,
            "interpretation": "contextual_comparator_not_independent_ACMG_evidence",
        }
        if finite.any():
            scores[f"context_{column}"] = oriented
    return scores, status


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    available = [column for column in columns if column in frame]
    if not available:
        return np.zeros(len(frame), dtype=bool)
    values = frame[available].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.isfinite(values).any(axis=1)


def _modality_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit source-valued modality coverage before missing-value imputation."""
    esm = frame["ESM_EXTRACTION_SUCCESS"].fillna(0).astype(bool).to_numpy()
    masked = _esm_zero_shot_scores(frame)
    if masked is not None:
        esm &= np.isfinite(masked)
    structure_columns = (
        "SASA",
        "RELATIVE_SASA",
        "PLDDT_SCORE",
        "LOCAL_CONTACT_COUNT_8A",
        "LOCAL_CONTACT_COUNT_12A",
        "LOCAL_LONG_RANGE_CONTACT_COUNT_8A",
        "LOCAL_MEAN_PLDDT_8A",
        "LOCAL_MIN_PLDDT_8A",
        "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
        "LOCAL_MEAN_DISTANCE_8A",
        "LOCAL_HYDROPHOBIC_FRACTION_8A",
        "LOCAL_CHARGED_FRACTION_8A",
    )
    structure = _finite_numeric(frame, structure_columns)
    if "HAS_STRUCTURE" in frame:
        declared_structure = (
            pd.to_numeric(frame["HAS_STRUCTURE"], errors="coerce").fillna(0).gt(0).to_numpy()
        )
        structure &= declared_structure
    conservation_columns = (
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
    )
    conservation = _finite_numeric(frame, conservation_columns)
    complete = esm & structure & conservation
    masks = {
        "esm_masked_marginal": esm,
        "structure": structure,
        "conservation": conservation,
        "complete_sequence_structure_conservation": complete,
    }
    report: dict[str, Any] = {
        name: {
            "n": int(values.sum()),
            "row_coverage": round(float(values.mean()), 6) if len(values) else 0.0,
        }
        for name, values in masks.items()
    }
    for modality, columns in (
        ("structure", structure_columns),
        ("conservation", conservation_columns),
    ):
        report[modality]["feature_coverage"] = {
            column: {
                "available_column": column in frame,
                "finite_n": (
                    int(
                        np.isfinite(
                            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                        ).sum()
                    )
                    if column in frame
                    else 0
                ),
                "row_coverage": (
                    round(
                        float(
                            np.isfinite(
                                pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                            ).mean()
                        ),
                        6,
                    )
                    if column in frame and len(frame)
                    else 0.0
                ),
            }
            for column in columns
        }
    missing = [name for name in ("structure", "conservation") if report[name]["n"] == 0]
    report["missing_modalities"] = missing
    report["full_multimodal_validation"] = bool(len(frame) and complete.all())
    report["interpretation"] = (
        "all rows contain sequence, structure, and conservation source values"
        if report["full_multimodal_validation"]
        else "models are evaluated with one or more missing/imputed modalities"
    )
    return report


def _esm_only_matrix(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[np.ndarray, list[str]]:
    extra_columns = [
        column
        for column in ("esm_variant_score", *MUTATION_FEATURE_COLS)
        if column in frame.columns
    ]
    extras = (
        frame[extra_columns].to_numpy(dtype=np.float32)
        if extra_columns
        else np.empty((len(frame), 0), dtype=np.float32)
    )
    names = [f"esm_embedding_{index}" for index in range(embeddings.shape[1])]
    return np.column_stack([embeddings, extras]), [*names, *extra_columns]


def _validate_internal_reference(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if not INTERNAL_OOF.exists():
        raise FileNotFoundError(INTERNAL_OOF)
    with np.load(INTERNAL_OOF, allow_pickle=True) as stored:
        row_key = f"{MODEL_TAG}__row_ids"
        fold_key = f"{MODEL_TAG}__fold_ids"
        if row_key not in stored or fold_key not in stored:
            raise KeyError("Internal OOF artifact lacks row or fold identifiers")
        if list(frame[C.ROW_ID_COL].astype(str)) != list(stored[row_key].astype(str)):
            raise ValueError("Internal OOF rows differ from Stage 10 input")
        fold_ids = stored[fold_key].astype(int)
        prefix = f"{MODEL_TAG}__"
        reserved = {"y", "groups", "row_ids", "fold_ids"}
        discovered = {
            key[len(prefix) :]
            for key in stored.files
            if key.startswith(prefix)
            and "__" not in key[len(prefix) :]
            and key[len(prefix) :] not in reserved
        }
        model_names = [name for name in MODEL_DISCOVERY_ORDER if name in discovered]
        model_names.extend(sorted(discovered - set(model_names)))
        if ENABLE_LORA:
            if f"{MODEL_TAG}__lora_esm" not in stored:
                raise KeyError("LoRA is enabled but missing from internal OOF results")
            if "lora_esm" not in model_names:
                model_names.append("lora_esm")
    if set(np.unique(fold_ids)) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Internal OOF fold identifiers are incomplete")
    if "lightgbm" not in model_names:
        raise ValueError("Internal OOF artifact lacks the required LightGBM reference")
    if not any(
        name in model_names for name in ("concatenation", "gated_fusion", "cross_attention")
    ):
        raise ValueError("Internal OOF artifact lacks a multimodal model")
    return fold_ids, model_names


def _validate_feature_schema(
    internal: pd.DataFrame, external: dict[str, DatasetPair]
) -> tuple[list[str], list[str]]:
    feature_names = C.select_features(internal)
    tabular_features = select_tabular_features(feature_names)
    for source, pair in external.items():
        missing = [column for column in feature_names if column not in pair.frame]
        if missing:
            raise ValueError(f"{source} misses model features: {missing}")
        selected = C.select_features(pair.frame)
        if selected != feature_names:
            raise ValueError(f"{source} feature order differs from internal schema: {selected}")
    return feature_names, tabular_features


def _predict_fold(
    fold: int,
    pair: DatasetPair,
    feature_names: list[str],
    tabular_features: list[str],
    model_names: list[str],
    reliability_components: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    fold_dir = MODEL_DIR / f"fold_{fold}"
    metadata_path = fold_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fold_features = metadata.get("feature_names", feature_names)
    fold_tabular = metadata.get("tabular_features", tabular_features)
    for role, names in (("model", fold_features), ("tabular", fold_tabular)):
        missing = [name for name in names if name not in pair.frame]
        if missing:
            raise ValueError(f"Fold {fold} {role} features are unavailable: {missing}")
    predictions: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    if "lightgbm" in model_names:
        tabular = pair.frame[fold_tabular].to_numpy(dtype=np.float32)
        lightgbm_path = fold_dir / "lightgbm.txt"
        calibrator_path = fold_dir / "lightgbm_calibrator.pkl"
        if not lightgbm_path.exists() or not calibrator_path.exists():
            raise FileNotFoundError(f"Missing LightGBM fold {fold} artifacts")
        lightgbm = Booster(model_file=str(lightgbm_path))
        calibrator = _load_pickle(calibrator_path)
        predictions["lightgbm"] = calibrator.predict(lightgbm.predict(tabular))
        thresholds["lightgbm"] = float(metadata["thresholds"]["lightgbm"])

    if "raw_esm_zero_shot" in model_names:
        raw_bundle = _load_pickle(fold_dir / "raw_esm_zero_shot.pkl")
        score_column = str(raw_bundle.get("score_column", "esm_variant_score"))
        if score_column not in pair.frame:
            raise KeyError(f"Fold {fold} raw ESM score column {score_column} is absent")
        raw_values = pd.to_numeric(pair.frame[score_column], errors="coerce").to_numpy(dtype=float)
        raw_values *= float(raw_bundle.get("pathogenic_direction", -1.0))
        if not np.isfinite(raw_values).all():
            raise ValueError(f"Fold {fold} raw ESM score is incomplete")
        raw_calibrator = raw_bundle["calibrator"]
        if hasattr(raw_calibrator, "predict_from_logits"):
            raw_probabilities = raw_calibrator.predict_from_logits(raw_values)
        else:
            raw_probabilities = raw_calibrator.predict(raw_values)
        predictions["raw_esm_zero_shot"] = raw_probabilities
        thresholds["raw_esm_zero_shot"] = float(raw_bundle["threshold"])

    publication_path = fold_dir / "publication_baselines.pkl"
    requested_publication = PUBLICATION_BASELINE_NAMES & set(model_names)
    if requested_publication:
        if not publication_path.exists():
            raise FileNotFoundError(publication_path)
        publication_bundles = _load_pickle(publication_path)
        for name in sorted(requested_publication):
            if name not in publication_bundles:
                raise KeyError(f"Fold {fold} publication bundle lacks {name}")
            bundle = publication_bundles[name]
            columns = bundle["feature_names"]
            missing = [column for column in columns if column not in pair.frame]
            if missing:
                raise KeyError(f"Fold {fold} {name} features are absent: {missing}")
            values = pair.frame[columns].to_numpy(dtype=np.float32)
            scaled = bundle["preprocessor"].transform(values)
            raw_probability = bundle["model"].predict_proba(scaled)[:, 1]
            predictions[name] = bundle["calibrator"].predict(raw_probability)
            thresholds[name] = float(bundle["threshold"])

    embedding_artifacts = {
        "esm_embedding_mutation": fold_dir / "esm_embedding_mutation.pkl",
        "esm_only": fold_dir / "esm_only.pkl",
    }
    for name, path in embedding_artifacts.items():
        if name not in model_names:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        esm_bundle = _load_pickle(path)
        chunk_size = 35000
        n_rows = len(pair.frame)
        if n_rows > chunk_size:
            raw_list = []
            for start_idx in range(0, n_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, n_rows)
                chunk_matrix, esm_names = _esm_only_matrix(
                    pair.frame.iloc[start_idx:end_idx], pair.embeddings[start_idx:end_idx]
                )
                if esm_bundle["feature_names"] != esm_names:
                    raise ValueError(f"Fold {fold} {name} schema differs")
                chunk_scaled = esm_bundle["preprocessor"].transform(chunk_matrix)
                chunk_raw = esm_bundle["model"].predict_proba(chunk_scaled)[:, 1]
                raw_list.append(chunk_raw)
                del chunk_matrix, chunk_scaled, chunk_raw
                gc.collect()
            esm_raw = np.concatenate(raw_list)
        else:
            esm_matrix, esm_names = _esm_only_matrix(pair.frame, pair.embeddings)
            if esm_bundle["feature_names"] != esm_names:
                raise ValueError(f"Fold {fold} {name} schema differs")
            esm_scaled = esm_bundle["preprocessor"].transform(esm_matrix)
            esm_raw = esm_bundle["model"].predict_proba(esm_scaled)[:, 1]
            del esm_matrix, esm_scaled
        predictions[name] = esm_bundle["calibrator"].predict(esm_raw)
        thresholds[name] = float(esm_bundle["threshold"])
        del esm_bundle, esm_raw

    for architecture in DEEP_MODEL_NAMES:
        if architecture not in model_names:
            continue
        model, preprocessors, names, threshold, bundle_metadata = C.load_deep_bundle(
            fold_dir / f"{architecture}.pt"
        )
        if int(bundle_metadata.get("fold", fold)) != fold:
            raise ValueError(f"Fold {fold} {architecture} bundle is inconsistent")
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError(f"Fold {fold} {architecture} feature schema is invalid")
        unavailable = [name for name in names if name not in pair.frame]
        if unavailable:
            raise KeyError(
                f"Fold {fold} {architecture} bundle features are unavailable: {unavailable}"
            )
        reliability_names = bundle_metadata.get("reliability_feature_names", [])
        if reliability_names is None:
            reliability_names = []
        if not isinstance(reliability_names, list) or any(
            name not in names for name in reliability_names
        ):
            raise ValueError(f"Fold {fold} {architecture} reliability feature metadata is invalid")
        if C.is_reliability_family(architecture):
            protocol = bundle_metadata.get("reliability_protocol")
            if not isinstance(protocol, dict) or not protocol.get(
                "availability_features_are_gate_only", False
            ):
                raise ValueError(f"Fold {fold} reliability bundle lacks its protocol contract")
            passthrough = C.reliability_passthrough_indices(names)
            if any(
                not np.isclose(preprocessors.bio.mean[index], 0.0)
                or not np.isclose(preprocessors.bio.scale[index], 1.0)
                for index in passthrough
            ):
                raise ValueError(f"Fold {fold} reliability gate feature scales were not preserved")
        bio = pair.frame[names].to_numpy(dtype=np.float32)
        transformed_bio = preprocessors.transform_bio(bio)
        chunk_size = 35000
        n_rows = len(pair.frame)
        if n_rows > chunk_size:
            pred_chunks = []
            rel_comp_chunks = (
                {k: [] for k in C.RELIABILITY_DIAGNOSTIC_COMPONENTS}
                if (architecture == C.RELIABILITY_ARCHITECTURE and reliability_components is not None)
                else None
            )
            for start_idx in range(0, n_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, n_rows)
                chunk_bio = transformed_bio[start_idx:end_idx]
                chunk_esm = preprocessors.transform_esm(pair.embeddings[start_idx:end_idx])
                if rel_comp_chunks is not None:
                    extracted = C.predict_reliability_components(
                        model,
                        chunk_bio,
                        chunk_esm,
                        batch_size=2048,
                    )
                    chunk_pred = extracted["model_probability"]
                    for k in C.RELIABILITY_DIAGNOSTIC_COMPONENTS:
                        rel_comp_chunks[k].append(extracted[k])
                else:
                    chunk_pred = C.predict(model, chunk_bio, chunk_esm, batch_size=2048)
                pred_chunks.append(chunk_pred)
                del chunk_bio, chunk_esm, chunk_pred
                gc.collect()
            predictions[architecture] = np.concatenate(pred_chunks)
            if rel_comp_chunks is not None:
                if reliability_components:
                    raise ValueError("Fold reliability component collector is not empty")
                reliability_components.update(
                    {name: np.concatenate(rel_comp_chunks[name]) for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS}
                )
        else:
            transformed_esm = preprocessors.transform_esm(pair.embeddings)
            predictions[architecture] = C.predict(
                model,
                transformed_bio,
                transformed_esm,
            )
            if architecture == C.RELIABILITY_ARCHITECTURE and reliability_components is not None:
                if reliability_components:
                    raise ValueError("Fold reliability component collector is not empty")
                extracted = C.predict_reliability_components(
                    model,
                    transformed_bio,
                    transformed_esm,
                )
                if not np.allclose(
                    extracted["model_probability"],
                    predictions[architecture],
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise RuntimeError(f"Fold {fold} reliability components differ from predictions")
                reliability_components.update(
                    {name: extracted[name] for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS}
                )
            del transformed_esm
        thresholds[architecture] = threshold
        del model, transformed_bio, bio
        if C.DEVICE == "cuda":
            torch.cuda.empty_cache()
    if "lora_esm" in model_names:
        model, threshold = C.load_lora_bundle(fold_dir / "lora_esm.pt")
        predictions["lora_esm"] = C.predict_lora(model, pair.frame)
        thresholds["lora_esm"] = threshold
        del model
        if C.DEVICE == "cuda":
            torch.cuda.empty_cache()
    if set(predictions) != set(model_names):
        missing = sorted(set(model_names) - set(predictions))
        unexpected = sorted(set(predictions) - set(model_names))
        raise RuntimeError(
            f"Fold {fold} prediction discovery mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name, values in predictions.items():
        if len(values) != len(pair.frame) or not np.isfinite(values).all():
            raise ValueError(f"Fold {fold} {name} predictions are invalid")
    return predictions, thresholds


def _aggregate_predictions(
    fold_probabilities: dict[str, list[np.ndarray]],
    fold_thresholds: dict[str, list[float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    probabilities: dict[str, np.ndarray] = {}
    decisions: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name, values in fold_probabilities.items():
        matrix = np.stack(values)
        threshold_array = np.asarray(fold_thresholds[name], dtype=float)
        probabilities[name] = matrix.mean(axis=0)
        thresholds[name] = float(threshold_array.mean())
        fold_decisions = matrix >= threshold_array[:, None]
        decisions[name] = (fold_decisions.sum(axis=0) > (len(threshold_array) / 2.0)).astype(
            np.int8
        )
    return probabilities, decisions, thresholds


def _fold_vote_confidence(
    fold_probabilities: np.ndarray,
    fold_thresholds: np.ndarray,
) -> np.ndarray:
    """Return the classifier's vote-margin confidence on a zero-to-one scale."""
    matrix = np.asarray(fold_probabilities, dtype=float)
    thresholds = np.asarray(fold_thresholds, dtype=float)
    if matrix.ndim != 2 or thresholds.ndim != 1:
        raise ValueError("Fold prediction confidence inputs have invalid dimensions")
    if matrix.shape[0] != len(thresholds) or matrix.shape[0] == 0:
        raise ValueError("Fold predictions and thresholds are misaligned")
    if not np.isfinite(matrix).all() or not np.isfinite(thresholds).all():
        raise ValueError("Fold prediction confidence inputs contain nonfinite values")
    vote_fraction = (matrix >= thresholds[:, None]).mean(axis=0)
    return np.abs(2.0 * vote_fraction - 1.0).astype(np.float64)


def _primary_gene(values: pd.Series) -> tuple[str, str]:
    genes = _normal_text(values)
    genes = genes[genes.ne("")]
    if genes.empty:
        return "UNKNOWN", "UNKNOWN"
    counts = genes.value_counts()
    top_count = int(counts.max())
    primary = sorted(counts[counts.eq(top_count)].index.astype(str))[0]
    return primary, ";".join(sorted(set(genes.astype(str))))


def _aggregate_clinvar_variants(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    ranking_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    fold_probabilities: dict[str, np.ndarray] | None = None,
    fold_thresholds: dict[str, np.ndarray] | None = None,
    reliability_components: dict[str, np.ndarray] | None = None,
    fold_reliability_components: dict[str, np.ndarray] | None = None,
) -> EvaluationUnit:
    """Collapse ClinVar annotation rows to unique genomic variants.

    Model probabilities are arithmetic means across all retained annotation
    mappings and internal folds.  Each fold first averages its annotation-level
    probabilities for the genomic variant, applies that fold's frozen internal
    threshold, and the final class is a strict fold majority.  A deterministic
    primary gene is used only for hierarchical bootstrap grouping.
    """
    row_count = len(frame)
    vectors = {**probabilities, **ranking_scores}
    if any(len(values) != row_count for values in vectors.values()):
        raise ValueError("ClinVar aggregation inputs are misaligned")
    genomic = _genomic_keys(frame)
    valid_key = genomic.ne("").to_numpy()
    working = frame.loc[valid_key].copy().reset_index(drop=True)
    working["genomic_variant_key"] = genomic.loc[valid_key].astype(str).to_numpy()
    selected_vectors = {
        name: np.asarray(values, dtype=float)[valid_key] for name, values in vectors.items()
    }
    selected_reliability_components: dict[str, np.ndarray] = {}
    if reliability_components is not None:
        if set(reliability_components) != set(C.RELIABILITY_DIAGNOSTIC_COMPONENTS):
            raise ValueError("ClinVar reliability component schema is incomplete")
        for name, values in reliability_components.items():
            array = np.asarray(values, dtype=float)
            if array.shape != (row_count,) or not np.isfinite(array).all():
                raise ValueError(f"ClinVar reliability component {name} is invalid")
            selected_reliability_components[name] = array[valid_key]
    selected_fold_probabilities: dict[str, np.ndarray] = {}
    selected_fold_thresholds: dict[str, np.ndarray] = {}
    if fold_probabilities is not None or fold_thresholds is not None:
        if fold_probabilities is None or fold_thresholds is None:
            raise ValueError("ClinVar fold probabilities and thresholds must be paired")
        if set(fold_probabilities) != set(probabilities) or set(fold_thresholds) != set(
            probabilities
        ):
            raise ValueError("ClinVar fold-level model names are inconsistent")
        for name in probabilities:
            matrix = np.asarray(fold_probabilities[name], dtype=float)
            frozen = np.asarray(fold_thresholds[name], dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != row_count:
                raise ValueError(f"ClinVar fold probabilities are invalid for {name}")
            if frozen.ndim != 1 or len(frozen) != matrix.shape[0]:
                raise ValueError(f"ClinVar fold thresholds are invalid for {name}")
            if not np.isfinite(matrix).all() or not np.isfinite(frozen).all():
                raise ValueError(f"ClinVar fold predictions are nonfinite for {name}")
            selected_fold_probabilities[name] = matrix[:, valid_key]
            selected_fold_thresholds[name] = frozen
    selected_fold_reliability_components: dict[str, np.ndarray] = {}
    if fold_reliability_components is not None:
        if not selected_reliability_components:
            raise ValueError("Fold-level ClinVar reliability components require aggregated values")
        if set(fold_reliability_components) != set(C.RELIABILITY_DIAGNOSTIC_COMPONENTS):
            raise ValueError("ClinVar fold reliability component schema is incomplete")
        for name, values in fold_reliability_components.items():
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != row_count:
                raise ValueError(f"ClinVar fold reliability component {name} is invalid")
            if not np.isfinite(matrix).all():
                raise ValueError(f"ClinVar fold reliability component {name} is nonfinite")
            selected_fold_reliability_components[name] = matrix[:, valid_key]
    label_counts = working.groupby("genomic_variant_key", sort=False)[C.LABEL_COL].nunique()
    conflicting_keys = set(label_counts[label_counts.gt(1)].index.astype(str))
    if conflicting_keys:
        keep = ~working["genomic_variant_key"].isin(conflicting_keys).to_numpy()
        working = working.loc[keep].reset_index(drop=True)
        selected_vectors = {name: values[keep] for name, values in selected_vectors.items()}
        selected_reliability_components = {
            name: values[keep] for name, values in selected_reliability_components.items()
        }
        selected_fold_probabilities = {
            name: values[:, keep] for name, values in selected_fold_probabilities.items()
        }
        selected_fold_reliability_components = {
            name: values[:, keep] for name, values in selected_fold_reliability_components.items()
        }
    if working.empty:
        return EvaluationUnit(
            frame=working,
            probabilities={},
            decisions={},
            ranking_scores={},
            audit={
                "input_annotation_rows": int(row_count),
                "rows_missing_genomic_key": int((~valid_key).sum()),
                "label_conflict_variants_excluded": int(len(conflicting_keys)),
                "unique_genomic_variants": 0,
            },
            decision_confidence={},
            reliability_components=(
                {
                    name: np.empty(0, dtype=np.float32)
                    for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS
                }
                if selected_reliability_components
                else None
            ),
            reliability_anchor_decisions=(
                np.empty(0, dtype=np.int8) if selected_reliability_components else None
            ),
        )

    records: list[dict[str, Any]] = []
    aggregated = {name: [] for name in selected_vectors}
    fold_majority_decisions = {name: [] for name in probabilities}
    vote_confidences = {name: [] for name in probabilities}
    aggregated_reliability = {name: [] for name in selected_reliability_components}
    reliability_anchor_decisions: list[int] = []
    grouped = working.groupby("genomic_variant_key", sort=True)
    for key, group in grouped:
        indices = group.index.to_numpy(dtype=int)
        primary_gene, all_genes = _primary_gene(group[C.GENE_COL])
        record: dict[str, Any] = {
            "genomic_variant_key": str(key),
            C.ROW_ID_COL: str(key),
            C.GENE_COL: primary_gene,
            "all_genes": all_genes,
            "gene_count": int(len(all_genes.split(";"))),
            "annotation_row_count": int(len(group)),
            C.LABEL_COL: int(pd.to_numeric(group[C.LABEL_COL]).iloc[0]),
            "EXT_SOURCE": "clinvar",
        }
        for column in ("chr", "pos", "ref", "alt", "variant_id"):
            if column in group:
                record[column] = group[column].iloc[0]
        records.append(record)
        for name, values in selected_vectors.items():
            selected = values[indices]
            finite = selected[np.isfinite(selected)]
            aggregated[name].append(float(finite.mean()) if finite.size else float("nan"))
        for name, values in selected_reliability_components.items():
            aggregated_reliability[name].append(float(values[indices].mean()))
        for name in probabilities:
            if name not in selected_fold_probabilities:
                continue
            per_fold_variant = selected_fold_probabilities[name][:, indices].mean(axis=1)
            votes = per_fold_variant >= selected_fold_thresholds[name]
            fold_majority_decisions[name].append(int(votes.sum() > (len(votes) / 2.0)))
            vote_confidences[name].append(abs(2.0 * float(votes.mean()) - 1.0))
        if selected_fold_reliability_components:
            anchor_per_fold = selected_fold_reliability_components["anchor_probability"][
                :, indices
            ].mean(axis=1)
            reliability_thresholds = selected_fold_thresholds.get(C.RELIABILITY_ARCHITECTURE)
            if reliability_thresholds is None:
                raise ValueError("ClinVar reliability anchor lacks frozen fold thresholds")
            anchor_votes = anchor_per_fold >= reliability_thresholds
            reliability_anchor_decisions.append(int(anchor_votes.sum() > (len(anchor_votes) / 2.0)))

    variant_frame = pd.DataFrame(records)
    variant_probabilities = {
        name: np.asarray(aggregated[name], dtype=float) for name in probabilities
    }
    variant_rankings = {name: np.asarray(aggregated[name], dtype=float) for name in ranking_scores}
    variant_decisions: dict[str, np.ndarray] = {}
    variant_decision_confidence: dict[str, np.ndarray] = {}
    for name, values in variant_probabilities.items():
        if name in selected_fold_probabilities:
            variant_decisions[name] = np.asarray(fold_majority_decisions[name], dtype=np.int8)
            variant_decision_confidence[name] = np.asarray(vote_confidences[name], dtype=np.float64)
        else:
            # Backward-compatible path for direct utility calls that do not
            # supply fold tensors. Production Stage 12 always supplies them.
            variant_decisions[name] = (values >= thresholds[name]).astype(np.int8)
    variant_reliability = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in aggregated_reliability.items()
    }
    if variant_reliability and not reliability_anchor_decisions:
        reliability_anchor_decisions = (
            (variant_reliability["anchor_probability"] >= thresholds[C.RELIABILITY_ARCHITECTURE])
            .astype(np.int8)
            .tolist()
        )
    annotation_counts = variant_frame["annotation_row_count"]
    audit = {
        "input_annotation_rows": int(row_count),
        "rows_with_valid_genomic_key": int(valid_key.sum()),
        "rows_missing_genomic_key": int((~valid_key).sum()),
        "label_conflict_variants_excluded": int(len(conflicting_keys)),
        "unique_genomic_variants": int(len(variant_frame)),
        "multi_annotation_variants": int(annotation_counts.gt(1).sum()),
        "multi_gene_variants": int(variant_frame["gene_count"].gt(1).sum()),
        "genomic_key_definition": "GRCh37 chromosome:position:reference:alternate",
        "probability_aggregation": "arithmetic_mean_across_retained_annotation_rows",
        "ranking_score_aggregation": "arithmetic_mean_across_retained_annotation_rows",
        "decision_aggregation": (
            "within_each_fold_mean_annotation_probability_then_frozen_fold_"
            "threshold_then_strict_fold_majority"
            if selected_fold_probabilities
            else "legacy_mean_probability_vs_mean_threshold_fallback"
        ),
        "bootstrap_gene_assignment": ("most_frequent_annotation_gene_with_lexical_tie_break"),
    }
    return EvaluationUnit(
        frame=variant_frame,
        probabilities=variant_probabilities,
        decisions=variant_decisions,
        ranking_scores=variant_rankings,
        audit=audit,
        decision_confidence=variant_decision_confidence,
        reliability_components=variant_reliability or None,
        reliability_anchor_decisions=(
            np.asarray(reliability_anchor_decisions, dtype=np.int8) if variant_reliability else None
        ),
    )


def _hierarchical_bootstrap_indices(
    groups: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> list[np.ndarray]:
    """Sample genes, then variants within each sampled gene, with replacement."""
    groups = np.asarray(groups, dtype=object)
    if groups.size == 0:
        return []
    codes, unique = pd.factorize(groups, sort=False)
    if (codes < 0).any():
        raise ValueError("Bootstrap groups contain missing values")
    mapping = {group: np.flatnonzero(codes == index) for index, group in enumerate(unique)}
    random_state = np.random.RandomState(seed)
    samples: list[np.ndarray] = []
    for _ in range(iterations):
        sampled_genes = random_state.choice(unique, len(unique), replace=True)
        pieces = []
        for gene in sampled_genes:
            variants = mapping[gene]
            pieces.append(random_state.choice(variants, len(variants), replace=True))
        samples.append(np.concatenate(pieces))
    return samples


def _hierarchical_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    decisions: np.ndarray,
    groups: np.ndarray,
    seed: int = RANDOM_STATE,
) -> dict[str, list[float] | None]:
    values: dict[str, list[float]] = {name: [] for name in ("mcc", "auroc", "auprc", "brier")}
    for indices in _hierarchical_bootstrap_indices(groups, seed=seed):
        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        sampled_decisions = decisions[indices]
        values["brier"].append(float(brier_score_loss(sampled_labels, sampled_probabilities)))
        if len(np.unique(sampled_labels)) == 2:
            values["mcc"].append(float(matthews_corrcoef(sampled_labels, sampled_decisions)))
            values["auroc"].append(float(roc_auc_score(sampled_labels, sampled_probabilities)))
            values["auprc"].append(
                float(average_precision_score(sampled_labels, sampled_probabilities))
            )
    return {
        name: (
            [round(float(value), 4) for value in np.percentile(samples, [2.5, 97.5])]
            if samples
            else None
        )
        for name, samples in values.items()
    }


def _rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Ranking metric inputs are invalid")
    both_classes = len(np.unique(labels)) == 2
    return {
        "output_scale": "raw_ranking_score",
        "score_orientation": "higher_is_more_deleterious",
        "threshold": None,
        "mcc": None,
        "auroc": (round(float(roc_auc_score(labels, scores)), 4) if both_classes else None),
        "auprc": (
            round(float(average_precision_score(labels, scores)), 4) if both_classes else None
        ),
        "brier": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def _hierarchical_rank_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    seed: int = RANDOM_STATE,
) -> dict[str, list[float] | None]:
    values = {"auroc": [], "auprc": []}
    for indices in _hierarchical_bootstrap_indices(groups, seed=seed):
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_scores = scores[indices]
        values["auroc"].append(float(roc_auc_score(sampled_labels, sampled_scores)))
        values["auprc"].append(float(average_precision_score(sampled_labels, sampled_scores)))
    return {
        metric: (
            [round(float(value), 4) for value in np.percentile(samples, [2.5, 97.5])]
            if samples
            else None
        )
        for metric, samples in values.items()
    }


def _contextual_predictor_benchmark(
    frame: pd.DataFrame,
    contextual_scores: dict[str, np.ndarray],
    model_probabilities: dict[str, np.ndarray],
    ranking_scores: dict[str, np.ndarray],
    set_name: str,
    artifact: dict[str, np.ndarray],
    inferential_reporting_allowed: bool = True,
) -> dict[str, Any]:
    """Evaluate predictor comparators individually and on one common subset."""
    if not contextual_scores:
        return {
            "available": False,
            "reason": "no_contextual_predictor_scores",
            "model_input": False,
        }
    labels = frame[C.LABEL_COL].to_numpy(dtype=int)
    groups = frame[C.GENE_COL].astype(str).to_numpy(dtype=object)
    individual: dict[str, Any] = {}
    finite_masks: dict[str, np.ndarray] = {}
    for name, scores in contextual_scores.items():
        finite = np.isfinite(scores)
        finite_masks[name] = finite
        record: dict[str, Any] = {
            "n": int(finite.sum()),
            "coverage": round(float(finite.mean()), 6),
            "positives": int(labels[finite].sum()),
            "genes": int(pd.Series(groups[finite]).nunique()),
            "model_input": False,
            "interpretation": "contextual_comparator_not_independent_ACMG_evidence",
        }
        if finite.sum() and len(np.unique(labels[finite])) == 2:
            metrics = _rank_metrics(labels[finite], scores[finite])
            metrics["ci95"] = (
                _hierarchical_rank_intervals(labels[finite], scores[finite], groups[finite])
                if inferential_reporting_allowed
                else None
            )
            record["metrics"] = metrics
        else:
            record["metrics"] = {"auroc": None, "auprc": None, "ci95": None}
        individual[name] = record

    common = np.logical_and.reduce(list(finite_masks.values()))
    common_prefix = f"{set_name}_contextual_common_coverage"
    common_result: dict[str, Any] = {
        "policy": "intersection_of_all_available_contextual_predictors",
        "n": int(common.sum()),
        "coverage": round(float(common.mean()), 6),
        "positives": int(labels[common].sum()),
        "genes": int(pd.Series(groups[common]).nunique()),
        "predictors": list(contextual_scores),
        "models": {},
        "artifact_prefix": common_prefix,
    }
    if common.sum() and len(np.unique(labels[common])) == 2:
        artifact[f"{common_prefix}__y"] = labels[common]
        artifact[f"{common_prefix}__groups"] = groups[common]
        artifact[f"{common_prefix}__row_ids"] = (
            frame.loc[common, C.ROW_ID_COL].astype(str).to_numpy()
        )
        common_scores = {
            **{name: scores[common] for name, scores in contextual_scores.items()},
            **{name: values[common] for name, values in ranking_scores.items()},
            **{name: values[common] for name, values in model_probabilities.items()},
        }
        for name, scores in common_scores.items():
            metrics = _rank_metrics(labels[common], scores)
            if name in model_probabilities:
                metrics["output_scale"] = "calibrated_probability"
            metrics["ci95"] = (
                _hierarchical_rank_intervals(labels[common], scores, groups[common])
                if inferential_reporting_allowed
                else None
            )
            common_result["models"][name] = metrics
            artifact[f"{common_prefix}__{name}"] = scores
    else:
        common_result["status"] = "insufficient_common_coverage_or_classes"
    return {
        "available": True,
        "analysis_unit": "unique_genomic_variant",
        "model_input": False,
        "score_directions": {
            f"context_{column}": specification["source_orientation"]
            for column, specification in CONTEXTUAL_PREDICTOR_DIRECTIONS.items()
            if f"context_{column}" in contextual_scores
        },
        "evaluation_orientation": "all scores transformed so higher_is_more_deleterious",
        "warning": (
            "These established predictors can contain ClinVar/population-derived "
            "information and are contextual comparators, not independent evidence."
        ),
        "reporting_mode": (
            "descriptive_with_gene_bootstrap_intervals"
            if inferential_reporting_allowed
            else "descriptive_only_underpowered_clinical_cohort"
        ),
        "individual_coverage": individual,
        "common_coverage": common_result,
    }


def _hierarchical_model_comparison(
    labels: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    first_decision: np.ndarray,
    second_decision: np.ndarray,
    groups: np.ndarray,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    randomization = C.paired_group_randomization_test(
        labels,
        first_probability,
        second_probability,
        first_decision,
        second_decision,
        groups,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=seed + 1,
    )
    differences = {name: [] for name in ("mcc", "auroc", "auprc")}
    for indices in _hierarchical_bootstrap_indices(groups, seed=seed):
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) == 2:
            differences["mcc"].append(
                float(matthews_corrcoef(sampled_labels, second_decision[indices]))
                - float(matthews_corrcoef(sampled_labels, first_decision[indices]))
            )
            differences["auroc"].append(
                float(roc_auc_score(sampled_labels, second_probability[indices]))
                - float(roc_auc_score(sampled_labels, first_probability[indices]))
            )
            differences["auprc"].append(
                float(average_precision_score(sampled_labels, second_probability[indices]))
                - float(average_precision_score(sampled_labels, first_probability[indices]))
            )
    output: dict[str, Any] = {}
    for metric, samples in differences.items():
        if not samples:
            output[metric] = None
            continue
        array = np.asarray(samples, dtype=float)
        inference = randomization[metric]
        output[metric] = {
            "mean_difference_second_minus_first": round(
                float(inference["observed_difference_second_minus_first"]), 4
            ),
            "bootstrap_mean_difference_second_minus_first": round(float(array.mean()), 4),
            "ci95": [round(float(value), 4) for value in np.percentile(array, [2.5, 97.5])],
            "ci_method": "descriptive_gene_then_variant_bootstrap_percentile",
            "two_sided_probability": inference["two_sided_probability"],
            "randomization_iterations": inference["randomization_iterations"],
            "inference_method": inference["inference_method"],
            "exchangeability_unit": "gene",
            "inference_scope": "conditional_on_frozen_model_predictions",
            "training_procedure_uncertainty_included": False,
        }
    return output


def _bootstrap_groups(frame: pd.DataFrame, source: str) -> np.ndarray:
    if source == "dms" and "ASSAY_ID" in frame:
        assay = frame["ASSAY_ID"].astype("string").fillna("").str.strip()
        if assay.ne("").all():
            return assay.to_numpy(dtype=object)
    return frame[C.GENE_COL].astype(str).to_numpy(dtype=object)


def _assay_macro_bootstrap(
    assays: dict[str, Any],
    model_names: list[str],
    metrics: tuple[str, ...],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> dict[str, dict[str, list[float] | None]]:
    assay_ids = np.asarray(sorted(assays), dtype=object)
    random_state = np.random.RandomState(seed)
    samples = {model: {metric: [] for metric in metrics} for model in model_names}
    for _ in range(iterations):
        selected = random_state.choice(assay_ids, len(assay_ids), replace=True)
        for model in model_names:
            for metric in metrics:
                values = [assays[assay]["models"][model].get(metric) for assay in selected]
                finite = [float(value) for value in values if value is not None]
                if finite:
                    samples[model][metric].append(float(np.mean(finite)))
    return {
        model: {
            metric: (
                [
                    round(float(value), 4)
                    for value in np.percentile(samples[model][metric], [2.5, 97.5])
                ]
                if samples[model][metric]
                else None
            )
            for metric in metrics
        }
        for model in model_names
    }


def _assay_paired_comparisons(
    assays: dict[str, Any],
    model_names: list[str],
    metrics: tuple[str, ...] = ("functional_spearman", "auroc"),
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pair_offset, (first, second) in enumerate(itertools.combinations(model_names, 2)):
        pair_key = f"{second}_minus_{first}"
        output[pair_key] = {
            "first": first,
            "second": second,
            "difference_orientation": "second_minus_first",
            "metrics": {},
        }
        for metric_offset, metric in enumerate(metrics):
            paired = [
                (
                    record["models"][first].get(metric),
                    record["models"][second].get(metric),
                )
                for record in assays.values()
            ]
            differences = np.asarray(
                [
                    float(second_value) - float(first_value)
                    for first_value, second_value in paired
                    if first_value is not None and second_value is not None
                ],
                dtype=float,
            )
            if differences.size < 2:
                output[pair_key]["metrics"][metric] = {
                    "paired_assays": int(differences.size),
                    "mean_difference": (
                        round(float(differences.mean()), 4) if differences.size else None
                    ),
                    "ci95": None,
                    "two_sided_probability": None,
                    "inference_status": "insufficient_paired_assays",
                    "minimum_paired_assays": 2,
                }
                continue
            random_state = np.random.RandomState(seed + pair_offset * 17 + metric_offset)
            bootstrap = [
                float(random_state.choice(differences, len(differences), replace=True).mean())
                for _ in range(iterations)
            ]
            observed = float(differences.mean())
            null_means = np.empty(iterations, dtype=float)
            for iteration in range(iterations):
                signs = random_state.choice((-1.0, 1.0), len(differences))
                null_means[iteration] = float(np.mean(differences * signs))
            exceedances = int(np.count_nonzero(np.abs(null_means) + 1e-15 >= abs(observed)))
            probability = (exceedances + 1.0) / (iterations + 1.0)
            output[pair_key]["metrics"][metric] = {
                "paired_assays": int(len(differences)),
                "mean_difference": round(observed, 4),
                "ci95": [round(float(value), 4) for value in np.percentile(bootstrap, [2.5, 97.5])],
                "ci_method": "descriptive_paired_assay_bootstrap_percentile",
                "two_sided_probability": round(float(probability), 6),
                "randomization_iterations": int(iterations),
                "inference_method": "paired_assay_sign_flip_randomization",
                "exchangeability_unit": "assay",
                "inference_status": "evaluated",
            }
    for metric in metrics:
        keys = [
            key
            for key, record in output.items()
            if record["metrics"].get(metric, {}).get("two_sided_probability") is not None
        ]
        if not keys:
            continue
        adjusted = multipletests(
            [output[key]["metrics"][metric]["two_sided_probability"] for key in keys],
            method="holm",
        )[1]
        for key, value in zip(keys, adjusted):
            output[key]["metrics"][metric]["holm_adjusted_probability"] = round(float(value), 4)
    return output


def _dms_assay_results(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    thresholds: dict[str, float],
    ranking_scores: dict[str, np.ndarray] | None = None,
    sampling_audit: dict[str, Any] | None = None,
    decision_confidence: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if "ASSAY_ID" not in frame:
        return {"available": False}
    ranking_scores = ranking_scores or {}
    decision_confidence = decision_confidence or {}
    assays: dict[str, Any] = {}
    assay_values = frame["ASSAY_ID"].astype(str).to_numpy()
    labels = frame[C.LABEL_COL].to_numpy(dtype=int)
    scores = (
        pd.to_numeric(frame["DMS_SCORE"], errors="coerce").to_numpy(dtype=float)
        if "DMS_SCORE" in frame
        else np.full(len(frame), np.nan)
    )
    for assay in sorted(pd.unique(assay_values)):
        index = np.flatnonzero(assay_values == assay)
        record: dict[str, Any] = {
            "n": int(len(index)),
            "positives": int(labels[index].sum()),
            "models": {},
        }
        for name, model_probabilities in probabilities.items():
            metrics = C.evaluate(
                labels[index],
                model_probabilities[index],
                thresholds[name],
                predictions=decisions[name][index],
                decision_confidence=(
                    decision_confidence[name][index] if name in decision_confidence else None
                ),
            )
            finite = np.isfinite(scores[index])
            if finite.sum() >= 3:
                correlation = spearmanr(
                    scores[index][finite],
                    1.0 - model_probabilities[index][finite],
                ).statistic
                metrics["functional_spearman"] = (
                    None if not np.isfinite(correlation) else round(float(correlation), 4)
                )
            else:
                metrics["functional_spearman"] = None
            record["models"][name] = metrics
        for name, model_scores in ranking_scores.items():
            metrics = _rank_metrics(labels[index], model_scores[index])
            finite = np.isfinite(scores[index])
            if finite.sum() >= 3:
                correlation = spearmanr(
                    scores[index][finite],
                    -model_scores[index][finite],
                ).statistic
                metrics["functional_spearman"] = (
                    None if not np.isfinite(correlation) else round(float(correlation), 4)
                )
            else:
                metrics["functional_spearman"] = None
            record["models"][name] = metrics
        assays[assay] = record
    model_names = [*probabilities, *ranking_scores]
    macro: dict[str, dict[str, float | None]] = {}
    macro_metrics = ("mcc", "auroc", "auprc", "f1", "functional_spearman")
    for name in model_names:
        macro[name] = {}
        for metric in macro_metrics:
            values = [
                record["models"][name].get(metric)
                for record in assays.values()
                if record["models"][name].get(metric) is not None
            ]
            macro[name][metric] = round(float(np.mean(values)), 4) if values else None
    primary_metrics = ("functional_spearman", "auroc")
    macro_ci95 = _assay_macro_bootstrap(assays, model_names, primary_metrics)
    return {
        "available": True,
        "analysis_unit": "assay",
        "assay_count": int(len(assays)),
        "primary_endpoints": [
            "equal_weight_assay_macro_functional_spearman",
            "equal_weight_assay_macro_auroc",
        ],
        "primary_endpoint_note": (
            "DMS_SCORE Spearman uses predicted functional score; AUROC uses the "
            "assay-specific damaging label. Uniform sampling probability is constant "
            "within each assay, so inverse-probability weighting does not change these "
            "within-assay rank endpoints. Pooled-row metrics are descriptive only."
        ),
        "bootstrap": {
            "scheme": "assay_resampling_with_replacement",
            "iterations": BOOTSTRAP_ITERATIONS,
            "paired_model_differences": True,
        },
        "sampling_caveats": sampling_audit or {},
        "assays": assays,
        "macro": macro,
        "macro_ci95": macro_ci95,
        "paired_comparisons": _assay_paired_comparisons(assays, model_names),
        "comparison_multiplicity_correction": "Holm within endpoint",
    }


def _dms_sampling_audit(path: Path = EXTERNAL_PREP_MANIFEST) -> dict[str, Any]:
    base = {
        "is_full_proteingym_benchmark": False,
        "prevalence_interpretation": (
            "DMS assay-cutoff labels and the assay mixture are not clinical or "
            "population prevalence"
        ),
        "caveat": (
            "Results describe the prepared eligible assay subset; equal-weight assay "
            "macro endpoints are primary and pooled calibration is not interpreted."
        ),
    }
    if not path.exists():
        return {
            **base,
            "manifest_available": False,
            "sampling_policy": "unknown",
            "sampling_uses_label": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = payload.get("extra", {})
    retained = extra.get("assay_rows", {})
    candidates = extra.get("assay_candidate_rows", {})
    assay_sampling = extra.get("assay_sampling", {})
    capped = [
        assay
        for assay, candidate_n in candidates.items()
        if int(candidate_n) > int(retained.get(assay, 0))
    ]
    declared_policy = extra.get("dms_sampling_policy")
    declared_uses_label = extra.get("dms_sampling_uses_label")
    legacy_fallback = declared_policy is None
    if legacy_fallback:
        sampling_policy = "legacy_deterministic_class_balanced_per_assay_cap"
        sampling_uses_label = True
        policy_note = (
            "Legacy manifest lacks a sampling declaration; the historical Stage 09 "
            "implementation sampled by binary label."
        )
    else:
        sampling_policy = str(declared_policy)
        sampling_uses_label = bool(declared_uses_label)
        policy_note = (
            "All eligible rows retained."
            if sampling_policy == "all"
            else "Deterministic uniform hash sample within each assay, independent of label."
        )
    candidate_total = int(sum(int(value) for value in candidates.values())) if candidates else None
    retained_total = int(sum(int(value) for value in retained.values())) if retained else None
    sampling_probabilities = [
        float(record["sampling_probability"])
        for record in assay_sampling.values()
        if record.get("sampling_probability") is not None
    ]
    sampling_weights = [
        float(record["sample_weight"])
        for record in assay_sampling.values()
        if record.get("sample_weight") is not None
    ]
    inconsistent_weights = sum(
        1
        for record in assay_sampling.values()
        if record.get("sampling_probability") not in (None, 0)
        and record.get("sample_weight") is not None
        and not np.isclose(
            float(record["sample_weight"]),
            1.0 / float(record["sampling_probability"]),
        )
    )
    return {
        **base,
        "manifest_available": True,
        "sampling_policy": sampling_policy,
        "sampling_uses_label": sampling_uses_label,
        "legacy_fallback": legacy_fallback,
        "policy_note": policy_note,
        "maximum_rows_per_assay": extra.get("dms_max_rows_per_assay"),
        "assays_retained": int(len(retained)),
        "assays_downsampled": int(len(capped)),
        "candidate_rows": candidate_total,
        "retained_rows": retained_total,
        "is_full_prepared_candidate_set": bool(
            candidate_total is not None
            and retained_total is not None
            and candidate_total == retained_total
        ),
        "sampling_probability_column": "DMS_SAMPLING_PROBABILITY",
        "sampling_weight_column": "DMS_SAMPLE_WEIGHT",
        "row_level_probabilities_and_weights_expected": not legacy_fallback,
        "sampling_probability_range": (
            [min(sampling_probabilities), max(sampling_probabilities)]
            if sampling_probabilities
            else None
        ),
        "sample_weight_range": (
            [min(sampling_weights), max(sampling_weights)] if sampling_weights else None
        ),
        "assays_with_sampling_metadata": int(len(assay_sampling)),
        "inverse_probability_weight_inconsistencies": int(inconsistent_weights),
    }


def _evaluate_set(
    source: str,
    policy: str,
    pair: DatasetPair,
    mask: np.ndarray,
    probabilities: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    thresholds: dict[str, float],
    artifact: dict[str, np.ndarray],
    ranking_scores: dict[str, np.ndarray] | None = None,
    contextual_scores: dict[str, np.ndarray] | None = None,
    dms_sampling: dict[str, Any] | None = None,
    fold_probabilities: dict[str, np.ndarray] | None = None,
    fold_thresholds: dict[str, np.ndarray] | None = None,
    reliability_components: dict[str, np.ndarray] | None = None,
    fold_reliability_components: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    row_frame = pair.frame.loc[mask].reset_index(drop=True)
    set_name = f"{source}_{policy}"
    if row_frame.empty:
        return (
            {
                "status": "empty",
                "source": source,
                "policy": policy,
                "n": 0,
                "row_n": 0,
            },
            [],
            pd.DataFrame(),
        )
    selected_probabilities = {name: values[mask] for name, values in probabilities.items()}
    selected_decisions = {name: values[mask] for name, values in decisions.items()}
    selected_rankings = {name: values[mask] for name, values in (ranking_scores or {}).items()}
    selected_contextual = {name: values[mask] for name, values in (contextual_scores or {}).items()}
    selected_reliability_components: dict[str, np.ndarray] = {}
    selected_fold_reliability_components: dict[str, np.ndarray] = {}
    if reliability_components is not None:
        if set(reliability_components) != set(C.RELIABILITY_DIAGNOSTIC_COMPONENTS):
            raise ValueError("External reliability component schema is incomplete")
        selected_reliability_components = {
            name: np.asarray(values, dtype=float)[mask]
            for name, values in reliability_components.items()
        }
    if fold_reliability_components is not None:
        if not selected_reliability_components:
            raise ValueError("Fold external reliability components require aggregated values")
        if set(fold_reliability_components) != set(C.RELIABILITY_DIAGNOSTIC_COMPONENTS):
            raise ValueError("External fold reliability component schema is incomplete")
        selected_fold_reliability_components = {
            name: np.asarray(values, dtype=float)[:, mask]
            for name, values in fold_reliability_components.items()
        }
    if (fold_probabilities is None) != (fold_thresholds is None):
        raise ValueError("External fold probabilities and thresholds must be paired")
    if fold_probabilities is not None and (
        set(fold_probabilities) != set(selected_probabilities)
        or set(fold_thresholds or {}) != set(selected_probabilities)
    ):
        raise ValueError("External fold-level model names are inconsistent")
    selected_fold_probabilities = {
        name: np.asarray(values, dtype=float)[:, mask]
        for name, values in (fold_probabilities or {}).items()
    }
    selected_fold_thresholds = {
        name: np.asarray(values, dtype=float) for name, values in (fold_thresholds or {}).items()
    }
    selected_confidence = {
        name: _fold_vote_confidence(
            selected_fold_probabilities[name], selected_fold_thresholds[name]
        )
        for name in selected_fold_probabilities
    }
    if source == "clinvar":
        units = _aggregate_clinvar_variants(
            row_frame,
            selected_probabilities,
            selected_decisions,
            {**selected_rankings, **selected_contextual},
            thresholds,
            selected_fold_probabilities or None,
            selected_fold_thresholds or None,
            selected_reliability_components or None,
            selected_fold_reliability_components or None,
        )
        frame = units.frame
        selected_probabilities = units.probabilities
        selected_decisions = units.decisions
        selected_rankings = {name: units.ranking_scores[name] for name in selected_rankings}
        selected_contextual = {name: units.ranking_scores[name] for name in selected_contextual}
        selected_confidence = units.decision_confidence or {}
        selected_reliability_components = units.reliability_components or {}
        reliability_anchor_decisions = units.reliability_anchor_decisions
        aggregation_audit = units.audit
        analysis_unit = "unique_genomic_variant"
        bootstrap_scheme = "hierarchical_gene_then_variant_with_replacement"
    else:
        frame = row_frame
        reliability_anchor_decisions = None
        if selected_reliability_components:
            if selected_fold_reliability_components:
                anchor_fold_probabilities = selected_fold_reliability_components[
                    "anchor_probability"
                ]
                reliability_thresholds = selected_fold_thresholds.get(C.RELIABILITY_ARCHITECTURE)
                if reliability_thresholds is None:
                    raise ValueError("External reliability anchor lacks frozen fold thresholds")
                anchor_votes = anchor_fold_probabilities >= reliability_thresholds[:, None]
                reliability_anchor_decisions = (
                    anchor_votes.sum(axis=0) > (len(reliability_thresholds) / 2.0)
                ).astype(np.int8)
            else:
                reliability_anchor_decisions = (
                    selected_reliability_components["anchor_probability"]
                    >= thresholds[C.RELIABILITY_ARCHITECTURE]
                ).astype(np.int8)
        aggregation_audit = {
            "input_rows": int(len(frame)),
            "unique_assay_variant_rows": int(len(frame)),
        }
        analysis_unit = "assay_variant_row"
        bootstrap_scheme = "assay_cluster_for_descriptive_pooled_metrics"
    if frame.empty:
        return (
            {
                "status": "empty_after_unit_aggregation",
                "source": source,
                "policy": policy,
                "n": 0,
                "row_n": int(len(row_frame)),
                "aggregation_audit": aggregation_audit,
            },
            [],
            pd.DataFrame(),
        )
    labels = frame[C.LABEL_COL].to_numpy(dtype=int)
    groups = _bootstrap_groups(frame, source)
    clinical_evidence = (
        PR.clinical_evidence_assessment(labels, groups) if source == "clinvar" else None
    )
    # Production always supplies frozen fold tensors.  The no-fold branch is a
    # backward-compatible utility path retained for historical direct callers;
    # it is never serialized as a Stage 12 publication artifact.
    publication_reporting_guard_enforced = fold_probabilities is not None
    inferential_reporting_allowed = bool(
        clinical_evidence is None
        or clinical_evidence["inferential_model_comparison_allowed"]
        or not publication_reporting_guard_enforced
    )
    artifact[f"{set_name}__y"] = labels
    artifact[f"{set_name}__groups"] = groups
    artifact[f"{set_name}__row_ids"] = frame[C.ROW_ID_COL].astype(str).to_numpy()
    if source == "clinvar":
        artifact[f"{set_name}__variant_ids"] = frame["genomic_variant_key"].astype(str).to_numpy()
        artifact[f"{set_name}__annotation_row_counts"] = frame["annotation_row_count"].to_numpy(
            dtype=np.int32
        )
    modality_coverage = _modality_coverage(row_frame)
    is_primary = policy == "exact_variant_disjoint"
    validation_scope = (
        "clinical_pathogenicity_at_unique_genomic_variant_level"
        if source == "clinvar"
        else (
            "full_multimodal_functional_transfer"
            if modality_coverage["full_multimodal_validation"]
            else "functional_transfer_with_missing_or_imputed_modalities"
        )
    )
    result: dict[str, Any] = {
        "status": "evaluated",
        "source": source,
        "policy": policy,
        "role": "primary" if is_primary else "secondary_stress_test",
        "analysis_unit": analysis_unit,
        "n": int(len(frame)),
        "row_n": int(len(row_frame)),
        "unique_variant_n": int(len(frame)) if source == "clinvar" else None,
        "positives": int(labels.sum()),
        "prevalence": round(float(labels.mean()), 6),
        "genes": int(frame[C.GENE_COL].nunique()),
        "assays": (
            int(frame["ASSAY_ID"].nunique()) if source == "dms" and "ASSAY_ID" in frame else None
        ),
        "esm_extraction_coverage": round(
            float(row_frame["ESM_EXTRACTION_SUCCESS"].fillna(0).mean()), 6
        ),
        "modality_coverage": modality_coverage,
        "full_multimodal_validation": modality_coverage["full_multimodal_validation"],
        "validation_scope": validation_scope,
        "aggregation_audit": aggregation_audit,
        "bootstrap": {
            "scheme": bootstrap_scheme,
            "iterations": BOOTSTRAP_ITERATIONS,
        },
        "calibration_interpretation": (
            (
                "clinical_pathogenicity_descriptive_only"
                if clinical_evidence and not clinical_evidence["clinical_calibration_claim_allowed"]
                else "clinical_pathogenicity"
            )
            if source == "clinvar"
            else "experimental_fitness_only"
        ),
        "models": {},
    }
    if selected_reliability_components:
        if C.RELIABILITY_ARCHITECTURE not in selected_probabilities:
            raise ValueError("Reliability diagnostics lack the proposed prediction")
        if reliability_anchor_decisions is None:
            raise ValueError("Reliability diagnostics lack anchor decisions")
        reliability_diagnostics = C.summarize_reliability_diagnostics(
            selected_reliability_components,
            labels,
            selected_probabilities[C.RELIABILITY_ARCHITECTURE],
            thresholds[C.RELIABILITY_ARCHITECTURE],
            selected_decisions[C.RELIABILITY_ARCHITECTURE],
            anchor_decisions=reliability_anchor_decisions,
        )
        reliability_diagnostics["evaluation_scope"] = (
            "external_frozen_deployment_folds_descriptive_only"
        )
        reliability_diagnostics["fold_aggregation"] = (
            "arithmetic_mean_component_and_calibrated_probability_across_five_"
            "frozen_internal_deployment_folds"
        )
        reliability_diagnostics["analysis_unit_aggregation"] = (
            "mean_across_retained_annotation_rows_after_fold_prediction"
            if source == "clinvar"
            else "none_assay_variant_row"
        )
        result["reliability_diagnostics"] = reliability_diagnostics
        component_prefix = f"{set_name}__reliability_components"
        for name, values in selected_reliability_components.items():
            artifact[f"{component_prefix}__{name}"] = np.asarray(values, dtype=np.float32)
        artifact[f"{component_prefix}__availability_stratum_code"] = (
            C.reliability_availability_codes(selected_reliability_components)
        )
        artifact[f"{component_prefix}__anchor_decisions"] = np.asarray(
            reliability_anchor_decisions, dtype=np.int8
        )
    if clinical_evidence is not None:
        result["publication_reporting_guard_enforced"] = publication_reporting_guard_enforced
        result["clinical_evidence_assessment"] = clinical_evidence
        result["clinical_utility_claim"] = {
            "allowed": False,
            "reason": (
                "underpowered_external_clinical_cohort"
                if clinical_evidence["status"] == "underpowered"
                else "no_patient_outcome_or_decision_analytic_validation"
            ),
        }
        result["conformal_prediction"] = PR.conformal_availability_report(
            calibration_labels=None,
            exchangeability_justification=None,
            evaluation_shift="post_cutoff_temporal_and_gene_distribution_shift",
        )
    if source == "dms":
        result["feature_access_interpretation"] = {
            "prepared_profile": sorted(
                row_frame.get(
                    "EXTERNAL_FEATURE_PROFILE",
                    pd.Series("unknown", index=row_frame.index),
                )
                .astype(str)
                .unique()
                .tolist()
            ),
            "training_to_external_modality_shift": not modality_coverage[
                "full_multimodal_validation"
            ],
            "fusion_superiority_claim_allowed": bool(
                modality_coverage["full_multimodal_validation"]
            ),
            "note": (
                "Structure/conservation-dependent models are evaluated under missing-"
                "modality shift. This benchmark can test functional transfer but cannot "
                "by itself establish full-multimodal fusion superiority."
            ),
        }
    if policy == "gene_disjoint":
        small = bool(len(frame) < 100 or labels.sum() < 20 or frame[C.GENE_COL].nunique() < 20)
        result["gene_disjoint_sample_warning"] = (
            "small secondary subset; do not promote over the predeclared primary set"
            if small
            else None
        )
    rows: list[dict[str, Any]] = []
    prediction_table = frame[
        [
            column
            for column in (
                C.ROW_ID_COL,
                C.GENE_COL,
                C.LABEL_COL,
                "EXT_SOURCE",
                "ASSAY_ID",
                "uniprot_id",
                "variant_id",
                "genomic_variant_key",
                "all_genes",
                "gene_count",
                "annotation_row_count",
            )
            if column in frame
        ]
    ].copy()
    prediction_table.insert(0, "evaluation_set", set_name)
    for name, values in selected_contextual.items():
        prediction_table[f"{name}_raw_deleteriousness_score"] = values
    for name, model_probabilities in selected_probabilities.items():
        model_decisions = selected_decisions[name]
        metrics = C.evaluate(
            labels,
            model_probabilities,
            thresholds[name],
            predictions=model_decisions,
            decision_confidence=selected_confidence.get(name),
        )
        metrics["ci95"] = (
            _hierarchical_intervals(labels, model_probabilities, model_decisions, groups)
            if source == "clinvar" and inferential_reporting_allowed
            else None
        )
        metrics["output_scale"] = "calibrated_probability"
        metrics["model_role"] = _model_reporting_role(name)
        metrics["uncertainty_interpretation"] = (
            "descriptive_only_underpowered_external_cohort"
            if source == "clinvar" and not inferential_reporting_allowed
            else "group_resampled_interval_conditional_on_frozen_predictions"
            if source == "clinvar"
            else "descriptive_pooled_metric"
        )
        metrics["selective_prediction_interpretation"] = {
            "status": "descriptive_only",
            "selection_confidence": (
                "majority_margin_of_frozen_internal_fold_decisions"
                if name in selected_confidence
                else "distance_from_frozen_internal_threshold"
            ),
            "threshold_or_coverage_tuned_on_external_labels": False,
            "clinical_utility_claimed": False,
            "note": (
                "Risk-coverage values describe this frozen cohort; they are not a "
                "prospective abstention guarantee."
            ),
        }
        metrics["calibration_reporting"] = {
            "status": (
                "experimental_label_calibration_not_clinical"
                if source == "dms"
                else "descriptive_only_underpowered_external_cohort"
                if not inferential_reporting_allowed
                else "descriptive_external_calibration_assessment"
            ),
            "recalibrated_on_external_labels": False,
            "clinical_calibration_claimed": bool(
                source == "clinvar"
                and clinical_evidence is not None
                and clinical_evidence["clinical_calibration_claim_allowed"]
            ),
        }
        if source == "dms":
            metrics["reporting_role"] = "descriptive_pooled_row_metric_without_inferential_interval"
        result["models"][name] = metrics
        artifact[f"{set_name}__{name}"] = model_probabilities
        artifact[f"{set_name}__{name}__decisions"] = model_decisions
        artifact[f"{set_name}__{name}__threshold"] = np.asarray(
            [thresholds[name]], dtype=np.float64
        )
        if name in selected_confidence:
            artifact[f"{set_name}__{name}__decision_confidence"] = selected_confidence[name]
        prediction_table[f"{name}_probability"] = model_probabilities
        prediction_table[f"{name}_decision"] = model_decisions
        rows.append(
            {
                "source": source,
                "policy": policy,
                "role": result["role"],
                "analysis_unit": analysis_unit,
                "model": name,
                "model_role": metrics["model_role"],
                "n": len(frame),
                "row_n": len(row_frame),
                "unique_variant_n": len(frame) if source == "clinvar" else None,
                "positives": int(labels.sum()),
                "genes": int(frame[C.GENE_COL].nunique()),
                "assays": result["assays"],
                "full_multimodal_validation": result["full_multimodal_validation"],
                "training_to_external_modality_shift": (
                    source == "dms" and not result["full_multimodal_validation"]
                ),
                "clinical_evidence_status": (
                    clinical_evidence["status"] if clinical_evidence else None
                ),
                "inferential_claim_allowed": (
                    inferential_reporting_allowed if source == "clinvar" else False
                ),
                "clinical_calibration_claim_allowed": (
                    clinical_evidence["clinical_calibration_claim_allowed"]
                    if clinical_evidence
                    else False
                ),
                "clinical_utility_claim_allowed": False,
                "esm_masked_marginal_coverage": modality_coverage["esm_masked_marginal"][
                    "row_coverage"
                ],
                "structure_coverage": modality_coverage["structure"]["row_coverage"],
                "conservation_coverage": modality_coverage["conservation"]["row_coverage"],
                **{
                    metric: metrics.get(metric)
                    for metric in (
                        "mcc",
                        "auroc",
                        "auprc",
                        "recall",
                        "precision",
                        "f1",
                        "brier",
                        "threshold",
                    )
                },
            }
        )
    for name, model_scores in selected_rankings.items():
        metrics = _rank_metrics(labels, model_scores)
        metrics["model_role"] = _model_reporting_role(name)
        metrics["ci95"] = (
            _hierarchical_rank_intervals(labels, model_scores, groups)
            if source == "clinvar" and inferential_reporting_allowed
            else None
        )
        result["models"][name] = metrics
        artifact[f"{set_name}__{name}"] = model_scores
        prediction_table[f"{name}_raw_deleteriousness_score"] = model_scores
        rows.append(
            {
                "source": source,
                "policy": policy,
                "role": result["role"],
                "analysis_unit": analysis_unit,
                "model": name,
                "model_role": metrics["model_role"],
                "n": len(frame),
                "row_n": len(row_frame),
                "unique_variant_n": len(frame) if source == "clinvar" else None,
                "positives": int(labels.sum()),
                "genes": int(frame[C.GENE_COL].nunique()),
                "assays": result["assays"],
                "full_multimodal_validation": result["full_multimodal_validation"],
                "training_to_external_modality_shift": (
                    source == "dms" and not result["full_multimodal_validation"]
                ),
                "clinical_evidence_status": (
                    clinical_evidence["status"] if clinical_evidence else None
                ),
                "inferential_claim_allowed": (
                    inferential_reporting_allowed if source == "clinvar" else False
                ),
                "clinical_calibration_claim_allowed": (
                    clinical_evidence["clinical_calibration_claim_allowed"]
                    if clinical_evidence
                    else False
                ),
                "clinical_utility_claim_allowed": False,
                "esm_masked_marginal_coverage": modality_coverage["esm_masked_marginal"][
                    "row_coverage"
                ],
                "structure_coverage": modality_coverage["structure"]["row_coverage"],
                "conservation_coverage": modality_coverage["conservation"]["row_coverage"],
                **{
                    metric: metrics.get(metric)
                    for metric in (
                        "mcc",
                        "auroc",
                        "auprc",
                        "recall",
                        "precision",
                        "f1",
                        "brier",
                        "threshold",
                    )
                },
            }
        )
    primary_baseline = next(
        (
            name
            for name in (
                "esm_conservation_logistic",
                "raw_esm_zero_shot",
                "lightgbm",
            )
            if name in selected_probabilities
        ),
        None,
    )
    proposed_models = [
        name
        for name in DEEP_MODEL_NAMES
        if name in selected_probabilities and name != primary_baseline
    ]
    can_compare = (
        source == "clinvar"
        and primary_baseline is not None
        and bool(proposed_models)
        and len(np.unique(labels)) == 2
        and len(np.unique(groups)) > 1
        and inferential_reporting_allowed
    )
    if can_compare:
        clustered = {
            f"{model}_minus_{primary_baseline}": _hierarchical_model_comparison(
                labels,
                selected_probabilities[primary_baseline],
                selected_probabilities[model],
                selected_decisions[primary_baseline],
                selected_decisions[model],
                groups,
                seed=RANDOM_STATE + offset,
            )
            for offset, model in enumerate(proposed_models, 1)
        }
        for metric in ("mcc", "auroc", "auprc"):
            keys = [key for key, value in clustered.items() if value.get(metric)]
            if not keys:
                continue
            adjusted = multipletests(
                [clustered[key][metric]["two_sided_probability"] for key in keys],
                method="holm",
            )[1]
            for key, value in zip(keys, adjusted):
                clustered[key][metric]["holm_adjusted_probability"] = round(float(value), 4)
        result["comparisons"] = {
            "primary_baseline": primary_baseline,
            "clustered_paired": clustered,
            "bootstrap_scheme": bootstrap_scheme,
            "multiplicity_correction": "Holm within endpoint",
            "inference_scope": "conditional_on_frozen_model_predictions",
            "rowwise_mcnemar_policy": (
                "omitted_because_row_independence_is_invalid_for_clustered_variants"
            ),
        }
    else:
        result["comparisons"] = {
            "status": "skipped",
            "reason": (
                "use_per_assay_paired_bootstrap"
                if source == "dms"
                else (
                    "underpowered_clinical_cohort_reporting_guard"
                    if not inferential_reporting_allowed
                    else "insufficient_classes_or_groups"
                )
            ),
        }
    if source == "dms":
        result["per_assay"] = _dms_assay_results(
            frame,
            selected_probabilities,
            selected_decisions,
            thresholds,
            selected_rankings,
            dms_sampling,
            selected_confidence,
        )
        result["primary_endpoints"] = result["per_assay"].get("primary_endpoints", [])
        result["pooled_row_metrics_are_primary"] = False
        macro = result["per_assay"].get("macro", {})
        macro_ci95 = result["per_assay"].get("macro_ci95", {})
        for row in rows:
            name = row["model"]
            row["primary_endpoint_unit"] = "equal_weight_assay_macro"
            row["assay_macro_auroc"] = macro.get(name, {}).get("auroc")
            row["assay_macro_functional_spearman"] = macro.get(name, {}).get("functional_spearman")
            for metric in ("auroc", "functional_spearman"):
                interval = macro_ci95.get(name, {}).get(metric)
                row[f"assay_macro_{metric}_ci95_low"] = interval[0] if interval else None
                row[f"assay_macro_{metric}_ci95_high"] = interval[1] if interval else None
    elif policy == "exact_variant_disjoint":
        contextual = _contextual_predictor_benchmark(
            frame,
            selected_contextual,
            selected_probabilities,
            selected_rankings,
            set_name,
            artifact,
            inferential_reporting_allowed,
        )
        result["contextual_predictor_benchmark"] = contextual
        if contextual.get("available"):
            for name, record in contextual["individual_coverage"].items():
                metrics = record["metrics"]
                rows.append(
                    {
                        "source": source,
                        "policy": f"{policy}_predictor_available_case",
                        "role": "contextual_comparator",
                        "analysis_unit": analysis_unit,
                        "model": name,
                        "n": record["n"],
                        "row_n": len(row_frame),
                        "unique_variant_n": record["n"],
                        "positives": record["positives"],
                        "genes": record["genes"],
                        "assays": None,
                        "full_multimodal_validation": False,
                        "auroc": metrics.get("auroc"),
                        "auprc": metrics.get("auprc"),
                    }
                )
            common = contextual["common_coverage"]
            for name, metrics in common.get("models", {}).items():
                rows.append(
                    {
                        "source": source,
                        "policy": f"{policy}_contextual_common_coverage",
                        "role": "coverage_matched_contextual_comparison",
                        "analysis_unit": analysis_unit,
                        "model": name,
                        "n": common["n"],
                        "row_n": len(row_frame),
                        "unique_variant_n": common["n"],
                        "positives": common["positives"],
                        "genes": common["genes"],
                        "assays": None,
                        "full_multimodal_validation": False,
                        "auroc": metrics.get("auroc"),
                        "auprc": metrics.get("auprc"),
                    }
                )
    else:
        result["contextual_predictor_benchmark"] = {
            "available": False,
            "reason": "reported_only_for_predeclared_primary_clinvar_policy",
        }
    return result, rows, prediction_table


def _stress_group_ids(frame: pd.DataFrame, source: str) -> tuple[np.ndarray, str]:
    """Return analysis-aware groups without consulting labels."""
    if source == "dms":
        if "ASSAY_ID" not in frame:
            raise KeyError("DMS missing-modality stress requires ASSAY_ID")
        groups = frame["ASSAY_ID"].astype("string").fillna("").str.strip()
        if groups.eq("").any():
            raise ValueError("DMS stress-test assay identifiers are blank")
        return groups.to_numpy(dtype=object), "assay"
    keys = _genomic_keys(frame)
    if keys.eq("").any():
        raise ValueError("ClinVar stress-test rows require genomic variant keys")
    working = pd.DataFrame(
        {
            "key": keys,
            "gene": frame[C.GENE_COL].astype("string").fillna("").str.strip(),
        }
    )
    primary_by_variant = {
        key: _primary_gene(group["gene"])[0] for key, group in working.groupby("key", sort=False)
    }
    groups = working["key"].map(primary_by_variant).astype(str).to_numpy(dtype=object)
    if np.any(groups == ""):
        raise ValueError("ClinVar stress-test primary genes are blank")
    return groups, "primary_gene_with_variant_consistent_assignment"


def _stress_affected_models(scenario: str, model_names: list[str]) -> list[str]:
    """Select models whose declared input modality is modified."""
    affected = {"lightgbm", *DEEP_MODEL_NAMES}
    if scenario in {"conservation", "structure_and_conservation"}:
        affected.update({"conservation_logistic", "esm_conservation_logistic"})
    if scenario in {"structure", "structure_and_conservation"}:
        # This audit-only negative control consumes availability indicators that
        # are updated consistently when structure is hidden.
        affected.add("availability_logistic")
    return [name for name in model_names if name in affected]


def _stress_anchor(model_names: list[str]) -> str | None:
    for name in (
        "raw_esm_zero_shot",
        "esm_score_logistic",
        "esm_embedding_mutation",
        "esm_only",
        "mutation_logistic",
    ):
        if name in model_names:
            return name
    return None


def _stress_level_key(level: float) -> str:
    return f"mask_{int(round(100.0 * float(level))):03d}pct"


def _stress_evaluation_unit(
    source: str,
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    decisions: dict[str, np.ndarray],
    thresholds: dict[str, float],
    fold_probabilities: dict[str, np.ndarray],
    fold_thresholds: dict[str, np.ndarray],
    row_mask: np.ndarray,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Aggregate stress predictions to the same unit as primary evaluation."""
    if source != "clinvar":
        confidence = {
            name: _fold_vote_confidence(fold_probabilities[name], fold_thresholds[name])
            for name in probabilities
        }
        return frame, probabilities, decisions, confidence, row_mask
    units = _aggregate_clinvar_variants(
        frame,
        probabilities,
        decisions,
        {},
        thresholds,
        fold_probabilities,
        fold_thresholds,
    )
    keys = _genomic_keys(frame)
    mask_by_key: dict[str, bool] = {}
    for key, values in pd.DataFrame(
        {"key": keys, "masked": np.asarray(row_mask, dtype=bool)}
    ).groupby("key", sort=False):
        unique = values["masked"].unique()
        if len(unique) != 1:
            raise RuntimeError("ClinVar missing-modality mask split annotation rows of one variant")
        mask_by_key[str(key)] = bool(unique[0])
    unit_mask = units.frame["genomic_variant_key"].map(mask_by_key).to_numpy(bool)
    return (
        units.frame,
        units.probabilities,
        units.decisions,
        units.decision_confidence or {},
        unit_mask,
    )


def _run_missing_modality_stress(
    source: str,
    pair: DatasetPair,
    primary_mask: np.ndarray,
    feature_names: list[str],
    tabular_features: list[str],
    model_names: list[str],
    original_fold_probabilities: dict[str, np.ndarray],
    original_fold_thresholds: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate frozen models on nested, label-independent modality masks."""
    selected = np.asarray(primary_mask, dtype=bool).copy()
    if source == "clinvar":
        selected &= _genomic_keys(pair.frame).ne("").to_numpy()
    indices = np.flatnonzero(selected)
    if not len(indices):
        return {
            "status": "not_evaluated",
            "reason": "empty_primary_set",
            "protocol_version": PR.STRESS_PROTOCOL_VERSION,
        }, []
    if len(indices) == len(pair.embeddings):
        base_embeddings = pair.embeddings
        base_fold_probabilities = original_fold_probabilities
    else:
        base_embeddings = np.asarray(pair.embeddings[indices])
        base_fold_probabilities = {
            name: np.asarray(values, dtype=np.float64)[:, indices]
            for name, values in original_fold_probabilities.items()
        }
    base_pair = DatasetPair(
        name=f"{source}_missing_modality_stress",
        frame=pair.frame.iloc[indices].reset_index(drop=True),
        embeddings=base_embeddings,
        extraction_coverage=float(pair.extraction_coverage),
    )
    fold_thresholds = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in original_fold_thresholds.items()
    }
    base_probabilities, base_decisions, base_thresholds = _aggregate_predictions(
        {name: list(values) for name, values in base_fold_probabilities.items()},
        {name: list(values) for name, values in fold_thresholds.items()},
    )
    stress_groups, grouping_unit = _stress_group_ids(base_pair.frame, source)
    anchor = _stress_anchor(model_names)
    candidates = [name for name in DEEP_MODEL_NAMES if name in model_names]
    report: dict[str, Any] = {
        "status": "evaluated",
        "protocol_version": PR.STRESS_PROTOCOL_VERSION,
        "prespecified_levels": list(PR.STRESS_LEVELS),
        "selection_uses_labels": False,
        "masks_are_nested_within_scenario": True,
        "grouping_unit": grouping_unit,
        "primary_set_only": True,
        "anchor_model": anchor,
        "candidate_models": candidates,
        "safe_fallback_policy": (
            "use_the_frozen_sequence_anchor_exactly_for_masked_evaluation_units"
            if anchor
            else "unavailable_no_sequence_only_anchor"
        ),
        "interpretation": (
            "Prespecified descriptive sensitivity analysis. It does not estimate a "
            "missing-data mechanism, retrain models, or establish clinical utility."
        ),
        "scenarios": {},
    }
    table_rows: list[dict[str, Any]] = []
    for scenario_offset, (scenario, columns) in enumerate(PR.STRESS_SCENARIOS.items()):
        coverage = PR.source_modality_coverage(base_pair.frame, columns)
        affected_models = _stress_affected_models(scenario, model_names)
        scenario_report: dict[str, Any] = {
            "status": (
                "evaluated"
                if coverage["finite_source_cells"] > 0
                else "not_applicable_no_source_values"
            ),
            "source_coverage_before_masking": coverage,
            "affected_models_recomputed": affected_models,
            "invariant_models_reused_exactly": sorted(set(model_names) - set(affected_models)),
            "levels": {},
        }
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for level in PR.STRESS_LEVELS:
            key = _stress_level_key(level)
            plan = PR.deterministic_group_mask(
                stress_groups,
                level,
                seed=RANDOM_STATE,
                namespace=f"{source}:{scenario}",
                grouping_unit=grouping_unit,
            )
            masked_frame, masking_audit = PR.apply_modality_mask(
                base_pair.frame, plan.row_mask, columns
            )
            use_original = bool(
                level == 0.0
                or coverage["finite_source_cells"] == 0
                or masking_audit["finite_source_cells_removed"] == 0
            )
            stress_fold_probabilities = {
                name: values.copy() for name, values in base_fold_probabilities.items()
            }
            if not use_original:
                masked_pair = DatasetPair(
                    name=f"{base_pair.name}:{scenario}:{key}",
                    frame=masked_frame,
                    embeddings=base_pair.embeddings,
                    extraction_coverage=base_pair.extraction_coverage,
                )
                for fold in range(1, N_FOLDS + 1):
                    predicted, predicted_thresholds = _predict_fold(
                        fold,
                        masked_pair,
                        feature_names,
                        tabular_features,
                        affected_models,
                    )
                    for name in affected_models:
                        expected_threshold = float(fold_thresholds[name][fold - 1])
                        if not np.isclose(
                            predicted_thresholds[name],
                            expected_threshold,
                            rtol=0.0,
                            atol=1e-12,
                        ):
                            raise RuntimeError(
                                f"Stress inference changed frozen threshold for {name}"
                            )
                        stress_fold_probabilities[name][fold - 1] = predicted[name]
            probabilities, decisions, thresholds = _aggregate_predictions(
                {name: list(values) for name, values in stress_fold_probabilities.items()},
                {name: list(values) for name, values in fold_thresholds.items()},
            )
            unit_frame, unit_probabilities, unit_decisions, _, unit_mask = _stress_evaluation_unit(
                source,
                masked_frame,
                probabilities,
                decisions,
                thresholds,
                stress_fold_probabilities,
                fold_thresholds,
                plan.row_mask,
            )
            labels = unit_frame[C.LABEL_COL].to_numpy(dtype=int)
            metric_groups = _bootstrap_groups(unit_frame, source)
            dms_scores = (
                pd.to_numeric(unit_frame["DMS_SCORE"], errors="coerce").to_numpy(float)
                if source == "dms" and "DMS_SCORE" in unit_frame
                else None
            )
            level_models: dict[str, Any] = {}
            for name in model_names:
                metrics = PR.descriptive_stress_metrics(
                    labels,
                    unit_probabilities[name],
                    unit_decisions[name],
                    metric_groups,
                    dms_functional_scores=dms_scores,
                )
                if level == 0.0:
                    baseline_metrics[name] = metrics
                metrics["change_from_unmasked"] = {
                    metric: (
                        None
                        if metrics.get(metric) is None or baseline_metrics[name].get(metric) is None
                        else round(
                            float(metrics[metric]) - float(baseline_metrics[name][metric]),
                            6,
                        )
                    )
                    for metric in (
                        "auroc",
                        "auprc",
                        "mcc",
                        "brier",
                        "group_macro_auroc",
                        "group_macro_functional_spearman",
                    )
                }
                level_models[name] = metrics
            fallback_metrics: dict[str, Any] = {}
            if anchor is not None:
                for candidate in candidates:
                    fallback_probability = PR.safe_fallback(
                        unit_probabilities[candidate],
                        unit_probabilities[anchor],
                        unit_mask,
                    )
                    fallback_decision = PR.safe_fallback(
                        unit_decisions[candidate],
                        unit_decisions[anchor],
                        unit_mask,
                    ).astype(np.int8)
                    metrics = PR.descriptive_stress_metrics(
                        labels,
                        fallback_probability,
                        fallback_decision,
                        metric_groups,
                        dms_functional_scores=dms_scores,
                    )
                    masked_difference = (
                        np.max(
                            np.abs(
                                fallback_probability[unit_mask]
                                - unit_probabilities[anchor][unit_mask]
                            )
                        )
                        if unit_mask.any()
                        else 0.0
                    )
                    metrics["anchor_identity_on_masked_units"] = {
                        "satisfied": bool(masked_difference <= 1e-12),
                        "maximum_absolute_probability_difference": round(
                            float(masked_difference), 12
                        ),
                    }
                    fallback_metrics[candidate] = metrics
            level_report = {
                "mask_plan": {
                    **plan.audit(),
                    "evaluation_unit_masked_n": int(unit_mask.sum()),
                    "evaluation_unit_n": int(len(unit_mask)),
                    "evaluation_unit_realized_fraction": round(
                        float(unit_mask.mean()) if len(unit_mask) else 0.0, 6
                    ),
                },
                "masking_audit": masking_audit,
                "inference_reused_unmasked_predictions": use_original,
                "models": level_models,
                "safe_fallback_models": fallback_metrics,
            }
            scenario_report["levels"][key] = level_report
            for model_role, records in (
                ("stress_model", level_models),
                ("safe_fallback", fallback_metrics),
            ):
                for name, metrics in records.items():
                    output_name = (
                        name
                        if model_role == "stress_model"
                        else f"{name}__safe_fallback_to__{anchor}"
                    )
                    table_rows.append(
                        {
                            "source": source,
                            "policy": f"missing_modality_{scenario}_{key}",
                            "role": "descriptive_missing_modality_stress",
                            "analysis_unit": (
                                "unique_genomic_variant"
                                if source == "clinvar"
                                else "assay_variant_row"
                            ),
                            "model": output_name,
                            "model_role": model_role,
                            "n": int(len(labels)),
                            "row_n": int(len(base_pair.frame)),
                            "unique_variant_n": (int(len(labels)) if source == "clinvar" else None),
                            "positives": int(labels.sum()),
                            "genes": (
                                int(len(np.unique(metric_groups))) if source == "clinvar" else None
                            ),
                            "assays": (
                                int(len(np.unique(metric_groups))) if source == "dms" else None
                            ),
                            "stress_scenario": scenario,
                            "target_mask_fraction": float(level),
                            "realized_mask_fraction": level_report["mask_plan"][
                                "evaluation_unit_realized_fraction"
                            ],
                            "auroc": metrics.get("auroc"),
                            "auprc": metrics.get("auprc"),
                            "mcc": metrics.get("mcc"),
                            "brier": metrics.get("brier"),
                            "assay_macro_auroc": metrics.get("group_macro_auroc"),
                            "assay_macro_functional_spearman": metrics.get(
                                "group_macro_functional_spearman"
                            ),
                            "inferential_claim_allowed": False,
                            "clinical_utility_claim_allowed": False,
                        }
                    )
        report["scenarios"][scenario] = scenario_report
    return report, table_rows


def main() -> None:
    """Evaluate frozen internal models externally."""
    ensure_directories(STAGE12_OUT)
    C.set_seeds()
    available_external_sources: list[str] = []
    for name, paths in EXTERNAL_INPUTS.items():
        availability = [path.exists() for path in paths]
        if all(availability):
            available_external_sources.append(name)
        elif any(availability):
            raise FileNotFoundError(f"Incomplete {name} external artifacts: {paths}")
        else:
            logger.warning("Skipping unavailable external source %s", name)
    if not available_external_sources:
        raise RuntimeError("No external ESM datasets are available")
    model_artifact_paths = _validate_upstream_chain(available_external_sources)
    C.load_tuned_parameters()
    internal = _load_pair("internal", INTERNAL_CSV, INTERNAL_NPY, INTERNAL_STATUS)
    _, model_names = _validate_internal_reference(internal.frame)
    if C.RELIABILITY_ARCHITECTURE not in model_names:
        raise RuntimeError(
            f"Publication Stage 12 requires the Stage 11 {C.RELIABILITY_ARCHITECTURE} model; "
            "rerun Stages 14 and 11 with the v4 publication architecture set"
        )
    external: dict[str, DatasetPair] = {}
    for name in available_external_sources:
        external[name] = _load_pair(name, *EXTERNAL_INPUTS[name], require_both_classes=False)
    feature_names, tabular_features = _validate_feature_schema(internal.frame, external)
    masks = {
        source: _deoverlap_masks(INTERNAL_UNIVERSE, pair.frame) for source, pair in external.items()
    }
    aggregated: dict[
        str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]
    ] = {}
    fold_level_probabilities: dict[str, dict[str, np.ndarray]] = {}
    fold_level_thresholds: dict[str, dict[str, np.ndarray]] = {}
    aggregated_reliability_components: dict[str, dict[str, np.ndarray]] = {}
    fold_level_reliability_components: dict[str, dict[str, np.ndarray]] = {}
    ranking_baselines: dict[str, dict[str, np.ndarray]] = {}
    ranking_baseline_status: dict[str, dict[str, Any]] = {}
    contextual_predictors: dict[str, dict[str, np.ndarray]] = {}
    contextual_predictor_status: dict[str, dict[str, Any]] = {}
    fold_threshold_report: dict[str, dict[str, list[float]]] = {}
    for source, pair in external.items():
        fold_probabilities = {name: [] for name in model_names}
        fold_thresholds = {name: [] for name in model_names}
        fold_reliability = {name: [] for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS}
        cache_dir = STAGE12_OUT / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(1, N_FOLDS + 1):
            cache_file = cache_dir / f"{source}_fold_{fold}.npz"
            extracted_components: dict[str, np.ndarray] = {}
            if cache_file.is_file():
                with np.load(cache_file) as stored:
                    predicted = {name: stored[f"pred__{name}"] for name in model_names}
                    thresholds = {name: float(stored[f"thresh__{name}"]) for name in model_names}
                    for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS:
                        extracted_components[name] = stored[f"rel__{name}"]
                logger.info("Loaded cached predictions for %s fold %d/%d", source, fold, N_FOLDS)
            else:
                predicted, thresholds = _predict_fold(
                    fold,
                    pair,
                    feature_names,
                    tabular_features,
                    model_names,
                    reliability_components=extracted_components,
                )
                save_dict = {f"pred__{name}": predicted[name] for name in model_names}
                save_dict.update({f"thresh__{name}": np.array(thresholds[name]) for name in model_names})
                save_dict.update({f"rel__{name}": extracted_components[name] for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS})
                _atomic_npz(save_dict, cache_file)
                logger.info("Predicted %s with fold %d/%d (saved to cache)", source, fold, N_FOLDS)
            if set(extracted_components) != set(C.RELIABILITY_DIAGNOSTIC_COMPONENTS):
                raise RuntimeError(f"Fold {fold} {source} reliability diagnostics are incomplete")
            for name in model_names:
                fold_probabilities[name].append(predicted[name])
                fold_thresholds[name].append(thresholds[name])
            for name in C.RELIABILITY_DIAGNOSTIC_COMPONENTS:
                fold_reliability[name].append(extracted_components[name])
        aggregated[source] = _aggregate_predictions(fold_probabilities, fold_thresholds)
        fold_level_probabilities[source] = {
            name: np.stack(values).astype(np.float64, copy=False)
            for name, values in fold_probabilities.items()
        }
        fold_level_thresholds[source] = {
            name: np.asarray(values, dtype=np.float64) for name, values in fold_thresholds.items()
        }
        fold_level_reliability_components[source] = {
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in fold_reliability.items()
        }
        aggregated_reliability_components[source] = {
            name: values.mean(axis=0, dtype=np.float64).astype(np.float32)
            for name, values in fold_level_reliability_components[source].items()
        }
        zero_shot = _esm_zero_shot_scores(pair.frame)
        if zero_shot is not None and np.isfinite(zero_shot).all():
            ranking_baselines[source] = {ZERO_SHOT_MODEL: zero_shot}
            ranking_baseline_status[source] = {
                "available": True,
                "model": "ESM2 raw masked-marginal zero-shot",
                "source_column": MASKED_MARGINAL_COL,
                "transformation": "deleteriousness_score=-masked_marginal",
                "calibrated": False,
            }
        else:
            finite_n = int(np.isfinite(zero_shot).sum()) if zero_shot is not None else 0
            ranking_baselines[source] = {}
            ranking_baseline_status[source] = {
                "available": False,
                "reason": "masked_marginal_missing_or_incomplete",
                "finite_rows": finite_n,
                "total_rows": int(len(pair.frame)),
            }
            logger.warning("%s lacks a complete raw masked-marginal zero-shot baseline", source)
        contextual_scores, contextual_status = _contextual_predictor_scores(pair.frame)
        contextual_predictors[source] = contextual_scores
        contextual_predictor_status[source] = contextual_status
        fold_threshold_report[source] = fold_thresholds
    dms_sampling = _dms_sampling_audit()
    results: dict[str, Any] = {
        "model_tag": MODEL_TAG,
        "models": model_names,
        "ranking_baselines": ranking_baseline_status,
        "contextual_predictors": contextual_predictor_status,
        "contextual_predictor_policy": (
            "evaluation-only rank comparators; excluded from every VariFuse model input"
        ),
        "prediction_aggregation": (
            "mean_calibrated_probability_with_majority_vote_of_frozen_fold_thresholds"
        ),
        "clinvar_annotation_decision_aggregation": (
            "within_each_fold_mean_annotation_probability_then_frozen_fold_"
            "threshold_then_strict_fold_majority"
        ),
        "threshold_source": "dedicated_internal_fold_partitions",
        "primary_evaluation_policy": {
            "clinvar": {
                "set": "clinvar_exact_variant_disjoint",
                "analysis_unit": "unique_genomic_variant",
                "annotation_aggregation": "mean_model_output",
                "bootstrap": "gene_then_variant_hierarchical",
            },
            "dms": {
                "set": "dms_exact_variant_disjoint",
                "analysis_unit": "assay",
                "endpoints": ["macro_functional_spearman", "macro_auroc"],
                "bootstrap": "paired_assay_resampling",
            },
        },
        "gene_disjoint_policy": (
            "secondary stress test only; never selected as primary because it is stricter"
        ),
        "uncertainty_reporting_policy": {
            "risk_coverage": ("descriptive_fixed_coverage_summary_using_frozen_fold_vote_margin"),
            "external_label_tuning": False,
            "conformal": ("withheld_without_a_distinct_sufficient_exchangeable_calibration_sample"),
            "clinical_utility": "never_inferred_from_discrimination_or_calibration_alone",
            "underpowered_clinvar_guard": {
                "minimum_n": PR.MIN_INFERENCE_N,
                "minimum_per_class": PR.MIN_INFERENCE_PER_CLASS,
                "minimum_groups": PR.MIN_INFERENCE_GROUPS,
            },
        },
        "reliability_diagnostic_policy": {
            "status": "descriptive_post_selection_not_primary",
            "strata_use_labels": False,
            "used_for_model_or_threshold_selection": False,
            "external_label_refitting": False,
            "fold_aggregation": ("mean_components_across_frozen_internal_deployment_folds"),
            "component_names": list(C.RELIABILITY_DIAGNOSTIC_COMPONENTS),
        },
        "sets": {},
        "fold_thresholds": fold_threshold_report,
        "robustness": {
            "protocol_version": PR.STRESS_PROTOCOL_VERSION,
            "sources": {},
        },
    }
    artifact: dict[str, np.ndarray] = {}
    table_rows: list[dict[str, Any]] = []
    reliability_table_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    for source, pair in external.items():
        probabilities, decisions, thresholds = aggregated[source]
        for policy, mask in masks[source].items():
            result, rows, prediction_table = _evaluate_set(
                source,
                policy,
                pair,
                mask,
                probabilities,
                decisions,
                thresholds,
                artifact,
                ranking_baselines[source],
                contextual_predictors[source],
                dms_sampling if source == "dms" else None,
                fold_level_probabilities[source],
                fold_level_thresholds[source],
                aggregated_reliability_components[source],
                fold_level_reliability_components[source],
            )
            set_name = f"{source}_{policy}"
            results["sets"][set_name] = result
            table_rows.extend(rows)
            if "reliability_diagnostics" in result:
                reliability_table_rows.extend(
                    C.reliability_diagnostic_rows(
                        result["reliability_diagnostics"],
                        {
                            "scope": "external_validation",
                            "source": source,
                            "policy": policy,
                            "role": result["role"],
                            "analysis_unit": result["analysis_unit"],
                            "model": C.RELIABILITY_ARCHITECTURE,
                        },
                    )
                )
            if not prediction_table.empty:
                prediction_tables.append(prediction_table)
            logger.info(
                "%s retained %d/%d rows",
                set_name,
                int(mask.sum()),
                len(mask),
            )
        robustness, robustness_rows = _run_missing_modality_stress(
            source,
            pair,
            masks[source]["exact_variant_disjoint"],
            feature_names,
            tabular_features,
            model_names,
            fold_level_probabilities[source],
            fold_level_thresholds[source],
        )
        results["robustness"]["sources"][source] = robustness
        table_rows.extend(robustness_rows)
        logger.info("Completed prespecified missing-modality stress for %s", source)
    if not table_rows:
        raise RuntimeError("No external evaluation rows remain after de-overlap")
    if not reliability_table_rows:
        raise RuntimeError("No external reliability diagnostics were generated")
    _atomic_json(results, RESULTS_FILE)
    _atomic_csv(pd.DataFrame(table_rows), TABLE_FILE)
    _atomic_npz(artifact, PREDICTION_FILE)
    _atomic_csv(pd.concat(prediction_tables, ignore_index=True), PREDICTION_TABLE)
    _atomic_csv(pd.DataFrame(reliability_table_rows), RELIABILITY_DIAGNOSTICS_TABLE)
    input_paths = [
        INTERNAL_CSV,
        INTERNAL_NPY,
        INTERNAL_STATUS,
        STAGE10_MANIFESTS["internal"],
        INTERNAL_OOF,
        INTERNAL_UNIVERSE,
        STAGE07_MANIFEST,
        EXTERNAL_PREP_MANIFEST,
        *[STAGE09_PREPARED_OUTPUTS[name] for name in available_external_sources],
        STAGE11_OUT / "results.json",
        STAGE11_OUT / "run_manifest.json",
        TUNING_BEST_JSON,
        STAGE14_OUT / "best_concatenation_params.json",
        STAGE14_OUT / "best_gated_fusion_params.json",
        STAGE14_OUT / "architecture_selection.json",
        STAGE14_OUT / "nested_inner_splits.json",
        STAGE14_OUT / "run_manifest.json",
        *[STAGE10_MANIFESTS[name] for name in available_external_sources],
    ]
    if "dms" in available_external_sources:
        input_paths.append(DMS_SEQUENCE_OUTPUT)
    for architecture in C.RELIABILITY_FAMILY_ARCHITECTURES:
        tuning_path = STAGE14_OUT / f"best_{architecture}_params.json"
        if tuning_path.is_file():
            input_paths.append(tuning_path)
    input_paths.extend(model_artifact_paths)
    for paths in EXTERNAL_INPUTS.values():
        input_paths.extend(paths)
    write_run_manifest(
        MANIFEST_FILE,
        "12_external_validation",
        input_paths,
        {
            "model_tag": MODEL_TAG,
            "models": model_names,
            "gpu_runtime": C.gpu_runtime_summary(),
            "ranking_baselines": ranking_baseline_status,
            "contextual_predictors": contextual_predictor_status,
            "feature_names": feature_names,
            "sets": {name: result.get("n", 0) for name, result in results["sets"].items()},
            "clinvar_and_dms_pooled": False,
            "exact_variant_deoverlap": True,
            "gene_disjoint_reported": True,
            "primary_sets": [
                "clinvar_exact_variant_disjoint",
                "dms_exact_variant_disjoint",
            ],
            "clinvar_analysis_unit": "unique_genomic_variant",
            "dms_primary_analysis_unit": "assay",
            "dms_sampling": dms_sampling,
            "stage11_model_artifacts_hashed": len(model_artifact_paths),
            "stage14_configuration_artifacts_hashed": True,
            "robustness_protocol_version": PR.STRESS_PROTOCOL_VERSION,
            "robustness_sources": sorted(results["robustness"]["sources"]),
            "robustness_inferential_claims": False,
            "reliability_diagnostics": {
                "status": "descriptive_post_selection_not_primary",
                "strata_use_labels": False,
                "used_for_model_or_threshold_selection": False,
                "component_names": list(C.RELIABILITY_DIAGNOSTIC_COMPONENTS),
            },
        },
        outputs=[
            RESULTS_FILE,
            TABLE_FILE,
            PREDICTION_FILE,
            PREDICTION_TABLE,
            RELIABILITY_DIAGNOSTICS_TABLE,
        ],
    )
    logger.info("Saved separate ClinVar and DMS external validation")


if __name__ == "__main__":
    main()
