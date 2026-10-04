from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from calendar import monthrange
from datetime import date
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _choice_from_env(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower().replace("-", "_")
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


# CHANGELOG 2026-09: added a hyphen-preserving choice reader.  ``_choice_from_env``
# rewrites "-" to "_", which silently corrupts settings whose accepted literals
# contain hyphens (``ESM_SCORING_MODE="masked-marginal"``).  Downstream code
# compares those values with ``==``, so normalisation would disable masked
# marginal scoring without any error message.
def _literal_choice_from_env(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def _float_from_env(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


# CHANGELOG 2026-09: added a validating integer reader.  Roughly twenty-five
# integer settings previously used a bare ``int(os.environ.get(...))``, so a
# typo such as ESM_LAYER="" raised an opaque ``ValueError`` at import time and a
# semantically invalid value such as ESM_MAX_BATCH_TOKENS="0" propagated
# silently into batch arithmetic.  Bounds are declared at the definition site.
def _int_from_env(
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.environ.get(name)
    text = default if raw is None else raw.strip()
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _iso_date_from_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must use exact ISO format YYYY-MM-DD, got {value!r}") from error
    return parsed.isoformat()


def source_release_status(
    source: str,
    declared_release: str | None,
    *,
    cutoff: str | None = None,
) -> dict[str, Any]:
    """Compare a declared YYYY-MM or YYYY-MM-DD source release to the cutoff.

    A month-precision release is conservatively represented by the final day of
    that month. Missing dates remain explicitly missing; filesystem timestamps
    are never substituted for source release dates.
    """
    cutoff_text = cutoff or TRAIN_CUTOFF_DATE
    cutoff_date = date.fromisoformat(cutoff_text)
    value = (declared_release or "").strip()
    if not value:
        return {
            "source": source,
            "declared_release": None,
            "precision": None,
            "comparison_date": None,
            "cutoff_date": cutoff_text,
            "on_or_before_cutoff": None,
        }
    try:
        if len(value) == 7:
            year, month = (int(part) for part in value.split("-"))
            comparison = date(year, month, monthrange(year, month)[1])
            precision = "month"
        else:
            comparison = date.fromisoformat(value)
            precision = "day"
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source} release must use YYYY-MM or YYYY-MM-DD, got {value!r}"
        ) from error
    return {
        "source": source,
        "declared_release": value,
        "precision": precision,
        "comparison_date": comparison.isoformat(),
        "cutoff_date": cutoff_text,
        "on_or_before_cutoff": comparison <= cutoff_date,
    }


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _path_from_env("VARIANT_PROJECT_ROOT", SOURCE_DIR.parent)
DATA_DIR = _path_from_env("VARIANT_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = _path_from_env("VARIANT_OUTPUT_DIR", PROJECT_ROOT / "outputs")
FIGURE_DIR = _path_from_env("VARIANT_FIGURE_DIR", PROJECT_ROOT / "figures")
EXTERNAL_DATA_DIR = _path_from_env(
    "VARIANT_EXTERNAL_DATA_DIR", DATA_DIR / "external"
)
PUBLICATION_SOURCE_DIR = _path_from_env(
    "VARIANT_PUBLICATION_SOURCE_DIR", DATA_DIR / "publication_sources"
)

STAGE01_OUT = OUTPUT_DIR / "01_dbnsfp"
STAGE02_OUT = OUTPUT_DIR / "02_missing_values"
STAGE03_OUT = OUTPUT_DIR / "03_duplicates"
STAGE04_OUT = OUTPUT_DIR / "04_feature_engineering"
STAGE05_OUT = OUTPUT_DIR / "05_leakage"
STAGE06_OUT = OUTPUT_DIR / "06_clean"
STAGE07_OUT = OUTPUT_DIR / "07_natural_prevalence"
STAGE08_OUT = OUTPUT_DIR / "08_prepare_esm"
STAGE09_OUT = OUTPUT_DIR / "09_prepare_external_esm"
STAGE10_OUT = OUTPUT_DIR / "10_esm_features"
STAGE11_OUT = OUTPUT_DIR / "11_train_and_evaluate"
STAGE12_OUT = OUTPUT_DIR / "12_external_validation"
STAGE13_OUT = FIGURE_DIR
STAGE14_OUT = OUTPUT_DIR / "14_tuning"
EDA_OUT = OUTPUT_DIR / "eda_figures"

OUTPUT_DIRECTORIES = (
    STAGE01_OUT,
    STAGE02_OUT,
    STAGE03_OUT,
    STAGE04_OUT,
    STAGE05_OUT,
    STAGE06_OUT,
    STAGE07_OUT,
    STAGE08_OUT,
    STAGE09_OUT,
    STAGE10_OUT,
    STAGE11_OUT,
    STAGE12_OUT,
    STAGE13_OUT,
    STAGE14_OUT,
    EDA_OUT,
)

DBNSFP_RELEASE = os.environ.get("DBNSFP_RELEASE", "5.3a")
CLINVAR_TRAIN_RELEASE = os.environ.get("CLINVAR_TRAIN_RELEASE", "2024-06")
CLINVAR_EXTERNAL_RELEASE = os.environ.get("CLINVAR_EXTERNAL_RELEASE", "2026-08")
CGC_RELEASE = os.environ.get("CGC_RELEASE", "v102")
CIVIC_RELEASE = os.environ.get("CIVIC_RELEASE", "2026-01-01")
PROTEINGYM_RELEASE = os.environ.get("PROTEINGYM_RELEASE", "v1.3")
UNIPROT_RELEASE = os.environ.get("UNIPROT_RELEASE", "2026-01-07")
ALPHAFOLD_RELEASE = os.environ.get("ALPHAFOLD_RELEASE", "v6")
TRAIN_CUTOFF_DATE = _iso_date_from_env("VARIANT_TRAIN_CUTOFF_DATE", "2024-06-30")
LABEL_TASK = _choice_from_env(
    "VARIANT_LABEL_TASK",
    "clinical",
    {"clinical", "somatic", "legacy_mixed"},
)
LABEL_POLICY_VERSION = (
    "source_separated_temporal_v3_grch37_projection_stable_clinvar_id_scv"
)
GENOMIC_VARIANT_ASSEMBLY = "GRCh37"
DBNSFP_PRIMARY_ASSEMBLY = "GRCh38"
DBNSFP_GRCH37_CHROM_COLUMN = "hg19_chr"
DBNSFP_GRCH37_POSITION_COLUMN = "hg19_pos(1-based)"
DBNSFP_COORDINATE_POLICY = "explicit_hg19_columns_no_primary_fallback_v1"
COORDINATE_CONTRACT = {
    "variant_id_assembly": GENOMIC_VARIANT_ASSEMBLY,
    "dbnsfp_primary_assembly": DBNSFP_PRIMARY_ASSEMBLY,
    "dbnsfp_target_chromosome_column": DBNSFP_GRCH37_CHROM_COLUMN,
    "dbnsfp_target_position_column": DBNSFP_GRCH37_POSITION_COLUMN,
    "dbnsfp_projection_policy": DBNSFP_COORDINATE_POLICY,
}
ALLOW_LEGACY_MIXED_LABELS = _bool_from_env("ALLOW_LEGACY_MIXED_LABELS", False)
ALLOW_POST_CUTOFF_TRAINING_EVIDENCE = _bool_from_env(
    "ALLOW_POST_CUTOFF_TRAINING_EVIDENCE", False
)
COSMIC_CMC_RELEASE_DATE = os.environ.get("COSMIC_CMC_RELEASE_DATE", "").strip() or None
CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION = _bool_from_env(
    "CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION", True
)
CLINVAR_REQUIRE_SCV_EVIDENCE = _bool_from_env(
    "CLINVAR_REQUIRE_SCV_EVIDENCE", True
)
CLINVAR_SCV_MIN_MATCHING = int(os.environ.get("CLINVAR_SCV_MIN_MATCHING", "1"))
CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS = int(
    os.environ.get("CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS", "1")
)
CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM = int(
    os.environ.get("CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM", "2")
)
if min(
    CLINVAR_SCV_MIN_MATCHING,
    CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS,
    CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM,
) < 1:
    raise ValueError("ClinVar SCV evidence thresholds must be positive integers")
FEATURE_COVERAGE_POLICY = _choice_from_env(
    "FEATURE_COVERAGE_POLICY", "warn", {"off", "warn", "error"}
)
FEATURE_COVERAGE_WARN_GAP = _float_from_env(
    "FEATURE_COVERAGE_WARN_GAP", 0.10, 0.0, 1.0
)
REQUIRE_TRANSCRIPT_MAPPING = _bool_from_env("REQUIRE_TRANSCRIPT_MAPPING", True)
TRANSCRIPT_SELECTION_POLICY = (
    "protein_mapping_then_clinvar_source_gene_then_mane_select_"
    "then_mane_plus_clinical_then_vep_canonical_v2"
)
REQUIRE_HOMOLOGY_GROUPS = _bool_from_env("REQUIRE_HOMOLOGY_GROUPS", True)
HOMOLOGY_MIN_SEQUENCE_IDENTITY = _float_from_env(
    "HOMOLOGY_MIN_SEQ_ID", 0.30, 0.0, 1.0
)
HOMOLOGY_MIN_COVERAGE = _float_from_env(
    "HOMOLOGY_MIN_COVERAGE", 0.80, 0.0, 1.0
)
HOMOLOGY_COVERAGE_MODE = 0
HOMOLOGY_CLUSTER_MODE = 2
ALLOW_SEQUENCE_ONLY_DMS = _bool_from_env("ALLOW_SEQUENCE_ONLY_DMS", True)
REQUIRE_EXTERNAL_CLINVAR = _bool_from_env("REQUIRE_EXTERNAL_CLINVAR", True)
REQUIRE_EXTERNAL_DMS = _bool_from_env("REQUIRE_EXTERNAL_DMS", True)

DBNSFP_FILE = _path_from_env(
    "DBNSFP_FILE", DATA_DIR / f"dbNSFP{DBNSFP_RELEASE}_grch37.gz"
)
CIVIC_FILE = _path_from_env(
    "CIVIC_FILE", DATA_DIR / "01-Jan-2026-VariantSummaries.tsv"
)
CGC_FILE = _path_from_env(
    "CGC_FILE", DATA_DIR / "Cosmic_CancerGeneCensus_v102_GRCh37.tsv"
)
COSMIC_CMC_FILE = _path_from_env("COSMIC_CMC_FILE", DATA_DIR / "cmc_export.tsv")
ONCOKB_FILE = _path_from_env(
    "ONCOKB_FILE", DATA_DIR / "oncokb_biomarker_drug_associations.tsv"
)
UNIPROT_FILE = _path_from_env(
    "UNIPROT_FILE", DATA_DIR / "uniprotkb_proteome_UP000005640_2026_01_07.txt"
)
ALPHAFOLD_DIR = _path_from_env(
    "ALPHAFOLD_DIR", DATA_DIR / "UP000005640_9606_HUMAN_v6"
)
CLINVAR_TRAIN_ARCHIVE = _path_from_env(
    "CLINVAR_TRAIN_ARCHIVE",
    EXTERNAL_DATA_DIR / f"clinvar_{CLINVAR_TRAIN_RELEASE}_grch37.tsv",
)
CLINVAR_EXTERNAL_ARCHIVE = _path_from_env(
    "CLINVAR_EXTERNAL_ARCHIVE",
    EXTERNAL_DATA_DIR / f"clinvar_{CLINVAR_EXTERNAL_RELEASE}_grch37.tsv",
)
CLINVAR_TRAIN_TRANSFORMATION_MANIFEST = _path_from_env(
    "CLINVAR_TRAIN_TRANSFORMATION_MANIFEST",
    CLINVAR_TRAIN_ARCHIVE.with_name(
        CLINVAR_TRAIN_ARCHIVE.name + ".transformation.json"
    ),
)
CLINVAR_EXTERNAL_TRANSFORMATION_MANIFEST = _path_from_env(
    "CLINVAR_EXTERNAL_TRANSFORMATION_MANIFEST",
    CLINVAR_EXTERNAL_ARCHIVE.with_name(
        CLINVAR_EXTERNAL_ARCHIVE.name + ".transformation.json"
    ),
)
CLINVAR_TRAIN_SUBMISSION_ARCHIVE = _path_from_env(
    "CLINVAR_TRAIN_SUBMISSION_ARCHIVE",
    PUBLICATION_SOURCE_DIR
    / "clinvar"
    / CLINVAR_TRAIN_RELEASE
    / "tab_delimited"
    / f"submission_summary_{CLINVAR_TRAIN_RELEASE}.txt.gz",
)
CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE = _path_from_env(
    "CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE",
    PUBLICATION_SOURCE_DIR
    / "clinvar"
    / CLINVAR_EXTERNAL_RELEASE
    / "tab_delimited"
    / f"submission_summary_{CLINVAR_EXTERNAL_RELEASE}.txt.gz",
)
PROTEINGYM_SOURCE_DIR = PUBLICATION_SOURCE_DIR / "proteingym" / "1.3"
PROTEINGYM_EXTRACTION_ROOT = _path_from_env(
    "PROTEINGYM_EXTRACTION_ROOT", PROTEINGYM_SOURCE_DIR / "extracted"
)
PROTEINGYM_DIR = _path_from_env(
    "PROTEINGYM_DIR",
    PROTEINGYM_EXTRACTION_ROOT / "DMS_ProteinGym_substitutions",
)
PROTEINGYM_METADATA = _path_from_env(
    "PROTEINGYM_METADATA", PROTEINGYM_SOURCE_DIR / "DMS_substitutions.csv"
)
PROTEINGYM_EXTRACTION_MANIFEST = _path_from_env(
    "PROTEINGYM_EXTRACTION_MANIFEST",
    PROTEINGYM_EXTRACTION_ROOT / "extraction_manifest.json",
)
PROTEINGYM_METADATA_PROVENANCE = _path_from_env(
    "PROTEINGYM_METADATA_PROVENANCE",
    PROTEINGYM_METADATA.with_name(PROTEINGYM_METADATA.name + ".provenance.json"),
)
DMS_LEGACY_FILE = _path_from_env(
    "DMS_LEGACY_FILE", EXTERNAL_DATA_DIR / "dms_scores.csv"
)

RANDOM_STATE = _int_from_env("VARIANT_RANDOM_STATE", 42, minimum=0)
REPRODUCIBLE = _bool_from_env("VARIANT_REPRODUCIBLE", True)
MODEL_TAG = "predictor_free"
HASH_LARGE_FILES = _bool_from_env("VARIANT_HASH_LARGE_FILES", False)
AUDIT_SAMPLE_MAX_ROWS = _int_from_env(
    "AUDIT_SAMPLE_MAX_ROWS", 10000, minimum=1
)

ESM_MODEL_NAME = os.environ.get("ESM_MODEL_NAME", "esm2_t33_650M_UR50D")
ESM_DEVICE = _choice_from_env("ESM_DEVICE", "auto", {"auto", "cuda", "cpu"})
# CHANGELOG 2026-09: the integer ESM settings below moved from bare ``int()`` to
# ``_int_from_env`` with explicit bounds.  Values are unchanged; only invalid
# overrides now fail loudly at import time instead of corrupting batch sizing.
ESM_LAYER = _int_from_env("ESM_LAYER", 33, minimum=0)
ESM_EMBED_DIM = _int_from_env("ESM_EMBED_DIM", 1280, minimum=1)
ESM_WINDOW_SIZE = _int_from_env("ESM_WINDOW_SIZE", 1022, minimum=1)
ESM_MAX_BATCH_TOKENS = _int_from_env("ESM_MAX_BATCH_TOKENS", 2048, minimum=1)
ESM_MAX_BATCH_ATTENTION = _int_from_env(
    "ESM_MAX_BATCH_ATTENTION", 2000000, minimum=1
)
ESM_MAX_BATCH_PROTEINS = _int_from_env("ESM_MAX_BATCH_PROTEINS", 4, minimum=1)
# Masked forwards have separate row and attention budgets. Attention is measured
# as batch rows * padded tokens squared, including BOS/EOS. Zero inherits the
# corresponding ESM_MAX_BATCH_* limit; the token budget applies in either case.
ESM_MAX_MASKS_PER_BATCH = _int_from_env(
    "ESM_MAX_MASKS_PER_BATCH", 0, minimum=0
)
ESM_MASK_BATCH_ATTENTION = _int_from_env(
    "ESM_MASK_BATCH_ATTENTION", 0, minimum=0
)
ESM_SCORING_MODE = _literal_choice_from_env(
    "ESM_SCORING_MODE",
    "masked-marginal",
    {"masked-marginal", "wt-marginal", "both"},
)
ESM_USE_FP16 = _bool_from_env("ESM_USE_FP16", True)
# CHANGELOG 2026-09 (performance): Stage 10 context caches store float32 ESM
# embeddings, which are effectively incompressible; ``np.savez_compressed`` spent
# zlib time on the critical path for a few percent of size.  Uncompressed
# ``np.savez`` is now the default and compression is available for disk-bound
# machines.  Reading is unaffected -- ``np.load`` handles both.
ESM_CACHE_COMPRESS = _bool_from_env("ESM_CACHE_COMPRESS", False)
ESM_INTERNAL_MAX_ROWS = _int_from_env(
    "ESM_INTERNAL_MAX_ROWS", 200000, minimum=1
)
ESM_SAMPLE_MIN_PER_CLASS = _int_from_env(
    "ESM_SAMPLE_MIN_PER_CLASS", 0, minimum=0
)
DMS_MAX_ROWS_PER_ASSAY = _int_from_env(
    "DMS_MAX_ROWS_PER_ASSAY", 500, minimum=1
)
DMS_SAMPLING_POLICY = _choice_from_env(
    "DMS_SAMPLING_POLICY", "hash_uniform", {"all", "hash_uniform"}
)

ENABLE_LORA = _bool_from_env("ENABLE_ESM_LORA", False)
LORA_RANK = _int_from_env("LORA_RANK", 8, minimum=1)
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = tuple(
    item.strip()
    for item in os.environ.get(
        "LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,out_proj"
    ).split(",")
    if item.strip()
)
ESM_LR = float(os.environ.get("ESM_LR", "1e-5"))
HEAD_LR = float(os.environ.get("HEAD_LR", "3e-4"))
ESM_UNFREEZE_AFTER = _int_from_env("ESM_UNFREEZE_AFTER", 0, minimum=0)
LORA_MAX_EPOCHS = _int_from_env("LORA_MAX_EPOCHS", 12, minimum=1)
LORA_PATIENCE = _int_from_env("LORA_PATIENCE", 3, minimum=1)
LORA_BATCH_SIZE = _int_from_env("LORA_BATCH_SIZE", 4, minimum=1)
LORA_GRAD_ACCUM_STEPS = _int_from_env("LORA_GRAD_ACCUM_STEPS", 8, minimum=1)
LORA_MAX_RESIDUES = _int_from_env("LORA_MAX_RESIDUES", 1022, minimum=1)
LORA_GRADIENT_CHECKPOINTING = _bool_from_env(
    "LORA_GRADIENT_CHECKPOINTING", True
)
# CHANGELOG 2026-09: LORA_PRECISION was an unvalidated free-text string compared
# with ``in {"fp16", "bf16"}`` in common.py, so a typo such as "f16" silently
# disabled autocast instead of raising.  Note that "bf16" is unusable on the
# Turing GTX 1660 target; it is retained for the Kaggle T4/A100 profiles.
LORA_PRECISION = _literal_choice_from_env(
    "LORA_PRECISION", "fp16", {"fp16", "bf16", "fp32"}
)
LORA_SEED = _int_from_env("LORA_SEED", RANDOM_STATE, minimum=0)

# Select the proposed model before training; source/configuration provenance
# prevents changing that choice when reporting an existing run.
PROPOSED_ARCHITECTURE = _choice_from_env(
    "VARIFUSE_PROPOSED_ARCHITECTURE",
    "reliability_residual",
    {"reliability_residual", "evidential_residual"},
)
TUNING_BEST_JSON = STAGE14_OUT / "best_cross_attention_params.json"
TUNING_STORAGE = os.environ.get(
    "TUNING_STORAGE", f"sqlite:///{(STAGE14_OUT / 'optuna.db').as_posix()}"
)
REQUIRE_TUNING_ARTIFACT = _bool_from_env("REQUIRE_TUNING_ARTIFACT", True)


def ensure_directories(*directories: Path) -> None:
    """Create requested output directories explicitly."""
    selected = directories or OUTPUT_DIRECTORIES
    for directory in selected:
        Path(directory).mkdir(parents=True, exist_ok=True)


def file_sha256(
    path: Path,
    block_size: int = 1 << 20,
    *,
    force: bool = False,
) -> str | None:
    """Return a streaming SHA256 hash for a file.

    Large inputs are hashed only when ``VARIANT_HASH_LARGE_FILES`` is enabled;
    manifests always include a cheap sampled fingerprint as well.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        if (
            path.stat().st_size > 5 * (1 << 30)
            and not HASH_LARGE_FILES
            and not force
        ):
            return None
    except OSError:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sampled_file_sha256(path: Path, sample_size: int = 1 << 20) -> str | None:
    """Fingerprint file size plus its first and last blocks cheaply."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        digest = hashlib.sha256(str(size).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(sample_size))
            if size > sample_size:
                handle.seek(max(0, size - sample_size))
                digest.update(handle.read(sample_size))
        return digest.hexdigest()
    except OSError:
        return None


def directory_fingerprint(path: Path) -> dict[str, Any] | None:
    """Fingerprint directory membership and file metadata deterministically."""
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    try:
        for item in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        ):
            relative = item.relative_to(path).as_posix()
            stat = item.stat()
            file_count += 1
            total_size += stat.st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode())
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode())
            digest.update(b"\n")
    except OSError:
        return None
    return {
        "metadata_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "content_hash_policy": "path_size_mtime",
    }


def _input_manifest_record(
    path: Path, *, exact_file_hash: bool = False
) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "exists": exists,
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if exists else None,
    }
    if path.is_file():
        record.update(
            {
                "sha256": file_sha256(path, force=exact_file_hash),
                "sample_sha256": sampled_file_sha256(path),
                "size_bytes": path.stat().st_size,
                "hash_policy": (
                    "full_sha256" if exact_file_hash else "size_sample_and_optional_full"
                ),
            }
        )
    elif path.is_dir():
        record["directory_fingerprint"] = directory_fingerprint(path)
    return record


def artifact_record(path: Path) -> dict[str, Any]:
    """Return the reproducible byte-level record used for pipeline artifacts."""
    return _input_manifest_record(Path(path).resolve(), exact_file_hash=True)


def validate_clinvar_snapshot_provenance(
    archive: Path,
    transformation_manifest: Path,
    expected_release: str,
) -> dict[str, Any]:
    """Authenticate a derived ClinVar snapshot against its acquisition chain.

    Absolute source paths inside the acquisition manifest are deliberately not
    treated as identity: the configured archive may be copied to another drive.
    Its exact bytes, release, upstream raw SHA256 and acquisition-sidecar SHA256
    remain mandatory.
    """
    archive = Path(archive).resolve()
    transformation_manifest = Path(transformation_manifest).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Configured ClinVar snapshot is missing: {archive}")
    if not transformation_manifest.is_file():
        raise FileNotFoundError(
            "Configured ClinVar snapshot lacks its transformation manifest: "
            f"{transformation_manifest}. Reacquire/transform the pinned release."
        )
    try:
        payload = json.loads(transformation_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid ClinVar transformation manifest {transformation_manifest}: {error}"
        ) from error
    if (
        payload.get("schema_version") != 1
        or payload.get("transformation")
        != "clinvar_variant_summary_grch37_stream_v1"
        or payload.get("release") != expected_release
    ):
        raise RuntimeError(
            "ClinVar transformation contract/release differs from the configured "
            f"snapshot {expected_release}: {transformation_manifest}"
        )
    output = payload.get("output")
    source = payload.get("input")
    if not isinstance(output, dict) or not isinstance(source, dict):
        raise RuntimeError("ClinVar transformation manifest lacks input/output records")
    expected_size = output.get("size_bytes")
    expected_sha256 = output.get("sha256")
    raw_sha256 = source.get("sha256")
    provenance_sha256 = source.get("acquisition_provenance_sha256")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    if (
        not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_sha256, str)
        or sha_pattern.fullmatch(expected_sha256.lower()) is None
        or not isinstance(raw_sha256, str)
        or sha_pattern.fullmatch(raw_sha256.lower()) is None
        or not isinstance(provenance_sha256, str)
        or sha_pattern.fullmatch(provenance_sha256.lower()) is None
        or source.get("version") != expected_release
        or source.get("provider") != "NCBI ClinVar"
    ):
        raise RuntimeError(
            "ClinVar transformation manifest lacks complete acquisition provenance"
        )
    observed_size = archive.stat().st_size
    observed_sha256 = file_sha256(archive, force=True)
    if observed_size != expected_size or observed_sha256 != expected_sha256.lower():
        raise RuntimeError(
            "Configured ClinVar snapshot bytes differ from its transformation "
            f"manifest: {archive}"
        )
    return {
        "status": "authenticated",
        "release": expected_release,
        "rows_written": output.get("rows_written"),
        "snapshot_size_bytes": observed_size,
        "snapshot_sha256": observed_sha256,
        "raw_archive_sha256": raw_sha256.lower(),
        "acquisition_provenance_sha256": provenance_sha256.lower(),
        "transformation_manifest_sha256": file_sha256(
            transformation_manifest, force=True
        ),
    }


def validate_clinvar_submission_provenance(
    archive: Path, expected_release: str
) -> dict[str, Any]:
    """Authenticate an archived submission_summary and acquisition sidecar."""
    archive = Path(archive).resolve()
    provenance = archive.with_name(archive.name + ".provenance.json")
    if not archive.is_file():
        raise FileNotFoundError(f"ClinVar submission archive is missing: {archive}")
    if not provenance.is_file():
        raise FileNotFoundError(
            "ClinVar submission archive lacks acquisition provenance: "
            f"{provenance}"
        )
    try:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid ClinVar submission provenance: {provenance}") from error
    observed_sha256 = file_sha256(archive, force=True)
    recorded_sha256 = payload.get("observed_checksums", {}).get("sha256")
    if (
        payload.get("provider") != "NCBI ClinVar"
        or payload.get("version") != expected_release
        or not isinstance(recorded_sha256, str)
        or recorded_sha256.lower() != observed_sha256
    ):
        raise RuntimeError(
            "ClinVar submission archive differs from its pinned acquisition "
            f"provenance: {archive}"
        )
    return {
        "status": "authenticated",
        "release": expected_release,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": observed_sha256,
        "acquisition_provenance_sha256": file_sha256(provenance, force=True),
        "source_id": payload.get("source_id"),
        "publisher_checksum": payload.get("publisher_checksum"),
    }


def validate_proteingym_provenance(
    dms_directory: Path,
    metadata_file: Path,
    extraction_manifest: Path,
    metadata_provenance: Path,
    expected_release: str,
) -> dict[str, Any]:
    """Verify every extracted ProteinGym assay and its official metadata."""
    dms_directory = Path(dms_directory).resolve()
    metadata_file = Path(metadata_file).resolve()
    extraction_manifest = Path(extraction_manifest).resolve()
    metadata_provenance = Path(metadata_provenance).resolve()
    required = (dms_directory, metadata_file, extraction_manifest, metadata_provenance)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "ProteinGym publication inputs/provenance are incomplete: " + ", ".join(missing)
        )
    try:
        extraction = json.loads(extraction_manifest.read_text(encoding="utf-8"))
        metadata_record = json.loads(metadata_provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid ProteinGym provenance JSON: {error}") from error
    normalized_release = expected_release.lower().removeprefix("v")
    archive = extraction.get("archive")
    members = extraction.get("members")
    if (
        extraction.get("schema_version") != 1
        or extraction.get("extraction_policy") != "safe_atomic_zip_v1"
        or not isinstance(archive, dict)
        or str(archive.get("version", "")).lower().removeprefix("v")
        != normalized_release
        or not isinstance(members, list)
    ):
        raise RuntimeError("ProteinGym extraction manifest contract/release differs")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    if (
        sha_pattern.fullmatch(str(archive.get("sha256", "")).lower()) is None
        or sha_pattern.fullmatch(
            str(archive.get("acquisition_provenance_sha256", "")).lower()
        )
        is None
        or not isinstance(archive.get("publisher_checksum"), dict)
    ):
        raise RuntimeError("ProteinGym archive lacks complete acquisition provenance")
    expected_files: dict[str, dict[str, Any]] = {}
    for record in members:
        if not isinstance(record, dict) or record.get("kind") != "file":
            continue
        member = PurePosixPath(str(record.get("path", "")))
        if (
            member.is_absolute()
            or ".." in member.parts
            or len(member.parts) < 2
            or member.parts[0] != dms_directory.name
        ):
            raise RuntimeError("ProteinGym extraction manifest has an unsafe member path")
        relative = PurePosixPath(*member.parts[1:]).as_posix()
        if relative in expected_files:
            raise RuntimeError("ProteinGym extraction manifest repeats a file member")
        expected_files[relative] = record
    observed_files = {
        path.relative_to(dms_directory).as_posix(): path
        for path in dms_directory.rglob("*")
        if path.is_file()
    }
    if set(observed_files) != set(expected_files):
        raise RuntimeError(
            "ProteinGym extracted assay inventory differs from its verified archive"
        )
    total_bytes = 0
    for relative, path in observed_files.items():
        record = expected_files[relative]
        expected_size = record.get("size_bytes")
        expected_sha256 = str(record.get("sha256", "")).lower()
        if (
            not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
            or sha_pattern.fullmatch(expected_sha256) is None
            or file_sha256(path, force=True) != expected_sha256
        ):
            raise RuntimeError(f"ProteinGym extracted assay differs: {relative}")
        total_bytes += expected_size
    if (
        extraction.get("file_count") != len(expected_files)
        or extraction.get("total_uncompressed_bytes") != total_bytes
    ):
        raise RuntimeError("ProteinGym extraction totals differ from member records")
    metadata_sha256 = file_sha256(metadata_file, force=True)
    observed_checksums = metadata_record.get("observed_checksums")
    if (
        metadata_record.get("schema_version") != 1
        or metadata_record.get("provider") != "ProteinGym"
        or str(metadata_record.get("version", "")).lower().removeprefix("v")
        != normalized_release
        or not isinstance(observed_checksums, dict)
        or observed_checksums.get("sha256") != metadata_sha256
        or metadata_record.get("observed_size_bytes") != metadata_file.stat().st_size
        or not isinstance(metadata_record.get("publisher_checksum"), dict)
    ):
        raise RuntimeError("ProteinGym metadata differs from its acquisition provenance")
    return {
        "status": "authenticated",
        "release": expected_release,
        "assay_files": len(expected_files),
        "assay_bytes": total_bytes,
        "archive_sha256": str(archive["sha256"]).lower(),
        "extraction_manifest_sha256": file_sha256(extraction_manifest, force=True),
        "metadata_sha256": metadata_sha256,
        "metadata_provenance_sha256": file_sha256(metadata_provenance, force=True),
    }


def output_artifact_id(path: Path, stage: str | None = None) -> str:
    """Return a drive-independent identifier for a generated pipeline artifact.

    Publication outputs live below ``VARIANT_OUTPUT_DIR``.  The detached fallback
    exists for isolated unit tests and explicitly custom, non-publication stage
    runs; normal manifests therefore use IDs such as
    ``04_feature_engineering/table.parquet`` on every machine.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(OUTPUT_DIR.resolve()).as_posix()
    except ValueError:
        stage_id = (stage or "detached").split(":", 1)[0].replace(" ", "_")
        return f"_detached/{stage_id}/{resolved.name}"


def _looks_absolute_artifact_id(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _artifact_records_match(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Compare artifact bytes without treating a harmless mtime change as corruption."""
    if expected.get("exists") is not True or observed.get("exists") is not True:
        return False
    if expected.get("kind") != observed.get("kind"):
        return False
    if expected.get("kind") == "file":
        for key in ("size_bytes", "sample_sha256"):
            if expected.get(key) is None or expected.get(key) != observed.get(key):
                return False
        expected_full = expected.get("sha256")
        observed_full = observed.get("sha256")
        # Upstream pipeline artifacts are always bound by a complete content
        # digest.  Size/head/tail-only legacy records are intentionally rejected.
        return (
            expected.get("hash_policy") == "full_sha256"
            and expected_full is not None
            and observed_full is not None
            and expected_full == observed_full
        )
    if expected.get("kind") == "directory":
        return expected.get("directory_fingerprint") == observed.get(
            "directory_fingerprint"
        )
    return False


def validate_upstream_manifest(
    manifest_path: Path,
    expected_stage: str,
    required_artifacts: Iterable[Path],
    *,
    required_source_files: Iterable[str] = (),
) -> dict[str, Any]:
    """Fail closed when an upstream artifact or publication contract is stale.

    The producing manifest must bind the exact bytes consumed by the downstream
    stage and must use the same label task, policy, and exact temporal cutoff as
    the current process. Legacy manifests that did not record outputs are not
    silently upgraded because their bytes cannot be authenticated.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Required upstream manifest is missing: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid upstream manifest: {manifest_path}") from error
    if payload.get("stage") != expected_stage:
        raise RuntimeError(
            f"Upstream manifest stage mismatch: expected {expected_stage!r}, "
            f"found {payload.get('stage')!r}"
        )
    chained_stages = {
        "02_remove_missing_values",
        "03_remove_duplicates",
        "04_feature_engineering",
        "05_remove_leakage",
        "06_clean_and_finalize",
        "07_dataset_balancing",
        "08_prepare_esm_dataset",
        "08b_build_homology_groups",
    }
    if (
        expected_stage in chained_stages
        and payload.get("extra", {}).get("upstream_validation") != "passed"
    ):
        raise RuntimeError(
            "Upstream stage was produced with lineage validation disabled; "
            "regenerate it through the publication pipeline"
        )
    expected_contract = {
        "label_task": LABEL_TASK,
        "label_policy_version": LABEL_POLICY_VERSION,
        "training_cutoff_date": TRAIN_CUTOFF_DATE,
    }
    mismatches = {
        key: {"manifest": payload.get(key), "current": value}
        for key, value in expected_contract.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Upstream label/temporal contract differs from the current process: "
            f"{mismatches}. Rerun from Stage 01 in a clean output directory."
        )
    if payload.get("coordinate_contract") != COORDINATE_CONTRACT:
        raise RuntimeError(
            "Upstream genome-assembly coordinate contract differs from the current "
            "publication run. dbNSFP primary GRCh38 coordinates must never be "
            "labelled as GRCh37; regenerate from Stage 01."
        )
    expected_mapping_contract = {
        "require_transcript_mapping": REQUIRE_TRANSCRIPT_MAPPING,
        "variant_selection": TRANSCRIPT_SELECTION_POLICY,
        "require_homology_groups": REQUIRE_HOMOLOGY_GROUPS,
    }
    if payload.get("internal_mapping_contract") != expected_mapping_contract:
        raise RuntimeError(
            "Upstream transcript/homology contract differs from the current "
            "publication run; regenerate the upstream stage"
        )
    if expected_stage == "09_prepare_external_esm_dataset":
        expected_external_contract = {
            "require_clinvar": REQUIRE_EXTERNAL_CLINVAR,
            "require_dms": REQUIRE_EXTERNAL_DMS,
            "strict_clinvar_post_cutoff_last_evaluated": (
                CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION
            ),
            "require_clinvar_scv_evidence": CLINVAR_REQUIRE_SCV_EVIDENCE,
            "clinvar_scv_min_matching": CLINVAR_SCV_MIN_MATCHING,
            "clinvar_scv_min_unique_submitters": (
                CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS
            ),
            "clinvar_scv_multiple_submitter_minimum": (
                CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM
            ),
            "dms_sampling_policy": DMS_SAMPLING_POLICY,
            "dms_max_rows_per_assay": DMS_MAX_ROWS_PER_ASSAY,
            "allow_sequence_only_dms": ALLOW_SEQUENCE_ONLY_DMS,
        }
        if payload.get("external_contract") != expected_external_contract:
            raise RuntimeError(
                "Upstream external-cohort contract differs from the current run; "
                "regenerate Stage 09"
            )
    recorded_outputs = payload.get("outputs")
    if not isinstance(recorded_outputs, dict):
        raise RuntimeError(
            f"Upstream manifest {manifest_path} does not bind output artifact bytes; "
            "legacy outputs must be regenerated"
        )
    absolute_keys = [
        key
        for key in recorded_outputs
        if isinstance(key, str) and _looks_absolute_artifact_id(key)
    ]
    if absolute_keys:
        raise RuntimeError(
            "Upstream manifest uses legacy absolute output paths and is not "
            "portable; regenerate the upstream stage with artifact manifest v2"
        )
    if payload.get("artifact_manifest_version") != 2:
        raise RuntimeError(
            "Upstream manifest predates exact portable artifact binding; regenerate "
            "the upstream stage"
        )
    for artifact in required_artifacts:
        resolved = Path(artifact).resolve()
        artifact_id = output_artifact_id(resolved, expected_stage)
        recorded = recorded_outputs.get(artifact_id)
        if recorded is None:
            raise RuntimeError(
                "Upstream manifest does not declare required artifact "
                f"{artifact_id} ({resolved})"
            )
        if not isinstance(recorded, dict) or recorded.get("artifact_id") != artifact_id:
            raise RuntimeError(
                f"Upstream artifact record is malformed for {artifact_id}"
            )
        observed = artifact_record(resolved)
        if not _artifact_records_match(recorded, observed):
            raise RuntimeError(
                f"Upstream artifact fingerprint mismatch for {resolved}; rerun "
                f"{expected_stage} before continuing"
            )
    source_hashes = payload.get("source_files")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("Upstream manifest lacks source-file provenance")
    producing_source = f"{expected_stage.split(':', 1)[0].removeprefix('stage_')}.py"
    stage_implementation_dependencies = {
        "01_dbnsfp_processor": {"clinvar_identity.py"},
        "09_prepare_external_esm_dataset": {
            "01_dbnsfp_processor.py",
            "04_feature_engineering.py",
            "08_prepare_esm_dataset.py",
            "clinvar_identity.py",
        },
        "11_train_and_evaluate": {"common.py"},
        "12_external_validation": {"common.py"},
        "14_tune_cross_attention": {"common.py"},
    }
    required_sources = {
        producing_source,
        "config.py",
        "schema.py",
        "table_io.py",
    }
    required_sources.update(
        stage_implementation_dependencies.get(expected_stage, set())
    )
    required_sources.update(str(name) for name in required_source_files)
    missing_sources = [
        name
        for name in sorted(required_sources)
        if name not in source_hashes or not (SOURCE_DIR / name).is_file()
    ]
    if missing_sources:
        raise RuntimeError(
            "Upstream manifest lacks required implementation provenance for "
            f"{missing_sources}; regenerate the upstream stage"
        )
    changed_sources = []
    for name in sorted(required_sources):
        source = SOURCE_DIR / name
        if source_hashes[name] != file_sha256(source):
            changed_sources.append(name)
    if changed_sources:
        raise RuntimeError(
            "Upstream implementation changed after its artifacts were created: "
            f"{changed_sources}. Regenerate the upstream stage."
        )
    return payload


def validate_manifest_input_bindings(
    manifest_path: Path,
    expected_stage: str,
    required_inputs: Iterable[Path],
) -> dict[str, Any]:
    """Cryptographically bind portable current files to a producer's inputs.

    Historical input-map keys are absolute producer-machine paths.  Matching by
    portable basename plus exact SHA256 allows an authenticated artifact bundle
    to move to Kaggle or another drive while still proving which bytes the
    producer consumed. Ambiguous basenames fail closed.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid manifest: {manifest_path}") from error
    if payload.get("stage") != expected_stage:
        raise RuntimeError(
            f"Manifest stage mismatch: expected {expected_stage!r}, "
            f"found {payload.get('stage')!r}"
        )
    records = payload.get("inputs")
    if not isinstance(records, dict):
        raise RuntimeError(f"Manifest {manifest_path} lacks input provenance")
    for required in required_inputs:
        path = Path(required).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required bound input is missing: {path}")
        observed_sha256 = file_sha256(path, force=True)
        if observed_sha256 is None:
            raise RuntimeError(f"Could not hash required bound input: {path}")
        candidates: list[tuple[str, dict[str, Any]]] = []
        for key, record in records.items():
            if not isinstance(key, str) or not isinstance(record, dict):
                continue
            portable_names = {
                PureWindowsPath(key).name,
                PurePosixPath(key).name,
            }
            if path.name in portable_names:
                candidates.append((key, record))
        matches = [
            (key, record)
            for key, record in candidates
            if record.get("exists") is True
            and record.get("kind") == "file"
            and record.get("sha256") == observed_sha256
            and record.get("size_bytes") == path.stat().st_size
        ]
        if len(matches) != 1:
            reason = (
                "no exact SHA256 match"
                if not matches
                else "ambiguous exact SHA256 matches"
            )
            raise RuntimeError(
                f"Manifest {manifest_path} does not cryptographically bind "
                f"portable input {path.name}: {reason}. Regenerate the producer "
                "artifact chain."
            )
    return payload


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "fair-esm",
        "lightgbm",
        "biopython",
        "shap",
        "optuna",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_manifest(
    stage: str,
    inputs: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
    outputs: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build reproducibility metadata for a stage."""
    resolved_inputs = [Path(path).resolve() for path in inputs]
    resolved_outputs = [Path(path).resolve() for path in outputs]
    source_files = sorted(SOURCE_DIR.glob("*.py"))
    manifest: dict[str, Any] = {
        "stage": stage,
        "artifact_manifest_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "seed": RANDOM_STATE,
        "reproducible": REPRODUCIBLE,
        "model_tag": MODEL_TAG,
        "label_task": LABEL_TASK,
        "label_policy_version": LABEL_POLICY_VERSION,
        "training_cutoff_date": TRAIN_CUTOFF_DATE,
        "coordinate_contract": COORDINATE_CONTRACT,
        "esm_model": ESM_MODEL_NAME,
        "esm_layer": ESM_LAYER,
        "esm_contract": {
            "device_request": ESM_DEVICE,
            "embedding_dimension": ESM_EMBED_DIM,
            "window_size": ESM_WINDOW_SIZE,
            "scoring_mode": ESM_SCORING_MODE,
            "fp16": ESM_USE_FP16,
        },
        "external_contract": {
            "require_clinvar": REQUIRE_EXTERNAL_CLINVAR,
            "require_dms": REQUIRE_EXTERNAL_DMS,
            "strict_clinvar_post_cutoff_last_evaluated": (
                CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION
            ),
            "require_clinvar_scv_evidence": CLINVAR_REQUIRE_SCV_EVIDENCE,
            "clinvar_scv_min_matching": CLINVAR_SCV_MIN_MATCHING,
            "clinvar_scv_min_unique_submitters": (
                CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS
            ),
            "clinvar_scv_multiple_submitter_minimum": (
                CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM
            ),
            "dms_sampling_policy": DMS_SAMPLING_POLICY,
            "dms_max_rows_per_assay": DMS_MAX_ROWS_PER_ASSAY,
            "allow_sequence_only_dms": ALLOW_SEQUENCE_ONLY_DMS,
        },
        "internal_mapping_contract": {
            "require_transcript_mapping": REQUIRE_TRANSCRIPT_MAPPING,
            "variant_selection": TRANSCRIPT_SELECTION_POLICY,
            "require_homology_groups": REQUIRE_HOMOLOGY_GROUPS,
        },
        "source_files": {
            path.name: file_sha256(path) for path in source_files
        },
        "data_releases": {
            "dbnsfp": DBNSFP_RELEASE,
            "clinvar_train": CLINVAR_TRAIN_RELEASE,
            "clinvar_external": CLINVAR_EXTERNAL_RELEASE,
            "cgc": CGC_RELEASE,
            "civic": CIVIC_RELEASE,
            "cosmic_cmc_date": COSMIC_CMC_RELEASE_DATE,
            "proteingym": PROTEINGYM_RELEASE,
            "uniprot": UNIPROT_RELEASE,
            "alphafold": ALPHAFOLD_RELEASE,
        },
        "inputs": {str(path): _input_manifest_record(path) for path in resolved_inputs},
        "outputs": {
            artifact_id: {
                **_input_manifest_record(path, exact_file_hash=True),
                "artifact_id": artifact_id,
            }
            for path in resolved_outputs
            for artifact_id in (output_artifact_id(path, stage),)
        },
        "packages": _package_versions(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def json_default(value: Any) -> Any:
    """Convert common scientific values for JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_run_manifest(
    path: Path,
    stage: str,
    inputs: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
    outputs: Iterable[Path] = (),
) -> None:
    """Write a stage manifest atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = build_run_manifest(stage, inputs, extra, outputs)
    temporary.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)
