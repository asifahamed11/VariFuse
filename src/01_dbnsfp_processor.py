from __future__ import annotations

import hashlib
import json
import logging
import gc
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from clinvar_identity import normalize_variation_ids
from config import (
    ALLOW_LEGACY_MIXED_LABELS,
    ALLOW_POST_CUTOFF_TRAINING_EVIDENCE,
    CGC_FILE,
    CIVIC_FILE,
    CIVIC_RELEASE,
    CLINVAR_TRAIN_ARCHIVE,
    CLINVAR_TRAIN_RELEASE,
    CLINVAR_TRAIN_TRANSFORMATION_MANIFEST,
    COORDINATE_CONTRACT,
    COSMIC_CMC_FILE,
    COSMIC_CMC_RELEASE_DATE,
    DBNSFP_FILE,
    DBNSFP_GRCH37_CHROM_COLUMN,
    DBNSFP_GRCH37_POSITION_COLUMN,
    GENOMIC_VARIANT_ASSEMBLY,
    LABEL_POLICY_VERSION,
    LABEL_TASK,
    ONCOKB_FILE,
    STAGE01_OUT,
    TRAIN_CUTOFF_DATE,
    ensure_directories,
    json_default,
    source_release_status,
    validate_clinvar_snapshot_provenance,
    write_run_manifest,
)
from schema import LABEL_COL, RAW_CONTEXTUAL_PREDICTOR_COLS, ROW_ID_COL, STANDARD_AA

warnings.simplefilter(action="ignore", category=FutureWarning)
try:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
except AttributeError:
    warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage01_dbnsfp")

CHUNK_SIZE = int(os.environ.get("DBNSFP_CHUNK_SIZE", "5000"))
if CHUNK_SIZE < 1_000:
    raise ValueError("DBNSFP_CHUNK_SIZE must be at least 1000")
OUTPUT_FILE = STAGE01_OUT / "somatic_variant_dbNSFP.csv"
METADATA_FILE = STAGE01_OUT / "somatic_variant_dbNSFP.json"
IDENTITY_AUDIT_FILE = STAGE01_OUT / "clinvar_training_identity_audit.csv"
MANIFEST_FILE = STAGE01_OUT / "run_manifest.json"
BA1_AF_THRESHOLD = 0.05
MIN_COSMIC_EVIDENCE = 10
MIN_COSMIC_RESCUE = 50
HIGH_CONFIDENCE_REVIEWS = {
    "practice guideline",
    "reviewed by expert panel",
    "criteria provided, multiple submitters, no conflicts",
}

RESCUE_HOTSPOTS = {
    "TP53": {
        175,
        179,
        193,
        213,
        220,
        234,
        236,
        238,
        241,
        244,
        245,
        248,
        249,
        273,
        277,
        278,
        280,
        281,
        282,
    },
    "KRAS": {12, 13, 61, 117, 146},
    "NRAS": {12, 13, 61, 117, 146},
    "HRAS": {12, 13, 61},
    "BRAF": {469, 594, 596, 597, 600, 601},
    "PIK3CA": {88, 93, 111, 345, 420, 453, 542, 545, 1043, 1047},
    "EGFR": {709, 719, 768, 769, 773, 790, 858, 861},
    "IDH1": {132},
    "IDH2": {140, 172},
    "FGFR3": {249, 373, 375},
    "CTNNB1": {32, 33, 34, 37, 41, 45},
    "AKT1": {17},
    "ERBB2": {755, 777},
    "KIT": {816, 822},
    "MET": {1010, 1268},
    "ALK": {1196, 1269},
    "ERBB3": {104, 107},
    "GNAS": {201, 227},
    "MAP2K1": {56, 124},
    "PDGFRA": {842, 845},
}


def _validate_label_configuration() -> dict[str, dict[str, Any]]:
    """Validate label-task and temporal-source contracts before scanning dbNSFP."""
    clinvar_status = source_release_status("ClinVar training snapshot", CLINVAR_TRAIN_RELEASE)
    if not clinvar_status["on_or_before_cutoff"]:
        raise RuntimeError(
            f"ClinVar training release is after VARIANT_TRAIN_CUTOFF_DATE: {clinvar_status}"
        )
    civic_status = source_release_status("CIViC", CIVIC_RELEASE)
    cosmic_status = source_release_status("COSMIC CMC", COSMIC_CMC_RELEASE_DATE)
    clinvar_status["used_for_labels"] = True
    civic_status["used_for_labels"] = LABEL_TASK == "legacy_mixed"
    cosmic_status["used_for_labels"] = LABEL_TASK == "legacy_mixed"
    if LABEL_TASK == "somatic":
        raise RuntimeError(
            "VARIANT_LABEL_TASK=somatic is guarded because this project has no "
            "temporally qualified, same-domain somatic negative cohort. Do not use "
            "ClinVar benign or gnomAD BA1 variants as silent somatic passengers. "
            "Provide and implement a declared somatic-negative source first."
        )
    if LABEL_TASK == "legacy_mixed":
        if not ALLOW_LEGACY_MIXED_LABELS:
            raise RuntimeError(
                "legacy_mixed combines clinical, population, and somatic evidence. "
                "Set ALLOW_LEGACY_MIXED_LABELS=1 only for an explicitly labelled "
                "legacy sensitivity analysis."
            )
        if cosmic_status["declared_release"] is None:
            raise RuntimeError(
                "COSMIC_CMC_RELEASE_DATE=YYYY-MM-DD is required when COSMIC creates "
                "training labels; filesystem timestamps are not accepted as provenance."
            )
        if not cosmic_status["on_or_before_cutoff"] and not ALLOW_POST_CUTOFF_TRAINING_EVIDENCE:
            raise RuntimeError(
                f"COSMIC CMC is after the training cutoff and cannot create labels: {cosmic_status}"
            )
    return {
        "clinvar_train": clinvar_status,
        "civic": civic_status,
        "cosmic_cmc": cosmic_status,
    }


DBNSFP_COLS = [
    "#chr",
    "pos(1-based)",
    DBNSFP_GRCH37_CHROM_COLUMN,
    DBNSFP_GRCH37_POSITION_COLUMN,
    "ref",
    "alt",
    "aaref",
    "aaalt",
    "aapos",
    "genename",
    "Ensembl_transcriptid",
    "VEP_canonical",
    "MANE",
    "HGVSp_snpEff",
    "HGVSc_snpEff",
    *RAW_CONTEXTUAL_PREDICTOR_COLS,
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "Interpro_domain",
    "MetaLR_pred",
    "gnomAD4.1_joint_AF",
    "gnomAD2.1.1_exomes_non_cancer_AC",
    "gnomAD2.1.1_exomes_non_cancer_AN",
]

DBNSFP_PRIMARY_CHROM_COLUMN = "#chr"
DBNSFP_PRIMARY_POSITION_COLUMN = "pos(1-based)"
VALID_GRCH37_CHROMOSOMES = {
    *(str(chromosome) for chromosome in range(1, 23)),
    "X",
    "Y",
    "MT",
}

TRANSCRIPT_COLUMNS = [
    "aaref",
    "aaalt",
    "aapos",
    "genename",
    "Ensembl_transcriptid",
    "VEP_canonical",
    "MANE",
    "HGVSp_snpEff",
    "HGVSc_snpEff",
    *(column for column in RAW_CONTEXTUAL_PREDICTOR_COLS if column != "CADD_phred"),
    "Interpro_domain",
    "MetaLR_pred",
]


def _normalize_chromosome(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype("string").str.strip().str.replace(r"^chr", "", case=False, regex=True)
    )
    return normalized.replace({"M": "MT", "23": "X", "24": "Y"})


def _normalize_variant_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["chr"] = _normalize_chromosome(result["chr"])
    result["pos"] = pd.to_numeric(result["pos"], errors="coerce").astype("Int64")
    result["ref"] = result["ref"].astype("string").str.strip().str.upper()
    result["alt"] = result["alt"].astype("string").str.strip().str.upper()
    result = result.dropna(subset=["chr", "pos", "ref", "alt"])
    normalized = [
        _trim_alleles(int(pos), str(ref), str(alt))
        for pos, ref, alt in zip(result["pos"], result["ref"], result["alt"])
    ]
    result["pos"] = pd.array([item[0] for item in normalized], dtype="Int64")
    result["ref"] = [item[1] for item in normalized]
    result["alt"] = [item[2] for item in normalized]
    return result


def _dbnsfp_grch37_mapping_mask(raw: pd.DataFrame) -> pd.Series:
    """Identify explicit, valid dbNSFP GRCh37 projections without fallback."""
    required = {
        DBNSFP_PRIMARY_CHROM_COLUMN,
        DBNSFP_PRIMARY_POSITION_COLUMN,
        DBNSFP_GRCH37_CHROM_COLUMN,
        DBNSFP_GRCH37_POSITION_COLUMN,
        "ref",
        "alt",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(
            "dbNSFP GRCh37 projection requires explicit hg19 columns; missing "
            f"{missing}. The primary #chr/pos columns are GRCh38 and must not be "
            "used as a fallback."
        )
    chromosome = _normalize_chromosome(raw[DBNSFP_GRCH37_CHROM_COLUMN])
    position = pd.to_numeric(raw[DBNSFP_GRCH37_POSITION_COLUMN], errors="coerce")
    reference = raw["ref"].astype("string").str.strip().str.upper()
    alternate = raw["alt"].astype("string").str.strip().str.upper()
    return (
        chromosome.isin(VALID_GRCH37_CHROMOSOMES)
        & position.notna()
        & position.gt(0)
        & reference.str.fullmatch(r"[ACGT]", na=False)
        & alternate.str.fullmatch(r"[ACGT]", na=False)
        & reference.ne(alternate)
    )


def project_dbnsfp_to_grch37(raw: pd.DataFrame) -> pd.DataFrame:
    """Project dbNSFP rows to GRCh37 using hg19 columns only.

    dbNSFP 5.x primary ``#chr``/``pos(1-based)`` coordinates are GRCh38.  They
    are retained in the input contract solely to detect schema drift and are
    never allowed to create a GRCh37 variant identifier.
    """
    mapped = _dbnsfp_grch37_mapping_mask(raw)
    projected = raw.loc[mapped].copy()
    projected = projected.drop(
        columns=[DBNSFP_PRIMARY_CHROM_COLUMN, DBNSFP_PRIMARY_POSITION_COLUMN]
    ).rename(
        columns={
            DBNSFP_GRCH37_CHROM_COLUMN: "chr",
            DBNSFP_GRCH37_POSITION_COLUMN: "pos",
        }
    )
    projected = _normalize_variant_frame(projected)
    if not projected.empty:
        invalid = ~projected["chr"].isin(VALID_GRCH37_CHROMOSOMES)
        if invalid.any():
            raise RuntimeError("dbNSFP GRCh37 projection produced unsupported chromosomes")
    return projected


def _trim_alleles(position: int, reference: str, alternate: str) -> tuple[int, str, str]:
    reference = reference.upper()
    alternate = alternate.upper()
    while len(reference) > 1 and len(alternate) > 1 and reference[-1] == alternate[-1]:
        reference = reference[:-1]
        alternate = alternate[:-1]
    while len(reference) > 1 and len(alternate) > 1 and reference[0] == alternate[0]:
        reference = reference[1:]
        alternate = alternate[1:]
        position += 1
    return position, reference, alternate


def _variant_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        GENOMIC_VARIANT_ASSEMBLY
        + ":"
        + frame["chr"].astype(str)
        + ":"
        + frame["pos"].astype(str)
        + ":"
        + frame["ref"].astype(str)
        + ":"
        + frame["alt"].astype(str)
    )


def explode_aligned_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Explode transcript fields without cross-products."""
    present = [column for column in columns if column in frame.columns]
    if frame.empty or not present:
        return frame, 0
    split = frame[present].apply(lambda column: column.astype("string").fillna("").str.split(";"))
    lengths = split.map(len)
    target = lengths.max(axis=1)
    valid = lengths.apply(lambda column: (column == 1) | (column == target), axis=0).all(axis=1)
    ambiguous = int((~valid).sum())
    frame = frame.loc[valid].copy()
    split = split.loc[valid]
    target = target.loc[valid]
    for column in present:
        frame[column] = [
            values * int(expected) if len(values) == 1 else values
            for values, expected in zip(split[column], target)
        ]
    frame = frame.explode(present, ignore_index=True)
    for column in present:
        frame[column] = frame[column].replace(
            {"": np.nan, ".": np.nan, "nan": np.nan, "NA": np.nan}
        )
    return frame, ambiguous


def _resolve_column(columns: list[str], aliases: tuple[str, ...]) -> str:
    lookup = {column.lower().lstrip("#"): column for column in columns}
    for alias in aliases:
        match = lookup.get(alias.lower().lstrip("#"))
        if match is not None:
            return match
    raise KeyError(f"Missing one of columns: {aliases}")


def _joined_audit_values(values: pd.Series) -> str | pd.NA:
    observed = sorted(
        {value for value in values.astype("string").dropna().str.strip().astype(str) if value}
    )
    return ";".join(observed) if observed else pd.NA


def _load_clinvar_training_labels(
    *, return_identity: bool = False
) -> (
    tuple[set[str], set[str]]
    | tuple[set[str], set[str], pd.DataFrame, dict[str, Any], pd.DataFrame]
):
    """Load high-confidence labels and stable identities from the train snapshot."""
    if not CLINVAR_TRAIN_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Declared ClinVar training archive is missing: {CLINVAR_TRAIN_ARCHIVE}"
        )
    columns = pd.read_csv(CLINVAR_TRAIN_ARCHIVE, sep="\t", nrows=0).columns.tolist()
    aliases = {
        "chr": ("Chromosome", "chr", "#chr"),
        "pos": ("PositionVCF", "Start", "pos"),
        "ref": ("ReferenceAlleleVCF", "ReferenceAllele", "ref"),
        "alt": ("AlternateAlleleVCF", "AlternateAllele", "alt"),
        "assembly": ("Assembly", "assembly"),
        "review": ("ReviewStatus", "review_status", "review"),
        "significance": (
            "ClinicalSignificance",
            "clinical_significance",
            "clinvar_clnsig",
        ),
        "variation_id": ("VariationID", "#VariationID"),
        "source_gene": ("GeneSymbol", "Gene"),
        "source_name": ("Name",),
    }
    resolved = {name: _resolve_column(columns, values) for name, values in aliases.items()}
    usecols = list(dict.fromkeys(resolved.values()))
    identity_chunks: list[pd.DataFrame] = []
    invalid_variation_id_rows = 0
    for raw in pd.read_csv(
        CLINVAR_TRAIN_ARCHIVE,
        sep="\t",
        usecols=usecols,
        dtype="string",
        chunksize=100_000,
        low_memory=False,
    ):
        frame = raw.rename(columns={value: key for key, value in resolved.items()})
        assembly = frame["assembly"].str.upper().str.replace(" ", "", regex=False)
        frame = frame[assembly.isin({"GRCH37", "GRCH37.P13", "HG19"})].copy()
        review = frame["review"].str.lower().str.strip()
        frame = frame[review.isin(HIGH_CONFIDENCE_REVIEWS)].copy()
        significance = frame["significance"].str.lower()
        pathogenic_hit = significance.str.contains("pathogenic", na=False)
        benign_hit = significance.str.contains("benign", na=False)
        conflicting = significance.str.contains("conflict", na=False) | (
            pathogenic_hit & benign_hit
        )
        pathogenic = pathogenic_hit & ~conflicting
        benign = benign_hit & ~conflicting
        frame[LABEL_COL] = np.select([pathogenic, benign], [1, 0], default=-1).astype(np.int8)
        frame = frame[pathogenic | benign].copy()
        if frame.empty:
            continue
        frame = _normalize_variant_frame(frame)
        snv = frame["ref"].str.fullmatch("[ACGT]") & frame["alt"].str.fullmatch("[ACGT]")
        frame = frame[snv & frame["ref"].ne(frame["alt"]) & frame["pos"].gt(0)].copy()
        frame["variant_id"] = _variant_keys(frame)
        frame["CLINVAR_VARIATION_ID"] = normalize_variation_ids(frame["variation_id"])
        invalid_variation_id_rows += int(frame["CLINVAR_VARIATION_ID"].isna().sum())
        frame = frame.dropna(subset=["CLINVAR_VARIATION_ID"]).copy()
        if frame.empty:
            continue
        frame["CLINVAR_SOURCE_GENE"] = frame["source_gene"].astype("string").str.strip()
        frame["CLINVAR_SOURCE_NAME"] = frame["source_name"].astype("string").str.strip()
        frame["CLINVAR_REVIEW_STATUS"] = frame["review"].astype("string").str.strip()
        identity_chunks.append(
            frame[
                [
                    "variant_id",
                    LABEL_COL,
                    "CLINVAR_VARIATION_ID",
                    "CLINVAR_SOURCE_GENE",
                    "CLINVAR_SOURCE_NAME",
                    "CLINVAR_REVIEW_STATUS",
                ]
            ]
        )
    if not identity_chunks:
        raise RuntimeError("ClinVar training archive yielded no stable variant identities")
    candidates = pd.concat(identity_chunks, ignore_index=True)
    stable_to_key = candidates.groupby("CLINVAR_VARIATION_ID", sort=False)["variant_id"].nunique()
    key_to_stable = candidates.groupby("variant_id", sort=False)["CLINVAR_VARIATION_ID"].nunique()
    stable_label_counts = candidates.groupby("CLINVAR_VARIATION_ID", sort=False)[
        LABEL_COL
    ].nunique()
    key_label_counts = candidates.groupby("variant_id", sort=False)[LABEL_COL].nunique()
    ambiguous_stable_ids = set(stable_to_key[stable_to_key.gt(1)].index.astype(str))
    ambiguous_keys = set(key_to_stable[key_to_stable.gt(1)].index.astype(str))
    stable_label_conflicts = set(stable_label_counts[stable_label_counts.gt(1)].index.astype(str))
    key_label_conflicts = set(key_label_counts[key_label_counts.gt(1)].index.astype(str))
    excluded = (
        candidates["CLINVAR_VARIATION_ID"].astype(str).isin(ambiguous_stable_ids)
        | candidates["variant_id"].astype(str).isin(ambiguous_keys)
        | candidates["CLINVAR_VARIATION_ID"].astype(str).isin(stable_label_conflicts)
        | candidates["variant_id"].astype(str).isin(key_label_conflicts)
    )
    identity_audit_rows = candidates.copy()
    identity_audit_rows["identity_exclusion_reason"] = np.select(
        [
            identity_audit_rows["CLINVAR_VARIATION_ID"].astype(str).isin(ambiguous_stable_ids),
            identity_audit_rows["variant_id"].astype(str).isin(ambiguous_keys),
            identity_audit_rows["CLINVAR_VARIATION_ID"].astype(str).isin(stable_label_conflicts),
            identity_audit_rows["variant_id"].astype(str).isin(key_label_conflicts),
        ],
        [
            "stable_id_multiple_grch37_loci",
            "grch37_locus_multiple_stable_ids",
            "stable_id_label_conflict",
            "grch37_locus_label_conflict",
        ],
        default="none",
    )
    identity_audit_rows["identity_status"] = np.where(excluded, "excluded", "retained")
    identity_audit_rows = (
        identity_audit_rows[
            [
                "CLINVAR_VARIATION_ID",
                "variant_id",
                LABEL_COL,
                "identity_status",
                "identity_exclusion_reason",
                "CLINVAR_SOURCE_GENE",
                "CLINVAR_SOURCE_NAME",
                "CLINVAR_REVIEW_STATUS",
            ]
        ]
        .sort_values(["CLINVAR_VARIATION_ID", "variant_id", LABEL_COL], kind="mergesort")
        .drop_duplicates(["CLINVAR_VARIATION_ID", "variant_id", LABEL_COL], keep="first")
        .reset_index(drop=True)
    )
    eligible = candidates.loc[~excluded].copy()
    identity = (
        eligible.groupby("variant_id", sort=False, as_index=False)
        .agg(
            {
                LABEL_COL: "first",
                "CLINVAR_VARIATION_ID": "first",
                "CLINVAR_SOURCE_GENE": _joined_audit_values,
                "CLINVAR_SOURCE_NAME": _joined_audit_values,
                "CLINVAR_REVIEW_STATUS": _joined_audit_values,
            }
        )
        .set_index("variant_id", verify_integrity=True)
    )
    pathogenic_keys = set(identity.index[identity[LABEL_COL].eq(1)].astype(str))
    benign_keys = set(identity.index[identity[LABEL_COL].eq(0)].astype(str))
    identity_audit = {
        "eligible_rows_before_identity_resolution": len(candidates),
        "invalid_or_missing_variation_id_rows": invalid_variation_id_rows,
        "ambiguous_stable_ids": len(ambiguous_stable_ids),
        "keys_with_multiple_stable_ids": len(ambiguous_keys),
        "stable_id_label_conflicts": len(stable_label_conflicts),
        "genomic_key_label_conflicts": len(key_label_conflicts),
        "excluded_rows_for_identity_ambiguity": int(excluded.sum()),
        "retained_unique_variants": len(identity),
        "stable_variation_id_required": True,
        "allele_id_used_as_fallback": False,
        "identity_audit_rows": len(identity_audit_rows),
    }
    logger.info(
        "Loaded ClinVar %s labels: %d pathogenic, %d benign; %d identity-ambiguous rows excluded",
        CLINVAR_TRAIN_RELEASE,
        len(pathogenic_keys),
        len(benign_keys),
        int(excluded.sum()),
    )
    if not pathogenic_keys or not benign_keys:
        raise RuntimeError("ClinVar training archive did not yield both label classes")
    if return_identity:
        return (
            pathogenic_keys,
            benign_keys,
            identity,
            identity_audit,
            identity_audit_rows,
        )
    return pathogenic_keys, benign_keys


def _load_civic() -> tuple[set[str], set[str]]:
    if LABEL_TASK != "legacy_mixed":
        logger.info("CIViC is excluded from the %s primary label task", LABEL_TASK)
        return set(), set()
    release_status = source_release_status("CIViC", CIVIC_RELEASE)
    if not release_status["on_or_before_cutoff"] and not ALLOW_POST_CUTOFF_TRAINING_EVIDENCE:
        logger.warning(
            "Excluding post-cutoff CIViC %s from labels (exact cutoff %s)",
            CIVIC_RELEASE,
            TRAIN_CUTOFF_DATE,
        )
        return set(), set()
    if not CIVIC_FILE.exists():
        logger.warning("CIViC file is missing")
        return set(), set()
    frame = pd.read_csv(CIVIC_FILE, sep="\t", low_memory=False)
    required = {
        "chromosome",
        "start",
        "reference_bases",
        "variant_bases",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"CIViC misses columns: {sorted(missing)}")
    frame = frame.rename(
        columns={
            "chromosome": "chr",
            "start": "pos",
            "reference_bases": "ref",
            "variant_bases": "alt",
        }
    )
    frame = _normalize_variant_frame(frame)
    frame["variant_id"] = _variant_keys(frame)
    qualified = pd.Series(False, index=frame.index)
    qualification_found = False
    for column in ("evidence_level", "Evidence Level", "evidenceLevel"):
        if column in frame.columns:
            qualification_found = True
            qualified |= frame[column].astype(str).str.upper().str[:1].isin({"A", "B"})
    for column in ("evidence_score", "evidence_rating", "rating"):
        if column in frame.columns:
            qualification_found = True
            qualified |= pd.to_numeric(frame[column], errors="coerce").ge(3)
    if not qualification_found:
        logger.warning("CIViC lacks evidence qualification fields")
    return set(frame["variant_id"]), set(frame.loc[qualified, "variant_id"])


def _load_cgc() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    empty = pd.DataFrame(columns=["genename", "ROLE_IN_CANCER", "TIER"])
    roles = {name: set() for name in ("cancer", "oncogene", "tsg", "tier1")}
    if not CGC_FILE.exists():
        logger.warning("Cancer Gene Census file is missing")
        return empty, roles
    frame = pd.read_csv(CGC_FILE, sep="\t", low_memory=False)
    required = {"GENE_SYMBOL", "ROLE_IN_CANCER", "TIER"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"CGC misses columns: {sorted(missing)}")
    frame["GENE_SYMBOL"] = frame["GENE_SYMBOL"].astype("string").str.strip()
    frame = frame[frame["GENE_SYMBOL"].notna() & frame["GENE_SYMBOL"].ne("")]
    genes = frame["GENE_SYMBOL"].astype(str)
    role_text = frame["ROLE_IN_CANCER"].astype(str)
    roles["cancer"] = set(genes)
    roles["oncogene"] = set(
        frame.loc[role_text.str.contains("oncogene", case=False, na=False), "GENE_SYMBOL"]
    )
    roles["tsg"] = set(
        frame.loc[role_text.str.contains("TSG", case=False, na=False), "GENE_SYMBOL"]
    )
    roles["tier1"] = set(
        frame.loc[pd.to_numeric(frame["TIER"], errors="coerce").eq(1), "GENE_SYMBOL"]
    )
    slim = frame[["GENE_SYMBOL", "ROLE_IN_CANCER", "TIER"]].copy()
    slim = slim.rename(columns={"GENE_SYMBOL": "genename"})
    slim = slim.drop_duplicates("genename")
    return slim, roles


def _load_cosmic() -> pd.DataFrame:
    columns = [
        "GENE_NAME",
        "AA_MUT_START",
        "AA_WT_ALLELE_SEQ",
        "AA_MUT_ALLELE_SEQ",
        "COSMIC_SAMPLE_MUTATED",
        "COSMIC_SAMPLE_TESTED",
    ]
    if not COSMIC_CMC_FILE.exists():
        logger.warning("COSMIC Mutant Census file is missing")
        return pd.DataFrame(
            columns=[
                "genename",
                "aapos",
                "aaref",
                "aaalt",
                "COSMIC_RECURRENCE",
                "COSMIC_TESTED",
                "COSMIC_FREQUENCY",
            ]
        )
    aggregates: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        COSMIC_CMC_FILE,
        sep="\t",
        usecols=columns,
        chunksize=50000,
        low_memory=False,
    ):
        chunk["AA_MUT_START"] = pd.to_numeric(chunk["AA_MUT_START"], errors="coerce").astype(
            "Int64"
        )
        chunk["COSMIC_SAMPLE_MUTATED"] = pd.to_numeric(
            chunk["COSMIC_SAMPLE_MUTATED"], errors="coerce"
        ).fillna(0)
        chunk["COSMIC_SAMPLE_TESTED"] = pd.to_numeric(
            chunk["COSMIC_SAMPLE_TESTED"], errors="coerce"
        ).fillna(0)
        chunk["AA_WT_ALLELE_SEQ"] = (
            chunk["AA_WT_ALLELE_SEQ"].astype("string").str.strip().str.upper()
        )
        chunk["AA_MUT_ALLELE_SEQ"] = (
            chunk["AA_MUT_ALLELE_SEQ"].astype("string").str.strip().str.upper()
        )
        standard = set(STANDARD_AA)
        grouped = (
            chunk[
                chunk["AA_WT_ALLELE_SEQ"].isin(standard)
                & chunk["AA_MUT_ALLELE_SEQ"].isin(standard)
                & chunk["AA_WT_ALLELE_SEQ"].ne(chunk["AA_MUT_ALLELE_SEQ"])
            ]
            .dropna(subset=["GENE_NAME", "AA_MUT_START"])
            .groupby(
                [
                    "GENE_NAME",
                    "AA_MUT_START",
                    "AA_WT_ALLELE_SEQ",
                    "AA_MUT_ALLELE_SEQ",
                ],
                as_index=False,
            )
            .agg(
                COSMIC_SAMPLE_MUTATED=("COSMIC_SAMPLE_MUTATED", "sum"),
                COSMIC_SAMPLE_TESTED=("COSMIC_SAMPLE_TESTED", "max"),
            )
        )
        aggregates.append(grouped)
    if not aggregates:
        return pd.DataFrame()
    merged = pd.concat(aggregates, ignore_index=True)
    merged = (
        merged.groupby(
            [
                "GENE_NAME",
                "AA_MUT_START",
                "AA_WT_ALLELE_SEQ",
                "AA_MUT_ALLELE_SEQ",
            ],
            as_index=False,
        )
        .agg(
            COSMIC_SAMPLE_MUTATED=("COSMIC_SAMPLE_MUTATED", "sum"),
            COSMIC_SAMPLE_TESTED=("COSMIC_SAMPLE_TESTED", "max"),
        )
        .rename(
            columns={
                "GENE_NAME": "genename",
                "AA_MUT_START": "aapos",
                "AA_WT_ALLELE_SEQ": "aaref",
                "AA_MUT_ALLELE_SEQ": "aaalt",
                "COSMIC_SAMPLE_MUTATED": "COSMIC_RECURRENCE",
                "COSMIC_SAMPLE_TESTED": "COSMIC_TESTED",
            }
        )
    )
    denominator = merged["COSMIC_TESTED"].replace(0, np.nan)
    merged["COSMIC_FREQUENCY"] = merged["COSMIC_RECURRENCE"] / denominator
    return merged


def _load_oncokb_genes() -> set[str]:
    if not ONCOKB_FILE.exists():
        logger.warning("OncoKB file is missing")
        return set()
    frame = pd.read_csv(ONCOKB_FILE, sep="\t", low_memory=False)
    if "Gene" not in frame.columns:
        raise KeyError("OncoKB misses Gene")
    return set(frame["Gene"].dropna().astype(str).str.strip())


def _clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "gnomAD4.1_joint_AF",
        *RAW_CONTEXTUAL_PREDICTOR_COLS,
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            continue
        values = (
            frame[column]
            .astype("string")
            .replace({".": np.nan, "..": np.nan, "": np.nan, "nan": np.nan, "NA": np.nan})
        )
        values = pd.to_numeric(values, errors="coerce")
        # GERP and phyloP are signed conservation statistics; -1 is a valid
        # observed value, not an annotation failure. Missing dbNSFP fields are
        # represented by '.', which was converted above.
        if column not in {"GERP++_RS", "phyloP100way_vertebrate"}:
            values = values.mask(values.eq(-1))
        values = values.where(np.isfinite(values))
        frame[column] = values.astype("float32")
    return frame


def _evidence_strings(frame: pd.DataFrame) -> pd.Series:
    sources = np.full(len(frame), "", dtype=object)
    mappings = (
        ("IS_CLINVAR_PATHOGENIC", "clinvar_pathogenic"),
        ("IS_CLINVAR_BENIGN", "clinvar_benign"),
        ("HAS_QUALIFIED_CIVIC", "civic_high_evidence"),
        ("HAS_HOTSPOT_SUPPORT", "curated_hotspot_cosmic_support"),
        ("HAS_BA1_EVIDENCE", "gnomad_ba1"),
    )
    for column, label in mappings:
        if frame[column].isna().any():
            raise ValueError(f"Evidence column {column} contains missing values")
        mask = frame[column].eq(1).to_numpy()
        sources[mask] = [f"{current};{label}" if current else label for current in sources[mask]]
    return pd.Series(sources, index=frame.index).replace("", "none")


def _stable_row_ids(frame: pd.DataFrame) -> pd.Series:
    values = (
        frame["variant_id"].astype(str)
        + "|"
        + frame["Ensembl_transcriptid"].fillna("no_transcript").astype(str)
        + "|"
        + frame["genename"].fillna("no_gene").astype(str)
        + "|p."
        + frame["aaref"].astype(str)
        + frame["aapos"].astype(str)
        + frame["aaalt"].astype(str)
    )
    return values.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:24])


def _process_chunk(
    raw: pd.DataFrame,
    clinvar_pathogenic_keys: set[str],
    clinvar_benign_keys: set[str],
    civic_keys: set[str],
    qualified_civic_keys: set[str],
    cgc: pd.DataFrame,
    roles: dict[str, set[str]],
    cosmic: pd.DataFrame,
    oncokb_genes: set[str],
    label_task: str = "clinical",
    clinvar_identity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    frame = project_dbnsfp_to_grch37(raw)
    if label_task == "clinical":
        # Only high-confidence ClinVar assertions can create labels in the
        # publication task. Filter on the normalized genomic key before the
        # expensive transcript explosion and numeric annotation; doing this
        # later produces the same saved rows but needlessly processes the
        # complete dbNSFP catalogue.
        variant_ids = _variant_keys(frame)
        labelled = variant_ids.isin(clinvar_pathogenic_keys) | variant_ids.isin(clinvar_benign_keys)
        frame = frame.loc[labelled].copy()
        if frame.empty:
            return frame, 0
    frame, ambiguous = explode_aligned_columns(frame, TRANSCRIPT_COLUMNS)
    frame["genename"] = frame["genename"].astype("string").str.strip()
    frame["aapos"] = pd.to_numeric(frame["aapos"], errors="coerce").astype("Int64")
    frame["aaref"] = frame["aaref"].astype("string").str.upper()
    frame["aaalt"] = frame["aaalt"].astype("string").str.upper()
    standard = set(STANDARD_AA)
    valid_missense = (
        frame["aaref"].isin(standard)
        & frame["aaalt"].isin(standard)
        & frame["aaref"].ne(frame["aaalt"])
        & frame["aapos"].gt(0)
    )
    frame = frame.loc[valid_missense].copy()
    if frame.empty:
        return frame, ambiguous
    frame = _clean_numeric(frame)
    non_cancer_ac = pd.to_numeric(frame["gnomAD2.1.1_exomes_non_cancer_AC"], errors="coerce")
    non_cancer_an = pd.to_numeric(frame["gnomAD2.1.1_exomes_non_cancer_AN"], errors="coerce")
    frame["gnomAD_non_cancer_AF"] = (non_cancer_ac / non_cancer_an).where(non_cancer_an.gt(0))
    frame["_population_af"] = frame["gnomAD_non_cancer_AF"].fillna(frame["gnomAD4.1_joint_AF"])
    frame["variant_id"] = _variant_keys(frame)
    if label_task == "clinical":
        if clinvar_identity is None:
            raise RuntimeError(
                "Clinical publication labels require stable ClinVar VariationID metadata"
            )
        if not clinvar_identity.index.is_unique:
            raise RuntimeError("ClinVar training identity map is not variant-unique")
        identity_columns = (
            "CLINVAR_VARIATION_ID",
            "CLINVAR_SOURCE_GENE",
            "CLINVAR_SOURCE_NAME",
            "CLINVAR_REVIEW_STATUS",
        )
        for column in identity_columns:
            if column not in clinvar_identity:
                raise RuntimeError(f"ClinVar training identity map lacks required column {column}")
            frame[column] = frame["variant_id"].map(clinvar_identity[column])
        if frame["CLINVAR_VARIATION_ID"].isna().any():
            raise RuntimeError(
                "A clinically labelled dbNSFP row lacks a stable ClinVar VariationID"
            )
    frame["transcript_variant_id"] = (
        frame["variant_id"].astype(str)
        + "|"
        + frame["Ensembl_transcriptid"].fillna("no_transcript").astype(str)
    )
    frame["variant_type"] = "missense"
    frame = frame.merge(cgc, on="genename", how="left", validate="many_to_one")
    frame["ROLE_IN_CANCER"] = frame["ROLE_IN_CANCER"].fillna("Not_in_CGC")
    frame["TIER"] = pd.to_numeric(frame["TIER"], errors="coerce").fillna(0).astype(int)
    if not cosmic.empty:
        frame = frame.merge(
            cosmic[
                [
                    "genename",
                    "aapos",
                    "aaref",
                    "aaalt",
                    "COSMIC_RECURRENCE",
                    "COSMIC_FREQUENCY",
                ]
            ],
            on=["genename", "aapos", "aaref", "aaalt"],
            how="left",
            validate="many_to_one",
        )
    if "COSMIC_RECURRENCE" in frame:
        frame["COSMIC_RECURRENCE"] = pd.to_numeric(
            frame["COSMIC_RECURRENCE"], errors="coerce"
        ).fillna(0)
    else:
        frame["COSMIC_RECURRENCE"] = 0.0
    if "COSMIC_FREQUENCY" not in frame:
        frame["COSMIC_FREQUENCY"] = np.nan
    frame["IS_CANCER_GENE"] = frame["genename"].isin(roles["cancer"]).astype(np.int8)
    frame["IS_ONCOGENE"] = frame["genename"].isin(roles["oncogene"]).astype(np.int8)
    frame["IS_TSG"] = frame["genename"].isin(roles["tsg"]).astype(np.int8)
    frame["IS_TIER1"] = frame["genename"].isin(roles["tier1"]).astype(np.int8)
    frame["IS_ONCOKB"] = frame["genename"].isin(oncokb_genes).astype(np.int8)
    hotspot = pd.Series(False, index=frame.index)
    for gene, positions in RESCUE_HOTSPOTS.items():
        hotspot |= frame["genename"].eq(gene) & frame["aapos"].isin(positions)
    frame["IS_KNOWN_HOTSPOT"] = hotspot.astype(np.int8)
    consensus_complete = (
        frame[["REVEL_score", "SIFT_score", "Polyphen2_HDIV_score"]].notna().all(axis=1)
    )
    frame["CONSENSUS_SCORE"] = (
        (
            frame["REVEL_score"].fillna(0.5) * 0.6
            + (1.0 - frame["SIFT_score"].fillna(0.5)) * 0.2
            + frame["Polyphen2_HDIV_score"].fillna(0.5) * 0.2
        )
        .where(consensus_complete)
        .astype("float32")
    )
    pathogenic = frame["variant_id"].isin(clinvar_pathogenic_keys)
    benign = frame["variant_id"].isin(clinvar_benign_keys)
    frame["IS_CLINVAR_PATHOGENIC"] = pathogenic.astype(np.int8)
    frame["IS_CLINVAR_BENIGN"] = benign.astype(np.int8)
    frame["HAS_CIVIC_EVIDENCE"] = frame["variant_id"].isin(civic_keys).astype(np.int8)
    frame["HAS_QUALIFIED_CIVIC"] = frame["variant_id"].isin(qualified_civic_keys).astype(np.int8)
    frame["HAS_HOTSPOT_SUPPORT"] = (
        hotspot & frame["COSMIC_RECURRENCE"].ge(MIN_COSMIC_EVIDENCE)
    ).astype(np.int8)
    frame["HAS_BA1_EVIDENCE"] = (
        frame["_population_af"].ge(BA1_AF_THRESHOLD).fillna(False).astype(np.int8)
    )
    if frame["HAS_BA1_EVIDENCE"].isna().any():
        raise RuntimeError("BA1 evidence contains missing values after assignment")
    if label_task == "clinical":
        # The publication-safe primary task uses clinical assertions only. BA1,
        # CIViC and COSMIC remain audit evidence but do not create class labels.
        positive_evidence = pathogenic
        benign_evidence = benign
    elif label_task == "legacy_mixed":
        positive_evidence = (
            pathogenic | frame["HAS_QUALIFIED_CIVIC"].eq(1) | frame["HAS_HOTSPOT_SUPPORT"].eq(1)
        )
        benign_evidence = benign | frame["HAS_BA1_EVIDENCE"].eq(1)
    elif label_task == "somatic":
        raise RuntimeError("Somatic labelling requires an explicit same-domain negative cohort")
    else:
        raise ValueError(f"Unsupported label task: {label_task!r}")
    conflict = positive_evidence & benign_evidence
    frame[LABEL_COL] = np.select(
        [positive_evidence & ~conflict, benign_evidence & ~conflict],
        [1, 0],
        default=-1,
    ).astype(np.int8)
    frame["EVIDENCE_SOURCES"] = _evidence_strings(frame)
    if label_task == "clinical":
        frame["EVIDENCE_SOURCE"] = np.select(
            [pathogenic, benign],
            ["clinvar_pathogenic", "clinvar_benign"],
            default="not_a_clinical_label",
        )
    else:
        frame["EVIDENCE_SOURCE"] = frame["EVIDENCE_SOURCES"]
    cosmic_rescue = frame["COSMIC_RECURRENCE"].ge(MIN_COSMIC_RESCUE)
    frame["WAS_RESCUED"] = (frame["HAS_CIVIC_EVIDENCE"].eq(1) | hotspot | cosmic_rescue).astype(
        np.int8
    )
    frame["RESCUE_REASON"] = np.select(
        [frame["HAS_CIVIC_EVIDENCE"].astype(bool), hotspot, cosmic_rescue],
        ["CIViC", "curated_hotspot", "COSMIC_recurrence"],
        default="none",
    )
    frame = frame.loc[frame[LABEL_COL].isin([0, 1])].copy()
    if frame.empty:
        return frame, ambiguous
    frame[ROW_ID_COL] = _stable_row_ids(frame)
    save_columns = [
        "chr",
        "pos",
        "ref",
        "alt",
        "variant_id",
        "transcript_variant_id",
        ROW_ID_COL,
        "CLINVAR_VARIATION_ID",
        "CLINVAR_SOURCE_GENE",
        "CLINVAR_SOURCE_NAME",
        "CLINVAR_REVIEW_STATUS",
        "genename",
        "Ensembl_transcriptid",
        "VEP_canonical",
        "MANE",
        "HGVSp_snpEff",
        "HGVSc_snpEff",
        "aapos",
        "aaref",
        "aaalt",
        "variant_type",
        "CONSENSUS_SCORE",
        *RAW_CONTEXTUAL_PREDICTOR_COLS,
        "MetaLR_pred",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "Interpro_domain",
        "ROLE_IN_CANCER",
        "TIER",
        "IS_CANCER_GENE",
        "IS_TIER1",
        "IS_ONCOGENE",
        "IS_TSG",
        "IS_ONCOKB",
        "IS_KNOWN_HOTSPOT",
        "COSMIC_RECURRENCE",
        "COSMIC_FREQUENCY",
        "IS_CLINVAR_PATHOGENIC",
        "IS_CLINVAR_BENIGN",
        "HAS_CIVIC_EVIDENCE",
        "HAS_QUALIFIED_CIVIC",
        "HAS_HOTSPOT_SUPPORT",
        "HAS_BA1_EVIDENCE",
        "WAS_RESCUED",
        "RESCUE_REASON",
        "EVIDENCE_SOURCE",
        "EVIDENCE_SOURCES",
        LABEL_COL,
    ]
    return frame[[column for column in save_columns if column in frame.columns]], ambiguous


def main() -> None:
    """Build the evidence-labelled missense dataset."""
    if not DBNSFP_FILE.exists():
        raise FileNotFoundError(DBNSFP_FILE)
    ensure_directories(STAGE01_OUT)
    temporal_source_audit = _validate_label_configuration()
    temporal_source_audit["clinvar_training_snapshot"] = validate_clinvar_snapshot_provenance(
        CLINVAR_TRAIN_ARCHIVE,
        CLINVAR_TRAIN_TRANSFORMATION_MANIFEST,
        CLINVAR_TRAIN_RELEASE,
    )
    (
        clinvar_pathogenic_keys,
        clinvar_benign_keys,
        clinvar_identity,
        clinvar_identity_audit,
        clinvar_identity_audit_rows,
    ) = _load_clinvar_training_labels(return_identity=True)
    civic_keys, qualified_civic_keys = _load_civic()
    if LABEL_TASK == "legacy_mixed":
        cgc, roles = _load_cgc()
        cosmic = _load_cosmic()
        oncokb_genes = _load_oncokb_genes()
    else:
        cgc = pd.DataFrame(columns=["genename", "ROLE_IN_CANCER", "TIER"])
        roles = {name: set() for name in ("cancer", "oncogene", "tsg", "tier1")}
        cosmic = pd.DataFrame()
        oncokb_genes = set()
    temporary = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    identity_audit_temporary = IDENTITY_AUDIT_FILE.with_suffix(IDENTITY_AUDIT_FILE.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    identity_audit_temporary.unlink(missing_ok=True)
    total_saved = 0
    ambiguous_rows = 0
    dbnsfp_source_rows = 0
    dbnsfp_grch37_projected_rows = 0
    first_chunk = True
    iterator = pd.read_csv(
        DBNSFP_FILE,
        sep="\t",
        compression="gzip",
        usecols=DBNSFP_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=True,
        dtype={
            DBNSFP_PRIMARY_CHROM_COLUMN: "string",
            DBNSFP_GRCH37_CHROM_COLUMN: "string",
        },
    )
    clinvar_identity_audit_rows.to_csv(identity_audit_temporary, index=False)
    try:
        for chunk_number, raw_chunk in enumerate(iterator, 1):
            dbnsfp_source_rows += len(raw_chunk)
            dbnsfp_grch37_projected_rows += int(_dbnsfp_grch37_mapping_mask(raw_chunk).sum())
            processed, ambiguous = _process_chunk(
                raw_chunk,
                clinvar_pathogenic_keys,
                clinvar_benign_keys,
                civic_keys,
                qualified_civic_keys,
                cgc,
                roles,
                cosmic,
                oncokb_genes,
                LABEL_TASK,
                clinvar_identity,
            )
            ambiguous_rows += ambiguous
            if not processed.empty:
                processed.to_csv(
                    temporary,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False
                total_saved += len(processed)
            if chunk_number % 10 == 0:
                logger.info(
                    "Processed %d chunks; retained %d rows",
                    chunk_number,
                    total_saved,
                )
            del processed
            del raw_chunk
            gc.collect()
        if first_chunk:
            raise RuntimeError("No labelled missense rows were produced")
        # --- Chunked metadata validation (avoids OOM on large files) ---
        logger.info("Validating output file (chunked read)...")
        class_counts: dict[int, int] = {}
        unique_variants: dict[int, set[str]] = {0: set(), 1: set()}
        source_counts: dict[str, int] = {}
        primary_source_counts: dict[str, int] = {}
        variant_type_counts: dict[str, int] = {}
        total_rows = 0
        has_missing_ids = False
        meta_reader = pd.read_csv(
            temporary,
            usecols=[
                LABEL_COL,
                "EVIDENCE_SOURCE",
                "EVIDENCE_SOURCES",
                "variant_type",
                "variant_id",
                ROW_ID_COL,
            ],
            chunksize=100_000,
            low_memory=False,
        )
        for meta_chunk in meta_reader:
            total_rows += len(meta_chunk)
            # Class counts
            for label, count in meta_chunk[LABEL_COL].value_counts().items():
                class_counts[int(label)] = class_counts.get(int(label), 0) + int(count)
                unique_variants[int(label)].update(
                    meta_chunk.loc[meta_chunk[LABEL_COL].eq(label), "variant_id"].astype(str)
                )
            # Missing row IDs
            if meta_chunk[ROW_ID_COL].isna().any():
                has_missing_ids = True
            # Source counts
            for values in meta_chunk["EVIDENCE_SOURCES"].dropna().astype(str):
                for source in values.split(";"):
                    source_counts[source] = source_counts.get(source, 0) + 1
            for source, count in meta_chunk["EVIDENCE_SOURCE"].value_counts().items():
                primary_source_counts[str(source)] = primary_source_counts.get(
                    str(source), 0
                ) + int(count)
            # Variant type counts
            for vtype, count in meta_chunk["variant_type"].value_counts().items():
                variant_type_counts[str(vtype)] = variant_type_counts.get(str(vtype), 0) + int(
                    count
                )
            del meta_chunk
            gc.collect()
        if set(class_counts) != {0, 1}:
            raise RuntimeError(f"Stage 01 produced invalid classes: {class_counts}")
        if has_missing_ids:
            raise RuntimeError("Stage 01 produced missing row identifiers")
        temporary.replace(OUTPUT_FILE)
        identity_audit_temporary.replace(IDENTITY_AUDIT_FILE)
    except BaseException:
        temporary.unlink(missing_ok=True)
        identity_audit_temporary.unlink(missing_ok=True)
        raise
    metadata = {
        "total": total_rows,
        "pathogenic": int(class_counts.get(1, 0)),
        "benign": int(class_counts.get(0, 0)),
        "missense_only": True,
        "ambiguous_transcript_rows_excluded": ambiguous_rows,
        "coordinate_projection": {
            **COORDINATE_CONTRACT,
            "dbnsfp_source_rows": dbnsfp_source_rows,
            "dbnsfp_grch37_projected_rows": dbnsfp_grch37_projected_rows,
            "dbnsfp_rows_without_valid_grch37_projection": (
                dbnsfp_source_rows - dbnsfp_grch37_projected_rows
            ),
        },
        "clinvar_identity_audit": clinvar_identity_audit,
        "clinvar_identity_audit_file": str(IDENTITY_AUDIT_FILE),
        "labeling": {
            "scheme": LABEL_POLICY_VERSION,
            "label_task": LABEL_TASK,
            "primary_policy": (
                "clinvar_high_confidence_pathogenic_vs_benign"
                if LABEL_TASK == "clinical"
                else "legacy_mixed_explicit_opt_in"
            ),
            "training_cutoff_date": TRAIN_CUTOFF_DATE,
            "clinvar_training_release": CLINVAR_TRAIN_RELEASE,
            "clinvar_high_confidence_reviews_only": True,
            "ba1_af_threshold": BA1_AF_THRESHOLD,
            "cosmic_hotspot_support": MIN_COSMIC_EVIDENCE,
            "cosmic_exact_amino_acid_substitution": True,
            "civic_requires_qualification": True,
            "post_cutoff_evidence_allowed": ALLOW_POST_CUTOFF_TRAINING_EVIDENCE,
            "cosmic_recurrence_alone_labels_pathogenic": False,
            "source_date_audit": temporal_source_audit,
        },
        "unique_variant_class_counts": {
            str(label): len(values) for label, values in unique_variants.items()
        },
        "primary_label_sources": primary_source_counts,
        "label_sources": primary_source_counts,
        "all_audit_evidence_sources": source_counts,
        "variants": variant_type_counts,
    }
    metadata_temporary = METADATA_FILE.with_suffix(METADATA_FILE.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, default=json_default), encoding="utf-8"
    )
    metadata_temporary.replace(METADATA_FILE)
    manifest_inputs = [
        DBNSFP_FILE,
        CLINVAR_TRAIN_ARCHIVE,
        CLINVAR_TRAIN_TRANSFORMATION_MANIFEST,
    ]
    if LABEL_TASK == "legacy_mixed":
        manifest_inputs.extend([CIVIC_FILE, CGC_FILE, COSMIC_CMC_FILE, ONCOKB_FILE])
    write_run_manifest(
        MANIFEST_FILE,
        "01_dbnsfp_processor",
        manifest_inputs,
        {
            "output_rows": total_saved,
            "dbnsfp_chunk_size": CHUNK_SIZE,
            "ambiguous_transcript_rows_excluded": ambiguous_rows,
            "label_task": LABEL_TASK,
            "label_policy_version": LABEL_POLICY_VERSION,
            "training_cutoff_date": TRAIN_CUTOFF_DATE,
            "temporal_source_audit": temporal_source_audit,
            "coordinate_projection": {
                **COORDINATE_CONTRACT,
                "dbnsfp_source_rows": dbnsfp_source_rows,
                "dbnsfp_grch37_projected_rows": dbnsfp_grch37_projected_rows,
                "dbnsfp_rows_without_valid_grch37_projection": (
                    dbnsfp_source_rows - dbnsfp_grch37_projected_rows
                ),
            },
            "clinvar_identity_audit": clinvar_identity_audit,
            "unique_variant_class_counts": {
                str(label): len(values) for label, values in unique_variants.items()
            },
        },
        outputs=[OUTPUT_FILE, METADATA_FILE, IDENTITY_AUDIT_FILE],
    )
    logger.info("Saved %d labelled missense rows", total_saved)


if __name__ == "__main__":
    main()
