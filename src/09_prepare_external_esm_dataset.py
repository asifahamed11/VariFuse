from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from clinvar_identity import normalize_variation_ids
from config import (
    ALLOW_SEQUENCE_ONLY_DMS,
    ALPHAFOLD_DIR,
    AUDIT_SAMPLE_MAX_ROWS,
    CLINVAR_EXTERNAL_ARCHIVE,
    CLINVAR_EXTERNAL_RELEASE,
    CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE,
    CLINVAR_EXTERNAL_TRANSFORMATION_MANIFEST,
    CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION,
    CLINVAR_REQUIRE_SCV_EVIDENCE,
    CLINVAR_SCV_MIN_MATCHING,
    CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS,
    CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM,
    CLINVAR_TRAIN_ARCHIVE,
    CLINVAR_TRAIN_RELEASE,
    CLINVAR_TRAIN_SUBMISSION_ARCHIVE,
    CLINVAR_TRAIN_TRANSFORMATION_MANIFEST,
    COORDINATE_CONTRACT,
    DBNSFP_FILE,
    DBNSFP_GRCH37_CHROM_COLUMN,
    DBNSFP_GRCH37_POSITION_COLUMN,
    DMS_MAX_ROWS_PER_ASSAY,
    DMS_LEGACY_FILE,
    DMS_SAMPLING_POLICY,
    FEATURE_COVERAGE_POLICY,
    FEATURE_COVERAGE_WARN_GAP,
    PROTEINGYM_DIR,
    PROTEINGYM_EXTRACTION_MANIFEST,
    PROTEINGYM_METADATA,
    PROTEINGYM_METADATA_PROVENANCE,
    PROTEINGYM_RELEASE,
    REQUIRE_EXTERNAL_CLINVAR,
    REQUIRE_EXTERNAL_DMS,
    STAGE09_OUT,
    TRAIN_CUTOFF_DATE,
    TRANSCRIPT_SELECTION_POLICY,
    UNIPROT_FILE,
    ensure_directories,
    json_default,
    source_release_status,
    validate_clinvar_snapshot_provenance,
    validate_clinvar_submission_provenance,
    validate_proteingym_provenance,
    write_run_manifest,
)
from schema import (
    AVAILABILITY_FEATURE_COLS,
    BASE_FEATURE_ALLOWLIST,
    GENE_COL,
    LABEL_COL,
    MUTATION_FEATURE_COLS,
    RAW_CONTEXTUAL_PREDICTOR_COLS,
    ROW_ID_COL,
    STANDARD_AA,
)

stage01 = importlib.import_module("01_dbnsfp_processor")
stage04 = importlib.import_module("04_feature_engineering")
stage08 = importlib.import_module("08_prepare_esm_dataset")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage09_external")

CLINVAR_OUTPUT = STAGE09_OUT / "clinvar_esm_ready.csv"
DMS_OUTPUT = STAGE09_OUT / "dms_esm_ready.csv"
DMS_SEQUENCE_OUTPUT = STAGE09_OUT / "dms_sequences.parquet"
LOSS_REPORT = STAGE09_OUT / "external_mapping_loss.csv"
DISAGREEMENT_FILE = STAGE09_OUT / "clinvar_dms_disagreements.csv"
CLINVAR_TRANSCRIPT_AUDIT = STAGE09_OUT / "clinvar_transcript_selection_audit.csv"
CLINVAR_SCV_EVIDENCE_AUDIT = STAGE09_OUT / "clinvar_scv_evidence_audit.csv"
SUMMARY_FILE = STAGE09_OUT / "external_preparation_summary.json"
MANIFEST_FILE = STAGE09_OUT / "run_manifest.json"
CLINVAR_SCV_AUDIT_TOOL = (
    Path(__file__).resolve().parents[1] / "tools" / "audit_clinvar_submissions.py"
)

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
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "Interpro_domain",
    *RAW_CONTEXTUAL_PREDICTOR_COLS,
]

HIGH_CONFIDENCE_REVIEWS = {
    "practice guideline",
    "reviewed by expert panel",
    "criteria provided, multiple submitters, no conflicts",
}
CLINVAR_PATHOGENIC_AGGREGATES = {
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
    "likely pathogenic/pathogenic",
}
CLINVAR_BENIGN_AGGREGATES = {
    "benign",
    "likely benign",
    "benign/likely benign",
    "likely benign/benign",
}
CLINVAR_IDENTITY_POLICY = "stable_variation_id_then_unique_genomic_fallback_v2"
CLINVAR_SCV_POLICY = (
    "current_high_confidence_aggregate_and_post_cutoff_new_or_versioned_"
    "matching_contributing_scv_v1"
)
CLINVAR_SCV_AUDIT_COLUMNS = (
    "CLINVAR_VARIATION_ID",
    "CLINVAR_GENOMIC_KEY",
    "CLINVAR_TEMPORAL_KEY",
    "CLINVAR_IDENTITY_SOURCE",
    LABEL_COL,
    "significance",
    "review",
    "BASELINE_LABEL",
    "BASELINE_CLINVAR_VARIATION_ID",
    "BASELINE_CLINVAR_GENOMIC_KEY",
    "TEMPORAL_STATUS",
    "TEMPORAL_MATCH_SOURCE",
    "TEMPORAL_ASSERTION_POLICY",
    "last_evaluated",
    "CLINVAR_RELEASE",
    "TRAIN_CUTOFF_DATE",
    "candidate_class",
    "current_contributing_matching_scv_count",
    "current_contributing_opposing_scv_count",
    "current_contributing_ambiguous_scv_count",
    "current_contributing_unique_submitter_count",
    "current_contributing_matching_unique_submitter_count",
    "post_cutoff_new_matching_scv_count",
    "post_cutoff_updated_matching_scv_count",
    "post_cutoff_matching_event_scv_count",
    "current_contributing_matching_scv_ids",
    "current_contributing_opposing_scv_ids",
    "post_cutoff_matching_event_scv_ids",
    "post_cutoff_matching_event_dates",
    "scv_evidence_pass",
    "scv_evidence_reason",
    "SCV_REQUIRED_MATCHING_UNIQUE_SUBMITTERS",
    "SCV_REVIEW_SUBMITTER_PASS",
    "SCV_EVIDENCE_PASS",
    "SCV_EVIDENCE_REASON",
)

AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
CLINVAR_MISSENSE_PATTERN = re.compile(
    r"(?P<transcript>N[MR]_[0-9]+(?:\.[0-9]+)?).*?"
    r"p\.\(?(?P<reference>[A-Za-z]{3})(?P<position>[0-9]+)"
    r"(?P<alternate>[A-Za-z]{3})\)?",
    flags=re.IGNORECASE,
)

EXTERNAL_TABULAR_FEATURES = tuple(
    feature
    for feature in BASE_FEATURE_ALLOWLIST
    if feature not in {"esm_variant_score", "ESM_EXTRACTION_SUCCESS", *MUTATION_FEATURE_COLS}
    and not feature.endswith("__missing")
)
CRITICAL_DMS_CONTEXT_FEATURES = (
    "GERP++_RS",
    "SASA",
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
    "HAS_STRUCTURE",
)
FEATURE_COVERAGE_AUDIT_COLUMNS = (
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SASA",
    "RELATIVE_SASA",
    "PLDDT_SCORE",
    "LOCAL_CONTACT_COUNT_8A",
    "LOCAL_CONTACT_COUNT_12A",
    "LOCAL_MEAN_PLDDT_8A",
    "LOCAL_MIN_PLDDT_8A",
    "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
    "HAS_PROTEIN_MAPPING",
    "STRUCTURE_FILE_AVAILABLE",
    "HAS_STRUCTURE",
    "HAS_DOMAIN_ANNOTATION",
    "HAS_ACTIVE_SITE_ANNOTATION",
    "HAS_BINDING_SITE_ANNOTATION",
    "HAS_TRANSMEMBRANE_ANNOTATION",
)


def _resolve_column(
    columns: list[str], aliases: tuple[str, ...], required: bool = True
) -> str | None:
    lookup = {column.lower().lstrip("#"): column for column in columns}
    for alias in aliases:
        match = lookup.get(alias.lower().lstrip("#"))
        if match is not None:
            return match
    if required:
        raise KeyError(f"Missing one of columns: {aliases}")
    return None


def _normalise_clinvar_variation_ids(values: pd.Series) -> pd.Series:
    """Return positive, canonical ClinVar VariationIDs without using AlleleID."""
    return normalize_variation_ids(values)


def _ensure_clinvar_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize identity columns, including legacy/mocked snapshot frames.

    Production readers populate these columns through
    :func:`_attach_clinvar_identity`.  Keeping this small normalization at the
    temporal boundary makes tests and previously materialized in-memory frames
    explicit genomic fallbacks instead of crashing or inventing an AlleleID.
    """
    if "variant_id" not in frame:
        raise RuntimeError("ClinVar temporal input lacks the genomic variant_id")
    working = frame.copy()
    raw_variation = working.get(
        "CLINVAR_VARIATION_ID",
        working.get(
            "clinvar_variation_id",
            pd.Series(pd.NA, index=working.index, dtype="string"),
        ),
    )
    working["CLINVAR_VARIATION_ID"] = _normalise_clinvar_variation_ids(raw_variation)
    working["CLINVAR_SOURCE_GENE"] = working.get(
        "CLINVAR_SOURCE_GENE",
        working.get(
            "gene_symbol",
            pd.Series(pd.NA, index=working.index, dtype="string"),
        ),
    ).astype("string")
    working["CLINVAR_SOURCE_NAME"] = working.get(
        "CLINVAR_SOURCE_NAME",
        working.get("name", pd.Series(pd.NA, index=working.index, dtype="string")),
    ).astype("string")
    working["CLINVAR_REVIEW_STATUS"] = working.get(
        "CLINVAR_REVIEW_STATUS",
        working.get("review", pd.Series(pd.NA, index=working.index, dtype="string")),
    ).astype("string")
    genomic = working["variant_id"].astype("string")
    if (genomic.isna() | genomic.str.strip().eq("")).any():
        raise RuntimeError("ClinVar temporal input contains a missing genomic key")
    if "CLINVAR_GENOMIC_KEY" in working:
        supplied = working["CLINVAR_GENOMIC_KEY"].astype("string")
        inconsistent = supplied.notna() & supplied.ne(genomic)
        if inconsistent.any():
            raise RuntimeError(
                "ClinVar temporal input contains genomic identity keys that differ from variant_id"
            )
    working["CLINVAR_GENOMIC_KEY"] = genomic
    stable = working["CLINVAR_VARIATION_ID"].notna()
    working["CLINVAR_IDENTITY_SOURCE"] = np.where(
        stable, "stable_variation_id", "genomic_fallback_no_variation_id"
    )
    working["CLINVAR_TEMPORAL_KEY"] = np.where(
        stable,
        "VariationID:" + working["CLINVAR_VARIATION_ID"].fillna(""),
        "Genomic:" + working["CLINVAR_GENOMIC_KEY"].fillna(""),
    )
    return working


def _attach_clinvar_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign stable temporal keys and exclude ambiguous within-snapshot mappings."""
    working = frame.copy()
    raw_variation = working.get(
        "clinvar_variation_id", pd.Series(pd.NA, index=working.index, dtype="string")
    )
    working["CLINVAR_VARIATION_ID"] = _normalise_clinvar_variation_ids(raw_variation)
    working["CLINVAR_SOURCE_GENE"] = working.get(
        "gene_symbol", pd.Series(pd.NA, index=working.index, dtype="string")
    ).astype("string")
    working["CLINVAR_SOURCE_NAME"] = working.get(
        "name", pd.Series(pd.NA, index=working.index, dtype="string")
    ).astype("string")
    working["CLINVAR_REVIEW_STATUS"] = working.get(
        "review", pd.Series(pd.NA, index=working.index, dtype="string")
    ).astype("string")
    working["CLINVAR_GENOMIC_KEY"] = working["variant_id"].astype("string")
    if (
        working["CLINVAR_GENOMIC_KEY"].isna() | working["CLINVAR_GENOMIC_KEY"].str.strip().eq("")
    ).any():
        raise RuntimeError("ClinVar snapshot contains a missing genomic key")
    stable = working["CLINVAR_VARIATION_ID"].notna()
    working["CLINVAR_IDENTITY_SOURCE"] = np.where(
        stable, "stable_variation_id", "genomic_fallback_no_variation_id"
    )
    working["CLINVAR_TEMPORAL_KEY"] = np.where(
        stable,
        "VariationID:" + working["CLINVAR_VARIATION_ID"].fillna(""),
        "Genomic:" + working["CLINVAR_GENOMIC_KEY"].fillna(""),
    )

    stable_rows = working.loc[stable]
    ambiguous_stable_ids = set(
        stable_rows.groupby("CLINVAR_VARIATION_ID", sort=False)["CLINVAR_GENOMIC_KEY"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    ambiguous_genomic_keys = set(
        stable_rows.groupby("CLINVAR_GENOMIC_KEY", sort=False)["CLINVAR_VARIATION_ID"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    temporal_label_conflicts = set(
        working.groupby("CLINVAR_TEMPORAL_KEY", sort=False)[LABEL_COL]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    genomic_label_conflicts = set(
        working.groupby("CLINVAR_GENOMIC_KEY", sort=False)[LABEL_COL]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    ambiguous = (
        working["CLINVAR_VARIATION_ID"].astype(str).isin(ambiguous_stable_ids)
        | working["CLINVAR_GENOMIC_KEY"].astype(str).isin(ambiguous_genomic_keys)
        | working["CLINVAR_TEMPORAL_KEY"].astype(str).isin(temporal_label_conflicts)
        | working["CLINVAR_GENOMIC_KEY"].astype(str).isin(genomic_label_conflicts)
    )
    rows_before = len(working)
    working = working.loc[~ambiguous].copy()

    review_rank = {
        "practice guideline": 3,
        "reviewed by expert panel": 2,
        "criteria provided, multiple submitters, no conflicts": 1,
    }
    working["_review_rank"] = working["review"].astype(str).str.lower().str.strip().map(review_rank)
    if "last_evaluated" in working:
        working["_last_evaluated"] = pd.to_datetime(
            working["last_evaluated"], errors="coerce", format="mixed"
        )
    else:
        working["_last_evaluated"] = pd.NaT
    working["_stable_rank"] = working["CLINVAR_VARIATION_ID"].notna().astype(int)
    working = working.sort_values(
        ["CLINVAR_TEMPORAL_KEY", "_review_rank", "_last_evaluated"],
        ascending=[True, False, False],
        kind="mergesort",
    ).drop_duplicates("CLINVAR_TEMPORAL_KEY", keep="first")
    before_genomic_dedup = len(working)
    working = working.sort_values(
        ["CLINVAR_GENOMIC_KEY", "_stable_rank", "_review_rank", "_last_evaluated"],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop_duplicates("CLINVAR_GENOMIC_KEY", keep="first")
    audit = {
        "identity_policy": CLINVAR_IDENTITY_POLICY,
        "eligible_rows_before_identity_checks": int(rows_before),
        "rows_with_stable_variation_id": int(stable.sum()),
        "rows_without_stable_variation_id": int((~stable).sum()),
        "ambiguous_stable_variation_ids": int(len(ambiguous_stable_ids)),
        "ambiguous_stable_variation_id_values": sorted(ambiguous_stable_ids),
        "ambiguous_genomic_keys_with_multiple_stable_ids": int(len(ambiguous_genomic_keys)),
        "ambiguous_genomic_key_values": sorted(ambiguous_genomic_keys),
        "temporal_identity_label_conflicts": int(len(temporal_label_conflicts)),
        "temporal_identity_label_conflict_values": sorted(temporal_label_conflicts),
        "genomic_label_conflicts": int(len(genomic_label_conflicts)),
        "genomic_label_conflict_values": sorted(genomic_label_conflicts),
        "rows_excluded_for_identity_ambiguity_or_conflict": int(ambiguous.sum()),
        "genomic_fallback_rows_shadowed_by_stable_identity": int(
            before_genomic_dedup - len(working)
        ),
        "retained_unique_temporal_identities": int(len(working)),
        "retained_stable_variation_ids": int(working["CLINVAR_VARIATION_ID"].notna().sum()),
        "retained_genomic_fallback_identities": int(working["CLINVAR_VARIATION_ID"].isna().sum()),
    }
    working = working.drop(columns=["_review_rank", "_last_evaluated", "_stable_rank"]).sort_values(
        "CLINVAR_TEMPORAL_KEY", kind="mergesort"
    )
    working.attrs["clinvar_identity_audit"] = audit
    return working


def _validate_external_temporal_configuration() -> dict[str, dict[str, Any]]:
    train_status = source_release_status("ClinVar training snapshot", CLINVAR_TRAIN_RELEASE)
    external_status = source_release_status("ClinVar external snapshot", CLINVAR_EXTERNAL_RELEASE)
    if not train_status["on_or_before_cutoff"]:
        raise RuntimeError(
            f"ClinVar training snapshot is after the exact training cutoff: {train_status}"
        )
    if external_status["on_or_before_cutoff"]:
        raise RuntimeError(
            f"ClinVar external snapshot must be after the exact training cutoff: {external_status}"
        )
    return {"clinvar_train": train_status, "clinvar_external": external_status}


def _read_clinvar(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    aliases = {
        "chr": ("Chromosome", "#Chromosome", "chr"),
        "pos": ("PositionVCF", "Start", "pos"),
        "ref": ("ReferenceAlleleVCF", "ReferenceAllele", "ref"),
        "alt": ("AlternateAlleleVCF", "AlternateAllele", "alt"),
        "significance": ("ClinicalSignificance", "clinvar_clnsig"),
        "review": ("ReviewStatus", "review_status"),
        "assembly": ("Assembly", "assembly"),
        "last_evaluated": ("LastEvaluated", "DateLastEvaluated"),
        # AlleleID is a different ClinVar identifier and must never silently
        # substitute for the stable VariationID temporal key.
        "clinvar_variation_id": ("VariationID", "#VariationID"),
        "name": ("Name",),
        "gene_symbol": ("GeneSymbol", "Gene"),
    }
    resolved = {
        name: _resolve_column(
            header,
            values,
            required=name in {"chr", "pos", "ref", "alt", "significance", "review"},
        )
        for name, values in aliases.items()
    }
    usecols = [column for column in resolved.values() if column is not None]
    chunks: list[pd.DataFrame] = []
    rename = {column: name for name, column in resolved.items() if column is not None}
    for raw in pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype="string",
        chunksize=100_000,
        low_memory=False,
    ):
        chunk = raw.rename(columns=rename)
        if "assembly" in chunk:
            assembly = chunk["assembly"].str.upper().str.replace(" ", "", regex=False)
            chunk = chunk[assembly.isin({"GRCH37", "GRCH37.P13", "HG19"})].copy()
        review = chunk["review"].str.lower().str.strip()
        chunk = chunk[review.isin(HIGH_CONFIDENCE_REVIEWS)].copy()
        significance = (
            chunk["significance"]
            .astype("string")
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        pathogenic = significance.isin(CLINVAR_PATHOGENIC_AGGREGATES)
        benign = significance.isin(CLINVAR_BENIGN_AGGREGATES)
        chunk = chunk[pathogenic | benign].copy()
        chunk[LABEL_COL] = pathogenic.reindex(chunk.index).astype(np.int8)
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=[*resolved, LABEL_COL])
    frame = pd.concat(chunks, ignore_index=True)
    frame = stage01._normalize_variant_frame(frame)
    frame["variant_id"] = stage01._variant_keys(frame)
    return _attach_clinvar_identity(frame)


def _parse_clinvar_name_consequences(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover RefSeq missense consequences absent from a fixed dbNSFP release.

    Stage 04 authenticates the parsed transcript against UniProt RefSeq
    cross-references and independently checks the reference residue.  Rows that
    cannot pass both checks remain excluded from the primary external cohort.
    """
    columns = [
        "variant_id",
        "genename",
        "Ensembl_transcriptid",
        "VEP_canonical",
        "MANE",
        "HGVSp_snpEff",
        "HGVSc_snpEff",
        "aapos",
        "aaref",
        "aaalt",
        "Interpro_domain",
        "CLINVAR_CONSEQUENCE_SOURCE",
        "TRANSCRIPT_NAMESPACE",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        *RAW_CONTEXTUAL_PREDICTOR_COLS,
    ]
    if frame.empty or "name" not in frame:
        return pd.DataFrame(columns=columns)
    parsed: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        name = str(getattr(row, "name", ""))
        match = CLINVAR_MISSENSE_PATTERN.search(name)
        if match is None:
            continue
        reference = AA_THREE_TO_ONE.get(match.group("reference").upper())
        alternate = AA_THREE_TO_ONE.get(match.group("alternate").upper())
        if reference is None or alternate is None or reference == alternate:
            continue
        gene = str(getattr(row, "gene_symbol", "")).strip()
        gene = re.split(r"[;,|]", gene, maxsplit=1)[0].strip()
        if not gene or gene.lower() in {"nan", "na", "-"}:
            continue
        transcript = match.group("transcript")
        position = int(match.group("position"))
        coding_match = re.search(r":(c\.[^\s(]+)", name, flags=re.IGNORECASE)
        parsed.append(
            {
                "variant_id": str(getattr(row, "variant_id")),
                "genename": gene,
                # This compatibility column can carry Ensembl or RefSeq IDs;
                # Stage 04 normalizes versions and verifies the namespace.
                "Ensembl_transcriptid": transcript,
                "VEP_canonical": "",
                "MANE": "",
                "HGVSp_snpEff": (
                    f"p.{match.group('reference').title()}"
                    f"{position}{match.group('alternate').title()}"
                ),
                "HGVSc_snpEff": coding_match.group(1) if coding_match else "",
                "aapos": position,
                "aaref": reference,
                "aaalt": alternate,
                "Interpro_domain": np.nan,
                "CLINVAR_CONSEQUENCE_SOURCE": "clinvar_name_refseq",
                "TRANSCRIPT_NAMESPACE": "RefSeq",
            }
        )
    result = pd.DataFrame(parsed)
    if result.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in result:
            result[column] = np.nan
    key = [
        "variant_id",
        "genename",
        "Ensembl_transcriptid",
        "aapos",
        "aaref",
        "aaalt",
    ]
    return result[columns].sort_values(key, kind="mergesort").drop_duplicates(key)


def _exclude_cross_snapshot_identity_ambiguity(
    older: pd.DataFrame, current: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fail closed when stable IDs and genomic alleles remap across releases."""
    older = _ensure_clinvar_identity_columns(older)
    current = _ensure_clinvar_identity_columns(current)
    combined = pd.concat(
        [
            older[["CLINVAR_VARIATION_ID", "CLINVAR_GENOMIC_KEY"]].assign(_snapshot="baseline"),
            current[["CLINVAR_VARIATION_ID", "CLINVAR_GENOMIC_KEY"]].assign(_snapshot="endpoint"),
        ],
        ignore_index=True,
    )
    stable = combined.dropna(subset=["CLINVAR_VARIATION_ID"])
    remapped_stable_ids = set(
        stable.groupby("CLINVAR_VARIATION_ID", sort=False)["CLINVAR_GENOMIC_KEY"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    remapped_genomic_keys = set(
        stable.groupby("CLINVAR_GENOMIC_KEY", sort=False)["CLINVAR_VARIATION_ID"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
    )
    excluded = current["CLINVAR_VARIATION_ID"].astype(str).isin(remapped_stable_ids) | current[
        "CLINVAR_GENOMIC_KEY"
    ].astype(str).isin(remapped_genomic_keys)
    filtered = current.loc[~excluded].copy()
    audit = {
        "stable_variation_ids_remapped_between_snapshots": int(len(remapped_stable_ids)),
        "remapped_stable_variation_id_values": sorted(remapped_stable_ids),
        "genomic_keys_linked_to_multiple_stable_ids_between_snapshots": int(
            len(remapped_genomic_keys)
        ),
        "remapped_genomic_key_values": sorted(remapped_genomic_keys),
        "endpoint_rows_excluded_for_cross_snapshot_remapping": int(excluded.sum()),
        "excluded_endpoint_temporal_keys": sorted(
            current.loc[excluded, "CLINVAR_TEMPORAL_KEY"].astype(str).tolist()
        ),
    }
    return filtered, audit


def _match_clinvar_temporal_identities(older: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Match by stable VariationID first and genomic allele only for missing IDs."""
    older = _ensure_clinvar_identity_columns(older)
    current = _ensure_clinvar_identity_columns(current)
    if current.empty:
        output = current.copy()
        output["BASELINE_LABEL"] = pd.Series(dtype="float64")
        output["BASELINE_CLINVAR_VARIATION_ID"] = pd.Series(dtype="string")
        output["BASELINE_CLINVAR_GENOMIC_KEY"] = pd.Series(dtype="string")
        output["TEMPORAL_MATCH_SOURCE"] = pd.Series(dtype="string")
        return output
    baseline_stable = {
        str(row.CLINVAR_VARIATION_ID): row
        for row in older.itertuples(index=False)
        if pd.notna(row.CLINVAR_VARIATION_ID)
    }
    baseline_genomic = {str(row.CLINVAR_GENOMIC_KEY): row for row in older.itertuples(index=False)}
    records: list[dict[str, Any]] = []
    for row in current.itertuples(index=False):
        stable_id = str(row.CLINVAR_VARIATION_ID) if pd.notna(row.CLINVAR_VARIATION_ID) else None
        genomic_key = str(row.CLINVAR_GENOMIC_KEY)
        matched = baseline_stable.get(stable_id) if stable_id is not None else None
        match_source = "stable_variation_id" if matched is not None else None
        if matched is None:
            genomic_match = baseline_genomic.get(genomic_key)
            # Genomic matching is a fallback only when at least one side lacks
            # a stable VariationID. Different non-missing IDs were excluded as
            # ambiguous remapping before this step.
            if genomic_match is not None and (
                stable_id is None or pd.isna(genomic_match.CLINVAR_VARIATION_ID)
            ):
                matched = genomic_match
                match_source = (
                    "genomic_fallback_endpoint_missing_variation_id"
                    if stable_id is None
                    else "genomic_fallback_baseline_missing_variation_id"
                )
        records.append(
            {
                "BASELINE_LABEL": (
                    int(getattr(matched, LABEL_COL)) if matched is not None else np.nan
                ),
                "BASELINE_CLINVAR_VARIATION_ID": (
                    str(matched.CLINVAR_VARIATION_ID)
                    if matched is not None and pd.notna(matched.CLINVAR_VARIATION_ID)
                    else pd.NA
                ),
                "BASELINE_CLINVAR_GENOMIC_KEY": (
                    str(matched.CLINVAR_GENOMIC_KEY) if matched is not None else pd.NA
                ),
                "TEMPORAL_MATCH_SOURCE": match_source or "absent_at_baseline",
            }
        )
    matched_frame = current.reset_index(drop=True).copy()
    match_table = pd.DataFrame(records, index=matched_frame.index)
    for column in match_table:
        matched_frame[column] = match_table[column]
    return matched_frame


def _run_scv_candidate_screen(
    candidates: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the provenance-checked streaming SCV screen from the tools module."""
    baseline_provenance = validate_clinvar_submission_provenance(
        CLINVAR_TRAIN_SUBMISSION_ARCHIVE, CLINVAR_TRAIN_RELEASE
    )
    endpoint_provenance = validate_clinvar_submission_provenance(
        CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE, CLINVAR_EXTERNAL_RELEASE
    )
    specification = importlib.util.spec_from_file_location(
        "varifuse_clinvar_scv_audit", CLINVAR_SCV_AUDIT_TOOL
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load ClinVar SCV audit module: {CLINVAR_SCV_AUDIT_TOOL}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    records, scan = module.screen_candidate_variations(
        CLINVAR_TRAIN_SUBMISSION_ARCHIVE,
        CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE,
        candidates,
        TRAIN_CUTOFF_DATE,
        min_matching_scvs=CLINVAR_SCV_MIN_MATCHING,
        min_unique_submitters=CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS,
        require_no_opposition=True,
        require_post_cutoff_event=True,
    )
    return records, {
        "baseline_submission_archive": baseline_provenance,
        "endpoint_submission_archive": endpoint_provenance,
        "stream_scan": scan,
    }


def _apply_clinvar_scv_evidence(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Require label-consistent, post-cutoff versioned SCV evidence."""
    if candidates.empty:
        return (
            candidates.copy(),
            pd.DataFrame(),
            {
                "status": "empty_candidate_set",
                "policy": CLINVAR_SCV_POLICY,
            },
        )
    if not CLINVAR_REQUIRE_SCV_EVIDENCE:
        output = candidates.copy()
        output["SCV_EVIDENCE_PASS"] = pd.NA
        return (
            output,
            pd.DataFrame(),
            {
                "status": "disabled_by_configuration",
                "policy": CLINVAR_SCV_POLICY,
                "candidate_rows": int(len(output)),
            },
        )

    candidates = candidates.reset_index(drop=True).copy()
    stable = candidates["CLINVAR_VARIATION_ID"].notna()
    stable_candidates = candidates.loc[stable].copy()
    if stable_candidates["CLINVAR_VARIATION_ID"].duplicated().any():
        raise RuntimeError("ClinVar SCV candidates contain duplicate stable VariationIDs")
    candidate_classes = {
        int(variation_id): "pathogenic" if int(label) == 1 else "benign"
        for variation_id, label in stable_candidates[
            ["CLINVAR_VARIATION_ID", LABEL_COL]
        ].itertuples(index=False)
    }
    if not candidate_classes:
        audit = candidates.copy()
        audit["scv_evidence_pass"] = 0
        audit["scv_evidence_reason"] = "missing_stable_variation_id_scv_screen_unavailable"
        audit["SCV_EVIDENCE_PASS"] = 0
        audit["SCV_EVIDENCE_REASON"] = audit["scv_evidence_reason"]
        return (
            candidates.iloc[0:0].copy(),
            audit.reindex(columns=CLINVAR_SCV_AUDIT_COLUMNS),
            {
                "status": "no_candidates_with_stable_variation_id",
                "policy": CLINVAR_SCV_POLICY,
                "candidate_rows": int(len(candidates)),
                "excluded_genomic_fallback_rows": int(len(candidates)),
            },
        )

    records, provenance = _run_scv_candidate_screen(candidate_classes)
    screen = pd.DataFrame(records)
    expected_ids = {str(value) for value in candidate_classes}
    if screen.empty or "VariationID" not in screen:
        raise RuntimeError("ClinVar SCV screen returned no auditable records")
    screen["CLINVAR_VARIATION_ID"] = _normalise_clinvar_variation_ids(screen["VariationID"])
    observed_ids = set(screen["CLINVAR_VARIATION_ID"].dropna().astype(str))
    if (
        screen["CLINVAR_VARIATION_ID"].isna().any()
        or observed_ids != expected_ids
        or screen["CLINVAR_VARIATION_ID"].duplicated().any()
    ):
        raise RuntimeError("ClinVar SCV screen VariationIDs are incomplete or duplicated")
    observed_classes = dict(
        zip(
            screen["CLINVAR_VARIATION_ID"].astype(str),
            screen["candidate_class"].astype(str).str.lower(),
        )
    )
    expected_classes = {
        str(variation_id): candidate_class
        for variation_id, candidate_class in candidate_classes.items()
    }
    if observed_classes != expected_classes:
        raise RuntimeError("ClinVar SCV screen candidate classes do not match labels")
    screen = screen.drop(columns=["VariationID"])
    merged = candidates.merge(
        screen,
        on="CLINVAR_VARIATION_ID",
        how="left",
        validate="one_to_one",
    )
    stable = merged["CLINVAR_VARIATION_ID"].notna()
    missing_stable_screen = stable & merged["scv_evidence_pass"].isna()
    if missing_stable_screen.any():
        raise RuntimeError("Stable ClinVar candidates are missing SCV evidence records")

    count_columns = (
        "current_contributing_matching_scv_count",
        "current_contributing_opposing_scv_count",
        "current_contributing_ambiguous_scv_count",
        "current_contributing_matching_unique_submitter_count",
        "post_cutoff_matching_event_scv_count",
    )
    screen_counts: dict[str, pd.Series] = {}
    for column in count_columns:
        values = pd.to_numeric(merged[column], errors="coerce")
        invalid = stable & (values.isna() | values.lt(0) | values.mod(1).fillna(1).ne(0))
        if invalid.any():
            raise RuntimeError(f"ClinVar SCV screen returned invalid {column}")
        screen_counts[column] = values.fillna(0)

    recomputed_tool_pass = (
        screen_counts["current_contributing_matching_scv_count"].ge(CLINVAR_SCV_MIN_MATCHING)
        & screen_counts["current_contributing_matching_unique_submitter_count"].ge(
            CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS
        )
        & screen_counts["current_contributing_opposing_scv_count"].eq(0)
        & screen_counts["current_contributing_ambiguous_scv_count"].eq(0)
        & screen_counts["post_cutoff_matching_event_scv_count"].gt(0)
    )
    reported_tool_pass = pd.to_numeric(merged["scv_evidence_pass"], errors="coerce").eq(1)
    if (stable & recomputed_tool_pass.ne(reported_tool_pass)).any():
        raise RuntimeError(
            "ClinVar SCV screen pass flags disagree with independently recomputed evidence counts"
        )

    multiple_submitter_review = (
        merged["review"]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq("criteria provided, multiple submitters, no conflicts")
    )
    required_submitters = np.where(
        multiple_submitter_review,
        max(
            CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS,
            CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM,
        ),
        CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS,
    )
    matching_submitters = screen_counts["current_contributing_matching_unique_submitter_count"]
    review_submitter_pass = matching_submitters.ge(required_submitters)
    tool_pass = reported_tool_pass
    final_pass = stable & tool_pass & review_submitter_pass
    merged["SCV_REQUIRED_MATCHING_UNIQUE_SUBMITTERS"] = required_submitters.astype(np.int16)
    merged["SCV_REVIEW_SUBMITTER_PASS"] = review_submitter_pass.astype(np.int8)
    merged["SCV_EVIDENCE_PASS"] = final_pass.astype(np.int8)
    reason = merged["scv_evidence_reason"].astype("string")
    reason = reason.mask(~stable, "missing_stable_variation_id_scv_screen_unavailable")
    reason = reason.mask(
        stable & tool_pass & ~review_submitter_pass,
        "insufficient_unique_matching_submitters_for_aggregate_review_status",
    )
    merged["SCV_EVIDENCE_REASON"] = reason
    final_assertion_policy = (
        "current_high_confidence_aggregate_with_post_cutoff_new_or_versioned_matching_scv_evidence"
    )
    merged.loc[final_pass, "TEMPORAL_ASSERTION_POLICY"] = final_assertion_policy
    audit = merged.reindex(columns=CLINVAR_SCV_AUDIT_COLUMNS).copy()
    retained = merged.loc[final_pass].copy()
    summary = {
        "status": "applied",
        "policy": CLINVAR_SCV_POLICY,
        "model_outputs_or_predictor_availability_used_for_selection": False,
        "candidate_rows_before_scv_screen": int(len(candidates)),
        "stable_variation_id_candidates_screened": int(len(stable_candidates)),
        "genomic_fallback_candidates_excluded_without_scv_identity": int((~stable).sum()),
        "retained_rows": int(len(retained)),
        "excluded_rows": int(len(candidates) - len(retained)),
        "reason_counts": audit["SCV_EVIDENCE_REASON"].value_counts(dropna=False).to_dict(),
        "retained_temporal_status_counts": retained["TEMPORAL_STATUS"]
        .value_counts(dropna=False)
        .to_dict(),
        "retained_review_status_counts": retained["review"]
        .astype(str)
        .str.lower()
        .str.strip()
        .value_counts(dropna=False)
        .to_dict(),
        "post_cutoff_new_matching_scvs_retained": int(
            pd.to_numeric(retained["post_cutoff_new_matching_scv_count"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "post_cutoff_updated_matching_scvs_retained": int(
            pd.to_numeric(retained["post_cutoff_updated_matching_scv_count"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "thresholds": {
            "minimum_matching_scvs": CLINVAR_SCV_MIN_MATCHING,
            "minimum_unique_matching_submitters": (CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS),
            "multiple_submitter_review_minimum": (CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM),
            "require_no_current_contributing_opposition_or_ambiguity": True,
            "require_post_cutoff_new_or_version_increased_matching_scv": True,
        },
        "provenance": provenance,
    }
    return retained, audit, summary


def load_temporal_clinvar(
    *, return_audit: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load high-confidence temporal candidates using stable ClinVar identity."""
    _validate_external_temporal_configuration()
    if not CLINVAR_EXTERNAL_ARCHIVE.exists():
        logger.warning("ClinVar external archive is missing")
        empty = pd.DataFrame()
        if return_audit:
            return empty, {"status": "external_snapshot_missing"}, empty
        return empty
    current = _read_clinvar(CLINVAR_EXTERNAL_ARCHIVE)
    if not CLINVAR_TRAIN_ARCHIVE.exists():
        raise FileNotFoundError(
            "Temporal ClinVar validation requires the exact training archive: "
            f"{CLINVAR_TRAIN_ARCHIVE}"
        )
    older = _read_clinvar(CLINVAR_TRAIN_ARCHIVE)
    current_identity_audit = dict(current.attrs.get("clinvar_identity_audit", {}))
    older_identity_audit = dict(older.attrs.get("clinvar_identity_audit", {}))
    current, cross_snapshot_audit = _exclude_cross_snapshot_identity_ambiguity(older, current)
    current = _match_clinvar_temporal_identities(older, current)
    previous = pd.to_numeric(current["BASELINE_LABEL"], errors="coerce")
    current["TEMPORAL_STATUS"] = np.where(
        previous.isna(),
        "new_variant",
        np.where(previous.ne(current[LABEL_COL]), "reclassified", "unchanged"),
    )
    temporal_match_counts_before_filter = (
        current["TEMPORAL_MATCH_SOURCE"].value_counts(dropna=False).to_dict()
    )
    temporal_status_counts_before_filter = (
        current["TEMPORAL_STATUS"].value_counts(dropna=False).to_dict()
    )
    if CLINVAR_REQUIRE_SCV_EVIDENCE:
        # A stable aggregate label can still have genuinely new, independent
        # post-cutoff evidence.  The SCV version/date screen is the temporal
        # gate, so unchanged aggregate rows remain eligible for that screen.
        aggregate_candidate_policy = "all_current_high_confidence_aggregates_then_scv_temporal_gate"
    else:
        current = current[current["TEMPORAL_STATUS"].ne("unchanged")].copy()
        aggregate_candidate_policy = "aggregate_new_or_reclassified_without_scv_temporal_gate"
    rows_before_date_filter = int(len(current))
    if CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION:
        if "last_evaluated" not in current:
            raise RuntimeError("Strict ClinVar temporal validation requires a LastEvaluated column")
        evaluated = pd.to_datetime(current["last_evaluated"], errors="coerce", format="mixed")
        cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
        eligible = evaluated.gt(cutoff)
        excluded = int((~eligible).sum())
        if excluded:
            logger.warning(
                "Excluded %d ClinVar temporal candidates without a post-cutoff "
                "aggregate LastEvaluated date",
                excluded,
            )
        current = current.loc[eligible].copy()
        current["TEMPORAL_ASSERTION_POLICY"] = "last_evaluated_after_cutoff"
    else:
        current["TEMPORAL_ASSERTION_POLICY"] = "snapshot_difference_only"
    rows_after_date_filter = int(len(current))
    status_counts_after_date_filter = (
        current["TEMPORAL_STATUS"].value_counts(dropna=False).to_dict()
    )
    current["EXT_SOURCE"] = "clinvar"
    current["CLINVAR_RELEASE"] = CLINVAR_EXTERNAL_RELEASE
    current["TRAIN_CUTOFF_DATE"] = TRAIN_CUTOFF_DATE
    current, scv_audit, scv_summary = _apply_clinvar_scv_evidence(current)
    temporal_audit = {
        "identity_policy": CLINVAR_IDENTITY_POLICY,
        "baseline_snapshot_identity": older_identity_audit,
        "endpoint_snapshot_identity": current_identity_audit,
        "cross_snapshot_identity": cross_snapshot_audit,
        "temporal_match_source_counts_before_candidate_filter": (
            temporal_match_counts_before_filter
        ),
        "temporal_status_counts_before_candidate_filter": (temporal_status_counts_before_filter),
        "aggregate_candidate_policy": aggregate_candidate_policy,
        "rows_before_aggregate_date_filter": rows_before_date_filter,
        "rows_after_aggregate_date_filter_before_scv": rows_after_date_filter,
        "temporal_status_counts_after_date_filter_before_scv": (status_counts_after_date_filter),
        "temporal_match_source_counts": current["TEMPORAL_MATCH_SOURCE"]
        .value_counts(dropna=False)
        .to_dict(),
        "temporal_status_counts_retained_after_scv": current["TEMPORAL_STATUS"]
        .value_counts(dropna=False)
        .to_dict(),
        "retained_rows": int(len(current)),
        "stable_variation_id_rows": int(current["CLINVAR_VARIATION_ID"].notna().sum()),
        "genomic_fallback_rows": int(current["CLINVAR_VARIATION_ID"].isna().sum()),
        "scv_evidence": scv_summary,
    }
    current.attrs["temporal_clinvar_audit"] = temporal_audit
    if return_audit:
        return current, temporal_audit, scv_audit
    return current


def scan_dbnsfp_for_clinvar(keys: set[str]) -> pd.DataFrame:
    """Extract exact ClinVar matches from dbNSFP."""
    if not keys:
        return pd.DataFrame()
    if not DBNSFP_FILE.exists():
        raise FileNotFoundError(DBNSFP_FILE)
    matched: list[pd.DataFrame] = []
    source_rows = 0
    projected_rows = 0
    iterator = pd.read_csv(
        DBNSFP_FILE,
        sep="\t",
        compression="gzip",
        usecols=DBNSFP_COLS,
        chunksize=50000,
        low_memory=True,
        dtype={
            stage01.DBNSFP_PRIMARY_CHROM_COLUMN: "string",
            DBNSFP_GRCH37_CHROM_COLUMN: "string",
        },
    )
    for chunk_number, raw in enumerate(iterator, 1):
        source_rows += len(raw)
        frame = stage01.project_dbnsfp_to_grch37(raw)
        projected_rows += len(frame)
        frame["variant_id"] = stage01._variant_keys(frame)
        frame = frame[frame["variant_id"].isin(keys)].copy()
        if frame.empty:
            continue
        frame, ambiguous = stage01.explode_aligned_columns(frame, stage01.TRANSCRIPT_COLUMNS)
        if ambiguous:
            logger.warning("Excluded %d ambiguous ClinVar transcript rows", ambiguous)
        frame["aapos"] = pd.to_numeric(frame["aapos"], errors="coerce").astype("Int64")
        frame["aaref"] = frame["aaref"].astype("string").str.upper()
        frame["aaalt"] = frame["aaalt"].astype("string").str.upper()
        standard = set(STANDARD_AA)
        frame = frame[
            frame["aaref"].isin(standard)
            & frame["aaalt"].isin(standard)
            & frame["aaref"].ne(frame["aaalt"])
            & frame["aapos"].gt(0)
        ].copy()
        for column in (
            "GERP++_RS",
            "phyloP100way_vertebrate",
            "phastCons100way_vertebrate",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        matched.append(frame)
        if chunk_number % 20 == 0:
            logger.info("Scanned %d dbNSFP chunks", chunk_number)
    result = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
    coordinate_audit = {
        **COORDINATE_CONTRACT,
        "dbnsfp_source_rows": source_rows,
        "dbnsfp_grch37_projected_rows": projected_rows,
        "dbnsfp_rows_without_valid_grch37_projection": source_rows - projected_rows,
        "matched_annotation_rows_before_deduplication": len(result),
    }
    if result.empty:
        result.attrs["dbnsfp_coordinate_projection"] = coordinate_audit
        return result
    subset = [
        "variant_id",
        "genename",
        "Ensembl_transcriptid",
        "aapos",
        "aaref",
        "aaalt",
    ]
    result = result.sort_values(subset, kind="mergesort").drop_duplicates(subset)
    coordinate_audit["matched_annotation_rows_after_deduplication"] = len(result)
    result.attrs["dbnsfp_coordinate_projection"] = coordinate_audit
    return result


def _stable_identifier(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _select_primary_clinvar_consequences(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one ClinVar consequence after transcript/protein validation."""
    selected, full_audit = stage04.select_primary_mapped_consequences(
        frame,
        require_transcript_mapping=True,
    )
    audit_columns = [
        column
        for column in (
            "variant_id",
            ROW_ID_COL,
            GENE_COL,
            "Ensembl_transcriptid",
            "MANE",
            "VEP_canonical",
            "HGVSp_snpEff",
            "HGVSc_snpEff",
            "aapos",
            "aaref",
            "aaalt",
            LABEL_COL,
            "CLINVAR_VARIATION_ID",
            "CLINVAR_GENOMIC_KEY",
            "CLINVAR_TEMPORAL_KEY",
            "CLINVAR_IDENTITY_SOURCE",
            "BASELINE_CLINVAR_VARIATION_ID",
            "BASELINE_CLINVAR_GENOMIC_KEY",
            "TEMPORAL_MATCH_SOURCE",
            "TEMPORAL_STATUS",
            "TEMPORAL_ASSERTION_POLICY",
            "last_evaluated",
            "current_contributing_matching_scv_count",
            "current_contributing_opposing_scv_count",
            "current_contributing_ambiguous_scv_count",
            "current_contributing_matching_unique_submitter_count",
            "post_cutoff_new_matching_scv_count",
            "post_cutoff_updated_matching_scv_count",
            "post_cutoff_matching_event_scv_count",
            "post_cutoff_matching_event_scv_ids",
            "post_cutoff_matching_event_dates",
            "SCV_REQUIRED_MATCHING_UNIQUE_SUBMITTERS",
            "SCV_REVIEW_SUBMITTER_PASS",
            "SCV_EVIDENCE_PASS",
            "SCV_EVIDENCE_REASON",
            "uniprot_id",
            "PROTEIN_MAPPING_STATUS",
            "MAPPING_TRANSCRIPT_MATCH",
            "PRIMARY_MAPPING_ELIGIBLE",
            "MAPPING_CONFIDENCE",
            "PRIMARY_CONSEQUENCE_RANK",
            "PRIMARY_CONSEQUENCE_SELECTED",
            "TRANSCRIPT_CANDIDATE_COUNT",
            "TRANSCRIPT_MAPPED_CANDIDATE_COUNT",
            "CONSEQUENCE_SELECTION_POLICY",
            "TRANSCRIPT_SELECTION_OUTCOME",
            "CLINVAR_CONSEQUENCE_SOURCE",
            "TRANSCRIPT_NAMESPACE",
        )
        if column in full_audit
    ]
    return selected, full_audit[audit_columns].copy()


def prepare_clinvar(
    resources: Any,
    *,
    return_selection_audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare temporal ClinVar records."""
    clinvar, temporal_audit, scv_evidence_audit = load_temporal_clinvar(return_audit=True)
    dbnsfp_coordinate_audit: dict[str, Any] = {}

    def attach_audit_metadata(frame: pd.DataFrame) -> pd.DataFrame:
        frame.attrs["temporal_clinvar_audit"] = temporal_audit
        frame.attrs["clinvar_scv_evidence_audit"] = scv_evidence_audit
        frame.attrs["dbnsfp_coordinate_projection"] = dbnsfp_coordinate_audit
        return frame

    if clinvar.empty:
        empty = attach_audit_metadata(pd.DataFrame())
        return (empty, empty, empty) if return_selection_audit else (empty, empty)
    dbnsfp = scan_dbnsfp_for_clinvar(set(clinvar["variant_id"]))
    dbnsfp_coordinate_audit = dict(dbnsfp.attrs.get("dbnsfp_coordinate_projection", {}))
    parsed_clinvar = _parse_clinvar_name_consequences(clinvar)
    if dbnsfp.empty and parsed_clinvar.empty:
        logger.warning("No temporal ClinVar rows yielded a protein consequence")
        empty = attach_audit_metadata(pd.DataFrame())
        return (empty, empty, empty) if return_selection_audit else (empty, empty)
    feature_columns = [
        "variant_id",
        "genename",
        "Ensembl_transcriptid",
        "VEP_canonical",
        "MANE",
        "HGVSp_snpEff",
        "HGVSc_snpEff",
        "aapos",
        "aaref",
        "aaalt",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "phastCons100way_vertebrate",
        "Interpro_domain",
        *RAW_CONTEXTUAL_PREDICTOR_COLS,
        "CLINVAR_CONSEQUENCE_SOURCE",
        "TRANSCRIPT_NAMESPACE",
    ]
    if not dbnsfp.empty:
        dbnsfp = dbnsfp.copy()
        dbnsfp["CLINVAR_CONSEQUENCE_SOURCE"] = "dbnsfp_transcript"
        dbnsfp["TRANSCRIPT_NAMESPACE"] = "Ensembl"
    candidates = pd.concat(
        [candidate for candidate in (dbnsfp, parsed_clinvar) if not candidate.empty],
        ignore_index=True,
        sort=False,
    )
    for column in feature_columns:
        if column not in candidates:
            candidates[column] = np.nan
    candidate_key = [
        "variant_id",
        "genename",
        "Ensembl_transcriptid",
        "aapos",
        "aaref",
        "aaalt",
    ]
    candidates = candidates.sort_values(
        [*candidate_key, "CLINVAR_CONSEQUENCE_SOURCE"], kind="mergesort"
    ).drop_duplicates(candidate_key, keep="first")
    merged = clinvar.merge(
        candidates[feature_columns],
        on="variant_id",
        how="inner",
        validate="one_to_many",
    )
    for column in RAW_CONTEXTUAL_PREDICTOR_COLS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    consensus_complete = (
        merged[["REVEL_score", "SIFT_score", "Polyphen2_HDIV_score"]].notna().all(axis=1)
    )
    merged["CONSENSUS_SCORE"] = (
        (
            merged["REVEL_score"].fillna(0.5) * 0.6
            + (1.0 - merged["SIFT_score"].fillna(0.5)) * 0.2
            + merged["Polyphen2_HDIV_score"].fillna(0.5) * 0.2
        )
        .where(consensus_complete)
        .astype(np.float32)
    )
    merged["transcript_variant_id"] = (
        merged["variant_id"].astype(str)
        + "|"
        + merged["Ensembl_transcriptid"].fillna("no_transcript").astype(str)
    )
    merged[ROW_ID_COL] = [
        _stable_identifier(f"clinvar|{variant}|{transcript}|{gene}|{ref}{position}{alt}")
        for variant, transcript, gene, ref, position, alt in zip(
            merged["variant_id"],
            merged["Ensembl_transcriptid"].fillna("no_transcript"),
            merged[GENE_COL],
            merged["aaref"],
            merged["aapos"],
            merged["aaalt"],
        )
    ]
    annotated = stage04.annotate_variants(merged, resources=resources)
    selected, selection_audit = _select_primary_clinvar_consequences(annotated)
    ready, audit = stage08._prepare_chunk(
        selected,
        require_transcript_mapping=True,
    )
    aligned = _align_external_schema(ready)
    if not aligned.empty and aligned["variant_id"].duplicated().any():
        raise RuntimeError("Primary external ClinVar output must have one row per variant")
    if not aligned.empty and aligned["CLINVAR_TEMPORAL_KEY"].duplicated().any():
        raise RuntimeError(
            "Primary external ClinVar output must have one row per temporal identity"
        )
    if CLINVAR_REQUIRE_SCV_EVIDENCE and not aligned.empty:
        stable_ids = aligned["CLINVAR_VARIATION_ID"]
        scv_pass = pd.to_numeric(aligned["SCV_EVIDENCE_PASS"], errors="coerce")
        if stable_ids.isna().any() or stable_ids.duplicated().any():
            raise RuntimeError("SCV-gated ClinVar output requires unique stable VariationIDs")
        if not scv_pass.eq(1).all():
            raise RuntimeError("ClinVar output contains rows that failed the SCV gate")
    if not aligned.empty:
        aligned["EXTERNAL_FEATURE_PROFILE"] = "matched_human_annotation"
    aligned = attach_audit_metadata(aligned)
    selection_audit.attrs["temporal_clinvar_audit"] = temporal_audit
    if return_selection_audit:
        return aligned, audit, selection_audit
    return aligned, audit


def _metadata_lookup(metadata: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    by_assay: dict[str, pd.Series] = {}
    by_file: dict[str, pd.Series] = {}
    if metadata.empty:
        return by_assay, by_file
    for _, row in metadata.iterrows():
        for column in ("DMS_id", "assay_id", "ProteinGym_ID"):
            if column in metadata and pd.notna(row.get(column)):
                by_assay[str(row[column])] = row
        for column in ("DMS_filename", "filename"):
            if column in metadata and pd.notna(row.get(column)):
                by_file[Path(str(row[column])).name] = row
    return by_assay, by_file


def _first_value(row: pd.Series | None, aliases: tuple[str, ...]) -> Any:
    if row is None:
        return None
    for column in aliases:
        value = row.get(column)
        if pd.notna(value):
            return value
    return None


def _normalize_dms_frame(
    raw: pd.DataFrame,
    assay_id: str,
    metadata_row: pd.Series | None,
    accession_to_gene: dict[str, str],
) -> pd.DataFrame:
    mutation_column = _resolve_column(
        raw.columns.tolist(), ("mutant", "mutation", "variant"), required=False
    )
    if mutation_column is not None:
        mutation = (
            raw[mutation_column]
            .astype(str)
            .str.extract(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
        )
    else:
        reference_column = _resolve_column(
            raw.columns.tolist(), ("aaref", "aa_ref", "wildtype", "wt"), required=False
        )
        position_column = _resolve_column(
            raw.columns.tolist(),
            ("aapos", "aa_pos", "position", "residue_position"),
            required=False,
        )
        alternate_column = _resolve_column(
            raw.columns.tolist(),
            ("aaalt", "aa_alt", "mutant_aa", "alternate"),
            required=False,
        )
        if None in (reference_column, position_column, alternate_column):
            raise KeyError("Missing a combined mutation column or split amino-acid columns")
        standard = set(STANDARD_AA)
        reference = raw[reference_column].astype("string").str.strip().str.upper()
        alternate = raw[alternate_column].astype("string").str.strip().str.upper()
        mutation = pd.DataFrame(
            {
                0: reference.where(reference.isin(standard)),
                1: pd.to_numeric(raw[position_column], errors="coerce"),
                2: alternate.where(alternate.isin(standard)),
            },
            index=raw.index,
        )
    score_column = _resolve_column(raw.columns.tolist(), ("DMS_score", "dms_score", "score"))
    binary_column = _resolve_column(
        raw.columns.tolist(), ("DMS_score_bin", "dms_score_bin", "functional_bin"), required=False
    )
    score = pd.to_numeric(raw[score_column], errors="coerce")
    if binary_column is not None:
        functional = pd.to_numeric(raw[binary_column], errors="coerce")
        valid_binary = functional.isin([0, 1])
        label = 1 - functional
        label_method = "proteingym_assay_bin"
    else:
        cutoff = _first_value(
            metadata_row,
            ("binarization_cutoff", "DMS_binarization_cutoff", "cutoff"),
        )
        cutoff = pd.to_numeric(pd.Series([cutoff]), errors="coerce").iloc[0]
        if pd.isna(cutoff):
            logger.warning("Skipping %s without an assay cutoff", assay_id)
            return pd.DataFrame()
        valid_binary = score.notna()
        label = score.lt(float(cutoff)).astype(float)
        label_method = "assay_specific_cutoff"
    accession_column = _resolve_column(
        raw.columns.tolist(), ("UniProt_ID", "uniprot_id", "accession"), required=False
    )
    gene_column = _resolve_column(
        raw.columns.tolist(), ("genename", "gene_name", "Gene", "gene"), required=False
    )
    accession_default = _first_value(
        metadata_row,
        ("UniProt_ID", "uniprot_id", "target_uniprot", "accession"),
    )
    gene_default = _first_value(
        metadata_row,
        ("genename", "gene_name", "Gene", "target_gene"),
    )
    if accession_column is not None:
        accession = raw[accession_column].astype("string")
    else:
        accession = pd.Series(accession_default, index=raw.index, dtype="string")
    if gene_column is not None:
        gene = raw[gene_column].astype("string")
    else:
        gene = pd.Series(gene_default, index=raw.index, dtype="string")
    gene = gene.fillna(accession.map(accession_to_gene))
    gene = gene.fillna(accession.str.split("_").str[0].map(accession_to_gene))
    gene = gene.fillna(accession.str.split("_").str[0])
    target_sequence = _first_value(metadata_row, ("target_seq", "target_sequence", "sequence"))
    target_sequence = (
        "".join(str(target_sequence).split()).upper()
        if target_sequence is not None and pd.notna(target_sequence)
        else ""
    )
    frame = pd.DataFrame(
        {
            GENE_COL: gene,
            "aaref": mutation[0],
            "aapos": pd.to_numeric(mutation[1], errors="coerce").astype("Int64"),
            "aaalt": mutation[2],
            LABEL_COL: label,
            "DMS_SCORE": score,
            "ASSAY_ID": assay_id,
            "uniprot_id_hint": accession,
            "uniprot_id": accession.str.split("_").str[0],
            "protein_sequence": target_sequence,
            "PROTEIN_MAPPING_STATUS": "provided_target_sequence",
            "HAS_PROTEIN_MAPPING": int(bool(target_sequence)),
            "MAPPING_TRANSCRIPT_MATCH": 0,
            "UNIPROT_REVIEWED": 0,
            "DMS_LABEL_METHOD": label_method,
            "DMS_SCORE_DIRECTION": "higher_is_more_functional",
            "EXT_SOURCE": "dms",
            "PROTEINGYM_RELEASE": PROTEINGYM_RELEASE,
        }
    )
    valid = (
        mutation.notna().all(axis=1)
        & valid_binary
        & frame[GENE_COL].notna()
        & frame["protein_sequence"].astype(str).str.len().gt(0)
        & frame["aaref"].ne(frame["aaalt"])
    )
    frame = frame[valid].copy()
    frame[LABEL_COL] = frame[LABEL_COL].astype(np.int8)
    frame["variant_id"] = [
        f"PROTEIN:{accession if pd.notna(accession) else gene}:{ref}{position}{alt}"
        for accession, gene, ref, position, alt in zip(
            frame["uniprot_id_hint"],
            frame[GENE_COL],
            frame["aaref"],
            frame["aapos"],
            frame["aaalt"],
        )
    ]
    frame["transcript_variant_id"] = frame["variant_id"]
    frame[ROW_ID_COL] = [
        _stable_identifier(f"dms|{assay_id}|{variant}") for variant in frame["variant_id"]
    ]
    conflict = frame.groupby(["ASSAY_ID", "variant_id"])[LABEL_COL].transform("nunique").gt(1)
    frame = frame[~conflict].copy()
    return frame.sort_values(
        ["ASSAY_ID", "variant_id", "DMS_SCORE"], kind="mergesort"
    ).drop_duplicates(["ASSAY_ID", "variant_id"], keep="first")


def _sample_dms_assay(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep a deterministic label-independent uniform subset of one assay."""
    candidate_rows = len(frame)
    if candidate_rows == 0:
        return frame.copy()
    use_all = DMS_SAMPLING_POLICY == "all" or max_rows <= 0 or candidate_rows <= max_rows
    if use_all:
        sampled = frame.copy()
        policy = "all_rows"
    else:
        sampled = frame.copy()
        sampled["_sample_key"] = sampled[ROW_ID_COL].map(
            lambda value: _stable_identifier(f"uniform_sample|{value}")
        )
        sampled = sampled.sort_values("_sample_key", kind="mergesort").head(max_rows)
        sampled = sampled.drop(columns="_sample_key")
        policy = "hash_uniform_without_label"
    retained_rows = len(sampled)
    probability = retained_rows / candidate_rows
    sampled["DMS_CANDIDATE_ROWS"] = candidate_rows
    sampled["DMS_RETAINED_ROWS"] = retained_rows
    sampled["DMS_SAMPLING_PROBABILITY"] = np.float32(probability)
    sampled["DMS_SAMPLE_WEIGHT"] = np.float32(1.0 / probability)
    sampled["DMS_SAMPLING_POLICY"] = policy
    return sampled.sort_values(["ASSAY_ID", "variant_id"], kind="mergesort")


def iter_dms_assays(resources: Any) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield normalized ProteinGym assays."""
    metadata = (
        pd.read_csv(PROTEINGYM_METADATA, low_memory=False)
        if PROTEINGYM_METADATA.exists()
        else pd.DataFrame()
    )
    by_assay, by_file = _metadata_lookup(metadata)
    accession_to_gene = {
        accession: record.gene_name
        for accession, record in resources.records.items()
        if record.gene_name
    }
    yielded = False
    seen_assays: set[str] = set()
    official_directory_present = PROTEINGYM_DIR.exists()
    if official_directory_present:
        for path in sorted(PROTEINGYM_DIR.rglob("*.csv")):
            assay_id = path.stem
            if assay_id in seen_assays:
                logger.warning("Skipping duplicate assay %s", assay_id)
                continue
            metadata_row = by_file.get(path.name, by_assay.get(assay_id))
            raw = pd.read_csv(path, low_memory=False)
            normalized = _normalize_dms_frame(raw, assay_id, metadata_row, accession_to_gene)
            if not normalized.empty:
                yielded = True
                seen_assays.add(assay_id)
                yield assay_id, normalized
    if DMS_LEGACY_FILE.exists() and not official_directory_present:
        raw = pd.read_csv(DMS_LEGACY_FILE, low_memory=False)
        assay_column = _resolve_column(
            raw.columns.tolist(), ("ASSAY_ID", "assay_id", "DMS_id"), required=False
        )
        if assay_column is None:
            raw["ASSAY_ID"] = "legacy_assay"
            assay_column = "ASSAY_ID"
        for assay_id, assay in raw.groupby(assay_column, sort=True):
            assay_id = str(assay_id)
            if assay_id in seen_assays:
                logger.warning("Skipping duplicate assay %s", assay_id)
                continue
            normalized = _normalize_dms_frame(
                assay, assay_id, by_assay.get(assay_id), accession_to_gene
            )
            if not normalized.empty:
                yielded = True
                seen_assays.add(assay_id)
                yield assay_id, normalized
    elif DMS_LEGACY_FILE.exists():
        logger.info(
            "Ignoring legacy DMS file because the authenticated ProteinGym release is present"
        )
    if not yielded:
        logger.warning("No classifiable DMS assays were found")


def _align_external_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    excluded = {
        "esm_variant_score",
        "ESM_EXTRACTION_SUCCESS",
        *MUTATION_FEATURE_COLS,
    }
    for feature in BASE_FEATURE_ALLOWLIST:
        if feature in excluded or feature.endswith("__missing"):
            continue
        if feature not in result:
            result[feature] = np.nan
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
        result[feature] = result[feature].where(np.isfinite(result[feature]))
        if feature in AVAILABILITY_FEATURE_COLS:
            # Absent annotations cannot inherit the training cohort's median
            # availability. This is essential for sequence-only DMS fallback.
            result[feature] = result[feature].fillna(0).astype(np.float32)
        result[f"{feature}__missing"] = result[feature].isna().astype("int8")
    return result


def _coverage_fraction(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return 0.0
    values = frame[column]
    if column.startswith("HAS_") or column == "STRUCTURE_FILE_AVAILABLE":
        return float(pd.to_numeric(values, errors="coerce").fillna(0).gt(0).mean())
    return float(values.notna().mean())


def _class_feature_coverage_audit(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"rows": len(frame), "features": {}, "flagged": []}
    for column in FEATURE_COVERAGE_AUDIT_COLUMNS:
        if column not in frame:
            continue
        by_class = {
            str(label): _coverage_fraction(frame.loc[frame[LABEL_COL].eq(label)], column)
            for label in (0, 1)
        }
        gap = abs(by_class["1"] - by_class["0"])
        report["features"][column] = {"by_class": by_class, "absolute_gap": gap}
        if gap >= FEATURE_COVERAGE_WARN_GAP:
            report["flagged"].append(column)
    if report["flagged"] and FEATURE_COVERAGE_POLICY != "off":
        message = (
            "External feature-coverage gaps exceed "
            f"{FEATURE_COVERAGE_WARN_GAP:.3f}: {report['flagged']}"
        )
        if FEATURE_COVERAGE_POLICY == "error":
            raise RuntimeError(message)
        logger.warning(message)
    report["policy"] = FEATURE_COVERAGE_POLICY
    report["gap_threshold"] = FEATURE_COVERAGE_WARN_GAP
    return report


def _append_csv(frame: pd.DataFrame, path: Path, first: bool) -> bool:
    if frame.empty:
        return first
    frame.to_csv(
        path,
        mode="w" if first else "a",
        header=first,
        index=False,
    )
    return False


def _mutation_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[GENE_COL].astype(str)
        + ":"
        + frame["aa_pos"].astype(int).astype(str)
        + ":"
        + frame["aa_ref"].astype(str)
        + ":"
        + frame["aa_alt"].astype(str)
    )


def main() -> None:
    """Prepare separate ClinVar and DMS datasets."""
    ensure_directories(STAGE09_OUT)
    temporal_source_audit = _validate_external_temporal_configuration()
    temporal_source_audit["clinvar_train_snapshot"] = validate_clinvar_snapshot_provenance(
        CLINVAR_TRAIN_ARCHIVE,
        CLINVAR_TRAIN_TRANSFORMATION_MANIFEST,
        CLINVAR_TRAIN_RELEASE,
    )
    temporal_source_audit["clinvar_external_snapshot"] = validate_clinvar_snapshot_provenance(
        CLINVAR_EXTERNAL_ARCHIVE,
        CLINVAR_EXTERNAL_TRANSFORMATION_MANIFEST,
        CLINVAR_EXTERNAL_RELEASE,
    )
    proteingym_provenance: dict[str, Any] | None = None
    if REQUIRE_EXTERNAL_DMS or any(
        path.exists()
        for path in (
            PROTEINGYM_DIR,
            PROTEINGYM_METADATA,
            PROTEINGYM_EXTRACTION_MANIFEST,
            PROTEINGYM_METADATA_PROVENANCE,
        )
    ):
        proteingym_provenance = validate_proteingym_provenance(
            PROTEINGYM_DIR,
            PROTEINGYM_METADATA,
            PROTEINGYM_EXTRACTION_MANIFEST,
            PROTEINGYM_METADATA_PROVENANCE,
            PROTEINGYM_RELEASE,
        )
    for path in (
        CLINVAR_OUTPUT,
        DMS_OUTPUT,
        DMS_SEQUENCE_OUTPUT,
        LOSS_REPORT,
        DISAGREEMENT_FILE,
        CLINVAR_TRANSCRIPT_AUDIT,
        CLINVAR_SCV_EVIDENCE_AUDIT,
    ):
        path.unlink(missing_ok=True)
    resources = stage04.build_annotation_resources(UNIPROT_FILE, ALPHAFOLD_DIR, force_reparse=False)
    loss_temporary = LOSS_REPORT.with_suffix(LOSS_REPORT.suffix + ".tmp")
    disagreement_temporary = DISAGREEMENT_FILE.with_suffix(DISAGREEMENT_FILE.suffix + ".tmp")
    clinvar_temporary = CLINVAR_OUTPUT.with_suffix(CLINVAR_OUTPUT.suffix + ".tmp")
    dms_temporary = DMS_OUTPUT.with_suffix(DMS_OUTPUT.suffix + ".tmp")
    dms_sequence_temporary = DMS_SEQUENCE_OUTPUT.with_suffix(DMS_SEQUENCE_OUTPUT.suffix + ".tmp")
    selection_temporary = CLINVAR_TRANSCRIPT_AUDIT.with_suffix(
        CLINVAR_TRANSCRIPT_AUDIT.suffix + ".tmp"
    )
    scv_audit_temporary = CLINVAR_SCV_EVIDENCE_AUDIT.with_suffix(
        CLINVAR_SCV_EVIDENCE_AUDIT.suffix + ".tmp"
    )
    for path in (
        loss_temporary,
        disagreement_temporary,
        clinvar_temporary,
        dms_temporary,
        dms_sequence_temporary,
        selection_temporary,
        scv_audit_temporary,
    ):
        path.unlink(missing_ok=True)
    source_rows = Counter()
    source_classes: dict[str, Counter] = defaultdict(Counter)
    assay_rows = Counter()
    assay_candidate_rows = Counter()
    assay_candidate_classes: dict[str, Counter] = defaultdict(Counter)
    assay_retained_classes: dict[str, Counter] = defaultdict(Counter)
    assay_sampling: dict[str, dict[str, Any]] = {}
    loss_counts = Counter()
    clinvar_labels: dict[str, int] = {}
    first_loss = True
    loss_sample_rows = 0
    first_dms = True
    first_disagreement = True
    source_feature_rows = Counter()
    source_feature_coverage_counts: dict[str, Counter] = defaultdict(Counter)
    clinvar_coverage_audit: dict[str, Any] = {}
    clinvar_selection_summary: dict[str, Any] = {}
    clinvar_temporal_cohort_audit: dict[str, Any] = {}
    clinvar_dbnsfp_coordinate_audit: dict[str, Any] = {}
    dms_sequences: dict[str, str] = {}
    try:
        clinvar_ready, clinvar_audit, selection_audit = prepare_clinvar(
            resources, return_selection_audit=True
        )
        clinvar_temporal_cohort_audit = dict(clinvar_ready.attrs.get("temporal_clinvar_audit", {}))
        clinvar_dbnsfp_coordinate_audit = dict(
            clinvar_ready.attrs.get("dbnsfp_coordinate_projection", {})
        )
        raw_scv_audit = clinvar_ready.attrs.get("clinvar_scv_evidence_audit", pd.DataFrame())
        if not isinstance(raw_scv_audit, pd.DataFrame):
            raise RuntimeError("ClinVar SCV audit metadata is not tabular")
        raw_scv_audit.reindex(columns=CLINVAR_SCV_AUDIT_COLUMNS).to_csv(
            scv_audit_temporary, index=False
        )
        scv_provenance = clinvar_temporal_cohort_audit.get("scv_evidence", {}).get("provenance")
        if scv_provenance is not None:
            temporal_source_audit["clinvar_submission_archives"] = scv_provenance
        selection_audit.to_csv(selection_temporary, index=False)
        clinvar_selection_summary = {
            "candidate_consequence_rows": len(selection_audit),
            "candidate_unique_variants": int(
                selection_audit["variant_id"].nunique() if "variant_id" in selection_audit else 0
            ),
            "selected_consequence_rows": int(
                selection_audit.get(
                    "PRIMARY_CONSEQUENCE_SELECTED",
                    pd.Series(False, index=selection_audit.index),
                ).sum()
            ),
            "ready_unique_variants": int(
                clinvar_ready["variant_id"].nunique() if not clinvar_ready.empty else 0
            ),
            "primary_output_variant_unique": bool(
                clinvar_ready.empty or clinvar_ready["variant_id"].is_unique
            ),
            "primary_output_temporal_identity_unique": bool(
                clinvar_ready.empty or clinvar_ready["CLINVAR_TEMPORAL_KEY"].is_unique
            ),
            "selection_policy": TRANSCRIPT_SELECTION_POLICY,
            "selection_occurs_after_protein_mapping": True,
            "audit_file": str(CLINVAR_TRANSCRIPT_AUDIT),
            "candidate_source_counts": (
                selection_audit["CLINVAR_CONSEQUENCE_SOURCE"].value_counts(dropna=False).to_dict()
                if "CLINVAR_CONSEQUENCE_SOURCE" in selection_audit
                else {}
            ),
            "selected_source_counts": (
                selection_audit.loc[
                    selection_audit.get(
                        "PRIMARY_CONSEQUENCE_SELECTED",
                        pd.Series(False, index=selection_audit.index),
                    ).astype(bool),
                    "CLINVAR_CONSEQUENCE_SOURCE",
                ]
                .value_counts(dropna=False)
                .to_dict()
                if "CLINVAR_CONSEQUENCE_SOURCE" in selection_audit
                else {}
            ),
        }
        if not clinvar_audit.empty:
            loss_counts.update(clinvar_audit["outcome"].value_counts().to_dict())
            sample = clinvar_audit[clinvar_audit["outcome"].ne("kept")].head(
                AUDIT_SAMPLE_MAX_ROWS - loss_sample_rows
            )
            first_loss = _append_csv(sample, loss_temporary, first_loss)
            loss_sample_rows += len(sample)
        if not clinvar_ready.empty:
            clinvar_coverage_audit = _class_feature_coverage_audit(clinvar_ready)
            clinvar_ready.to_csv(clinvar_temporary, index=False)
            source_rows["clinvar"] = len(clinvar_ready)
            source_classes["clinvar"].update(clinvar_ready[LABEL_COL].value_counts().to_dict())
            clinvar_labels = dict(
                zip(_mutation_key(clinvar_ready), clinvar_ready[LABEL_COL].astype(int))
            )
            source_feature_rows["clinvar"] += len(clinvar_ready)
            for feature in EXTERNAL_TABULAR_FEATURES:
                source_feature_coverage_counts["clinvar"][feature] += int(
                    round(_coverage_fraction(clinvar_ready, feature) * len(clinvar_ready))
                )
        elif REQUIRE_EXTERNAL_CLINVAR:
            raise RuntimeError(
                "No external ClinVar variants survived the temporal, consequence, "
                "mapping and feature contracts"
            )
        for assay_id, assay in iter_dms_assays(resources):
            ready, audit = stage08._prepare_chunk(
                assay,
                require_transcript_mapping=False,
            )
            ready = _align_external_schema(ready)
            loss_counts.update(audit["outcome"].value_counts().to_dict())
            if loss_sample_rows < AUDIT_SAMPLE_MAX_ROWS:
                sample = audit[audit["outcome"].ne("kept")].head(
                    AUDIT_SAMPLE_MAX_ROWS - loss_sample_rows
                )
                first_loss = _append_csv(sample, loss_temporary, first_loss)
                loss_sample_rows += len(sample)
            if ready.empty:
                continue
            ready = ready.sort_values(
                ["ASSAY_ID", "variant_id", ROW_ID_COL], kind="mergesort"
            ).drop_duplicates(["ASSAY_ID", "variant_id"], keep="first")
            assay_candidate_rows[assay_id] += len(ready)
            assay_candidate_classes[assay_id].update(ready[LABEL_COL].value_counts().to_dict())
            ready = _sample_dms_assay(ready, DMS_MAX_ROWS_PER_ASSAY)
            ready["EXTERNAL_FEATURE_PROFILE"] = "sequence_and_mutation_only"
            for sequence_hash, protein_sequence in (
                ready[["sequence_hash", "protein_sequence"]]
                .drop_duplicates()
                .itertuples(index=False)
            ):
                sequence_hash = str(sequence_hash)
                protein_sequence = str(protein_sequence)
                previous_sequence = dms_sequences.get(sequence_hash)
                if previous_sequence is not None and previous_sequence != protein_sequence:
                    raise RuntimeError(
                        "DMS sequence_hash collision detected while normalizing sequences"
                    )
                dms_sequences[sequence_hash] = protein_sequence
            first_dms = _append_csv(
                ready.drop(columns=["protein_sequence", "mutation_window"], errors="ignore"),
                dms_temporary,
                first_dms,
            )
            source_rows["dms"] += len(ready)
            source_classes["dms"].update(ready[LABEL_COL].value_counts().to_dict())
            assay_rows[assay_id] += len(ready)
            assay_retained_classes[assay_id].update(ready[LABEL_COL].value_counts().to_dict())
            assay_sampling[assay_id] = {
                "candidate_rows": int(ready["DMS_CANDIDATE_ROWS"].iloc[0]),
                "retained_rows": int(ready["DMS_RETAINED_ROWS"].iloc[0]),
                "sampling_probability": float(ready["DMS_SAMPLING_PROBABILITY"].iloc[0]),
                "sample_weight": float(ready["DMS_SAMPLE_WEIGHT"].iloc[0]),
                "policy": str(ready["DMS_SAMPLING_POLICY"].iloc[0]),
            }
            source_feature_rows["dms"] += len(ready)
            for feature in EXTERNAL_TABULAR_FEATURES:
                source_feature_coverage_counts["dms"][feature] += int(
                    round(_coverage_fraction(ready, feature) * len(ready))
                )
            if clinvar_labels:
                keys = _mutation_key(ready)
                comparison = pd.DataFrame(
                    {
                        ROW_ID_COL: ready[ROW_ID_COL],
                        "mutation_key": keys,
                        "clinvar_label": keys.map(clinvar_labels),
                        "dms_label": ready[LABEL_COL].to_numpy(dtype=int),
                        "ASSAY_ID": ready["ASSAY_ID"],
                    }
                )
                disagreement = comparison[
                    comparison["clinvar_label"].notna()
                    & comparison["clinvar_label"].ne(comparison["dms_label"])
                ]
                first_disagreement = _append_csv(
                    disagreement, disagreement_temporary, first_disagreement
                )
        if REQUIRE_EXTERNAL_DMS and source_rows["dms"] == 0:
            raise RuntimeError("No ProteinGym DMS rows survived assay normalization and mapping")
        dms_rows = source_feature_rows["dms"]
        dms_critical_coverage = {
            feature: (
                source_feature_coverage_counts["dms"][feature] / dms_rows if dms_rows else 0.0
            )
            for feature in CRITICAL_DMS_CONTEXT_FEATURES
        }
        unavailable_dms_features = [
            feature for feature, coverage in dms_critical_coverage.items() if coverage == 0
        ]
        if dms_rows and unavailable_dms_features:
            message = (
                "DMS is sequence/mutation-only; context features are unavailable: "
                f"{unavailable_dms_features}. Full-fusion calibration metrics are not "
                "methodologically comparable to the internal cohort."
            )
            if not ALLOW_SEQUENCE_ONLY_DMS:
                raise RuntimeError(message)
            logger.warning(message)
        if not clinvar_ready.empty:
            clinvar_temporary.replace(CLINVAR_OUTPUT)
        selection_temporary.replace(CLINVAR_TRANSCRIPT_AUDIT)
        scv_audit_temporary.replace(CLINVAR_SCV_EVIDENCE_AUDIT)
        if not first_dms:
            dms_temporary.replace(DMS_OUTPUT)
            sequence_frame = pd.DataFrame(
                sorted(dms_sequences.items()),
                columns=["sequence_hash", "protein_sequence"],
            )
            if sequence_frame.empty or sequence_frame["sequence_hash"].duplicated().any():
                raise RuntimeError("DMS sequence normalization produced an invalid table")
            sequence_frame.to_parquet(
                dms_sequence_temporary,
                index=False,
                compression="zstd",
            )
            dms_sequence_temporary.replace(DMS_SEQUENCE_OUTPUT)
        if not first_loss:
            loss_temporary.replace(LOSS_REPORT)
        else:
            pd.DataFrame(
                columns=[
                    ROW_ID_COL,
                    GENE_COL,
                    LABEL_COL,
                    "source",
                    "sequence_length",
                    "length_band",
                    "mapping_status",
                    "outcome",
                ]
            ).to_csv(loss_temporary, index=False)
            loss_temporary.replace(LOSS_REPORT)
        if not first_disagreement:
            disagreement_temporary.replace(DISAGREEMENT_FILE)
        else:
            pd.DataFrame(
                columns=[
                    ROW_ID_COL,
                    "mutation_key",
                    "clinvar_label",
                    "dms_label",
                    "ASSAY_ID",
                ]
            ).to_csv(disagreement_temporary, index=False)
            disagreement_temporary.replace(DISAGREEMENT_FILE)
        if not source_rows:
            raise RuntimeError("No external rows survived preparation")
    except BaseException:
        for path in (
            loss_temporary,
            disagreement_temporary,
            clinvar_temporary,
            dms_temporary,
            dms_sequence_temporary,
            selection_temporary,
            scv_audit_temporary,
        ):
            path.unlink(missing_ok=True)
        raise
    summary = {
        "source_rows": dict(source_rows),
        "source_classes": {source: dict(counts) for source, counts in source_classes.items()},
        "assay_rows": dict(assay_rows),
        "assay_candidate_rows": dict(assay_candidate_rows),
        "assay_candidate_classes": {
            assay: dict(counts) for assay, counts in assay_candidate_classes.items()
        },
        "assay_retained_classes": {
            assay: dict(counts) for assay, counts in assay_retained_classes.items()
        },
        "assay_sampling": assay_sampling,
        "mapping_outcomes": dict(loss_counts),
        "mapping_loss_sample_rows": loss_sample_rows,
        "clinvar_release": CLINVAR_EXTERNAL_RELEASE,
        "training_cutoff_date": TRAIN_CUTOFF_DATE,
        "temporal_source_audit": temporal_source_audit,
        "clinvar_design": (
            "stable_variation_id_primary_current_high_confidence_aggregate_"
            "post_cutoff_new_or_versioned_nonconflicting_scv_evidence"
            if CLINVAR_REQUIRE_SCV_EVIDENCE
            else "stable_variation_id_primary_snapshot_new_or_reclassified"
        ),
        "clinvar_identity_policy": CLINVAR_IDENTITY_POLICY,
        "clinvar_scv_policy": CLINVAR_SCV_POLICY,
        "clinvar_temporal_cohort_audit": clinvar_temporal_cohort_audit,
        "clinvar_dbnsfp_coordinate_projection": clinvar_dbnsfp_coordinate_audit,
        "clinvar_scv_evidence_audit": {
            "file": str(CLINVAR_SCV_EVIDENCE_AUDIT),
            "columns": list(CLINVAR_SCV_AUDIT_COLUMNS),
            "selection_uses_model_outputs_or_predictor_availability": False,
        },
        "clinvar_transcript_selection": clinvar_selection_summary,
        "clinvar_feature_coverage_audit": clinvar_coverage_audit,
        "dms_release": PROTEINGYM_RELEASE,
        "proteingym_provenance": proteingym_provenance,
        "dms_direction": "higher_is_better_for_measured_assay_phenotype",
        "dms_binary_interpretation": "below_assay_cutoff_not_clinical_pathogenicity",
        "dms_global_median_used": False,
        "dms_max_rows_per_assay": DMS_MAX_ROWS_PER_ASSAY,
        "dms_sampling_policy": DMS_SAMPLING_POLICY,
        "dms_sampling_uses_label": False,
        "dms_sequence_only_allowed": ALLOW_SEQUENCE_ONLY_DMS,
        "dms_sequences": "ProteinGym target_seq",
        "dms_sequence_storage": {
            "policy": "content_addressed_unique_sequence_table",
            "unique_sequences": len(dms_sequences),
            "table": str(DMS_SEQUENCE_OUTPUT),
            "prepared_rows_repeat_sequences": False,
        },
        "feature_coverage_by_source": {
            source: {
                feature: counts[feature] / source_feature_rows[source]
                for feature in EXTERNAL_TABULAR_FEATURES
            }
            for source, counts in source_feature_coverage_counts.items()
            if source_feature_rows[source]
        },
        "sources_pooled_for_evaluation": False,
    }
    summary_temporary = SUMMARY_FILE.with_suffix(SUMMARY_FILE.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2, default=json_default), encoding="utf-8"
    )
    summary_temporary.replace(SUMMARY_FILE)
    write_run_manifest(
        MANIFEST_FILE,
        "09_prepare_external_esm_dataset",
        [
            CLINVAR_TRAIN_ARCHIVE,
            CLINVAR_TRAIN_TRANSFORMATION_MANIFEST,
            CLINVAR_EXTERNAL_ARCHIVE,
            CLINVAR_EXTERNAL_TRANSFORMATION_MANIFEST,
            *(
                [
                    CLINVAR_TRAIN_SUBMISSION_ARCHIVE,
                    CLINVAR_TRAIN_SUBMISSION_ARCHIVE.with_name(
                        CLINVAR_TRAIN_SUBMISSION_ARCHIVE.name + ".provenance.json"
                    ),
                    CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE,
                    CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE.with_name(
                        CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE.name + ".provenance.json"
                    ),
                    CLINVAR_SCV_AUDIT_TOOL,
                ]
                if CLINVAR_REQUIRE_SCV_EVIDENCE
                else []
            ),
            DBNSFP_FILE,
            PROTEINGYM_METADATA,
            PROTEINGYM_METADATA_PROVENANCE,
            PROTEINGYM_DIR,
            PROTEINGYM_EXTRACTION_MANIFEST,
            *(
                [DMS_LEGACY_FILE]
                if DMS_LEGACY_FILE.exists() and not PROTEINGYM_DIR.exists()
                else []
            ),
            UNIPROT_FILE,
            ALPHAFOLD_DIR,
        ],
        summary,
        outputs=[
            path
            for path in (
                CLINVAR_OUTPUT,
                DMS_OUTPUT,
                DMS_SEQUENCE_OUTPUT,
                LOSS_REPORT,
                DISAGREEMENT_FILE,
                CLINVAR_TRANSCRIPT_AUDIT,
                CLINVAR_SCV_EVIDENCE_AUDIT,
                SUMMARY_FILE,
            )
            if path.exists()
        ],
    )
    logger.info("Prepared external rows: %s", dict(source_rows))


if __name__ == "__main__":
    main()
