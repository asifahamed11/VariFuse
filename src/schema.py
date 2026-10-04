from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.schema")

LABEL_COL = "LABEL_PATHOGENIC"
GENE_COL = "genename"
ROW_ID_COL = "row_id"
STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")

RAW_CONTEXTUAL_PREDICTOR_COLS = (
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "CADD_phred",
    "REVEL_score",
    "AlphaMissense_score",
    "PrimateAI_score",
    "MetaRNN_score",
    "BayesDel_noAF_score",
    "VEST4_score",
    "MutPred2_score",
    "MPC_score",
    "ClinPred_score",
    "DEOGEN2_score",
    "LIST-S2_score",
    "VARITY_R_score",
    "VARITY_ER_score",
    "VARITY_R_LOO_score",
    "VARITY_ER_LOO_score",
)
PREDICTOR_COLS = (
    *RAW_CONTEXTUAL_PREDICTOR_COLS,
    "CONSENSUS_SCORE",
    "MetaLR_pred",
)
REQUIRED_MAPPING_COLS = (
    "uniprot_id",
    "PROTEIN_MAPPING_STATUS",
    "MAPPING_TRANSCRIPT_MATCH",
    "PRIMARY_MAPPING_ELIGIBLE",
    "MAPPING_CONFIDENCE",
)
TRANSCRIPT_SELECTION_COLS = (
    "PRIMARY_CONSEQUENCE_RANK",
    "PRIMARY_CONSEQUENCE_SELECTED",
    "TRANSCRIPT_CANDIDATE_COUNT",
    "TRANSCRIPT_MAPPED_CANDIDATE_COUNT",
    "SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT",
    "CLINVAR_SOURCE_GENE_MATCH",
    "CONSEQUENCE_SELECTION_POLICY",
)
LABEL_PROXY_COLS = (
    "COSMIC_FREQUENCY",
    "COSMIC_RECURRENCE",
    "EVIDENCE_SOURCE",
    "EVIDENCE_SOURCES",
    "RESCUE_REASON",
    "WAS_RESCUED",
    "IS_CLINVAR_PATHOGENIC",
    "IS_CLINVAR_BENIGN",
    "IS_KNOWN_HOTSPOT",
    "IS_CANCER_GENE",
    "IS_ONCOGENE",
    "IS_TSG",
    "IS_TIER1",
    "IS_ONCOKB",
    "TIER",
    "ROLE_IN_CANCER",
    "HAS_CIVIC_EVIDENCE",
    "HAS_QUALIFIED_CIVIC",
    "HAS_HOTSPOT_SUPPORT",
    "HAS_BA1_EVIDENCE",
)
ID_COLS = (
    "chr",
    "pos",
    "ref",
    "alt",
    "variant_id",
    "transcript_variant_id",
    "CLINVAR_VARIATION_ID",
    "CLINVAR_SOURCE_GENE",
    "CLINVAR_SOURCE_NAME",
    "CLINVAR_REVIEW_STATUS",
    ROW_ID_COL,
    "aa_pos",
    "aa_ref",
    "aa_alt",
    "aapos",
    "aaref",
    "aaalt",
    "protein_sequence",
    "mutation_window",
    "window_aa_pos",
    "sequence_hash",
    "uniprot_id",
    "Ensembl_transcriptid",
    "HGVSp_snpEff",
    "HGVSc_snpEff",
    "EXT_SOURCE",
    "ASSAY_ID",
)
MUTATION_FEATURE_COLS = tuple(
    [f"aa_ref_is_{aa}" for aa in STANDARD_AA]
    + [f"aa_alt_is_{aa}" for aa in STANDARD_AA]
    + [
        "aa_hydrophobicity_delta",
        "aa_volume_delta",
        "aa_charge_delta",
    ]
)
BIOLOGICAL_FEATURE_ALLOWLIST = (
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SASA",
    "RELATIVE_SASA",
    "PLDDT_SCORE",
    # Rotation/translation-invariant local AlphaFold environment descriptors.
    # These are computed from C-alpha neighbourhoods around the substituted
    # residue and keep the primary model lightweight enough for local training.
    "LOCAL_CONTACT_COUNT_8A",
    "LOCAL_CONTACT_COUNT_12A",
    "LOCAL_LONG_RANGE_CONTACT_COUNT_8A",
    "LOCAL_MEAN_PLDDT_8A",
    "LOCAL_MIN_PLDDT_8A",
    "LOCAL_CONFIDENT_CONTACT_FRACTION_8A",
    "LOCAL_MEAN_DISTANCE_8A",
    "LOCAL_HYDROPHOBIC_FRACTION_8A",
    "LOCAL_CHARGED_FRACTION_8A",
    "IS_IN_DOMAIN",
    "DISTANCE_TO_ACTIVE_SITE",
    "IS_ACTIVE_SITE",
    "IS_BINDING_SITE",
    "IS_TRANSMEMBRANE",
    "esm_variant_score",
    *MUTATION_FEATURE_COLS,
)
AVAILABILITY_FEATURE_COLS = (
    "HAS_PROTEIN_MAPPING",
    "STRUCTURE_FILE_AVAILABLE",
    "HAS_STRUCTURE",
    "LOW_CONFIDENCE_STRUCTURE",
    "HAS_DOMAIN_ANNOTATION",
    "HAS_ACTIVE_SITE_ANNOTATION",
    "HAS_BINDING_SITE_ANNOTATION",
    "HAS_TRANSMEMBRANE_ANNOTATION",
    "ESM_EXTRACTION_SUCCESS",
)
# Storage/cleaning schema remains broad so audit features are preserved.
BASE_FEATURE_ALLOWLIST = tuple(
    dict.fromkeys((*BIOLOGICAL_FEATURE_ALLOWLIST, *AVAILABILITY_FEATURE_COLS))
)
MISSING_INDICATOR_COLS = tuple(
    f"{name}__missing"
    for name in (*BIOLOGICAL_FEATURE_ALLOWLIST, *AVAILABILITY_FEATURE_COLS)
)
# The primary publication model deliberately excludes availability/missingness
# shortcuts. They remain in the data for a mandatory diagnostic baseline.
MODEL_FEATURE_ALLOWLIST = tuple(dict.fromkeys(BIOLOGICAL_FEATURE_ALLOWLIST))


def select_model_features(df: pd.DataFrame) -> list[str]:
    """Select only approved numeric model features."""
    forbidden = set(PREDICTOR_COLS) | set(LABEL_PROXY_COLS) | set(ID_COLS)
    selected = [
        column
        for column in MODEL_FEATURE_ALLOWLIST
        if column in df.columns
        and column not in forbidden
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not selected:
        raise ValueError("No approved model features are available")
    unexpected = set(selected) & set(PREDICTOR_COLS)
    if unexpected:
        raise RuntimeError(f"Predictor leakage detected: {unexpected}")
    logger.info("Predictor-free feature set: %d features", len(selected))
    return selected


def select_tabular_features(feature_names: list[str]) -> list[str]:
    """Return the schema shared by tabular-only training and inference."""
    return [
        column
        for column in feature_names
        if not column.startswith("esm_variant_score")
        and column != "ESM_EXTRACTION_SUCCESS"
    ]


def select_availability_features(df: pd.DataFrame) -> list[str]:
    """Select annotation-coverage features for the audit-only shortcut baseline."""
    candidates = (*AVAILABILITY_FEATURE_COLS, *MISSING_INDICATOR_COLS)
    return [
        column
        for column in candidates
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]


def add_mutation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode reference and alternate amino acids."""
    result = df.copy()
    reference = result["aa_ref"].astype(str).str.upper()
    alternate = result["aa_alt"].astype(str).str.upper()
    for amino_acid in STANDARD_AA:
        result[f"aa_ref_is_{amino_acid}"] = (reference == amino_acid).astype(np.int8)
        result[f"aa_alt_is_{amino_acid}"] = (alternate == amino_acid).astype(np.int8)
    hydrophobicity = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }
    volume = {
        "A": 88.6,
        "C": 108.5,
        "D": 111.1,
        "E": 138.4,
        "F": 189.9,
        "G": 60.1,
        "H": 153.2,
        "I": 166.7,
        "K": 168.6,
        "L": 166.7,
        "M": 162.9,
        "N": 114.1,
        "P": 112.7,
        "Q": 143.8,
        "R": 173.4,
        "S": 89.0,
        "T": 116.1,
        "V": 140.0,
        "W": 227.8,
        "Y": 193.6,
    }
    charge = {amino_acid: 0.0 for amino_acid in STANDARD_AA}
    charge.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})
    result["aa_hydrophobicity_delta"] = alternate.map(hydrophobicity) - reference.map(
        hydrophobicity
    )
    result["aa_volume_delta"] = alternate.map(volume) - reference.map(volume)
    result["aa_charge_delta"] = alternate.map(charge) - reference.map(charge)
    return result
