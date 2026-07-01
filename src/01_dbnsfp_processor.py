"""Stage 01 - parse and filter dbNSFP records into the base somatic-variant table."""

import pandas as pd
from config import DATA_DIR, STAGE01_OUT
import numpy as np
import json
import warnings
import gc
import os
from collections import Counter

warnings.filterwarnings("ignore")

CHUNK_SIZE = 5000
OUTPUT_FILE = STAGE01_OUT / "somatic_variant_dbNSFP.csv"
METADATA_FILE = STAGE01_OUT / "somatic_variant_dbNSFP.json"

GNOMAD_AF_THRESHOLD = 0.0001
MIN_COSMIC_RESCUE = 50
MIN_COSMIC_EVIDENCE = 10
MIN_CADD_SCORE = 10

RESCUE_HOTSPOTS = {
    "TP53": [
        175,
        245,
        248,
        249,
        273,
        282,
        220,
        179,
        193,
        213,
        234,
        236,
        238,
        241,
        244,
        277,
        278,
        280,
        281,
    ],
    "KRAS": [12, 13, 61, 117, 146],
    "NRAS": [12, 13, 61, 117, 146],
    "HRAS": [12, 13, 61],
    "BRAF": [600, 469, 594, 596, 597, 601],
    "PIK3CA": [542, 545, 1047, 88, 93, 111, 345, 420, 453, 1043],
    "EGFR": [719, 790, 858, 861, 709, 768, 769, 773],
    "IDH1": [132],
    "IDH2": [140, 172],
    "FGFR3": [249, 373, 375],
    "CTNNB1": [32, 33, 34, 37, 41, 45],
    "AKT1": [17],
    "ERBB2": [755, 777],
    "KIT": [816, 822],
    "MET": [1010, 1268],
    "ALK": [1196, 1269],
    "ERBB3": [104, 107],
    "GNAS": [201, 227],
    "MAP2K1": [56, 124],
    "PDGFRA": [842, 845],
}

DBNSFP_COLS = [
    "#chr",
    "pos(1-based)",
    "ref",
    "alt",
    "aaref",
    "aaalt",
    "aapos",
    "genename",
    "Ensembl_transcriptid",
    "HGVSp_snpEff",
    "HGVSc_snpEff",
    "SIFT_score",
    "Polyphen2_HDIV_score",
    "REVEL_score",
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "Interpro_domain",
    "CADD_phred",
    "MetaLR_pred",
    "clinvar_clnsig",
    "gnomAD4.1_joint_AF",
    "gnomAD2.1.1_exomes_non_cancer_AC",
    "gnomAD2.1.1_exomes_non_cancer_AN",
]


def explode_semicolon_columns(variant_df, cols_to_explode):
    """Handle semicolon-delimited multi-value fields in dbNSFP"""
    if variant_df.empty:
        return variant_df

    for col in cols_to_explode:
        if col in variant_df.columns:
            variant_df[col] = variant_df[col].astype(str).str.split(";")

    try:
        variant_df = variant_df.explode(cols_to_explode)
    except ValueError:
        for col in cols_to_explode:
            variant_df = variant_df.explode(col)

    for col in cols_to_explode:
        if col in variant_df.columns:
            variant_df[col] = variant_df[col].replace([".", "nan", "", None], np.nan)

    return variant_df


print("Loading CIViC rescue variants")
civic_tsv = pd.read_csv(DATA_DIR / "01-Jan-2026-VariantSummaries.tsv", sep="\t", low_memory=False)
rescue_list_civic = civic_tsv[
    ["gene", "chromosome", "start", "reference_bases", "variant_bases"]
].copy()
rescue_list_civic = rescue_list_civic.dropna(subset=["chromosome", "start"])
rescue_list_civic["rescue_key"] = (
    rescue_list_civic["chromosome"].astype(str)
    + ":"
    + rescue_list_civic["start"].astype(int).astype(str)
    + ":"
    + rescue_list_civic["reference_bases"].fillna("")
    + ":"
    + rescue_list_civic["variant_bases"].fillna("")
)
rescue_keys_civic = set(rescue_list_civic["rescue_key"].unique())
print(f"Loaded {len(rescue_keys_civic):,} CIViC rescue variants")
del civic_tsv, rescue_list_civic
gc.collect()

print("Loading Cancer Gene Census and COSMIC data")
cgc = pd.read_csv(DATA_DIR / "Cosmic_CancerGeneCensus_v102_GRCh37.tsv", sep="\t")
cancer_genes = set(cgc["GENE_SYMBOL"].unique())
oncogenes = set(
    cgc[cgc["ROLE_IN_CANCER"].str.contains("oncogene", na=False, case=False)]["GENE_SYMBOL"]
)
tsgs = set(cgc[cgc["ROLE_IN_CANCER"].str.contains("TSG", na=False, case=False)]["GENE_SYMBOL"])
tier1_genes = set(cgc[cgc["TIER"] == 1]["GENE_SYMBOL"])

cgc_slim = cgc[["GENE_SYMBOL", "ROLE_IN_CANCER", "TIER"]].copy()
cgc_slim.columns = ["genename", "ROLE_IN_CANCER", "TIER"]
del cgc
gc.collect()

print("Aggregating COSMIC Mutant Census")
cmc_chunks = []
for cosmic_chunk in pd.read_csv(
    DATA_DIR / "cmc_export.tsv",
    sep="\t",
    usecols=[
        "GENE_NAME",
        "AA_MUT_START",
        "COSMIC_SAMPLE_MUTATED",
        "COSMIC_SAMPLE_TESTED",
    ],
    chunksize=50000,
    low_memory=False,
):
    chunk_agg = (
        cosmic_chunk.groupby(["GENE_NAME", "AA_MUT_START"])
        .agg({"COSMIC_SAMPLE_MUTATED": "sum", "COSMIC_SAMPLE_TESTED": "max"})
        .reset_index()
    )
    cmc_chunks.append(chunk_agg)

cmc_agg = (
    pd.concat(cmc_chunks, ignore_index=True)
    .groupby(["GENE_NAME", "AA_MUT_START"])
    .agg({"COSMIC_SAMPLE_MUTATED": "sum", "COSMIC_SAMPLE_TESTED": "max"})
    .reset_index()
)
cmc_agg.columns = ["genename", "aapos", "COSMIC_RECURRENCE", "COSMIC_TESTED"]
cmc_agg["COSMIC_FREQUENCY"] = cmc_agg["COSMIC_RECURRENCE"] / cmc_agg["COSMIC_TESTED"].replace(0, 1)

cosmic_rescue = cmc_agg[cmc_agg["COSMIC_RECURRENCE"] >= MIN_COSMIC_RESCUE].copy()
cosmic_rescue["rescue_gene_pos"] = (
    cosmic_rescue["genename"] + "_" + cosmic_rescue["aapos"].astype(str)
)
cosmic_rescue_keys = set(cosmic_rescue["rescue_gene_pos"].unique())
print(f"Aggregated COSMIC data - {len(cosmic_rescue):,} rescue variants identified")
del cmc_chunks, cosmic_rescue
gc.collect()

oncokb = pd.read_csv(DATA_DIR / "oncokb_biomarker_drug_associations.tsv", sep="\t")
oncokb_genes = set(oncokb["Gene"].unique())
del oncokb
gc.collect()

print("\nStarting streaming dbNSFP processing")
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

file_path = DATA_DIR / "dbNSFP5.3a_grch37.gz"
chunk_iterator = pd.read_csv(
    file_path,
    sep="\t",
    compression="gzip",
    usecols=DBNSFP_COLS,
    chunksize=CHUNK_SIZE,
    low_memory=True,
    dtype={"#chr": str},
)

total_saved = 0
first_chunk_flag = True

for i, raw_chunk in enumerate(chunk_iterator):

    raw_chunk.rename(columns={"#chr": "chr", "pos(1-based)": "pos"}, inplace=True)

    cols_to_explode = [
        "genename",
        "aapos",
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "REVEL_score",
        "Interpro_domain",
    ]
    raw_chunk = explode_semicolon_columns(raw_chunk, cols_to_explode)
    raw_chunk.reset_index(drop=True, inplace=True)

    raw_chunk["genename"] = raw_chunk["genename"].str.strip()
    raw_chunk["aapos"] = pd.to_numeric(raw_chunk["aapos"], errors="coerce")

    numeric_cols = [
        "gnomAD4.1_joint_AF",
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "REVEL_score",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "CADD_phred",
    ]

    for col in numeric_cols:
        if col in raw_chunk.columns:
            raw_chunk[col] = (
                raw_chunk[col].astype(str).replace([".", "..", "", "nan", "NA"], np.nan)
            )
            raw_chunk[col] = pd.to_numeric(raw_chunk[col], errors="coerce")

            if col == "GERP++_RS":
                raw_chunk.loc[raw_chunk[col] > 100, col] = np.nan
            else:
                raw_chunk.loc[raw_chunk[col] > 100, col] = np.nan
                raw_chunk.loc[raw_chunk[col] == -1, col] = np.nan

            raw_chunk[col] = raw_chunk[col].astype("float32")

    raw_chunk["gnomAD_non_cancer_AF"] = np.nan
    non_cancer_mask = (
        pd.to_numeric(raw_chunk["gnomAD2.1.1_exomes_non_cancer_AN"], errors="coerce") > 0
    )
    if non_cancer_mask.any():
        raw_chunk.loc[non_cancer_mask, "gnomAD_non_cancer_AF"] = (
            pd.to_numeric(
                raw_chunk.loc[non_cancer_mask, "gnomAD2.1.1_exomes_non_cancer_AC"],
                errors="coerce",
            )
            / pd.to_numeric(
                raw_chunk.loc[non_cancer_mask, "gnomAD2.1.1_exomes_non_cancer_AN"],
                errors="coerce",
            )
        ).astype("float32")

    raw_chunk["rescue_key"] = (
        raw_chunk["chr"].astype(str)
        + ":"
        + raw_chunk["pos"].astype(str)
        + ":"
        + raw_chunk["ref"]
        + ":"
        + raw_chunk["alt"]
    )
    raw_chunk["rescue_gene_pos"] = raw_chunk["genename"] + "_" + raw_chunk["aapos"].astype(str)

    in_civic = raw_chunk["rescue_key"].isin(rescue_keys_civic)
    in_cosmic = raw_chunk["rescue_gene_pos"].isin(cosmic_rescue_keys)

    in_hotspot = pd.Series(False, index=raw_chunk.index)
    possible_hotspots = raw_chunk[raw_chunk["genename"].isin(RESCUE_HOTSPOTS.keys())]
    if not possible_hotspots.empty:
        for gene, positions in RESCUE_HOTSPOTS.items():
            mask = (raw_chunk["genename"] == gene) & (raw_chunk["aapos"].isin(positions))
            in_hotspot = in_hotspot | mask

    is_rescued = in_civic | in_hotspot | in_cosmic

    af_to_use = raw_chunk["gnomAD_non_cancer_AF"].fillna(raw_chunk["gnomAD4.1_joint_AF"])
    passes_af = (af_to_use.isna()) | (af_to_use < GNOMAD_AF_THRESHOLD)

    quality_ok = (
        (raw_chunk["CADD_phred"] >= MIN_CADD_SCORE) | (raw_chunk["CADD_phred"].isna()) | is_rescued
    )
    keep_variant = passes_af | is_rescued

    somatic_subset = raw_chunk[keep_variant & quality_ok].copy()

    if somatic_subset.empty:
        del raw_chunk
        continue

    somatic_subset["WAS_RESCUED"] = is_rescued[somatic_subset.index].astype(int)
    somatic_subset["RESCUE_REASON"] = "None"
    somatic_subset.loc[in_civic[somatic_subset.index], "RESCUE_REASON"] = "CIViC"
    somatic_subset.loc[in_hotspot[somatic_subset.index], "RESCUE_REASON"] = "Hotspot"
    somatic_subset.loc[in_cosmic[somatic_subset.index], "RESCUE_REASON"] = "COSMIC"

    somatic_subset["variant_type"] = "missense"
    nonsense_mask = (somatic_subset["aaalt"].isin(["*", "X"])) | (
        somatic_subset["HGVSp_snpEff"].str.contains("Ter|\\*", na=False, regex=True)
    )
    somatic_subset.loc[nonsense_mask, "variant_type"] = "nonsense"
    frameshift_mask = somatic_subset["HGVSp_snpEff"].str.contains("fs", na=False, case=False)
    somatic_subset.loc[frameshift_mask, "variant_type"] = "frameshift"
    synonymous_mask = (somatic_subset["aaref"] == somatic_subset["aaalt"]) & somatic_subset[
        "aaref"
    ].notna()
    somatic_subset.loc[synonymous_mask, "variant_type"] = "synonymous"

    somatic_subset = somatic_subset.merge(cgc_slim, on="genename", how="left")
    somatic_subset["ROLE_IN_CANCER"].fillna("Not_in_CGC", inplace=True)
    somatic_subset["TIER"] = somatic_subset["TIER"].fillna(0).astype(int)

    somatic_subset = somatic_subset.merge(
        cmc_agg[["genename", "aapos", "COSMIC_RECURRENCE", "COSMIC_FREQUENCY"]],
        on=["genename", "aapos"],
        how="left",
    )
    somatic_subset["COSMIC_RECURRENCE"].fillna(0, inplace=True)

    somatic_subset["IS_CANCER_GENE"] = somatic_subset["genename"].isin(cancer_genes).astype(int)
    somatic_subset["IS_ONCOGENE"] = somatic_subset["genename"].isin(oncogenes).astype(int)
    somatic_subset["IS_TSG"] = somatic_subset["genename"].isin(tsgs).astype(int)
    somatic_subset["IS_TIER1"] = somatic_subset["genename"].isin(tier1_genes).astype(int)
    somatic_subset["IS_ONCOKB"] = somatic_subset["genename"].isin(oncokb_genes).astype(int)
    somatic_subset["IS_KNOWN_HOTSPOT"] = in_hotspot[somatic_subset.index].astype(int)

    somatic_subset["CONSENSUS_SCORE"] = (
        somatic_subset["REVEL_score"].fillna(0.5) * 0.6
        + (1 - somatic_subset["SIFT_score"].fillna(0.5)) * 0.2
        + somatic_subset["Polyphen2_HDIV_score"].fillna(0.5) * 0.2
    ).astype("float32")

    somatic_subset["IS_CLINVAR_PATHOGENIC"] = (
        somatic_subset["clinvar_clnsig"]
        .astype(str)
        .str.contains("Pathogenic|Likely_pathogenic", na=False, case=False)
        .astype(int)
    )

    labels = pd.Series(-1, index=somatic_subset.index)
    evidence = pd.Series("None", index=somatic_subset.index)

    mask = somatic_subset["IS_KNOWN_HOTSPOT"] == 1
    labels[mask] = 1
    evidence[mask] = "known_hotspot"

    mask = (somatic_subset["IS_CLINVAR_PATHOGENIC"] == 1) & (labels == -1)
    labels[mask] = 1
    evidence[mask] = "clinvar_pathogenic"

    mask = (
        (somatic_subset["COSMIC_RECURRENCE"] >= 10)
        & (somatic_subset["IS_CANCER_GENE"] == 1)
        & (labels == -1)
    )
    labels[mask] = 1
    evidence[mask] = "cosmic_driver"

    mask = (
        (somatic_subset["variant_type"].isin(["nonsense", "frameshift"]))
        & (somatic_subset["IS_TSG"] == 1)
        & (labels == -1)
    )
    labels[mask] = 1
    evidence[mask] = "lof_tsg_driver"

    mask_passenger = (
        (somatic_subset["COSMIC_RECURRENCE"] > 0)
        & (somatic_subset["COSMIC_RECURRENCE"] <= 2)
        & (somatic_subset["IS_CLINVAR_PATHOGENIC"] == 0)
        & (somatic_subset["CONSENSUS_SCORE"] < 0.4)
        & (somatic_subset["WAS_RESCUED"] == 0)
        & (labels == -1)
    )

    labels[mask_passenger] = 0
    evidence[mask_passenger] = "somatic_passenger_cosmic"

    somatic_subset["LABEL_PATHOGENIC"] = labels
    somatic_subset["EVIDENCE_SOURCE"] = evidence

    final_variants = somatic_subset[somatic_subset["LABEL_PATHOGENIC"] != -1].copy()

    if not final_variants.empty:
        save_cols = [
            "chr",
            "pos",
            "ref",
            "alt",
            "genename",
            "aapos",
            "aaref",
            "aaalt",
            "variant_type",
            "CONSENSUS_SCORE",
            "REVEL_score",
            "CADD_phred",
            "SIFT_score",
            "Polyphen2_HDIV_score",
            "GERP++_RS",
            "phyloP100way_vertebrate",
            "Interpro_domain",
            "ROLE_IN_CANCER",
            "TIER",
            "IS_CANCER_GENE",
            "IS_TIER1",
            "IS_ONCOGENE",
            "IS_TSG",
            "IS_KNOWN_HOTSPOT",
            "COSMIC_RECURRENCE",
            "COSMIC_FREQUENCY",
            "IS_CLINVAR_PATHOGENIC",
            "WAS_RESCUED",
            "RESCUE_REASON",
            "EVIDENCE_SOURCE",
            "LABEL_PATHOGENIC",
        ]
        existing_cols = [c for c in save_cols if c in final_variants.columns]

        final_variants[existing_cols].to_csv(
            OUTPUT_FILE, mode="a", index=False, header=first_chunk_flag
        )
        first_chunk_flag = False
        total_saved += len(final_variants)

    del raw_chunk, somatic_subset, final_variants
    if i % 10 == 0:
        gc.collect()
        print(f"Processed chunk {i} | Saved rows: {total_saved:,}", end="\r")

print(f"\n\nProcessing complete. Total variants saved: {total_saved:,}")

print("Generating dataset metadata")
metadata_df = pd.read_csv(
    OUTPUT_FILE, usecols=["LABEL_PATHOGENIC", "EVIDENCE_SOURCE", "variant_type"]
)

metadata = {
    "total": int(len(metadata_df)),
    "pathogenic": int((metadata_df["LABEL_PATHOGENIC"] == 1).sum()),
    "benign": int((metadata_df["LABEL_PATHOGENIC"] == 0).sum()),
    "sources": metadata_df["EVIDENCE_SOURCE"].value_counts().to_dict(),
    "variants": metadata_df["variant_type"].value_counts().to_dict(),
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to {METADATA_FILE}")
print("Pipeline finished")
