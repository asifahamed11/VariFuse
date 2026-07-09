import pandas as pd
import os
import re
import sys
import importlib
import numpy as np
from pathlib import Path

# Add src to path to import 04_feature_engineering which starts with a number
sys.path.append(str(Path(__file__).resolve().parent))
feat_eng = importlib.import_module("04_feature_engineering")
annotate_variants = feat_eng.annotate_variants

from config import DATA_DIR, STAGE04_OUT

CLINVAR_TSV = DATA_DIR / "external" / "clinvar_recent.tsv"
DMS_CSV = DATA_DIR / "external" / "dms_scores.csv"
STAGE04_CSV = (
    STAGE04_OUT
    / "somatic_variant_dbNSFP_Removes_missing_values_Deduplication_Structural_Functional.csv"
)
UNIPROT_FILE = DATA_DIR / "uniprotkb_proteome_UP000005640_2026_01_07.txt"
ALPHAFOLD_DIR = DATA_DIR / "UP000005640_9606_HUMAN_v6"
OUT_CSV = DATA_DIR / "external" / "external_ready_for_esm.csv"
DBNSFP_FILE = DATA_DIR / "dbNSFP5.3a_grch37.gz"

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
    if variant_df.empty:
        return variant_df

    present = [c for c in cols_to_explode if c in variant_df.columns]
    if not present:
        return variant_df

    for col in present:
        variant_df[col] = variant_df[col].astype(str).str.split(";")

    try:
        # Fast path: all lists per row already equal length -> stays paired.
        variant_df = variant_df.explode(present)
    except ValueError:
        # Safe path: pad ragged lists to equal length so a single explode
        # keeps gene / aapos / score pairing intact (NO cross-join).
        def _pad_row(row):
            lengths = [len(x) if isinstance(x, list) else 1 for x in row]
            max_len = max(lengths) if lengths else 1
            padded = []
            for x in row:
                if isinstance(x, list):
                    padded.append(x + [np.nan] * (max_len - len(x)))
                else:
                    padded.append([x] + [np.nan] * (max_len - 1))
            return padded

        padded = variant_df[present].apply(_pad_row, axis=1, result_type="expand")
        padded.columns = present
        for col in present:
            variant_df[col] = padded[col].values
        variant_df = variant_df.explode(present)

    variant_df.reset_index(drop=True, inplace=True)
    for col in present:
        variant_df[col] = variant_df[col].replace([".", "nan", "", None], np.nan)
    return variant_df


def process_dbnsfp_chunk(raw_chunk):
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
            raw_chunk.loc[raw_chunk[col] > 100, col] = np.nan
            raw_chunk.loc[raw_chunk[col] == -1, col] = np.nan
            raw_chunk[col] = raw_chunk[col].astype("float32")
    raw_chunk["variant_type"] = "missense"
    nonsense_mask = (raw_chunk["aaalt"].isin(["*", "X"])) | (
        raw_chunk["HGVSp_snpEff"].str.contains("Ter|\\*", na=False, regex=True)
    )
    raw_chunk.loc[nonsense_mask, "variant_type"] = "nonsense"
    frameshift_mask = raw_chunk["HGVSp_snpEff"].str.contains("fs", na=False, case=False)
    raw_chunk.loc[frameshift_mask, "variant_type"] = "frameshift"
    synonymous_mask = (raw_chunk["aaref"] == raw_chunk["aaalt"]) & raw_chunk[
        "aaref"
    ].notna()
    raw_chunk.loc[synonymous_mask, "variant_type"] = "synonymous"
    return raw_chunk


def main():
    cv_mapped = pd.DataFrame()
    dms_mapped = pd.DataFrame()
    cv_search_keys = set()
    dms_search_keys = set()

    if os.path.exists(CLINVAR_TSV):
        print("\nProcessing ClinVar Data...")
        try:
            cv = pd.read_csv(
                CLINVAR_TSV,
                sep="\t",
                usecols=[
                    "Chromosome",
                    "PositionVCF",
                    "ReferenceAlleleVCF",
                    "AlternateAlleleVCF",
                    "ClinicalSignificance",
                ],
                quoting=3,
                dtype=str,
            )
            sig_col = "ClinicalSignificance"
        except ValueError:
            cv = pd.read_csv(
                CLINVAR_TSV,
                sep="\t",
                usecols=[
                    "Chromosome",
                    "PositionVCF",
                    "ReferenceAlleleVCF",
                    "AlternateAlleleVCF",
                    "clinvar_clnsig",
                ],
                quoting=3,
                dtype=str,
            )
            sig_col = "clinvar_clnsig"
        if sig_col in cv.columns:
            sig = cv[sig_col].astype(str).str.lower()
            conflicting = sig.str.contains("conflict", na=False)
            path = sig.str.contains("pathogenic", na=False) & ~conflicting
            benign = sig.str.contains("benign", na=False) & ~conflicting
            cv = cv[path | benign].copy()
            cv["LABEL_PATHOGENIC"] = path[path | benign].astype(int)
        cv = cv.rename(
            columns={
                "Chromosome": "chr",
                "PositionVCF": "pos",
                "ReferenceAlleleVCF": "ref",
                "AlternateAlleleVCF": "alt",
            }
        )
        for col in ["chr", "pos", "ref", "alt"]:
            if col in cv.columns:
                cv[col] = cv[col].astype(str)
        cv_mapped = cv[["chr", "pos", "ref", "alt", "LABEL_PATHOGENIC"]].copy()
        cv_mapped["EXT_SOURCE"] = "clinvar"
        cv_mapped["chr_pos_ref_alt"] = (
            cv_mapped["chr"]
            + ":"
            + cv_mapped["pos"]
            + ":"
            + cv_mapped["ref"]
            + ":"
            + cv_mapped["alt"]
        )
        cv_search_keys = set(cv_mapped["chr_pos_ref_alt"].unique())
        print(
            f"-> Set up {len(cv_search_keys)} unique genomic keys from ClinVar to scan DbNSFP."
        )

    if os.path.exists(DMS_CSV):
        print("\nProcessing DMS Data...")
        dms = pd.read_csv(DMS_CSV)
        cutoff = dms["dms_score"].median()
        dms["LABEL_PATHOGENIC"] = (dms["dms_score"] <= cutoff).astype(int)
        dms_mapped = dms[
            ["genename", "aapos", "aaref", "aaalt", "LABEL_PATHOGENIC"]
        ].copy()
        dms_mapped["EXT_SOURCE"] = "dms"
        dms_mapped["genename_aapos"] = (
            dms_mapped["genename"].astype(str)
            + ":"
            + dms_mapped["aapos"].astype(float).astype("Int64").astype(str)
        )
        dms_search_keys = set(dms_mapped["genename_aapos"].unique())
        print(
            f"-> Set up {len(dms_search_keys)} unique gene+pos keys from DMS to scan DbNSFP."
        )

    if cv_mapped.empty and dms_mapped.empty:
        print(
            "No external data found or matched. Please check your data/external/ folder."
        )
        return

    print(
        "\nScanning dbNSFP bulk file for external variants (this ensures fully independent DbNSFP features)..."
    )
    if not os.path.exists(DBNSFP_FILE):
        print(f"Error: Could not find dbNSFP file at {DBNSFP_FILE}")
        return

    chunk_iterator = pd.read_csv(
        DBNSFP_FILE,
        sep="\t",
        compression="gzip",
        usecols=DBNSFP_COLS,
        chunksize=50000,
        low_memory=True,
        dtype={"#chr": str, "pos(1-based)": str, "ref": str, "alt": str},
    )

    extracted_dbnsfp_chunks = []
    for i, raw_chunk in enumerate(chunk_iterator):
        if i % 10 == 0:
            print(f"  Scanned {i * 50000:,} rows from dbNSFP...")
        raw_chunk["chr_pos_ref_alt"] = (
            raw_chunk["#chr"]
            + ":"
            + raw_chunk["pos(1-based)"]
            + ":"
            + raw_chunk["ref"]
            + ":"
            + raw_chunk["alt"]
        )
        pre_filter = raw_chunk["chr_pos_ref_alt"].isin(cv_search_keys)
        if len(dms_search_keys) > 0:
            gn_regex = "|".join({re.escape(k.split(":")[0]) for k in dms_search_keys})
            pos_regex = "|".join(
                {r"\b" + re.escape(k.split(":")[1]) + r"\b" for k in dms_search_keys}
            )
            gn_matches = raw_chunk["genename"].str.contains(
                gn_regex, na=False, regex=True
            )
            pos_matches = (
                raw_chunk["aapos"]
                .astype(str)
                .str.contains(pos_regex, na=False, regex=True)
            )
            pre_filter = pre_filter | (gn_matches & pos_matches)
        filtered_chunk = raw_chunk[pre_filter].copy()
        if not filtered_chunk.empty:
            processed = process_dbnsfp_chunk(filtered_chunk)
            final_filter = pd.Series(False, index=processed.index)
            if len(cv_search_keys) > 0:
                final_filter |= processed["chr_pos_ref_alt"].isin(cv_search_keys)
            if len(dms_search_keys) > 0:
                processed["genename_aapos_exact"] = (
                    processed["genename"].astype(str)
                    + ":"
                    + processed["aapos"].astype(float).astype("Int64").astype(str)
                )
                final_filter |= processed["genename_aapos_exact"].isin(dms_search_keys)
            processed = processed[final_filter].copy()
            if not processed.empty:
                extracted_dbnsfp_chunks.append(processed)
    print("Finished scanning dbNSFP.")

    dbnsfp_features = pd.DataFrame()
    if extracted_dbnsfp_chunks:
        dbnsfp_features = pd.concat(extracted_dbnsfp_chunks, ignore_index=True)
        dbnsfp_features = dbnsfp_features.drop_duplicates(
            subset=["chr_pos_ref_alt", "genename", "aapos", "aaref", "aaalt"]
        )
        print(f"-> Extracted {len(dbnsfp_features)} matching records from dbNSFP.")
    else:
        print("-> Warning: No matching records found in dbNSFP.")

    external_parts = []
    if not cv_mapped.empty:
        if not dbnsfp_features.empty:
            db_feats = dbnsfp_features.drop(
                columns=["genename_aapos_exact", "chr", "pos", "ref", "alt"],
                errors="ignore",
            )
            cv_final = pd.merge(cv_mapped, db_feats, on="chr_pos_ref_alt", how="inner")
            cv_final = cv_final.drop(columns=["chr_pos_ref_alt"])
            external_parts.append(cv_final)
            print(f"-> ClinVar variants after DbNSFP merge: {len(cv_final)}")
            if len(cv_final) == 0:
                print(
                    "-> CRITICAL WARNING: ClinVar branch silently emptied during merge."
                )
        else:
            print("-> Warning: No dbnsfp_features to merge with ClinVar.")

    if not dms_mapped.empty:
        if not dbnsfp_features.empty:
            dms_final = pd.merge(
                dms_mapped,
                dbnsfp_features.drop(columns=["chr_pos_ref_alt"], errors="ignore"),
                on=["genename", "aapos", "aaref", "aaalt"],
                how="inner",
            )
            dms_final = dms_final.drop(
                columns=["genename_aapos", "genename_aapos_exact"], errors="ignore"
            )
            external_parts.append(dms_final)
            print(f"-> DMS variants after DbNSFP merge: {len(dms_final)}")
            if len(dms_final) == 0:
                print("-> CRITICAL WARNING: DMS branch silently emptied during merge.")
        else:
            print("-> Warning: No dbnsfp_features to merge with DMS.")

    if not external_parts:
        print("No external data survived DbNSFP merge.")
        return

    final_ext = pd.concat(external_parts, ignore_index=True)
    final_ext = final_ext.drop_duplicates(
        subset=["genename", "aapos", "aaref", "aaalt"]
    )
    print(
        f"\nTotal combined variants before structural feature engineering: {len(final_ext)}"
    )

    print("\nRunning Independent Structural Feature Engineering (Stage 04)...")
    final_ext = annotate_variants(
        dataset=final_ext,
        uniprot_file=str(UNIPROT_FILE),
        alphafold_dir=str(ALPHAFOLD_DIR),
    )
    final_ext = final_ext.rename(
        columns={"aapos": "aa_pos", "aaref": "aa_ref", "aaalt": "aa_alt"}
    )

    print("\nMapping Protein Sequences from UniProt for ESM-2...")
    _GN_PATTERN = re.compile(r"Name=([^;{\s]+)")
    unique_genes = set(final_ext["genename"].dropna().unique())
    gene_to_seq = {}
    current_gn, in_seq, seq_lines = None, False, []
    with open(UNIPROT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("GN"):
                m = _GN_PATTERN.search(line)
                if m and m.group(1) in unique_genes:
                    current_gn = m.group(1)
            elif line.startswith("SQ"):
                if current_gn:
                    in_seq = True
                    seq_lines = []
            elif line.startswith("//"):
                if current_gn and in_seq:
                    seq = "".join(seq_lines).replace(" ", "").strip()
                    gene_to_seq.setdefault(current_gn, seq)
                current_gn, in_seq, seq_lines = None, False, []
            elif in_seq:
                seq_lines.append(line.strip())

    final_ext["protein_sequence"] = final_ext["genename"].map(gene_to_seq)
    final_ext = final_ext.dropna(subset=["protein_sequence"])

    def _ref_ok(row):
        try:
            pos = int(row["aa_pos"])
        except Exception:
            return False
        seq = row["protein_sequence"]
        return (
            isinstance(seq, str)
            and 1 <= pos <= len(seq)
            and seq[pos - 1] == row["aa_ref"]
        )

    final_ext = final_ext[final_ext.apply(_ref_ok, axis=1)].reset_index(drop=True)
    if final_ext.empty:
        print("No external rows survived protein-sequence mapping.")
        return

    final_ext.to_csv(OUT_CSV, index=False)
    print(f"Saved external ESM-ready dataset -> {OUT_CSV}")
    print("Done. Run Stage 10 in external mode next (python 10_extract_esm_features.py --dataset both).")


if __name__ == "__main__":
    main()