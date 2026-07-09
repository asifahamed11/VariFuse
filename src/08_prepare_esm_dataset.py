import re
import pandas as pd
from config import STAGE07_OUT, STAGE08_OUT, DATA_DIR

INPUT_FILE = STAGE07_OUT / "Final_Dataset_Balanced.csv"
OUTPUT_FILE = STAGE08_OUT / "Final_Dataset_Balanced_ESM.csv"
UNIPROT_FILE = DATA_DIR / "uniprotkb_proteome_UP000005640_2026_01_07.txt"

# UniProt gene-name pattern, same convention as Stage 04
_GN_PATTERN = re.compile(r"Name=([^;{\s]+)")


def parse_uniprot_sequences(unique_genes: set) -> dict[str, str]:
    """Return {gene_name: canonical_sequence} for genes we care about.

    UniProt .txt lines look like:
        GN   Name=BRCA2; Synonyms=FACD, ...
        SQ   SEQUENCE   3418 AA;  384780 MW;  ...
        <sequence block, space-separated>
        //
    """
    gene_to_seq: dict[str, str] = {}
    current_gn = None
    seq_lines: list[str] = []
    in_seq = False

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
                    # keep the first (canonical) sequence seen for a gene
                    gene_to_seq.setdefault(current_gn, seq)
                current_gn = None
                seq_lines = []
                in_seq = False
            elif in_seq:
                seq_lines.append(line.strip())

    return gene_to_seq


def _ref_matches(row) -> bool:
    """True if the mapped sequence has aa_ref at position aa_pos (1-indexed)."""
    seq = row["protein_sequence"]
    try:
        pos = int(row["aa_pos"])
    except (TypeError, ValueError):
        return False
    return (
        isinstance(seq, str)
        and 1 <= pos <= len(seq)
        and seq[pos - 1] == str(row["aa_ref"])
    )


def prepare_esm():
    print("Loading balanced dataset...")
    df = pd.read_csv(INPUT_FILE)

    required = ["aapos", "aaref", "aaalt"]
    if not all(c in df.columns for c in required):
        print("Error: aapos, aaref, and aaalt columns are missing!")
        print("Ensure they were kept in Stage 05 and re-run stages 05-07.")
        return

    print("Renaming mutation columns for ESM...")
    df = df.rename(columns={"aapos": "aa_pos", "aaref": "aa_ref", "aaalt": "aa_alt"})

    n0 = len(df)
    df = df.dropna(subset=["aa_pos", "aa_ref", "aa_alt"])
    df["aa_pos"] = pd.to_numeric(df["aa_pos"], errors="coerce")
    df = df.dropna(subset=["aa_pos"])
    df["aa_pos"] = df["aa_pos"].astype(int)
    print(f"Dropped {n0 - len(df)} rows with missing/invalid mutation info.")

    unique_genes = set(df["genename"].dropna().unique())
    print(f"Found {len(unique_genes)} unique genes. Parsing UniProt...")
    gene_to_seq = parse_uniprot_sequences(unique_genes)
    print(f"Extracted sequences for {len(gene_to_seq)} / {len(unique_genes)} genes.")
    if not gene_to_seq:
        print("FATAL: no sequences parsed. Check the UniProt file path/format.")
        return

    print("Mapping sequences to dataset...")
    df["protein_sequence"] = df["genename"].map(gene_to_seq)

    n1 = len(df)
    df = df.dropna(subset=["protein_sequence"])
    print(f"Dropped {n1 - len(df)} rows with no UniProt sequence.")

    # Critical: reference-AA consistency (guards against isoform mismatch)
    n2 = len(df)
    df = df[df.apply(_ref_matches, axis=1)].reset_index(drop=True)
    print(f"Dropped {n2 - len(df)} rows where seq[aa_pos] != aa_ref "
          f"(isoform/position mismatch).")

    print(f"Final ESM-ready rows: {len(df):,}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved -> {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    prepare_esm()