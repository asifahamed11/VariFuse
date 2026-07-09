"""Stage 05 - drop annotation columns that leak the target label."""
import pandas as pd
from config import STAGE04_OUT, STAGE05_OUT
import gc
from pathlib import Path

INPUT_FILE = (STAGE04_OUT /
              "somatic_variant_dbNSFP_Removes_missing_values_Deduplication_Structural_Functional.csv")
OUTPUT_FILE = STAGE05_OUT / "somatic_variant_Leakage_Removed.csv"

LEAKAGE_COLS = [
    "COSMIC_FREQUENCY",
    "COSMIC_RECURRENCE",
    "EVIDENCE_SOURCE",
    "RESCUE_REASON",
    "WAS_RESCUED",
    "IS_CLINVAR_PATHOGENIC",
    "IS_CLINVAR_BENIGN",
    "IS_KNOWN_HOTSPOT",
]

ID_COLS = [
    "chr",
    "pos",
    "ref",
    "alt",
    "Interpro_domain",
    "variant_type",
    "ROLE_IN_CANCER",
]


def drop_columns_chunked(chunksize=50000):
    print(f"Starting feature selection: {INPUT_FILE}")
    print(f"Chunk size: {chunksize:,} rows")
    try:
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        all_cols = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
        cols_to_drop = LEAKAGE_COLS + ID_COLS
        use_cols = [c for c in all_cols if c not in cols_to_drop]

        print(f"Total columns: {len(all_cols)}")
        print(f"Dropping: {len([c for c in cols_to_drop if c in all_cols])} columns")
        print(f"Retaining: {len(use_cols)} columns")
        print("\nRetained features:")
        for i, col in enumerate(use_cols, 1):
            print(f" {i}. {col}")

        first_chunk = True
        total_rows = 0
        chunk_num = 0
        print("\nProcessing chunks...")
        for chunk in pd.read_csv(INPUT_FILE, usecols=use_cols,
                                 chunksize=chunksize, low_memory=False):
            chunk_num += 1
            total_rows += len(chunk)
            if first_chunk:
                chunk.to_csv(OUTPUT_FILE, mode="w", index=False, header=True)
                first_chunk = False
                print(f" Chunk {chunk_num}: {len(chunk):,} rows (with header)")
            else:
                chunk.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
                print(f" Chunk {chunk_num}: {len(chunk):,} rows (appended)")
            del chunk
            gc.collect()

        print(f"\nProcessing complete. Total rows: {total_rows:,}")
        print(f"Output saved to: {OUTPUT_FILE}")
        final_df = pd.read_csv(OUTPUT_FILE, nrows=3)
        print(f" Shape: {final_df.shape}")
        print(f" Columns: {list(final_df.columns)}")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    drop_columns_chunked(chunksize=50000)