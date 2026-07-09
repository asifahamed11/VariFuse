"""Stage 06 - final cleaning and column preparation of the modelling dataset."""

import pandas as pd
from config import STAGE05_OUT, STAGE06_OUT
import numpy as np
from pathlib import Path

INPUT_FILE = STAGE05_OUT / "somatic_variant_Leakage_Removed.csv"
OUTPUT_FILE = STAGE06_OUT / "somatic_variant_Cleaned.csv"


def clean_dataset():
    print("FINAL DATASET CLEANING & PREPARATION")

    try:
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

        print(f"Loading dataset: {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
        print(f"Original Shape: {df.shape} (Rows, Columns)\n")

        if "DOMAIN_NAME" in df.columns:
            df.drop(columns=["DOMAIN_NAME"], inplace=True)
            print(" Step 1: Dropped 'DOMAIN_NAME' column.")
        else:
            print(" Step 1: 'DOMAIN_NAME' column not found (skipped).")

        u_cols_dropped = []
        for col in df.columns:
            if (
                df[col].dtype == "object"
                and df[col].nunique() == 1
                and df[col].unique()[0] == "U"
            ):
                u_cols_dropped.append(col)

        if u_cols_dropped:
            df.drop(columns=u_cols_dropped, inplace=True)
            print(f" Step 2: Dropped columns with only 'U' values: {u_cols_dropped}")
        else:
            print(" Step 2: No columns found with only 'U' values.")


        if "SASA" in df.columns:
            initial_count = len(df)
            df = df[df["SASA"] != -1]
            removed_count = initial_count - len(df)
            print(
                f"Step 3: Removed {removed_count:,} rows with missing structural data (-1)."
            )
        else:
            print("Warning: 'SASA' column not found, could not filter -1 rows.")

        # Final Save
        print("\n" + "-" * 60)
        print(f"Final Shape: {df.shape} (Rows, Columns)")
        print(f"Saving to: {OUTPUT_FILE}")

        df.to_csv(OUTPUT_FILE, index=False)

    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    clean_dataset()
