"""Stage 07 - balance the pathogenic/benign classes for training."""

import pandas as pd
from config import STAGE06_OUT, STAGE07_OUT
import numpy as np
from pathlib import Path
import gc

INPUT_FILE = STAGE06_OUT / "somatic_variant_Cleaned.csv"
OUTPUT_FILE = STAGE07_OUT / "Final_Dataset_Balanced.csv"

CHUNK_SIZE = 50_000


def balance_dataset():
    print(f"Loading dataset: {INPUT_FILE}")

    try:
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

        
        print("\nPass 1: Counting class distribution...")
        n_pathogenic = 0
        n_benign = 0
        total = 0

        for chunk in pd.read_csv(INPUT_FILE, usecols=["LABEL_PATHOGENIC"],
                                 chunksize=CHUNK_SIZE):
            n_pathogenic += int((chunk["LABEL_PATHOGENIC"] == 1).sum())
            n_benign += int((chunk["LABEL_PATHOGENIC"] == 0).sum())
            total += len(chunk)
            del chunk

        print(f"\nOriginal class distribution:")
        print(f"   Total samples: {total:,}")
        print(f"   Benign (0): {n_benign:,} ({n_benign/total*100:.2f}%)")
        print(f"   Pathogenic (1): {n_pathogenic:,} ({n_pathogenic/total*100:.2f}%)")

        sample_size = min(n_pathogenic, n_benign)
        print(f"\nBalancing dataset via downsampling to {sample_size:,} per class...")

        benign_sample_frac = sample_size / max(n_benign, 1)

        pathogenic_parts = []
        benign_parts = []
        rng = np.random.RandomState(42)

        print("Pass 2: Collecting balanced samples in chunks...")
        chunk_num = 0
        for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE,
                                 low_memory=False):
            chunk_num += 1

            
            for col in chunk.select_dtypes(include=["float64"]).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast="float")

            path_rows = chunk[chunk["LABEL_PATHOGENIC"] == 1]
            if len(path_rows) > 0:
                pathogenic_parts.append(path_rows)

            ben_rows = chunk[chunk["LABEL_PATHOGENIC"] == 0]
            if len(ben_rows) > 0:
               
                n_sample = max(1, int(len(ben_rows) * benign_sample_frac))
                n_sample = min(n_sample, len(ben_rows))
                sampled = ben_rows.sample(n=n_sample, random_state=rng)
                benign_parts.append(sampled)

            del chunk, path_rows, ben_rows
            if chunk_num % 20 == 0:
                gc.collect()
                print(f"   Processed chunk {chunk_num}...", end="\r")

        print(f"   Processed {chunk_num} chunks total.")


        pathogenic_df = pd.concat(pathogenic_parts, ignore_index=True)
        del pathogenic_parts
        gc.collect()

        benign_df = pd.concat(benign_parts, ignore_index=True)
        del benign_parts
        gc.collect()

        
        if len(benign_df) > sample_size:
            benign_df = benign_df.sample(n=sample_size, random_state=42)
        elif len(benign_df) < sample_size:
            print(f"   Note: collected {len(benign_df):,} benign "
                  f"(target was {sample_size:,})")

        balanced_df = pd.concat([pathogenic_df, benign_df], ignore_index=True)
        del pathogenic_df, benign_df
        gc.collect()

        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\nBalanced class distribution:")
        new_counts = balanced_df["LABEL_PATHOGENIC"].value_counts()
        print(f"   Total samples: {len(balanced_df):,}")
        print(f"   Benign (0): {new_counts.get(0, 0):,}")
        print(f"   Pathogenic (1): {new_counts.get(1, 0):,}")

        print(f"\nSaving to: {OUTPUT_FILE}")
        balanced_df.to_csv(OUTPUT_FILE, index=False)

        print("Balancing complete")

    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    balance_dataset()

