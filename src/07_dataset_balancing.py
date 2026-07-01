import pandas as pd
from config import STAGE06_OUT, STAGE07_OUT
import numpy as np
from pathlib import Path

INPUT_FILE = STAGE06_OUT / 'somatic_variant_Cleaned.csv'
OUTPUT_FILE = STAGE07_OUT / 'Final_Dataset_Balanced.csv'

def balance_dataset():
    print(f"Loading dataset: {INPUT_FILE}")
    
    try:
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        variant_df = pd.read_csv(INPUT_FILE, low_memory=False)
        
        print("\nOriginal class distribution:")
        print("-" * 40)
        counts = variant_df['LABEL_PATHOGENIC'].value_counts()
        total = len(variant_df)
        
        n_benign = counts.get(0, 0)
        n_pathogenic = counts.get(1, 0)
        
        print(f"   Total samples: {total:,}")
        print(f"   Benign (0): {n_benign:,} ({n_benign/total*100:.2f}%)")
        print(f"   Pathogenic (1): {n_pathogenic:,} ({n_pathogenic/total*100:.2f}%)")
        print("-" * 40)
        
        print("\nBalancing dataset via downsampling...")
        
        pathogenic_variants = variant_df[variant_df['LABEL_PATHOGENIC'] == 1]
        benign_variants = variant_df[variant_df['LABEL_PATHOGENIC'] == 0]
        
        sample_size = len(pathogenic_variants)
        benign_downsampled = benign_variants.sample(n=sample_size, random_state=42)
        
        balanced_df = pd.concat([pathogenic_variants, benign_downsampled])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("\nBalanced class distribution:")
        print("-" * 40)
        new_counts = balanced_df['LABEL_PATHOGENIC'].value_counts()
        print(f"   Total samples: {len(balanced_df):,}")
        print(f"   Benign (0): {new_counts.get(0, 0):,}")
        print(f"   Pathogenic (1): {new_counts.get(1, 0):,}")
        print("-" * 40)
        
        print(f"\nSaving to: {OUTPUT_FILE}")
        balanced_df.to_csv(OUTPUT_FILE, index=False)
        
        print("Balancing complete")
        
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    balance_dataset()
