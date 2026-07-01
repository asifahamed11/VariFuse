#2. Removes all rows with any missing values
import pandas as pd
from config import DATA_DIR, STAGE01_OUT, STAGE02_OUT, STAGE03_OUT, STAGE04_OUT, STAGE05_OUT, STAGE06_OUT, STAGE07_OUT, STAGE08_OUT, STAGE09_OUT, EDA_OUT
import gc

INPUT_FILE = STAGE01_OUT / 'somatic_variant_dbNSFP.csv' 
OUTPUT_FILE = 'somatic_variant_dbNSFP_Removes_missing_values.csv'

COLUMNS_TO_RETAIN_WITH_MISSING = [
    'COSMIC_FREQUENCY', 
    'COSMIC_RECURRENCE', 
    'EVIDENCE_SOURCE', 
    'RESCUE_REASON', 
    'WAS_RESCUED',
    'IS_CLINVAR_PATHOGENIC',
    'IS_KNOWN_HOTSPOT',
    'genename', 
    'aapos', 
    'aaref', 
    'aaalt', 
    'Interpro_domain', 
    'variant_type',
    'ROLE_IN_CANCER'
] 


def clean_dataset_selective(chunksize=100000):
    """
    Remove rows with missing values in non-exempt columns
    
    Process data in chunks to handle large files efficiently
    """
    print(f"Starting selective cleaning: {INPUT_FILE}")
    print(f"Preserving rows with missing values in: {len(COLUMNS_TO_RETAIN_WITH_MISSING)} columns")
    print(f"Chunk size: {chunksize:,}\n")
    
    try:
        first_chunk = True
        total_original = 0
        total_kept = 0
        chunk_num = 0
        
        print("Processing chunks...")
        
        for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize, low_memory=False):
            chunk_num += 1
            original_rows = len(chunk)
            total_original += original_rows
            
            all_columns = chunk.columns.tolist()
            columns_to_check = [col for col in all_columns if col not in COLUMNS_TO_RETAIN_WITH_MISSING]
            
            chunk_clean = chunk.dropna(how='any', subset=columns_to_check)
            
            kept_rows = len(chunk_clean)
            total_kept += kept_rows
            
            if not chunk_clean.empty:
                if first_chunk:
                    chunk_clean.to_csv(OUTPUT_FILE, mode='w', index=False, header=True)
                    first_chunk = False
                else:
                    chunk_clean.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
            
            if chunk_num % 10 == 0:
                drop_in_chunk = original_rows - kept_rows
                print(f"   Chunk {chunk_num}: {original_rows:,} -> {kept_rows:,} (dropped: {drop_in_chunk:,})")
                
            del chunk, chunk_clean
            gc.collect()
        
        dropped = total_original - total_kept
        drop_pct = (dropped / total_original) * 100 if total_original > 0 else 0
        
        print(f"\nCleaning complete")
        print(f"   Total input rows: {total_original:,}")
        print(f"   Rows retained: {total_kept:,}")
        print(f"   Rows dropped: {dropped:,} ({drop_pct:.2f}%)")
        print(f"\nOutput saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found")
    except Exception as e:
        print(f"\nError occurred: {e}")


if __name__ == "__main__":
    clean_dataset_selective()