#3. Removes duplicate variants
import pandas as pd
from config import STAGE02_OUT, STAGE03_OUT
import gc

INPUT_FILE = STAGE02_OUT / 'somatic_variant_dbNSFP_Removes_missing_values.csv'
OUTPUT_FILE = STAGE03_OUT / 'somatic_variant_dbNSFP_Removes_missing_values_Deduplication.csv'


def remove_duplicates(chunksize=100000):
    print(f"Loading dataset: {INPUT_FILE}")
    print(f"Chunk size: {chunksize:,}\n")
    
    try:
        print("Pass 1: Reading and sorting chunks...")
        all_chunks = []
        chunk_num = 0
        total_rows = 0
        
        for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize, low_memory=False):
            chunk_num += 1
            total_rows += len(chunk)
            
           
            if 'CONSENSUS_SCORE' in chunk.columns:
                chunk = chunk.sort_values(
                    by=['chr', 'pos', 'ref', 'alt', 'CONSENSUS_SCORE'], 
                    ascending=[True, True, True, True, False]
                )
            
            all_chunks.append(chunk)
            
            if chunk_num % 10 == 0:
                print(f"   Processed chunk {chunk_num}: {total_rows:,} total rows")
            
            del chunk
            gc.collect()
        
        print(f"\nTotal variants loaded: {total_rows:,}")
        
        print("\nPass 2: Combining and deduplicating...")
        variant_df = pd.concat(all_chunks, ignore_index=True)
        del all_chunks
        gc.collect()
        
        
        if 'CONSENSUS_SCORE' in variant_df.columns:
            print("   Sorting by consensus score (highest first)")
            variant_df = variant_df.sort_values(
                by=['chr', 'pos', 'ref', 'alt', 'CONSENSUS_SCORE'], 
                ascending=[True, True, True, True, False]
            )
        
        # Remove duplicates
        print("   Removing duplicates...")
        deduplicated_df = variant_df.drop_duplicates(
            subset=['chr', 'pos', 'ref', 'alt'], 
            keep='first'
        )
        
        kept_rows = len(deduplicated_df)
        dropped_rows = total_rows - kept_rows
        
        print(f"\nUnique variants retained: {kept_rows:,}")
        print(f"Duplicates removed: {dropped_rows:,}")
        
        print(f"\nSaving to: {OUTPUT_FILE}")
        deduplicated_df.to_csv(OUTPUT_FILE, index=False)
        
        print("Deduplication complete")
        
        del variant_df, deduplicated_df
        gc.collect()

    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    remove_duplicates(chunksize=50000)