import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("data/external/urn_mavedb_00001234-c-1_scores.csv")
OUTPUT_FILE = Path("data/external/dms_scores.csv")
TARGET_GENE = "TP53"  # Change this if using a different DMS assay

aa_map = {
    "Ala":"A", "Arg":"R", "Asn":"N", "Asp":"D", "Cys":"C",
    "Gln":"Q", "Glu":"E", "Gly":"G", "His":"H", "Ile":"I",
    "Leu":"L", "Lys":"K", "Met":"M", "Phe":"F", "Pro":"P",
    "Ser":"S", "Thr":"T", "Trp":"W", "Tyr":"Y", "Val":"V", "Ter":"*"
}

print("Loading MaveDB raw dataset...")
df = pd.read_csv(INPUT_FILE)

records = []

pattern = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter)")

for idx, row in df.iterrows():
    hgvs = str(row.get("hgvs_pro", ""))
    score = row.get("score", pd.NA)
    
    if pd.isna(score):
        continue
        
    match = pattern.search(hgvs)
    if match:
        ref_3 = match.group(1)
        pos = match.group(2)
        alt_3 = match.group(3)
        
        if ref_3 in aa_map and alt_3 in aa_map:
            records.append({
                "genename": TARGET_GENE,         
                "aapos": int(pos),
                "aaref": aa_map[ref_3],
                "aaalt": aa_map[alt_3],
                "dms_score": float(score)
            })

out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_FILE, index=False)

print(f"Data Cleaning Successful!")
print(f"Formatted {len(out_df)} variants.")
print(f"Saved to -> {OUTPUT_FILE}")