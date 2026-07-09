# Datasets

Raw and external inputs are large and **not tracked in Git**. Point the pipeline at a local
copy:

    export TFDFE_DATA_DIR=/path/to/your/Datasets

`src/config.py` reads this automatically and falls back to this `data/` folder.

## Expected layout

    data/
    ├── dbNSFP5.3a_grch37.gz                          # stages 01, 09
    ├── uniprotkb_proteome_UP000005640_2026_01_07.txt # stages 04, 08, 09
    ├── UP000005640_9606_HUMAN_v6/                    # AlphaFold structures (stages 04, 09)
    └── external/
        ├── clinvar_recent.tsv                        # stage 09
        └── dms_scores.csv                            # stage 09

## Generated (not inputs)

Written by the pipeline into `data/external/`, no download needed:

- `external_ready_for_esm.csv` (Stage 09)
- `external_with_ESM_Score.csv`, `external_esm_embeddings.npy` (Stage 10)

A `.cache/` folder may appear after Stage 04; auto-generated, safe to delete.

## Setup

1. Download: **[Google Drive Link](https://drive.google.com/drive/folders/1sWNL6u6Fj5UEpQuplooFZFaDzfc0eUwS?usp=sharing)**
2. Place files under `data/` (and `data/external/`) per the layout above.
