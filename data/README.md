# Datasets

Raw inputs are large and not tracked in Git. Point the pipeline at a local copy:

```bash
export VARIFUSE_DATA_DIR=/path/to/your/Datasets
```

`src/config.py` reads this automatically and falls back to this `data/` folder.

## Expected layout

```
data/
├── dbNSFP5.3a_grch37.gz                              # stages 01, 09
├── uniprotkb_proteome_UP000005640_2026_01_07.txt     # stages 04, 08, 09
├── UP000005640_9606_HUMAN_v6/                        # AlphaFold structures (04, 09)
└── external/
    ├── clinvar_recent.tsv                            # stage 09
    └── dms_scores.csv                                # stage 09
```

## Generated (not inputs)

Written into `data/external/` by the pipeline:

- `external_ready_for_esm.csv` (Stage 09)
- `external_with_ESM_Score.csv`, `external_esm_embeddings.npy` (Stage 10)

A `.cache/` folder may appear after Stage 04. Auto-generated, safe to delete.

## Setup

1. Download: [Google Drive](https://drive.google.com/drive/folders/1sWNL6u6Fj5UEpQuplooFZFaDzfc0eUwS?usp=sharing)
2. Place files under `data/` (and `data/external/`) per the layout above.

The external set is built gene-disjoint from the internal data: Stage 09 drops shared genes, and with `STRICT_VARIANT_DEOVERLAP` also removes exact `chr:pos:ref:alt` matches.
