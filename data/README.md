# Datasets

This folder holds the raw and external input files required to run the pipeline. They are
large and **excluded from Git** via `.gitignore`.

Point the pipeline at a local copy without editing any code:

```bash
export TFDFE_DATA_DIR=/path/to/your/Datasets
```

`src/config.py` reads `TFDFE_DATA_DIR` automatically and falls back to this `data/` folder.

## Expected layout

```text
data/
├── dbNSFP5.3a_grch37.gz                          # dbNSFP bulk file (stages 01 and 09)
├── uniprotkb_proteome_UP000005640_2026_01_07.txt # UniProt human proteome (stages 04, 08, 09)
├── UP000005640_9606_HUMAN_v6/                    # AlphaFold structures (.cif / .pdb per protein)
└── external/                                     # inputs for the independent external set
    ├── clinvar_recent.tsv                        # ClinVar variant summaries (stage 09)
    └── dms_scores.csv                            # deep mutational scanning scores (stage 09)
```

## What each stage reads

- **Stage 01 / 09** — `dbNSFP5.3a_grch37.gz` is scanned for variant features. Stage 09 also
  reads `external/clinvar_recent.tsv` and `external/dms_scores.csv` to define an independent
  external validation set, then re-derives features from dbNSFP so nothing leaks from the
  internal data.
- **Stage 04 / 08 / 09** — `uniprotkb_proteome_UP000005640_2026_01_07.txt` supplies canonical
  protein sequences (for ref-AA consistency and ESM-2 input) and `UP000005640_9606_HUMAN_v6/`
  supplies AlphaFold structures for SASA / pLDDT / structural features.

## Generated files (not inputs)

The following are produced by the pipeline inside `data/external/` and do **not** need to be
downloaded:

- `external_ready_for_esm.csv` — written by Stage 09.
- `external_with_ESM_Score.csv` and `external_esm_embeddings.npy` — written by Stage 10
  (`--dataset external` / `both`).

A `.cache/` folder may also appear after Stage 04's first run; it is auto-generated and safe
to delete.

## Setup

1. Download the datasets: **[Google Drive Link](https://drive.google.com/drive/folders/1sWNL6u6Fj5UEpQuplooFZFaDzfc0eUwS?usp=sharing)**
2. Place them into this `data/` folder (and `data/external/`) matching the layout above, or
   set `TFDFE_DATA_DIR` to an existing local copy.
