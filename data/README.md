# Datasets

This folder holds the raw data files required to run the pipeline (~76GB total).

**Note:** excluded from Git via `.gitignore` — too large to track.

## Expected layout

```
data/
├── 01-Jan-2026-VariantSummaries.tsv
├── Cosmic_CancerGeneCensus_v102_GRCh37.tsv
├── cmc_export.tsv
├── oncokb_biomarker_drug_associations.tsv
├── dbNSFP5.3a_grch37.gz
├── uniprotkb_proteome_UP000005640_2026_01_07.txt
├── UP000005640_9606_HUMAN_v6/    # AlphaFold structures (.cif / .pdb per protein)
├── hg19.fa
└── hg19.fa.fai
```

(`.cache/` appears here too after the first run of script 04 — auto-generated, safe to delete.)

## Setup

1. Download the datasets: **[Google Drive Link](https://drive.google.com/drive/folders/1sWNL6u6Fj5UEpQuplooFZFaDzfc0eUwS?usp=sharing)**
2. Place them directly into this `data/` folder, matching the layout above.
   `src/config.py` reads this env var automatically — no code edits needed.
