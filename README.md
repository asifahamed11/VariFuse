# VariFuse

Evidence-aware somatic variant pathogenicity classifier fusing ESM-2 protein embeddings
with tabular biological features.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Two models are trained on an **identical feature set, CV split, and recall-aware
threshold** for a fair comparison:

- **LightGBM** over biological, structural, and ESM-derived features.
- **Cross-Attention fusion net** that cross-attends scalar features with ESM-2
  residue-embedding tokens (seed-averaged ensemble, EMA weights).

## Features

- **Predictors:** REVEL, CADD, SIFT, PolyPhen2, GERP++, phyloP
- **Structural (AlphaFold):** SASA, relative SASA, pLDDT, domain / site flags
- **Protein LM:** ESM-2 (`esm2_t33_650M_UR50D`, 1280-d) embeddings + variant score
- **Cancer annotations:** cancer-gene / tier1 / oncogene / TSG flags

Leakage-prone columns are dropped before modeling. A **leakage-audit** config also ablates
the meta-predictor scores to measure Type-1 circularity.

## Pipeline

Scripts in `src/` run in order; each reads the previous output and writes to its own
`outputs/` subfolder (folder numbers are fixed in `config.py`).

| # | Script | Stage |
| --- | --- | --- |
| 01 | `01_dbnsfp.py` | Parse/filter dbNSFP |
| 02 | `02_missing_values.py` | Drop missing values |
| 03 | `03_duplicates.py` | Deduplicate variants |
| 04 | `04_feature_engineering.py` | UniProt + AlphaFold features |
| 05 | `05_leakage.py` | Drop leaking columns |
| 06 | `06_clean.py` | Final cleaning |
| 07 | `07_dataset_balancing.py` | Balance classes |
| 08 | `08_prepare_esm_dataset.py` | Map sequences, ref-AA check |
| 09 | `09_prepare_external_esm_dataset.py` | Build independent ClinVar/DMS external set |
| 10 | `10_extract_esm_features.py` | ESM-2 embeddings (internal + external) |
| 11 | `11_train_and_evaluate.py` | LightGBM vs Cross-Attention (gene-level CV, SHAP, McNemar) |
| 12 | `12_external_validation.py` | Gene-disjoint external validation |
| 13 | `13_generate_figures.py` | Publication figures (PNG) |

**09 runs before 10** so both sets are embedded in one pass (model loads once):

    python 10_extract_esm_features.py   # defaults to --dataset both

## External validation

Stage 12 tests the frozen-threshold models on a **gene-disjoint** ClinVar/DMS set built in
Stage 09 with its own dbNSFP-derived features (no internal leakage). Shared genes and, with
`STRICT_VARIANT_DEOVERLAP`, exact `chr:pos:ref:alt` matches are removed. Both the primary
and leakage-audit configs are reported.

## Quick start

    pip install -r requirements.txt
    export TFDFE_DATA_DIR=/path/to/your/Datasets   # or place files under data/

ESM extraction (Stage 10) uses `fair-esm` + `torch`; a CUDA GPU is recommended (FP16 auto).
See **[`data/README.md`](data/README.md)** for the expected data layout.

## Outputs

- `outputs/10_model_evaluation/` → `results.json`, `comparison_table.csv`, `oof_predictions.npz`, `shap_values_*.npz`
- `outputs/12_external_validation/` → `external_validation.json`, `external_validation_table.csv`, `external_predictions.npz`
- `figures/` → `fig01`–`fig12` (ROC/PR, confusion, SHAP, Type-1 circularity, external analogues)

## License

See [LICENSE](LICENSE).
