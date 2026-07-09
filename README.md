# VariFuse

Somatic variant pathogenicity classifier that fuses ESM-2 protein embeddings with tabular biological features.

Two models are trained on the same feature set, CV split, and threshold, so the comparison is fair:

- **LightGBM** over biological, structural, and ESM-derived features
- **Cross-Attention fusion net** that cross-attends scalar features against ESM-2 residue tokens (seed-averaged ensemble, EMA weights)

## Features

- Predictors: REVEL, CADD, SIFT, PolyPhen2, GERP++, phyloP
- Structural (AlphaFold): SASA, relative SASA, pLDDT, domain/site flags
- Protein LM: ESM-2 (`esm2_t33_650M_UR50D`, 1280-d) embeddings + variant score
- Cancer annotations: cancer-gene / tier1 / oncogene / TSG flags

Leakage-prone columns are dropped before modeling. A leakage-audit config also ablates the meta-predictor scores to measure Type-1 circularity.

## Pipeline

Scripts in `src/` run in order. Each reads the previous output and writes to its own `outputs/` subfolder.

| # | Script | Stage |
| --- | --- | --- |
| 01 | `01_dbnsfp_processor.py` | Parse/filter dbNSFP |
| 02 | `02_remove_missing_values.py` | Drop missing values |
| 03 | `03_remove_duplicates.py` | Deduplicate variants |
| 04 | `04_feature_engineering.py` | UniProt + AlphaFold features |
| 05 | `05_remove_leakage.py` | Drop leaking columns |
| 06 | `06_clean_and_finalize.py` | Final cleaning |
| 07 | `07_dataset_balancing.py` | Balance classes |
| 08 | `08_prepare_esm_dataset.py` | Map sequences, ref-AA check |
| 09 | `09_prepare_external_esm_dataset.py` | Build external ClinVar/DMS set |
| 10 | `10_extract_esm_features.py` | ESM-2 embeddings (internal + external) |
| 11 | `11_train_and_evaluate.py` | LightGBM vs Cross-Attention (gene-level CV, SHAP, McNemar) |
| 12 | `12_external_validation.py` | Gene-disjoint external validation |
| 13 | `13_generate_figures.py` | Figures |

Stage 09 runs before 10 so both sets get embedded in one pass:

```bash
python src/10_extract_esm_features.py --dataset both
```

## Setup

```bash
git clone https://github.com/asifahamed11/VariFuse.git
cd VariFuse
pip install -r requirements.txt
export VARIFUSE_DATA_DIR=/path/to/your/Datasets
```

ESM extraction uses `fair-esm` + `torch`. A CUDA GPU is recommended (FP16 auto). See [`data/README.md`](data/README.md) for the expected data layout.

## Run

```bash
python src/09_prepare_external_esm_dataset.py
python src/10_extract_esm_features.py --dataset both
python src/11_train_and_evaluate.py
python src/12_external_validation.py
python src/13_generate_figures.py
```

## Evaluation

5-fold gene-level `StratifiedGroupKFold` with a nested inner fold for early stopping. Thresholds maximize MCC subject to recall ≥ 0.90, falling back to max-F₂. Reports MCC, AUROC, AUPRC, Brier, precision/recall/F1, and a McNemar test between the two models. SHAP is computed for LightGBM.

## Outputs

- `outputs/10_model_evaluation/` , `results.json`, `comparison_table.csv`, `oof_predictions.npz`, `shap_values_*.npz`
- `outputs/12_external_validation/` , `external_validation.json`, `external_validation_table.csv`, `external_predictions.npz`
- `figures/` , `fig01`–`fig12`

## License

MIT. See [LICENSE](LICENSE).
