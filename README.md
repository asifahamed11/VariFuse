# TF-DFE: ESM-2 Cross-Attention Fusion vs Gradient Boosting for Evidence-Aware Somatic Variant Pathogenicity Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/github/license/asifahamed11/TF-DFE)

This repository contains the preprocessing, feature engineering, protein-language-model
embedding, modeling, external validation, and figure-generation scripts for an
evidence-aware somatic variant pathogenicity classifier.

Two models are trained and compared **on an identical feature set, CV protocol, and
recall-aware decision threshold** so the comparison stays fair:

- **LightGBM** gradient-boosted trees over biological + structural + ESM-derived features.
- **Cross-Attention Fusion network** that tokenizes scalar features and cross-attends
  them with ESM-2 residue-embedding slot tokens (seed-averaged ensemble, EMA weights).

## Feature views

- **Biological predictors:** REVEL, CADD, SIFT, PolyPhen2, plus evolutionary
  conservation (GERP++, phyloP).
- **Structural / functional (3D):** AlphaFold-derived SASA, relative SASA, pLDDT,
  domain / active-site / binding-site / transmembrane flags, distance to active site.
- **Protein language model:** ESM-2 (`esm2_t33_650M_UR50D`, 1280-dim) residue
  embeddings and a wt-marginal `esm_variant_score` per variant.
- **Cancer annotations:** IS_CANCER_GENE, IS_TIER1, IS_ONCOGENE, IS_TSG.

Leakage-prone columns (raw coordinates, consensus score, tier, domain name, etc.) are
dropped before modeling, and a **leakage-audit** configuration additionally ablates the
meta-predictor scores (SIFT/PolyPhen2/CADD/REVEL) to quantify Type-1 circularity.

## Repository structure

```text
TF-DFE/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                        # centralized input/output paths (STAGExx_OUT anchors)
│   ├── common.py                        # shared schema, models, training, metrics, artifacts
│   ├── 01_dbnsfp.py                     # parse/filter dbNSFP into the base variant table
│   ├── 02_missing_values.py             # selective missing-value removal
│   ├── 03_duplicates.py                 # deduplicate on chr/pos/ref/alt
│   ├── 04_feature_engineering.py        # UniProt + AlphaFold structural/functional features
│   ├── 05_leakage.py                    # drop label-leaking annotation columns
│   ├── 06_clean.py                      # final cleaning and column preparation
│   ├── 07_dataset_balancing.py          # downsample to balanced pathogenic/benign
│   ├── 08_prepare_esm_dataset.py        # map UniProt sequences, ref-AA consistency check
│   ├── 09_prepare_external_esm_dataset.py  # independent ClinVar/DMS external set (dbNSFP-scanned)
│   ├── 10_extract_esm_features.py       # ESM-2 residue embeddings + variant score (internal+external)
│   ├── 11_train_and_evaluate.py         # LightGBM vs Cross-Attention, gene-level CV, SHAP, McNemar
│   ├── 12_external_validation.py        # frozen-threshold, gene-disjoint external validation
│   └── 13_generate_figures.py           # publication figures (PNG)
└── data/                                # raw + external inputs (git-ignored, see data/README.md)
```

## Pipeline stages

Scripts run in numerical order. Each reads the previous stage's output and writes to its
own `outputs/` subfolder (folder names are fixed by `config.py`; a script's filename
number does not always equal its output-folder number after the external branch was
inserted).

| Script | Stage |
| --- | --- |
| `01_dbnsfp.py` | Parse and filter dbNSFP into the base variant table |
| `02_missing_values.py` | Drop rows with missing values (outside retained columns) |
| `03_duplicates.py` | Remove duplicate variants |
| `04_feature_engineering.py` | UniProt + AlphaFold structural / functional features |
| `05_leakage.py` | Drop label-leaking annotation columns |
| `06_clean.py` | Final cleaning and column preparation |
| `07_dataset_balancing.py` | Balance the pathogenic / benign classes |
| `08_prepare_esm_dataset.py` | Map UniProt sequences; drop isoform/position mismatches |
| `09_prepare_external_esm_dataset.py` | Build an independent ClinVar/DMS external set with its own dbNSFP-derived features |
| `10_extract_esm_features.py` | Extract ESM-2 residue embeddings + variant score (internal **and** external) |
| `11_train_and_evaluate.py` | Train/evaluate LightGBM vs Cross-Attention (gene-level CV, SHAP, McNemar) |
| `12_external_validation.py` | Gene-disjoint external validation at a frozen internal threshold |
| `13_generate_figures.py` | Render all publication figures (PNG) |

### Run order

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
```

**Stage 09 runs before Stage 10 on purpose.** The external-set preparation writes
`data/external/external_ready_for_esm.csv`, so by the time you reach the ESM extractor you
can embed the internal and external sets in a single pass (the 650M model loads only once):

```bash
python 10_extract_esm_features.py            # defaults to --dataset both
# equivalent to: python 10_extract_esm_features.py --dataset both
```

## External validation

Stage 12 evaluates the frozen-threshold models on a **gene-disjoint** external set built
independently in Stage 09 from ClinVar and DMS, with its own dbNSFP-scanned features so no
internal feature values leak in. It removes any external gene seen in training and, with
`STRICT_VARIANT_DEOVERLAP`, also drops exact `chr:pos:ref:alt` matches. Both the primary
(`bio_full`) and leakage-audit (`bio_minus_predictor_scores`) configurations are reported,
and external probability vectors are saved for the external ROC/PR/calibration figures.

## Installing dependencies

```bash
pip install -r requirements.txt
```

ESM-2 feature extraction (Stage 10) uses `fair-esm` and `torch`; a CUDA GPU is strongly
recommended (FP16 is enabled automatically when CUDA is available). LightGBM and SHAP are
used in Stage 11.

## Data setup

Raw and external inputs are large and **not tracked in Git**. Point the pipeline at a local
copy without editing any code:

```bash
export TFDFE_DATA_DIR=/path/to/your/Datasets
```

See **[`data/README.md`](data/README.md)** for the expected folder layout, including the
`data/external/` inputs consumed by Stages 09 and 12.

## Outputs

All artifacts land in `outputs/` (created on first run):

- `10_model_evaluation/` → `results.json`, `comparison_table.csv`, `oof_predictions.npz`,
  `shap_values_*.npz`
- `12_external_validation/` → `external_validation.json`, `external_validation_table.csv`,
  `external_predictions.npz`
- `figures/` → `fig01`–`fig12` PNGs (label evidence, correlation, structural violins,
  ESM embedding projection, ROC/PR, confusion matrices, SHAP, Type-1 circularity, and the
  external ROC/PR/calibration analogues)

## License

MIT. See [LICENSE](LICENSE).
