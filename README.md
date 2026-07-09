# VariFuse

**Evidence-aware somatic variant pathogenicity classifier** that fuses ESM-2 protein
language-model embeddings with tabular biological, structural, and predictor features.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ESM--2](https://img.shields.io/badge/ESM--2-650M-8A2BE2)

VariFuse trains **two models on an identical feature set, cross-validation split, and
recall-aware decision threshold** so the comparison between them is genuinely fair:

- **LightGBM** — gradient-boosted trees over biological, structural, and ESM-derived features.
- **Cross-Attention Fusion Net** — a transformer that cross-attends scalar feature tokens
  against ESM-2 residue-embedding tokens (seed-averaged ensemble with EMA weights).

Everything is built around **methodological rigor**: gene-disjoint external validation,
a leakage audit that quantifies Type-1 circularity, and nested cross-validation with
gene-level grouping to prevent train/test contamination.

---

## Why this repo is different

Most variant-effect benchmarks quietly leak information: the same gene shows up in train
and test, or meta-predictor scores (REVEL, CADD, ...) were themselves trained on the labels
you're now evaluating against. VariFuse is built to expose and control for this:

- **Gene-level CV** (`StratifiedGroupKFold`) — no gene appears in both train and validation.
  Disjointness is asserted at runtime, not assumed.
- **Leakage audit** — every experiment is run twice: once with meta-predictor scores
  (`bio_full`) and once with them ablated (`bio_minus_predictor_scores`). The difference in
  MCC is reported directly as **Type-1 MCC inflation**.
- **Gene-disjoint external validation** — frozen-threshold models are tested on an
  independent ClinVar/DMS set. Shared genes (and, optionally, exact `chr:pos:ref:alt`
  matches) are removed before scoring.
- **Recall-aware thresholds** — a clinical setting cares about not missing pathogenic
  variants, so the operating point maximizes MCC subject to **recall ≥ 0.90**.

---

## Model architecture

The Cross-Attention Fusion Net (`src/common.py`) is a higher-capacity design that actually
uses the full 1280-dim ESM-2 residue embedding rather than collapsing it to a scalar:

```
scalar features ──▶ FeatureTokenizer (FT-Transformer style, 1 token/feature)
ESM-2 embedding ──▶ Linear projection ──▶ 16 slot tokens
        │
        ▼
  3 × bidirectional Cross-Attention blocks   (bio ⇄ esm, each with FFN + LayerNorm)
        │
        ▼
  [CLS | bio tokens | esm tokens] ──▶ 2 × self-attention fusion layers
        │
        ▼
  readout = concat(CLS, mean-pool) ──▶ MLP head ──▶ logit
```

| Component | Value |
| --- | --- |
| Token width (`D_MODEL`) | 160 |
| Attention heads | 8 |
| Cross-attention blocks | 3 (bidirectional bio ⇄ esm) |
| Self-attention fusion layers | 2 |
| ESM slot tokens | 16 |
| Dropout | 0.15 |

**Training recipe:** AdamW, linear warmup (5 epochs) → cosine decay, EMA weights
(decay 0.999), input mixup (α = 0.2), light label smoothing (0.02), gradient clipping (1.0),
`pos_weight`-balanced BCE loss, and AUROC-based early stopping evaluated on the EMA weights.
Each fold trains a **3-member seed-averaged ensemble** (logit averaging) for stability.

LightGBM hyperparameters are fixed and shared across all configs (`n_estimators=3000`,
`learning_rate=0.02`, `num_leaves=63`, early stopping at 150 rounds) so it never gets an
unfair tuning advantage over the deep model.

---

## Features

- **Meta-predictors:** REVEL, CADD, SIFT, PolyPhen2, GERP++, phyloP
- **Structural (AlphaFold):** SASA, relative SASA, pLDDT, domain / site flags
- **Protein LM:** ESM-2 (`esm2_t33_650M_UR50D`, 1280-d) residue embeddings + wt-marginal
  variant score (log-ratio of alt vs. ref amino-acid likelihood at the mutated position)
- **Cancer annotations:** cancer-gene / tier1 / oncogene / TSG flags

Known leakage-prone columns (`chr`, `pos`, `ref`, `alt`, `CONSENSUS_SCORE`, `TIER`,
`DOMAIN_NAME`, `SECONDARY_STRUCTURE`, `IS_CLINVAR_BENIGN`) are dropped before modeling, and
the DMS score is treated strictly as a label source, never a feature.

---

## Pipeline

Scripts in `src/` run in order. Each reads the previous stage's output and writes to its own
`outputs/` subfolder (folder numbering is fixed in `config.py`).

| # | Script | Stage |
| --- | --- | --- |
| 01 | `01_dbnsfp_processor.py` | Parse / filter dbNSFP |
| 02 | `02_remove_missing_values.py` | Drop missing values |
| 03 | `03_remove_duplicates.py` | Deduplicate variants |
| 04 | `04_feature_engineering.py` | UniProt + AlphaFold structural features |
| 05 | `05_remove_leakage.py` | Drop leaking columns |
| 06 | `06_clean_and_finalize.py` | Final cleaning |
| 07 | `07_dataset_balancing.py` | Balance classes |
| 08 | `08_prepare_esm_dataset.py` | Map sequences, reference-AA sanity check |
| 09 | `09_prepare_external_esm_dataset.py` | Build independent ClinVar/DMS external set |
| 10 | `10_extract_esm_features.py` | ESM-2 embeddings (internal + external) |
| 11 | `11_train_and_evaluate.py` | LightGBM vs Cross-Attention (gene-level CV, SHAP, McNemar) |
| 12 | `12_external_validation.py` | Gene-disjoint external validation |
| 13 | `13_generate_figures.py` | Publication figures (PNG) |

> **Run order note:** Stage 09 runs *before* Stage 10 so both the internal and external
> sets are embedded in a single pass (the ESM-2 model is loaded once):
> ```bash
> python src/10_extract_esm_features.py --dataset both   # default
> ```

---

## Installation

```bash
git clone https://github.com/asifahamed11/VariFuse.git
cd VariFuse
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

**GPU strongly recommended.** Stage 10 (ESM-2 extraction) uses `fair-esm` + `torch`; a CUDA
GPU is auto-detected and runs in FP16. On CPU it will work but be slow. For a specific CUDA
build of PyTorch, follow the selector at
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) before installing the
rest of the requirements.

---

## Data setup

Raw inputs are large and **not tracked in Git**. Point the pipeline at a local copy:

```bash
export VARIFUSE_DATA_DIR=/path/to/your/Datasets
```

`src/config.py` reads this automatically and falls back to the repo's `data/` folder.
See **[`data/README.md`](data/README.md)** for the full expected layout and download link.

---

## Quick start

```bash
# 1. Feature pipeline (stages 01–08)
python src/01_dbnsfp_processor.py
# ... run 02 through 08 in order ...

# 2. Build the external set, then embed both sets in one pass
python src/09_prepare_external_esm_dataset.py
python src/10_extract_esm_features.py --dataset both

# 3. Train + evaluate (LightGBM vs Cross-Attention, gene-level CV)
python src/11_train_and_evaluate.py

# 4. External validation + figures
python src/12_external_validation.py
python src/13_generate_figures.py
```

---

## Evaluation & decision policy

- **Cross-validation:** 5-fold `StratifiedGroupKFold` grouped by gene, with a nested inner
  fold for deep-model early stopping and LightGBM's validation set. Gene disjointness between
  train and validation is asserted every fold.
- **Threshold selection:** maximize **MCC subject to recall ≥ 0.90**; if that recall floor is
  unreachable, fall back to the max-F₂ threshold.
- **Selective abstention:** predictions within ±0.10 of the threshold are treated as
  low-confidence; a "confident-only" metric block is reported alongside the full metrics.
- **Metrics:** MCC, AUROC, AUPRC, Brier score, precision, recall, F1, plus a **McNemar test**
  on the paired thresholded predictions of the two models.
- **Interpretability:** SHAP values are computed for the LightGBM model and saved for the
  figure stage.

---

## Outputs

- `outputs/10_model_evaluation/` → `results.json`, `comparison_table.csv`,
  `oof_predictions.npz`, `shap_values_*.npz`
- `outputs/12_external_validation/` → `external_validation.json`,
  `external_validation_table.csv`, `external_predictions.npz`
- `figures/` → `fig01`–`fig12` (ROC/PR, confusion matrices, SHAP, Type-1 circularity,
  external-set analogues)

`results.json` includes a `type1_mcc_inflation` block reporting, per model, how much MCC is
attributable to meta-predictor leakage (primary − leakage-audit MCC).

---

## Project structure

```
VariFuse/
├── src/
│   ├── config.py                 # paths, stage-folder mapping (env-driven)
│   ├── common.py                 # fusion net, training loop, metrics, thresholding
│   ├── 01_dbnsfp_processor.py
│   ├── ... (stages 02–09)
│   ├── 10_extract_esm_features.py
│   ├── 11_train_and_evaluate.py
│   ├── 12_external_validation.py
│   └── 13_generate_figures.py
├── data/                         # inputs (git-ignored) — see data/README.md
├── outputs/                      # generated per-stage artifacts
├── figures/                      # generated publication figures
├── requirements.txt
└── README.md
```

---

## Reproducibility

All stochastic components are seeded (`RANDOM_STATE = 42`): NumPy, PyTorch, CV splits, and
ensemble members (each member uses a distinct offset seed). TF32 is enabled on CUDA for
throughput; expect run-to-run variation to be negligible but not bit-exact on GPU.

---

## License

Released under the [MIT License](LICENSE).
