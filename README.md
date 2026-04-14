# TF-DFE: Topology-Guided Multi-View Ensemble Learning for Evidence-Aware Somatic Variant Pathogenicity Classification in Cancer Genomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paper:** Topology-Guided Multi-View Ensemble Learning for Evidence-Aware Somatic Variant Pathogenicity Classification in Cancer Genomics   
> **Authors:** Asif Ahamed, Md. Tanvir Hasan, Most. Alisa Tabassum, Ahammad Hossain*, A.H.M. Rahmatullah Imon  
> **Corresponding Author:** Ahammad Hossain — ahammadstatru@gmail.com

---

## Overview

TF-DFE (Topo-Fractal Dynamic Fuzzy Ensemble) is a scalable, evidence-aware framework for classifying somatic single nucleotide variants (SNVs) as pathogenic drivers or benign passengers. The framework combines:

- **Standard biological predictors** - REVEL, CADD, SIFT, PolyPhen2, GERP++, phyloP, structural/UniProt features
- **Fractal sequence features** - Frequency Chaos Game Representation (FCGR) at k=3 and k=4 (320 dimensions)
- **Topological features** - Persistent homology via giotto-tda (6 dimensions)
- **Dynamic ensemble selection** - Enhanced KNORA-Eliminate strategy

**Final dataset:** 207,266 balanced somatic SNVs curated from dbNSFP v5.3a (GRCh37)  
**Performance:** MCC = 0.8702 | AUROC = 0.9666 | AUPRC = 0.9768 | Accuracy = 93.48%

---

## Repository Structure

```
TF-DFE/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_somatic_variant_processor.py       # dbNSFP parsing, CIViC/COSMIC rescue, labeling
│   ├── 02_remove_missing_values.py           # Selective missing value removal
│   ├── 03_remove_duplicates.py               # Deduplication on chr/pos/ref/alt
│   ├── 04_structural_feature_engineering.py  # UniProt + AlphaFold2 + SASA features
│   ├── 05_remove_leakage_columns.py          # Leakage column removal
│   ├── 06_dataset_balancing.py               # Downsample to balanced 207,266 variants
│   ├── 07_eda.py                             # Exploratory data analysis + figures
│   └── 08_tf_dfe_model.py                    # Main TF-DFE model, evaluation, SHAP
└── data/
    └── Final_Dataset.csv                     # Balanced dataset (207,266 variants, 24 features)
```

---

## Pipeline

Run the scripts in order. Each script outputs a CSV that is used as input to the next step.

```
dbNSFP5.3a_grch37.gz
        │
        ▼
01_somatic_variant_processor.py  →  somatic_variant_dbNSFP.csv  (19,613,687 variants)
        │
        ▼
02_remove_missing_values.py      →  somatic_variant_dbNSFP_Removes_missing_values.csv  (9,241,413)
        │
        ▼
05_remove_leakage_columns.py     →  somatic_variant_dbNSFP_Removes_leakage.csv
        │
        ▼
03_remove_duplicates.py          →  somatic_variant_dbNSFP_Removes_leakage_missing_values_Deduplication.csv  (6,461,982)
        │
        ▼
04_structural_feature_engineering.py  →  Final_Dataset_UniProt_Removes_leakage.csv  (adds UniProt/AlphaFold2 features)
        │
        ▼
06_dataset_balancing.py          →  data/Final_Dataset.csv  (207,266 balanced variants)
        │
        ▼
07_eda.py                        →  EDA_Results/  (figures and statistics)
        │
        ▼
08_tf_dfe_model.py               →  results_publication/  (model results, figures, SHAP)
```

---

## External Data Requirements

These large/licensed files must be downloaded separately before running the pipeline.

| File | Source | Used In |
|------|--------|---------|
| `dbNSFP5.3a_grch37.gz` | [dbNSFP](https://dbnsfp.org) | Script 01 |
| `01-Jan-2026-VariantSummaries.tsv` | [CIViC Releases](https://civicdb.org/releases) | Script 01 |
| `Cosmic_CancerGeneCensus_v102_GRCh37.tsv` | [COSMIC](https://cancer.sanger.ac.uk/cosmic) | Script 01 |
| `cmc_export.tsv` | [COSMIC Mutant Census](https://cancer.sanger.ac.uk/cosmic) | Script 01 |
| `oncokb_biomarker_drug_associations.tsv` | [OncoKB](https://www.oncokb.org) | Script 01 |
| `uniprot_sprot_human.dat` | [UniProt](https://www.uniprot.org/downloads) | Script 04 |
| AlphaFold2 PDB files (`AF-*-model_v*.pdb`) | [AlphaFold DB](https://alphafold.ebi.ac.uk/download) | Script 04 |

> **Note:** Place all external files in the same directory as the scripts before running, or update the file paths inside each script accordingly.

---

## Installation

```bash
git clone https://github.com/asifahamed11/TF-DFE.git
cd TF-DFE
pip install -r requirements.txt
```

Python 3.8 or higher is required.

---

## Quick Start (Skip to Model Training)

If you only want to reproduce the model results using the pre-processed dataset:

```bash
# Dataset is already available in data/Final_Dataset.csv
# Update DATA_PATH in script 08 if needed, then run:
python scripts/08_tf_dfe_model.py
```

Results will be saved to `results_publication/`.

---

## Dataset

The final balanced dataset (`data/Final_Dataset.csv`) contains **207,266 somatic SNVs** (103,633 pathogenic / 103,633 benign) with the following features:

| Feature Group | Features | Dimensions |
|---------------|----------|------------|
| Genomic coordinates | chr, pos, ref, alt | 4 |
| Pathogenicity scores | REVEL, CADD, SIFT, PolyPhen2, GERP++, phyloP | 6 |
| Cancer gene annotations | IS_CANCER_GENE, IS_TIER1, IS_ONCOGENE, IS_TSG | 4 |
| Structural features | SASA, RELATIVE_SASA, PLDDT_SCORE | 3 |
| UniProt functional sites | IS_IN_DOMAIN, IS_ACTIVE_SITE, IS_BINDING_SITE, IS_TRANSMEMBRANE, DISTANCE_TO_ACTIVE_SITE | 5 |
| Label | LABEL_PATHOGENIC (0=benign, 1=pathogenic) | 1 |

FCGR (320-dim) and TDA (6-dim) features are computed on-the-fly inside `08_tf_dfe_model.py`.

---

## Reproducing Paper Figures

All figures in the manuscript are generated by `08_tf_dfe_model.py` and saved to `results_publication/` as both `.png` and `.tiff`.

EDA figures (Fig 4a, 4b, 4c, 5, 5b) are generated by `07_eda.py` and saved to `EDA_Results/`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Ahammad Hossain** (Corresponding Author)  
Department of Computer Science & Engineering, Varendra University, Bangladesh  
Email: ahammadstatru@gmail.com  
ORCID: [0000-0001-9081-5905](https://orcid.org/0000-0001-9081-5905)
