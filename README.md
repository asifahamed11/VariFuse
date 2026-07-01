# Pathogenicity Prediction Framework

This repository contains the preprocessing, feature engineering, and modeling scripts for the pathogenicity prediction project.

## Repository Structure

- `src/` : Contains all Python scripts, numbered in execution order (01 to 10).
- `src/config.py` : Centralized configuration for input and output paths.
- `data/` : Directory intended for raw datasets (e.g., dbNSFP, hg19.fa). Note that datasets are excluded from Git due to their size.
- `outputs/` : Directory for all generated CSVs, models, and figures.

## Data Setup

Due to GitHub file size limits, the 76GB of raw data is not tracked in this repository.

1. Download the datasets: **[TODO: paste your Google Drive link here]**
2. Place them in the `data/` folder (see `data/README.md` for the expected layout), **or**
3. Point at an existing local copy without touching any code: `export TFDFE_DATA_DIR=/path/to/your/Datasets` before running the scripts.

## Installing dependencies

`pip install -r requirements.txt`

Note: `biopython`, `giotto-tda`, and `pyfaidx` are core to the method (structural + topological features in scripts 04 and 08), not optional extras — install them even though the code itself guards each import with try/except for graceful fallback.

## Running the Pipeline

Run the scripts in numerical order from the `src/` directory. All outputs are saved automatically into organized subfolders within `outputs/` (created on first run).
