# Pathogenicity Prediction Framework

This repository contains the preprocessing, feature engineering, and modeling scripts for the pathogenicity prediction project.

## Repository Structure

- `src/` : Contains all Python scripts, numbered in execution order (01 to 10).
- `src/config.py` : Centralized configuration for input and output paths.
- `data/` : Directory intended for raw datasets (e.g., dbNSFP, hg19.fa). Note that datasets are excluded from Git due to their size.
- `outputs/` : Directory for all generated CSVs, models, and figures.

## Data Setup

Due to GitHub file size limits, the 76GB of raw data is not tracked in this repository.
Before running the scripts, please download the required datasets (e.g., dbNSFP database, hg19.fa, etc.) and place them in the `data/` folder, or update the `DATA_DIR` path in `src/config.py` to point to your existing dataset location.

## Running the Pipeline

Run the scripts in numerical order from the `src/` directory. All outputs will automatically be saved into organized subfolders within the `outputs/` directory.
