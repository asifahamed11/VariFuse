# Datasets

This folder is intended to hold the raw data files required to run the pipeline, such as the dbNSFP database, hg19.fa reference genome, etc.

**Note:** The actual dataset files are excluded from this Git repository via `.gitignore` because they are extremely large (76GB+).

To reproduce this project:
1. Download the datasets from [Insert Google Drive or source link here]
2. Place the datasets directly into this `data/` folder.
3. Ensure that `config.py` has `DATA_DIR` pointing to this folder.
