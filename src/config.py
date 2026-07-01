from pathlib import Path
import os

# Project root is the parent directory of the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw datasets (dbNSFP, hg19.fa, AlphaFold structures, etc).
# Default: <repo>/data  (see data/README.md for the Google Drive download link).
# Override without editing this file: set env var TFDFE_DATA_DIR to your local path.
DATA_DIR = Path(os.environ.get("TFDFE_DATA_DIR", PROJECT_ROOT / "data"))

# Centralized Outputs Directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Individual Stage Outputs
STAGE01_OUT = OUTPUT_DIR / "01_dbnsfp"
STAGE02_OUT = OUTPUT_DIR / "02_missing_values"
STAGE03_OUT = OUTPUT_DIR / "03_duplicates"
STAGE04_OUT = OUTPUT_DIR / "04_feature_engineering"
STAGE05_OUT = OUTPUT_DIR / "05_leakage"
STAGE06_OUT = OUTPUT_DIR / "06_clean"
STAGE07_OUT = OUTPUT_DIR / "07_balancing"
STAGE08_OUT = OUTPUT_DIR / "08_tda_fuzzy"
STAGE09_OUT = OUTPUT_DIR / "09_experiments"
EDA_OUT = OUTPUT_DIR / "eda_figures"

# Ensure all output directories exist when config is imported
for d in [
    STAGE01_OUT,
    STAGE02_OUT,
    STAGE03_OUT,
    STAGE04_OUT,
    STAGE05_OUT,
    STAGE06_OUT,
    STAGE07_OUT,
    STAGE08_OUT,
    STAGE09_OUT,
    EDA_OUT,
]:
    d.mkdir(parents=True, exist_ok=True)
