from pathlib import Path
import os

# Project root is the parent directory of the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Since datasets are currently kept in the original folder, we point there
# When uploading to GitHub, this should be changed to PROJECT_ROOT / "data"
# For local execution, it points to the old Datasets folder
DATA_DIR = Path(r"C:\Users\Admin\Desktop\Code\Datasets")
# Or fallback to local data/ folder if the above doesn't exist
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "data"

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
EDA_OUT     = OUTPUT_DIR / "eda_figures"

# Ensure all output directories exist when config is imported
for d in [STAGE01_OUT, STAGE02_OUT, STAGE03_OUT, STAGE04_OUT, STAGE05_OUT,
          STAGE06_OUT, STAGE07_OUT, STAGE08_OUT, STAGE09_OUT, EDA_OUT]:
    d.mkdir(parents=True, exist_ok=True)
