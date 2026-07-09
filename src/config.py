from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VARIFUSE_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Output folders.
#
# NOTE: the *constant names* (STAGExx_OUT) are semantic anchors used across the
# code and MUST stay stable. After the pipeline was re-numbered, the script
# filename number no longer always equals the folder number. Mapping below so
# nobody gets confused:
#
#   Script file (new #)              reads from        writes to
#   -------------------------------  ----------------  ----------------------
#   01_dbnsfp                        (raw dbNSFP)      STAGE01_OUT
#   02_missing_values                STAGE01_OUT       STAGE02_OUT
#   03_duplicates                    STAGE02_OUT       STAGE03_OUT
#   04_feature_engineering           STAGE03_OUT       STAGE04_OUT
#   05_leakage                       STAGE04_OUT       STAGE05_OUT
#   06_clean                         STAGE05_OUT       STAGE06_OUT
#   07_dataset_balancing             STAGE06_OUT       STAGE07_OUT
#   08_prepare_esm_dataset           STAGE07_OUT       STAGE08_OUT
#   09_prepare_external_esm_dataset  STAGE04_OUT/data  DATA_DIR/external
#   10_extract_esm_features          STAGE08_OUT +     STAGE09_OUT +
#                                    DATA_DIR/external  DATA_DIR/external
#   11_train_and_evaluate            STAGE09_OUT       STAGE10_OUT
#   12_external_validation           STAGE09_OUT +     STAGE12_OUT
#                                    DATA_DIR/external
#   13_generate_figures              many + STAGE12    STAGE13_OUT (figures)
# ---------------------------------------------------------------------------
STAGE01_OUT = OUTPUT_DIR / "01_dbnsfp"
STAGE02_OUT = OUTPUT_DIR / "02_missing_values"
STAGE03_OUT = OUTPUT_DIR / "03_duplicates"
STAGE04_OUT = OUTPUT_DIR / "04_feature_engineering"
STAGE05_OUT = OUTPUT_DIR / "05_leakage"
STAGE06_OUT = OUTPUT_DIR / "06_clean"
STAGE07_OUT = OUTPUT_DIR / "07_balancing"
STAGE08_OUT = OUTPUT_DIR / "08_prepare_esm"
STAGE09_OUT = OUTPUT_DIR / "09_esm_features"
STAGE10_OUT = OUTPUT_DIR / "10_model_evaluation"
# 11_train_and_evaluate writes into STAGE10_OUT (model eval artifacts).
# STAGE11_OUT is defined for completeness / future use; no stage writes here yet.
STAGE11_OUT = OUTPUT_DIR / "11_train_and_evaluate"
STAGE12_OUT = OUTPUT_DIR / "12_external_validation"
STAGE13_OUT = PROJECT_ROOT / "figures"
EDA_OUT = OUTPUT_DIR / "eda_figures"

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
    STAGE10_OUT,
    STAGE11_OUT,
    STAGE12_OUT,
    STAGE13_OUT,
    EDA_OUT,
]:
    d.mkdir(parents=True, exist_ok=True)
