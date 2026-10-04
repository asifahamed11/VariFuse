"""
VariFuse Full Pipeline Runner for Kaggle.
Runs Stage 14 -> Stage 11 -> Stage 12 -> Stage 13 sequentially,
and packages the final results into a single download zip.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/kaggle/working/VariFuse_2")


def run_stage_command(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f">>> STARTING: {name}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"{'='*70}\n", flush=True)
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    elapsed = round(time.time() - t0, 1)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} FAILED with exit code {proc.returncode} (elapsed: {elapsed}s)")
    print(f"\n>>> {name} FINISHED SUCCESSFULLY in {elapsed}s <<<\n", flush=True)


def main() -> None:
    env = os.environ.copy()

    # Auto-detect DMS data presence
    dms_embeddings = ROOT / "outputs" / "10_esm_features" / "dms_esm_embeddings.npy"
    has_dms = dms_embeddings.is_file() and dms_embeddings.stat().st_size > 1000

    env.update({
        "VARIANT_PROJECT_ROOT": str(ROOT),
        "VARIANT_OUTPUT_DIR": str(ROOT / "outputs"),
        "VARIANT_FIGURE_DIR": str(ROOT / "figures"),
        "TORCH_HOME": str(ROOT / "model_cache"),
        "VARIANT_LABEL_TASK": "clinical",
        "VARIANT_TRAIN_CUTOFF_DATE": "2024-06-30",
        "CLINVAR_TRAIN_RELEASE": "2024-06",
        "CLINVAR_EXTERNAL_RELEASE": "2026-08",
        "DBNSFP_RELEASE": "5.3a",
        "PROTEINGYM_RELEASE": "v1.3",
        "VARIANT_RANDOM_STATE": "42",
        "REQUIRE_TRANSCRIPT_MAPPING": "1",
        "REQUIRE_HOMOLOGY_GROUPS": "1",
        "REQUIRE_TUNING_ARTIFACT": "1",
        "ALLOW_LEGACY_MIXED_LABELS": "0",
        "ALLOW_POST_CUTOFF_TRAINING_EVIDENCE": "0",
        "VARIANT_REPRODUCIBLE": "1",
        "VARIANT_HASH_LARGE_FILES": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "VARIFUSE_USE_ALL_GPUS": "1",
        "VARIFUSE_REQUIRED_GPU_COUNT": "2",
        "VARIFUSE_MAX_GPU_COUNT": "2",
        "VARIFUSE_GPU_NAME_CONTAINS": "T4",
        "ESM_DEVICE": "cuda",
        "PYTHONUNBUFFERED": "1",
        "REQUIRE_EXTERNAL_CLINVAR": "1",
        "REQUIRE_EXTERNAL_DMS": "1" if has_dms else "0",
    })

    print(f"Pipeline initialized at: {ROOT}")
    print(f"External DMS embeddings present: {has_dms}")

    # =========================================================================
    # STEP 1: Stage 14 (Hyperparameter Tuning & Confirmatory Repeated CV)
    # =========================================================================
    stage14_script = ROOT / "src" / "14_tune_cross_attention.py"
    run_stage_command(
        "STAGE 14: TUNE ARCHITECTURES & CONFIRMATORY REPEATED CV",
        [
            sys.executable,
            "-u",
            str(stage14_script),
            "--confirmation-split-seeds",
            "1701",
            "2903",
            "4159",
            "6841",
            "7919",
            "--resume",
        ],
        env,
    )

    # =========================================================================
    # STEP 2: Stage 11 (Train & Evaluate Final Deployment Models)
    # =========================================================================
    stage11_script = ROOT / "src" / "11_train_and_evaluate.py"
    run_stage_command(
        "STAGE 11: TRAIN & EVALUATE FINAL MODELS",
        [sys.executable, "-u", str(stage11_script)],
        env,
    )

    # =========================================================================
    # STEP 3: Stage 12 (External Validation)
    # =========================================================================
    clinvar_ext = ROOT / "outputs" / "10_esm_features" / "clinvar_with_esm.parquet"
    if clinvar_ext.is_file():
        stage12_script = ROOT / "src" / "12_external_validation.py"
        run_stage_command(
            "STAGE 12: EXTERNAL VALIDATION (CLINVAR 2026-08)",
            [sys.executable, "-u", str(stage12_script)],
            env,
        )

        # =====================================================================
        # STEP 4: Stage 13 (Generate Publication Figures & Tables)
        # =====================================================================
        stage13_script = ROOT / "src" / "13_generate_figures.py"
        run_stage_command(
            "STAGE 13: GENERATE PUBLICATION FIGURES & TABLES",
            [sys.executable, "-u", str(stage13_script)],
            env,
        )
    else:
        print("\nNote: Stage 12 external dataset not found in package; skipping Stages 12 and 13 on Kaggle.", flush=True)

    # =========================================================================
    # STEP 5: Auto-Package All Outputs into a Single Zip
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print("ALL STAGES FINISHED! Packaging outputs for easy download ...", flush=True)
    archive_base = Path("/kaggle/working/VariFuse_Complete_Outputs")

    # Collect outputs and figures
    export_dir = Path("/kaggle/working/export_staging")
    export_dir.mkdir(parents=True, exist_ok=True)

    if (ROOT / "outputs").exists():
        shutil.copytree(ROOT / "outputs", export_dir / "outputs", dirs_exist_ok=True)
    if (ROOT / "figures").exists():
        shutil.copytree(ROOT / "figures", export_dir / "figures", dirs_exist_ok=True)

    shutil.make_archive(str(archive_base), "zip", str(export_dir))
    shutil.rmtree(export_dir, ignore_errors=True)

    zip_path = Path(str(archive_base) + ".zip")
    size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2) if zip_path.exists() else 0
    print("\n**********************************************************************")
    print("SUCCESS! Complete pipeline outputs saved at:")
    print(f"  {zip_path} ({size_mb} MB)")
    print("Download this file from the Kaggle Output panel.")
    print("**********************************************************************\n", flush=True)


if __name__ == "__main__":
    main()
