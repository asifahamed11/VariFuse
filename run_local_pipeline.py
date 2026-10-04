"""
VariFuse Local Automated Pipeline Runner (Windows / GTX 1660).
Executes Stage 14 (resumed) -> Stage 11 -> Stage 12 -> Stage 13 sequentially.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"i:\VariFuse_2")


LOG_FILE = ROOT / "pipeline_local.log"


def run_cmd(name: str, cmd: list[str], env: dict[str, str], log_f) -> None:
    header = f"\n{'='*75}\n>>> STARTING: {name}\nCommand: {' '.join(cmd)}\n{'='*75}\n"
    print(header, flush=True)
    log_f.write(header)
    log_f.flush()
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
        log_f.write(line)
        log_f.flush()
    proc.wait()

    elapsed = round(time.time() - t0, 1)
    if proc.returncode != 0:
        err = f"\n!!! {name} FAILED with exit code {proc.returncode} (elapsed: {elapsed}s) !!!\n"
        print(err, flush=True)
        log_f.write(err)
        log_f.flush()
        raise RuntimeError(err)
    success = f"\n>>> {name} FINISHED SUCCESSFULLY in {elapsed}s <<<\n"
    print(success, flush=True)
    log_f.write(success)
    log_f.flush()


def main() -> None:
    env = os.environ.copy()

    # Auto-detect DMS data presence locally
    dms_embeddings = ROOT / "outputs" / "10_esm_features" / "dms_esm_embeddings.npy"
    if not dms_embeddings.is_file():
        dms_pub = ROOT / "publication_runs" / "full_stepwise_v1" / "outputs" / "10_esm_features" / "dms_esm_embeddings.npy"
        has_dms = dms_pub.is_file()
    else:
        has_dms = True

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
        "ESM_DEVICE": "cuda",
        "PYTHONUNBUFFERED": "1",
        "REQUIRE_EXTERNAL_CLINVAR": "1",
        "REQUIRE_EXTERNAL_DMS": "1" if has_dms else "0",
        "DMS_SAMPLING_POLICY": "all",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    startup_msg = (
        f"Local VariFuse Pipeline initialized at: {ROOT}\n"
        f"Local GPU: NVIDIA GeForce GTX 1660\n"
        f"DMS Dataset Available: {has_dms}\n"
        f"Logging to: {LOG_FILE}\n"
    )
    print(startup_msg, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n{'#'*75}\n# NEW PIPELINE RUN STARTED AT {time.ctime()}\n{'#'*75}\n")
        log_f.write(startup_msg)
        log_f.flush()

        t_start = time.time()

        # Step 1: Stage 14 (Resume from current checkpoints or skip if complete)
        stage14_done = (
            ROOT / "outputs" / "14_tuning" / "architecture_selection.json"
        ).is_file() and (ROOT / "outputs" / "14_tuning" / "run_manifest.json").is_file()
        if stage14_done:
            msg = "\n>>> STAGE 14: ALREADY COMPLETED (skipping) <<<\n"
            print(msg, flush=True)
            log_f.write(msg)
            log_f.flush()
        else:
            run_cmd(
                "STAGE 14: TUNE ARCHITECTURES & CONFIRMATORY REPEATED CV",
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "src" / "14_tune_cross_attention.py"),
                    "--confirmation-split-seeds",
                    "1701",
                    "2903",
                    "4159",
                    "6841",
                    "7919",
                    "--resume",
                ],
                env,
                log_f,
            )

        # Step 2: Stage 11 (Train & Evaluate Final Deployment Models or skip if complete)
        stage11_done = (
            ROOT / "outputs" / "11_train_and_evaluate" / "oof_predictions.npz"
        ).is_file() and (ROOT / "outputs" / "11_train_and_evaluate" / "run_manifest.json").is_file()
        if stage11_done:
            msg = "\n>>> STAGE 11: ALREADY COMPLETED (skipping) <<<\n"
            print(msg, flush=True)
            log_f.write(msg)
            log_f.flush()
        else:
            run_cmd(
                "STAGE 11: TRAIN & EVALUATE FINAL MODELS",
                [sys.executable, "-u", str(ROOT / "src" / "11_train_and_evaluate.py")],
                env,
                log_f,
            )

        # Step 3: Stage 12 (External Validation)
        run_cmd(
            "STAGE 12: EXTERNAL VALIDATION (CLINVAR & DMS)",
            [sys.executable, "-u", str(ROOT / "src" / "12_external_validation.py")],
            env,
            log_f,
        )

        # Step 4: Stage 13 (Generate Figures & Tables)
        run_cmd(
            "STAGE 13: GENERATE PUBLICATION FIGURES & TABLES",
            [sys.executable, "-u", str(ROOT / "src" / "13_generate_figures.py")],
            env,
            log_f,
        )

        total_time = round((time.time() - t_start) / 60, 1)
        finish_msg = (
            f"\n{'*'*75}\n"
            f"ALL STAGES (14, 11, 12, 13) FINISHED SUCCESSFULLY in {total_time} minutes!\n"
            f"Final publication models:  {ROOT / 'outputs' / '11_training'}\n"
            f"External validation:       {ROOT / 'outputs' / '12_external_validation'}\n"
            f"Publication figures:       {ROOT / 'figures'}\n"
            f"{'*'*75}\n"
        )
        print(finish_msg, flush=True)
        log_f.write(finish_msg)
        log_f.flush()


if __name__ == "__main__":
    main()
