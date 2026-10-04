from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
LOCAL_MMSEQS_CANDIDATES = (
    ROOT / "tools" / "mmseqs2" / "mmseqs" / "bin" / "mmseqs.exe",
    ROOT / "tools" / "mmseqs2" / "mmseqs" / "mmseqs.bat",
)
PIPELINE = (
    ("01", "01_dbnsfp_processor.py"),
    ("02", "02_remove_missing_values.py"),
    ("03", "03_remove_duplicates.py"),
    ("04", "04_feature_engineering.py"),
    ("05", "05_remove_leakage.py"),
    ("06", "06_clean_and_finalize.py"),
    ("07", "07_dataset_balancing.py"),
    ("08", "08_prepare_esm_dataset.py"),
    ("08b", "08b_build_homology_groups.py"),
    ("09", "09_prepare_external_esm_dataset.py"),
    ("10", "10_extract_esm_features.py"),
    ("14", "14_tune_cross_attention.py"),
    ("11", "11_train_and_evaluate.py"),
    ("12", "12_external_validation.py"),
    ("13", "13_generate_figures.py"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run VariFuse in publication dependency order (including homology "
            "groups and Stage 14 tuning before deployment Stage 11)."
        )
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[stage for stage, _ in PIPELINE],
        help="Run only selected stage IDs while preserving dependency order.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--local-publication",
        action="store_true",
        help=(
            "Use a clean, resumable local publication profile: ESM2-650M FP16, "
            "one protein per GPU batch, all label-independent DMS rows, strict "
            "external/homology contracts, and project-local model cache."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Artifact run identifier for --local-publication. Defaults to a UTC "
            "timestamp and never reuses legacy outputs."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument(
        "--dms-policy", choices=["all", "hash_uniform"], help="Override DMS sampling."
    )
    parser.add_argument(
        "--esm-device", choices=["auto", "cuda", "cpu"], help="Local ESM device."
    )
    parser.add_argument(
        "--stage10-dataset",
        choices=["internal", "clinvar", "dms", "external", "all"],
        default="all",
        help="Subset used only when Stage 10 is selected.",
    )
    parser.add_argument(
        "--confirmation-split-seeds",
        nargs="+",
        type=int,
        help=(
            "Prespecified split seeds forwarded only to Stage 14 for fixed-config "
            "repeated confirmation (no repeated HPO)."
        ),
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help=(
            "Explicitly allow replacement inside non-empty output/figure directories. "
            "Publication reruns should normally use new VARIANT_OUTPUT_DIR and "
            "VARIANT_FIGURE_DIR values."
        ),
    )
    arguments = parser.parse_args()
    selected = set(arguments.stages or [stage for stage, _ in PIPELINE])
    if arguments.confirmation_split_seeds and "14" not in selected:
        raise ValueError("--confirmation-split-seeds requires Stage 14 to be selected")
    environment = os.environ.copy()
    if arguments.local_publication:
        run_id = arguments.run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("--run-id may contain only letters, digits, dot, dash, underscore")
        run_root = ROOT / "publication_runs" / run_id
        environment.update(
            {
                "VARIANT_OUTPUT_DIR": str(
                    (arguments.output_dir or run_root / "outputs").resolve()
                ),
                "VARIANT_FIGURE_DIR": str(
                    (arguments.figure_dir or run_root / "figures").resolve()
                ),
                "TORCH_HOME": str((ROOT / "model_cache").resolve()),
                "ESM_DEVICE": arguments.esm_device or "auto",
                "ESM_USE_FP16": "1",
                "ESM_MAX_BATCH_PROTEINS": "1",
                "ESM_MAX_BATCH_TOKENS": "1024",
                "ESM_MAX_BATCH_ATTENTION": "1048576",
                "DMS_SAMPLING_POLICY": arguments.dms_policy or "all",
                "DMS_MAX_ROWS_PER_ASSAY": "500",
                "VARIANT_LABEL_TASK": "clinical",
                "VARIANT_TRAIN_CUTOFF_DATE": "2024-06-30",
                "VARIANT_RANDOM_STATE": "42",
                "CLINVAR_TRAIN_RELEASE": "2024-06",
                "CLINVAR_EXTERNAL_RELEASE": "2026-08",
                "CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION": "1",
                "CLINVAR_REQUIRE_SCV_EVIDENCE": "1",
                "CLINVAR_SCV_MIN_MATCHING": "1",
                "CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS": "1",
                "CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM": "2",
                "DBNSFP_RELEASE": "5.3a",
                "PROTEINGYM_RELEASE": "v1.3",
                "ALLOW_LEGACY_MIXED_LABELS": "0",
                "ALLOW_POST_CUTOFF_TRAINING_EVIDENCE": "0",
                "REQUIRE_EXTERNAL_CLINVAR": "1",
                "REQUIRE_EXTERNAL_DMS": "1",
                "REQUIRE_TRANSCRIPT_MAPPING": "1",
                "REQUIRE_HOMOLOGY_GROUPS": "1",
                "ALLOW_SEQUENCE_ONLY_DMS": "1",
                "REQUIRE_TUNING_ARTIFACT": "1",
                "MMSEQS_THREADS": "1",
                "VARIANT_REPRODUCIBLE": "1",
                "VARIANT_HASH_LARGE_FILES": "1",
                # Required by deterministic CUDA matrix multiplication on
                # CUDA >= 10.2.  It must exist before the child imports torch.
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        if not environment.get("MMSEQS_EXECUTABLE"):
            bundled_mmseqs = next(
                (path for path in LOCAL_MMSEQS_CANDIDATES if path.is_file()), None
            )
            if bundled_mmseqs is not None:
                environment["MMSEQS_EXECUTABLE"] = str(bundled_mmseqs.resolve())
    else:
        if arguments.output_dir is not None:
            environment["VARIANT_OUTPUT_DIR"] = str(arguments.output_dir.resolve())
        if arguments.figure_dir is not None:
            environment["VARIANT_FIGURE_DIR"] = str(arguments.figure_dir.resolve())
        if arguments.dms_policy:
            environment["DMS_SAMPLING_POLICY"] = arguments.dms_policy
        if arguments.esm_device:
            environment["ESM_DEVICE"] = arguments.esm_device
    output_dir = Path(
        environment.get("VARIANT_OUTPUT_DIR", str(ROOT / "outputs"))
    ).resolve()
    figure_dir = Path(
        environment.get("VARIANT_FIGURE_DIR", str(ROOT / "figures"))
    ).resolve()
    nonempty_roots = [
        path
        for path in (output_dir, figure_dir)
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if (
        "01" in selected
        and not arguments.dry_run
        and nonempty_roots
        and not arguments.allow_existing_output
    ):
        raise RuntimeError(
            "Refusing to start Stage 01 with non-empty artifact roots: "
            f"{nonempty_roots}. Preserve the legacy results and set both "
            "VARIANT_OUTPUT_DIR and VARIANT_FIGURE_DIR to new directories, or pass "
            "--allow-existing-output only when replacement is intentional."
        )
    if arguments.allow_existing_output:
        environment["ALLOW_EXISTING_FIGURE_DIR"] = "1"
    if arguments.local_publication and "10" in selected:
        free_bytes = shutil.disk_usage(ROOT).free
        if free_bytes < 15 * 2**30:
            raise RuntimeError(
                "Local publication extraction requires at least 15 GiB free beside "
                f"the output root; found {free_bytes / 2**30:.2f} GiB"
            )
    if "08b" in selected:
        requested_mmseqs = environment.get("MMSEQS_EXECUTABLE", "mmseqs")
        if shutil.which(requested_mmseqs) is None:
            raise RuntimeError(
                "Stage 08b requires MMseqs2. Set MMSEQS_EXECUTABLE to the exact "
                "binary, or restore tools/mmseqs2/mmseqs/bin/mmseqs.exe."
            )
    print(
        f"Artifact root: {output_dir}\nFigure root: {figure_dir}\n"
        f"DMS policy: {environment.get('DMS_SAMPLING_POLICY', 'configured default')}\n"
        f"ESM device: {environment.get('ESM_DEVICE', 'auto')}\n"
        f"MMseqs2: {environment.get('MMSEQS_EXECUTABLE', 'PATH:mmseqs')}",
        flush=True,
    )
    for stage, filename in PIPELINE:
        if stage not in selected:
            continue
        command = [sys.executable, str(SOURCE / filename)]
        if stage == "10":
            command.extend(["--dataset", arguments.stage10_dataset])
        if stage == "14" and arguments.confirmation_split_seeds:
            command.extend(
                [
                    "--confirmation-split-seeds",
                    *[str(seed) for seed in arguments.confirmation_split_seeds],
                ]
            )
        print(f"[{stage}] {' '.join(command)}", flush=True)
        if not arguments.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
