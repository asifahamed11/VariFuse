"""Run the verified Stage 10 -> 14 -> 11 -> 12 -> 13 Kaggle workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
PROTOCOL = "fixed_nested_group_cv_v4_reliability_residual"
REQUIRED_ARCHITECTURES = {
    "concatenation",
    "gated_fusion",
    "cross_attention",
    "reliability_residual",
    "evidential_residual",
}
EXPECTED_OUTPUTS = {
    "stage14": [
        OUTPUT / "14_tuning" / f"best_{name}_params.json" for name in sorted(REQUIRED_ARCHITECTURES)
    ]
}
SCRIPTS = {
    "stage10": "10_extract_esm_features.py",
    "stage14": "14_tune_cross_attention.py",
    "stage11": "11_train_and_evaluate.py",
    "stage12": "12_external_validation.py",
    "stage13": "13_generate_figures.py",
}
ORDER = ("10-internal", "10-clinvar", "10-dms", "14", "11", "12", "13")
CONFIRMATION_SEEDS = [1701, 2903, 4159, 6841, 7919]


def _external_contract_environment(path: Path | None = None) -> dict[str, str]:
    path = path or OUTPUT / "09_prepare_external_esm/run_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "09_prepare_external_esm_dataset":
        raise ValueError("Expected the exact Stage 09 manifest")
    contract = manifest["external_contract"]
    names = {
        "require_clinvar": "REQUIRE_EXTERNAL_CLINVAR",
        "require_dms": "REQUIRE_EXTERNAL_DMS",
        "strict_clinvar_post_cutoff_last_evaluated": "CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION",
        "require_clinvar_scv_evidence": "CLINVAR_REQUIRE_SCV_EVIDENCE",
        "clinvar_scv_min_matching": "CLINVAR_SCV_MIN_MATCHING",
        "clinvar_scv_min_unique_submitters": "CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS",
        "clinvar_scv_multiple_submitter_minimum": "CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM",
        "dms_sampling_policy": "DMS_SAMPLING_POLICY",
        "dms_max_rows_per_assay": "DMS_MAX_ROWS_PER_ASSAY",
        "allow_sequence_only_dms": "ALLOW_SEQUENCE_ONLY_DMS",
    }
    return {
        environment: str(int(contract[name]))
        if isinstance(contract[name], bool)
        else str(contract[name])
        for name, environment in names.items()
    }


def environment() -> dict[str, str]:
    result = os.environ.copy()
    result.update(_external_contract_environment())
    result.update(
        {
            "VARIANT_PROJECT_ROOT": str(ROOT),
            "VARIANT_OUTPUT_DIR": str(OUTPUT),
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
            "ESM_USE_FP16": "1",
            "ESM_SCORING_MODE": "masked-marginal",
            "ESM_MAX_BATCH_PROTEINS": "2",
            "ESM_MAX_BATCH_TOKENS": "2048",
            "ESM_MAX_BATCH_ATTENTION": "2097152",
            "ESM_MAX_MASKS_PER_BATCH": "2",
            "ESM_MASK_BATCH_ATTENTION": "2097152",
            "ESM_INTERNAL_MAX_ROWS": "200000",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        }
    )
    return result


def run_stage(arguments: argparse.Namespace) -> int:
    env = environment()
    command = [sys.executable, "-u", str(ROOT / "src" / SCRIPTS[arguments.command])]
    if arguments.command == "stage10":
        command += ["--dataset", arguments.dataset, "--scoring-mode", arguments.scoring_mode]
    if arguments.command == "stage14":
        command += [
            "--confirmation-split-seeds",
            *map(str, getattr(arguments, "confirmation_seeds", CONFIRMATION_SEEDS)),
        ]
    command += getattr(arguments, "stage_arguments", [])
    print("Running:", " ".join(command), flush=True)
    if getattr(arguments, "dry_run", False):
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=True).returncode


def preflight(files_only: bool = False) -> None:
    env = environment()
    if files_only:
        env["ESM_DEVICE"] = "auto"
    os.environ.update(env)
    sys.path.insert(0, str(ROOT / "src"))
    config = importlib.import_module("config")
    config.validate_upstream_manifest(
        OUTPUT / "07_natural_prevalence/run_manifest.json",
        "07_dataset_balancing",
        [OUTPUT / "07_natural_prevalence/Final_Dataset_Natural_Prevalence.parquet"],
    )
    stage10 = importlib.import_module("10_extract_esm_features")
    for task in stage10._selected_tasks("all"):
        stage10._validate_task_upstream(task)
    for filename in ("esm2_t33_650M_UR50D.pt", "esm2_t33_650M_UR50D-contact-regression.pt"):
        if not (ROOT / "model_cache/hub/checkpoints" / filename).is_file():
            raise FileNotFoundError(filename)
    if not files_only:
        import torch
        from packaging.version import Version

        if Version(torch.__version__.split("+")[0]) < Version("2.4.1"):
            raise RuntimeError("A CUDA-enabled PyTorch >=2.4.1 environment is required")
        from gpu_runtime import cuda_inventory

        print(json.dumps(cuda_inventory("cuda"), indent=2))
        for name in (
            "esm",
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "sklearn",
            "lightgbm",
            "Bio",
            "matplotlib",
            "statsmodels",
            "shap",
            "optuna",
        ):
            importlib.import_module(name)
        sys.path.insert(0, str(ROOT / "tools"))
        from capture_environment import capture

        (ROOT / "kaggle_environment.json").write_text(
            json.dumps(capture(), indent=2), encoding="utf-8"
        )
    print("Preflight passed: exact prepared inputs and source provenance verified.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "all", *SCRIPTS])
    parser.add_argument(
        "--dataset", choices=["internal", "clinvar", "dms", "external", "all"], default="all"
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["masked-marginal", "wt-marginal", "both"],
        default="masked-marginal",
    )
    parser.add_argument("--start-at", choices=ORDER, default=ORDER[0])
    parser.add_argument("--confirmation-seeds", nargs="+", type=int, default=CONFIRMATION_SEEDS)
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments, extra = parser.parse_known_args()
    if arguments.command == "preflight":
        if extra:
            parser.error(f"Unexpected preflight arguments: {extra}")
        preflight(arguments.files_only)
        return
    if arguments.command == "all":
        if extra:
            parser.error("For custom training arguments use stage14 directly")
        if not arguments.dry_run:
            preflight()
        for entry in ORDER[ORDER.index(arguments.start_at) :]:
            stage, _, dataset = entry.partition("-")
            run_stage(
                argparse.Namespace(
                    **{
                        **vars(arguments),
                        "command": f"stage{stage}",
                        "dataset": dataset or "all",
                        "stage_arguments": [],
                    }
                )
            )
    else:
        arguments.stage_arguments = extra
        run_stage(arguments)


if __name__ == "__main__":
    main()
