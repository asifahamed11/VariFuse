from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_ENVIRONMENT = (
    "VARIANT_PROJECT_ROOT",
    "VARIANT_DATA_DIR",
    "VARIANT_EXTERNAL_DATA_DIR",
    "VARIANT_OUTPUT_DIR",
    "VARIANT_FIGURE_DIR",
    "VARIANT_LABEL_TASK",
    "VARIANT_TRAIN_CUTOFF_DATE",
    "VARIANT_RANDOM_STATE",
    "VARIANT_REPRODUCIBLE",
    "VARIANT_HASH_LARGE_FILES",
    "CLINVAR_TRAIN_RELEASE",
    "CLINVAR_EXTERNAL_RELEASE",
    "CLINVAR_TRAIN_SUBMISSION_ARCHIVE",
    "CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE",
    "CLINVAR_REQUIRE_SCV_EVIDENCE",
    "CLINVAR_SCV_MIN_MATCHING",
    "CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS",
    "CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM",
    "DBNSFP_RELEASE",
    "PROTEINGYM_RELEASE",
    "DMS_SAMPLING_POLICY",
    "DMS_MAX_ROWS_PER_ASSAY",
    "REQUIRE_EXTERNAL_CLINVAR",
    "REQUIRE_EXTERNAL_DMS",
    "REQUIRE_TRANSCRIPT_MAPPING",
    "ESM_MODEL_NAME",
    "ESM_LAYER",
    "ESM_SCORING_MODE",
    "ESM_USE_FP16",
    "CUBLAS_WORKSPACE_CONFIG",
    "MMSEQS_EXECUTABLE",
    "HOMOLOGY_CLUSTER_TSV",
    "HOMOLOGY_MIN_SEQ_ID",
    "HOMOLOGY_COVERAGE",
    "ENABLE_ESM_LORA",
    "TUNING_STORAGE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = [
        *sorted((ROOT / "src").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
        ROOT / "run_pipeline.py",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "requirements-publication.txt",
        ROOT / "tools" / "publication_data_catalog.json",
        ROOT / "kaggle" / "kaggle_runner.py",
    ]
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in paths
        if path.is_file()
    }


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return (completed.stdout.strip() or completed.stderr.strip()) or None


def capture() -> dict[str, object]:
    distributions = sorted(
        {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }.items(),
        key=lambda item: item[0].lower(),
    )
    gpu: dict[str, object] = {"torch_available": False}
    try:
        import torch

        gpu = {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    git_revision = None
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    mmseqs_executable = os.environ.get("MMSEQS_EXECUTABLE") or shutil.which("mmseqs")
    mmseqs_version = (
        _command_output([mmseqs_executable, "version"])
        if mmseqs_executable
        else None
    )
    nvidia_smi = shutil.which("nvidia-smi")
    driver_report = (
        _command_output(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        if nvidia_smi
        else None
    )
    source_hashes = _source_hashes()
    source_tree_digest = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_revision": git_revision,
        "packages": dict(distributions),
        "accelerator": gpu,
        "nvidia_smi": driver_report,
        "mmseqs": {
            "executable": mmseqs_executable,
            "version_output": mmseqs_version,
        },
        "scientific_environment": {
            name: os.environ[name]
            for name in SCIENTIFIC_ENVIRONMENT
            if name in os.environ
        },
        "source_sha256": source_hashes,
        "source_tree_sha256": source_tree_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("environment-lock.json"))
    arguments = parser.parse_args()
    payload = capture()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(arguments.output)
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
