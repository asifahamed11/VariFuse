"""Regenerate only the internal comparison figure after a layout-only change."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sampled_sha256(path: Path, sample_size: int = 1 << 20) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(sample_size))
        if size > sample_size:
            handle.seek(max(0, size - sample_size))
            digest.update(handle.read(sample_size))
    return digest.hexdigest()


def record(path: Path, artifact_id: str | None = None) -> dict[str, Any]:
    value = {
        "exists": True,
        "kind": "file",
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256(path),
        "sample_sha256": sampled_sha256(path),
        "size_bytes": path.stat().st_size,
        "hash_policy": "full_sha256",
    }
    if artifact_id is not None:
        value["artifact_id"] = artifact_id
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def backup(path: Path, output_root: Path, backup_root: Path) -> None:
    destination = backup_root / path.relative_to(output_root)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def refresh_input(manifest: dict[str, Any], suffix: str, path: Path) -> None:
    matches = [
        key
        for key in manifest.get("inputs", {})
        if key.replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Stage 13 manifest input is ambiguous: {suffix}")
    manifest["inputs"][matches[0]] = record(path)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=project / "outputs")
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=project / "audit_reports/final_run_20260915/original_before_repair",
    )
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    figure_root = output_root / "13_figures"
    source = project / "src/13_generate_figures.py"
    figure_manifest_path = figure_root / "figure_manifest.json"
    run_manifest_path = figure_root / "run_manifest.json"
    targets = [
        figure_root / f"internal_model_comparison.{extension}"
        for extension in ("png", "svg", "pdf")
    ]
    for path in [*targets, figure_manifest_path, run_manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
        backup(path, output_root, arguments.backup_root.resolve())

    os.environ.update(
        {
            "VARIANT_PROJECT_ROOT": str(project),
            "VARIANT_OUTPUT_DIR": str(output_root),
            "VARIANT_FIGURE_DIR": str(figure_root),
            "VARIANT_HASH_LARGE_FILES": "1",
            "MPLBACKEND": "Agg",
        }
    )
    sys.path.insert(0, str(project / "src"))
    stage13 = importlib.import_module("13_generate_figures")
    paths = stage13._save_figure(
        stage13.figure_model_comparison(), "internal_model_comparison"
    )

    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figure_manifest["figures"]["internal_model_comparison"].update(
        {
            "paths": paths,
            "layout": "horizontal_bars_with_readable_model_labels",
            "post_run_layout_repair": True,
        }
    )
    architecture_results_path = output_root / "14_tuning/architecture_selection.json"
    stage12_manifest_path = output_root / "12_external_validation/run_manifest.json"
    figure_manifest["stage14_validation"]["architecture_results_sha256"] = sha256(
        architecture_results_path
    )
    figure_manifest["stage12_validation"]["stage12_manifest_sha256"] = sha256(
        stage12_manifest_path
    )
    atomic_json(figure_manifest_path, figure_manifest)

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for path in targets + [figure_manifest_path]:
        matches = [
            key
            for key in run_manifest["outputs"]
            if key.replace("\\", "/").endswith("/" + path.name)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Stage 13 output record is ambiguous: {path.name}")
        run_manifest["outputs"][matches[0]] = record(path, matches[0])
    for suffix, path in (
        ("/14_tuning/architecture_selection.json", output_root / "14_tuning/architecture_selection.json"),
        ("/14_tuning/run_manifest.json", output_root / "14_tuning/run_manifest.json"),
        ("/11_train_and_evaluate/run_manifest.json", output_root / "11_train_and_evaluate/run_manifest.json"),
        ("/12_external_validation/run_manifest.json", output_root / "12_external_validation/run_manifest.json"),
    ):
        refresh_input(run_manifest, suffix, path)
    run_manifest.setdefault("extra", {})["post_run_figure_layout_repair"] = {
        "status": "applied_without_model_training_or_metric_recomputation",
        "figure": "internal_model_comparison",
        "layout": "horizontal_bars_with_readable_model_labels",
        "rendering_source_sha256": sha256(source),
        "original_files_preserved_at": str(arguments.backup_root.resolve()),
    }
    run_manifest["extra"]["stage14_validation"][
        "architecture_results_sha256"
    ] = sha256(architecture_results_path)
    run_manifest["extra"]["stage12_validation"]["stage12_manifest_sha256"] = sha256(
        stage12_manifest_path
    )
    atomic_json(run_manifest_path, run_manifest)
    print(json.dumps({"status": "complete", "files": paths}, indent=2))


if __name__ == "__main__":
    main()
