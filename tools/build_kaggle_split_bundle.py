"""Export a frozen, paired code/data bundle without touching an active run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import uuid
import zipfile
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path("VariFuse_2")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def patched_stage10(source: bytes) -> bytes:
    """Correct CLI provenance in the export only; preserve all other source bytes."""
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    before = (
        b"            arguments.scoring_mode," + newline + b"            validate_upstream=False,"
    )
    after = before.replace(b"False", b"True")
    if source.count(before) == 1:
        return source.replace(before, after)
    if source.count(after) == 1:
        return source
    raise ValueError("Stage 10 CLI changed; review the export patch before packaging")


def notebook() -> bytes:
    def cell(kind: str, source: str) -> dict:
        result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result

    cells = [
        cell(
            "markdown",
            "# VariFuse — full Kaggle run\nAttach the paired **Data** and **Code** datasets. Select **GPU T4 x2**, enable Internet for dependencies. Run the cells in order. Full data; no small-run budgets. See `kaggle/README_KAGGLE.md` for Bengali instructions and session recovery.\n",
        )
    ]
    cells.append(
        cell(
            "code",
            """from pathlib import Path
import subprocess, sys, zipfile, shutil
input_root = Path("/kaggle/input")
candidates = list(input_root.rglob("prepare_kaggle.py"))
if len(candidates) == 1:
    bootstrap = candidates[0]
elif len(candidates) == 0:
    archives = list(input_root.rglob("VariFuse_Kaggle_Code.zip"))
    if len(archives) != 1:
        raise RuntimeError("Attach exactly one VariFuse Code dataset")
    bootstrap = Path("/kaggle/working/varifuse_prepare.py")
    with zipfile.ZipFile(archives[0]) as archive:
        bootstrap.write_bytes(archive.read("VariFuse_2/kaggle/prepare_kaggle.py"))
else:
    raise RuntimeError("Multiple code datasets attached; keep one matching export")
subprocess.run([sys.executable, str(bootstrap)], check=True)
PROJECT = Path("/kaggle/working/VariFuse_2")
print("Free working space (GiB):", round(shutil.disk_usage(PROJECT).free / 2**30, 2))
""",
        )
    )
    cells.append(
        cell(
            "code",
            """subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(PROJECT / "kaggle/requirements-kaggle.txt")], check=True)
# If installation changed loaded packages, restart the session and rerun setup.
""",
        )
    )
    cells.append(
        cell(
            "code",
            """subprocess.run([sys.executable, str(PROJECT / "kaggle/kaggle_runner.py"), "preflight"], check=True)
""",
        )
    )
    cells.append(
        cell(
            "code",
            """def run_stage(*arguments):
    subprocess.run([sys.executable, "-u", str(PROJECT / "kaggle/kaggle_runner.py"), *arguments], cwd=PROJECT, check=True)
""",
        )
    )
    for label, args in [
        ("10 — internal (resumes exported completed cache)", '"stage10", "--dataset", "internal"'),
        ("10 — ClinVar", '"stage10", "--dataset", "clinvar"'),
        ("10 — ProteinGym DMS", '"stage10", "--dataset", "dms"'),
        ("14 — full nested tuning and confirmation seeds", '"stage14"'),
        ("11 — deployment model training", '"stage11"'),
        ("12 — external clinical and functional evaluation", '"stage12"'),
        ("13 — verified figures", '"stage13"'),
    ]:
        cells.append(cell("markdown", f"## {label}\n"))
        cells.append(cell("code", f"run_stage({args})\n"))
    cells.append(
        cell(
            "code",
            """import json
manifest = json.loads((PROJECT / "figures/figure_manifest.json").read_text())
print({key: manifest.get(key) for key in ("completed", "publication_complete")})
print("Preserve/download the full outputs and figures directories before the session ends.")
""",
        )
    )
    return json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        },
        indent=2,
    ).encode()


def valid_cache(path: Path) -> tuple[int, str | None]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            rows, embeddings = stored["rows"], stored["embeddings"]
            if (
                str(stored["model"].item()) != "esm2_t33_650M_UR50D"
                or int(stored["layer"].item()) != 33
                or not bool(stored["fp16"].item())
            ):
                return 0, "model/layer/precision mismatch"
            if str(stored["scoring_mode"].item()) not in {"masked-marginal", "both"}:
                return 0, "scoring mode mismatch"
            if (
                embeddings.shape != (len(rows), 1280)
                or not np.isfinite(embeddings).all()
                or not np.isfinite(stored["masked_scores"]).all()
            ):
                return 0, "incomplete numeric results"
            if "variants" not in stored or "context_hash" not in stored:
                return 0, "missing content identity"
            return len(rows), None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        return 0, str(error)


def write_zip(
    destination: Path, entries: list[tuple[str, Path | bytes]], kind: str, shared: dict
) -> dict:
    temporary = destination.with_suffix(".zip.building")
    records = []
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for index, (relative, source) in enumerate(entries, 1):
            name = (PREFIX / relative).as_posix()
            digest, count = hashlib.sha256(), 0
            if isinstance(source, bytes):
                archive.writestr(name, source)
                digest.update(source)
                count = len(source)
            else:
                before = source.stat()
                info = zipfile.ZipInfo.from_file(source, name)
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if source.suffix.lower() in {".pt", ".npz", ".parquet"}
                    else zipfile.ZIP_DEFLATED
                )
                with (
                    source.open("rb") as reader,
                    archive.open(info, "w", force_zip64=True) as writer,
                ):
                    while block := reader.read(8 << 20):
                        writer.write(block)
                        digest.update(block)
                        count += len(block)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed while packaging: {source}")
            records.append({"path": relative, "size_bytes": count, "sha256": digest.hexdigest()})
            if index % 500 == 0 or count > 1 << 30:
                print(f"Packed {kind}: {index}/{len(entries)} — {relative}", flush=True)
        manifest = {**shared, "format_version": 1, "kind": kind, "files": records}
        archive.writestr(
            (PREFIX / f"VARIFUSE_{kind.upper()}_MANIFEST.json").as_posix(),
            json.dumps(manifest, indent=2).encode(),
        )
    temporary.replace(destination)
    return {
        "file": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha(destination),
        "entries": len(records),
        "uncompressed_bytes": sum(r["size_bytes"] for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "publication_runs/full_stepwise_v1/outputs"
    )
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/kaggle_full_stepwise_v1")
    args = parser.parse_args()
    args.output_root, args.destination = args.output_root.resolve(), args.destination.resolve()
    args.destination.mkdir(parents=True, exist_ok=True)
    if any(
        (args.destination / f"VariFuse_Kaggle_{name}.zip").exists() for name in ("Code", "Data")
    ):
        raise FileExistsError("Paired export already exists; choose a fresh --destination")
    spec = importlib.util.spec_from_file_location(
        "old_bundle", ROOT / "tools/build_kaggle_bundle.py"
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validator.OUTPUT_ROOT = args.output_root
    validator.CHECKPOINT_ROOT = ROOT / "model_cache/hub/checkpoints"
    validator._validate_publication_contract()
    # Validate additional implementation dependencies used by the actual consumer.
    stage09 = json.loads(
        (args.output_root / "09_prepare_external_esm/run_manifest.json").read_text()
    )
    validator._validate_source_hashes(
        stage09,
        ("01_dbnsfp_processor.py", "04_feature_engineering.py", "08_prepare_esm_dataset.py"),
        "Stage 09 implementation dependencies",
    )
    data_entries = []
    for folder in ("07_natural_prevalence", "08_prepare_esm", "09_prepare_external_esm"):
        for path in sorted((args.output_root / folder).iterdir()):
            if path.is_file() and not path.name.endswith(".tmp"):
                data_entries.append(
                    ((Path("outputs") / path.relative_to(args.output_root)).as_posix(), path)
                )
    for path in sorted(validator.CHECKPOINT_ROOT.glob("esm2_t33_650M_UR50D*.pt")):
        data_entries.append(((Path("model_cache/hub/checkpoints") / path.name).as_posix(), path))
    # Atomic .npz files form the frozen resume snapshot. Never include the live
    # preallocated .npy.tmp, which has a final size even when rows are unfinished.
    cache_paths = sorted((args.output_root / "10_esm_features/cache/internal").glob("*.npz"))
    cached_rows, rejected = 0, []
    for path in cache_paths:
        rows, reason = valid_cache(path)
        if reason:
            rejected.append({"file": path.name, "reason": reason})
        else:
            cached_rows += rows
            data_entries.append(
                ((Path("outputs") / path.relative_to(args.output_root)).as_posix(), path)
            )
    source10 = (ROOT / "src/10_extract_esm_features.py").read_bytes()
    exported10 = patched_stage10(source10)
    code_entries = []
    for folder in ("src", "tests", "tools", "kaggle"):
        for path in sorted((ROOT / folder).glob("*.py")):
            code_entries.append(
                (
                    path.relative_to(ROOT).as_posix(),
                    exported10 if path.name == "10_extract_esm_features.py" else path,
                )
            )
    for name in (
        "README.md",
        "DATA_SOURCES.md",
        "REPRODUCIBILITY.md",
        "LOCAL_PUBLICATION_RUN.md",
        "SMALL_VALIDATION.md",
        "requirements.txt",
        "requirements-publication.txt",
        "pyproject.toml",
        "run_pipeline.py",
        "kaggle/README_KAGGLE.md",
        "kaggle/requirements-kaggle.txt",
        "tools/publication_data_catalog.json",
    ):
        path = ROOT / name
        if path.is_file():
            code_entries.append((name, path))
    nb = notebook()
    code_entries.append(("VariFuse_Kaggle.ipynb", nb))
    summary = json.loads(
        (args.output_root / "09_prepare_external_esm/external_preparation_summary.json").read_text()
    )
    shared = {
        "pair_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(args.output_root),
        "dataset_scope": "full_prepared_datasets_not_small_validation",
        "external_rows": summary["source_rows"],
        "dms_assays": len(summary["assay_rows"]),
        "internal_cache_files": len(cache_paths) - len(rejected),
        "internal_cached_rows": cached_rows,
        "cache_rejections": rejected,
        "required_order": ["10 internal", "10 clinvar", "10 dms", "14", "11", "12", "13"],
        "export_source_patch": {
            "file": "src/10_extract_esm_features.py",
            "change": "CLI extract_task(validate_upstream=True); production provenance correction only",
            "local_sha256": hashlib.sha256(source10).hexdigest(),
            "export_sha256": hashlib.sha256(exported10).hexdigest(),
            "local_source_modified": False,
        },
    }
    for entries in (code_entries, data_entries):
        paths = [name for name, _ in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate archive paths")
    started = time.monotonic()
    print(json.dumps(shared, indent=2), flush=True)
    outputs = [
        write_zip(args.destination / "VariFuse_Kaggle_Data.zip", data_entries, "data", shared),
        write_zip(args.destination / "VariFuse_Kaggle_Code.zip", code_entries, "code", shared),
    ]
    report = {**shared, "archives": outputs, "build_seconds": time.monotonic() - started}
    (args.destination / "EXPORT_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.destination / "SHA256SUMS.txt").write_text(
        "".join(f"{record['sha256']}  {Path(record['file']).name}\n" for record in outputs),
        encoding="ascii",
    )
    (args.destination / "VariFuse_Kaggle.ipynb").write_bytes(nb)
    (args.destination / "README_KAGGLE.md").write_bytes(
        (ROOT / "kaggle/README_KAGGLE.md").read_bytes()
    )
    print(json.dumps(report["archives"], indent=2), flush=True)


if __name__ == "__main__":
    main()
