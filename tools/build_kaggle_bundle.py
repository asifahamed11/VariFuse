from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "dist" / "VariFuse_Kaggle_Stages10_14_Offline.zip"
CHECKSUM_FILE = DESTINATION.with_suffix(DESTINATION.suffix + ".sha256")
UPLOAD_NOTE = DESTINATION.parent / "UPLOAD_THIS_FILE.txt"
ARCHIVE_ROOT = Path("VariFuse_2")
OUTPUT_ROOT = Path(
    os.environ.get("VARIANT_OUTPUT_DIR", str(ROOT / "outputs"))
).resolve()
CHECKPOINT_ROOT = Path(
    os.environ.get(
        "VARIANT_ESM_CHECKPOINT_DIR",
        str(
            OUTPUT_ROOT
            / "10_esm_features"
            / "torch_cache"
            / "hub"
            / "checkpoints"
        ),
    )
).resolve()
if not CHECKPOINT_ROOT.is_dir() and OUTPUT_ROOT != (ROOT / "outputs").resolve():
    CHECKPOINT_ROOT = (
        ROOT
        / "outputs"
        / "10_esm_features"
        / "torch_cache"
        / "hub"
        / "checkpoints"
    ).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate_bound_output(
    manifest: dict[str, object], path: Path, artifact_id: str
) -> None:
    outputs = manifest.get("outputs")
    record = outputs.get(artifact_id) if isinstance(outputs, dict) else None
    if (
        manifest.get("artifact_manifest_version") != 2
        or not isinstance(record, dict)
        or record.get("artifact_id") != artifact_id
        or record.get("exists") is not True
        or record.get("sha256") != _sha256(path)
        or record.get("size_bytes") != path.stat().st_size
    ):
        raise RuntimeError(f"Manifest does not bind Stage output {artifact_id}")


def _validate_all_bound_outputs(manifest: dict[str, object]) -> None:
    outputs = manifest.get("outputs")
    if manifest.get("artifact_manifest_version") != 2 or not isinstance(outputs, dict):
        raise RuntimeError("Manifest lacks portable v2 output bindings")
    for artifact_id in outputs:
        normalized = str(artifact_id).replace("\\", "/")
        parts = normalized.split("/")
        if not normalized or normalized.startswith("/") or ".." in parts:
            raise RuntimeError(f"Unsafe output artifact ID: {artifact_id}")
        path = OUTPUT_ROOT.joinpath(*parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_bound_output(manifest, path, str(artifact_id))


def _validate_source_hashes(
    manifest: dict[str, object], required: tuple[str, ...], description: str
) -> None:
    recorded = manifest.get("source_files")
    if not isinstance(recorded, dict):
        raise RuntimeError(f"{description} lacks source provenance")
    for name in required:
        path = ROOT / "src" / name
        if not path.is_file() or recorded.get(name) != _sha256(path):
            raise RuntimeError(
                f"{description} was produced by different code: {name}"
            )


def _validate_publication_contract() -> None:
    manifests = (
        OUTPUT_ROOT / "07_natural_prevalence" / "run_manifest.json",
        OUTPUT_ROOT / "08_prepare_esm" / "run_manifest.json",
        OUTPUT_ROOT / "08_prepare_esm" / "homology_run_manifest.json",
        OUTPUT_ROOT / "09_prepare_external_esm" / "run_manifest.json",
    )
    payloads = []
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("label_task") != "clinical":
            raise RuntimeError(
                f"Refusing to package non-clinical/legacy artifacts: {path}"
            )
        payloads.append(payload)
    stage07, stage08, homology, external = payloads
    for payload in payloads:
        _validate_all_bound_outputs(payload)
    if (
        stage07.get("stage") != "07_dataset_balancing"
        or stage08.get("stage") != "08_prepare_esm_dataset"
    ):
        raise RuntimeError("Stage 07/08 manifest identities are invalid")
    if homology.get("stage") != "08b_build_homology_groups" or external.get(
        "stage"
    ) != "09_prepare_external_esm_dataset":
        raise RuntimeError("Stage 08b/09 manifest identities are invalid")
    contract_fields = (
        "label_policy_version",
        "training_cutoff_date",
        "model_tag",
        "esm_model",
        "esm_layer",
        "data_releases",
    )
    mismatches = [
        field for field in contract_fields if homology.get(field) != external.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            f"Stage 08b and 09 publication contracts differ: {mismatches}"
        )
    homology_extra = homology.get("extra", {})
    external_extra = external.get("extra", {})
    source_rows = external_extra.get("source_rows", {})
    if (
        float(homology_extra.get("min_sequence_identity", -1)) != 0.30
        or float(homology_extra.get("minimum_coverage", -1)) != 0.80
        or homology_extra.get("coverage_mode") != 0
        or homology_extra.get("cluster_mode") != 2
        or external_extra.get("dms_sampling_uses_label") is not False
        or external_extra.get("dms_sampling_policy") not in {"all", "hash_uniform"}
        or not isinstance(source_rows, dict)
        or int(source_rows.get("clinvar", 0)) <= 0
        or int(source_rows.get("dms", 0)) <= 0
    ):
        raise RuntimeError("Stage 08b/09 publication profile is incomplete or unsafe")
    _validate_source_hashes(
        stage07,
        ("07_dataset_balancing.py", "config.py", "schema.py", "table_io.py"),
        "Stage 07",
    )
    _validate_source_hashes(
        stage08,
        ("08_prepare_esm_dataset.py", "config.py", "schema.py", "table_io.py"),
        "Stage 08",
    )
    _validate_source_hashes(
        homology,
        ("08b_build_homology_groups.py", "config.py", "schema.py", "table_io.py"),
        "Stage 08b",
    )
    for manifest, path, artifact_id in (
        (
            stage07,
            OUTPUT_ROOT
            / "07_natural_prevalence"
            / "Final_Dataset_Natural_Prevalence.parquet",
            "07_natural_prevalence/Final_Dataset_Natural_Prevalence.parquet",
        ),
        (
            stage08,
            OUTPUT_ROOT / "08_prepare_esm" / "internal_sequences.parquet",
            "08_prepare_esm/internal_sequences.parquet",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_bound_output(manifest, path, artifact_id)
    _validate_source_hashes(
        external,
        (
            "09_prepare_external_esm_dataset.py",
            "config.py",
            "schema.py",
            "table_io.py",
        ),
        "Stage 09",
    )
    for path in (
        OUTPUT_ROOT / "08_prepare_esm" / "internal_esm_ready_homology.parquet",
        OUTPUT_ROOT / "08_prepare_esm" / "homology_summary.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_bound_output(homology, path, f"08_prepare_esm/{path.name}")
    for path in (
        OUTPUT_ROOT / "09_prepare_external_esm" / "clinvar_esm_ready.csv",
        OUTPUT_ROOT / "09_prepare_external_esm" / "dms_esm_ready.csv",
        OUTPUT_ROOT / "09_prepare_external_esm" / "external_preparation_summary.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_bound_output(
            external, path, f"09_prepare_external_esm/{path.name}"
        )
    checkpoints = list(CHECKPOINT_ROOT.glob("esm2_t33_650M_UR50D*.pt"))
    if len(checkpoints) < 2:
        raise RuntimeError(
            "Offline bundle requires both the ESM2 model and contact-regression "
            f"checkpoints under {CHECKPOINT_ROOT}; set VARIANT_ESM_CHECKPOINT_DIR"
        )


def selected_files() -> list[tuple[Path, Path]]:
    _validate_publication_contract()
    selected: list[tuple[Path, Path]] = []

    def add(source: Path, relative: Path | None = None) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        selected.append((source, relative or source.relative_to(ROOT)))

    def add_output(source: Path) -> None:
        add(source, Path("outputs") / source.relative_to(OUTPUT_ROOT))

    for name in (
        "README.md",
        "REPRODUCIBILITY.md",
        "requirements.txt",
        "requirements-publication.txt",
        "pyproject.toml",
        "run_pipeline.py",
    ):
        add(ROOT / name)
    add(ROOT / "tools" / "capture_environment.py")
    add(ROOT / "kaggle" / "README_KAGGLE.md", Path("README_KAGGLE.md"))
    add(ROOT / "kaggle" / "requirements-kaggle.txt", Path("requirements-kaggle.txt"))
    add(ROOT / "kaggle" / "kaggle_runner.py", Path("kaggle_runner.py"))

    for path in sorted((ROOT / "src").glob("*.py")):
        add(path)
    for path in sorted((ROOT / "tests").glob("*.py")):
        add(path)
    for directory in (
        OUTPUT_ROOT / "07_natural_prevalence",
        OUTPUT_ROOT / "08_prepare_esm",
        OUTPUT_ROOT / "09_prepare_external_esm",
    ):
        for path in sorted(directory.glob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                add_output(path)

    checkpoint_archive = Path(
        "outputs/10_esm_features/torch_cache/hub/checkpoints"
    )
    for path in sorted(CHECKPOINT_ROOT.glob("*.pt")):
        add(path, checkpoint_archive / path.name)
    cache_dir = OUTPUT_ROOT / "10_esm_features" / "cache" / "internal"
    for path in sorted(cache_dir.glob("*.npz")):
        add_output(path)

    relative_paths = [relative.as_posix() for _, relative in selected]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Duplicate archive path")
    return selected


def write_entry(
    archive: zipfile.ZipFile,
    source: Path,
    relative: Path,
) -> dict[str, object]:
    stat_before = source.stat()
    archive_name = (ARCHIVE_ROOT / relative).as_posix()
    info = zipfile.ZipInfo.from_file(source, archive_name)
    already_compressed = source.suffix.lower() in {".pt", ".npz", ".parquet", ".gz"}
    info.compress_type = zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED
    digest = hashlib.sha256()
    with source.open("rb") as source_handle, archive.open(
        info, "w", force_zip64=True
    ) as archive_handle:
        while block := source_handle.read(8 << 20):
            digest.update(block)
            archive_handle.write(block)
    stat_after = source.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"Source changed during packaging: {source}")
    return {
        "path": relative.as_posix(),
        "size_bytes": stat_before.st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    files = selected_files()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(DESTINATION.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    started = time.monotonic()
    manifest_files: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for index, (source, relative) in enumerate(files, 1):
                manifest_files.append(write_entry(archive, source, relative))
                if index % 1000 == 0 or source.stat().st_size > (1 << 30):
                    print(f"Packed {index:,}/{len(files):,}: {relative}", flush=True)
            manifest = {
                "package": "VariFuse Kaggle Stages 10-14 Offline",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_output_root": str(OUTPUT_ROOT),
                "label_task": "clinical",
                "files": manifest_files,
                "notes": [
                    "ESM2 650M checkpoint included for offline Kaggle execution.",
                    "Exactly two Kaggle T4 GPUs are required and selected.",
                    "GPU model forward paths use transient two-device DataParallel.",
                    "Completed internal Stage 10 cache entries included for resume.",
                    "Partial *.tmp artifacts excluded.",
                    "Required run order: 10 internal, 10 ClinVar, 10 DMS, 14, 11, 12, 13.",
                    "Stage 08b/09 outputs are exact-hash bound to publication manifests.",
                ],
            }
            archive.writestr(
                (ARCHIVE_ROOT / "PACKAGE_MANIFEST.json").as_posix(),
                json.dumps(manifest, indent=2).encode("utf-8"),
            )
        os.replace(temporary, DESTINATION)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    elapsed = time.monotonic() - started
    archive_sha256 = _sha256(DESTINATION)
    CHECKSUM_FILE.write_text(
        f"{archive_sha256}  {DESTINATION.name}\n", encoding="ascii"
    )
    UPLOAD_NOTE.write_text(
        "Upload this exact file as a private Kaggle Dataset:\n"
        f"{DESTINATION}\n\n"
        "Kaggle accelerator setting: GPU T4 x2\n"
        f"SHA-256: {archive_sha256}\n",
        encoding="utf-8",
    )
    print(
        f"Created {DESTINATION} ({DESTINATION.stat().st_size:,} bytes) "
        f"with {len(files):,} files in {elapsed:.1f}s\n"
        f"SHA-256: {archive_sha256}"
    )


if __name__ == "__main__":
    main()
