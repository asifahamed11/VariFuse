#!/usr/bin/env python3
"""Safely extract a provenance-bound publication ZIP archive.

Every member is validated before any extraction.  Files are streamed into a
same-filesystem staging directory, hashed individually, and atomically renamed
only after the complete inventory and extraction manifest have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence


CHUNK_BYTES = 8 * 1024**2
PROVENANCE_SUFFIX = ".provenance.json"
MANIFEST_NAME = "extraction_manifest.json"
DEFAULT_MAX_MEMBERS = 10_000
DEFAULT_MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
DEFAULT_MAX_COMPRESSION_RATIO = 200.0
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class ExtractionError(RuntimeError):
    """Raised when an archive or extraction violates the safety contract."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_stream(handle: BinaryIO, destination: BinaryIO | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = handle.read(CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
        size += len(block)
        if destination is not None:
            destination.write(block)
    return digest.hexdigest(), size


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _hash_stream(handle)[0]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normalised_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise ExtractionError(f"invalid empty/NUL ZIP member name: {name!r}")
    canonical = name.replace("\\", "/")
    if canonical.startswith("/") or re.match(r"^[A-Za-z]:", canonical):
        raise ExtractionError(f"absolute ZIP member path is forbidden: {name!r}")
    path = PurePosixPath(canonical)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ExtractionError(f"unsafe ZIP member path: {name!r}")
    for component in path.parts:
        if component.endswith((" ", ".")) or ":" in component:
            raise ExtractionError(f"Windows-unsafe ZIP member component: {component!r}")
        basename = component.split(".", maxsplit=1)[0].upper()
        if basename in WINDOWS_RESERVED:
            raise ExtractionError(f"reserved Windows ZIP member component: {component!r}")
    return path


def _member_kind(info: zipfile.ZipInfo) -> str:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if info.is_dir():
        return "directory"
    if file_type == stat.S_IFLNK:
        raise ExtractionError(f"symbolic links are forbidden in ZIP archives: {info.filename!r}")
    if file_type not in {0, stat.S_IFREG}:
        raise ExtractionError(f"non-regular ZIP member is forbidden: {info.filename!r}")
    return "file"


def inspect_zip(
    archive: Path,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> list[dict[str, Any]]:
    """Return a validated exact member inventory without extracting."""

    if max_members <= 0 or max_uncompressed_bytes <= 0 or max_compression_ratio <= 0:
        raise ExtractionError("all archive safety ceilings must be positive")
    try:
        archive_handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExtractionError(f"cannot open ZIP archive {archive}: {exc}") from exc
    with archive_handle as bundle:
        infos = bundle.infolist()
        if not infos:
            raise ExtractionError(f"ZIP archive is empty: {archive}")
        if len(infos) > max_members:
            raise ExtractionError(f"ZIP has {len(infos)} members; limit is {max_members}")
        inventory: list[dict[str, Any]] = []
        names: set[str] = set()
        total_size = 0
        total_compressed = 0
        for info in infos:
            if info.flag_bits & 0x1:
                raise ExtractionError(f"encrypted ZIP member is unsupported: {info.filename!r}")
            path = _normalised_member_path(info.filename)
            collision_key = str(path).casefold()
            if collision_key in names:
                raise ExtractionError(f"duplicate/case-colliding ZIP member: {info.filename!r}")
            names.add(collision_key)
            kind = _member_kind(info)
            total_size += int(info.file_size)
            total_compressed += int(info.compress_size)
            if total_size > max_uncompressed_bytes:
                raise ExtractionError(
                    f"ZIP expands beyond ceiling {max_uncompressed_bytes} bytes"
                )
            if info.file_size and info.compress_size == 0:
                raise ExtractionError(f"impossible compression metadata: {info.filename!r}")
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > max_compression_ratio:
                raise ExtractionError(
                    f"ZIP member compression ratio {ratio:.1f} exceeds limit "
                    f"{max_compression_ratio:.1f}: {info.filename!r}"
                )
            inventory.append(
                {
                    "path": str(path),
                    "kind": kind,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                }
            )
        overall_ratio = total_size / max(total_compressed, 1)
        if overall_ratio > max_compression_ratio:
            raise ExtractionError(
                f"ZIP overall compression ratio {overall_ratio:.1f} exceeds limit "
                f"{max_compression_ratio:.1f}"
            )
        return inventory


def _load_acquisition_provenance(archive: Path, archive_sha256: str) -> tuple[dict, Path, str]:
    provenance_path = archive.with_name(archive.name + PROVENANCE_SUFFIX)
    if not provenance_path.is_file():
        raise ExtractionError(
            f"archive lacks acquisition provenance: {provenance_path}; acquire it with "
            "tools/acquire_publication_data.py"
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"invalid acquisition provenance {provenance_path}: {exc}") from exc
    recorded = provenance.get("observed_checksums", {}).get("sha256")
    if recorded != archive_sha256:
        raise ExtractionError(
            f"archive differs from acquisition provenance: recorded={recorded!r}, "
            f"observed={archive_sha256}"
        )
    return provenance, provenance_path, _hash_file(provenance_path)


def _member_target(staging: Path, member_path: PurePosixPath) -> Path:
    target = (staging / Path(*member_path.parts)).resolve()
    try:
        target.relative_to(staging.resolve())
    except ValueError as exc:
        raise ExtractionError(f"ZIP member escapes staging directory: {member_path}") from exc
    return target


def extract_zip(
    archive: Path,
    destination: Path,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
) -> dict[str, Any]:
    """Safely extract a ZIP into an atomically completed destination."""

    archive = archive.resolve()
    destination = destination.resolve()
    if not archive.is_file():
        raise ExtractionError(f"archive does not exist: {archive}")
    if destination.exists():
        raise ExtractionError(
            f"destination already exists: {destination}; use a new path or move it aside"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".partial-{os.getpid()}")
    if staging.exists():
        raise ExtractionError(f"staging directory already exists: {staging}")

    archive_sha256 = _hash_file(archive)
    provenance, provenance_path, provenance_sha256 = _load_acquisition_provenance(
        archive, archive_sha256
    )
    planned = inspect_zip(
        archive,
        max_members=max_members,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_compression_ratio=max_compression_ratio,
    )
    planned_lookup = {entry["path"]: entry for entry in planned}
    extracted: list[dict[str, Any]] = []
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive) as bundle:
            for index, info in enumerate(bundle.infolist(), start=1):
                path = _normalised_member_path(info.filename)
                planned_entry = planned_lookup[str(path)]
                target = _member_target(staging, path)
                if planned_entry["kind"] == "directory":
                    if target.exists() and not target.is_dir():
                        raise ExtractionError(f"directory member collides with a file: {path}")
                    target.mkdir(parents=True, exist_ok=True)
                    extracted.append({**planned_entry, "sha256": None})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, "r") as source, target.open("xb") as output:
                    sha256, size = _hash_stream(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                if size != planned_entry["size_bytes"]:
                    raise ExtractionError(
                        f"extracted size mismatch for {path}: {size} != "
                        f"{planned_entry['size_bytes']}"
                    )
                extracted.append({**planned_entry, "sha256": sha256})
                if index % 50 == 0:
                    print(f"Extracted {index}/{len(planned)} members", flush=True)
        manifest = {
            "schema_version": 1,
            "extraction_policy": "safe_atomic_zip_v1",
            "created_at_utc": _utc_now(),
            "archive": {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "sha256": archive_sha256,
                "acquisition_provenance": str(provenance_path),
                "acquisition_provenance_sha256": provenance_sha256,
                "source_id": provenance.get("source_id"),
                "provider": provenance.get("provider"),
                "version": provenance.get("version"),
                "publisher_checksum": provenance.get("publisher_checksum"),
            },
            "destination": str(destination),
            "limits": {
                "max_members": max_members,
                "max_uncompressed_bytes": max_uncompressed_bytes,
                "max_compression_ratio": max_compression_ratio,
            },
            "member_count": len(extracted),
            "file_count": sum(entry["kind"] == "file" for entry in extracted),
            "total_uncompressed_bytes": sum(entry["size_bytes"] for entry in extracted),
            "members": extracted,
        }
        _atomic_json(staging / MANIFEST_NAME, manifest)
        os.replace(staging, destination)
        return manifest
    except Exception as exc:
        raise ExtractionError(
            f"extraction failed; incomplete staging was retained at {staging}: {exc}"
        ) from exc


def _filesystem_inventory(destination: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            raise ExtractionError(f"symbolic link found in extraction: {path}")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise ExtractionError(f"non-regular extracted object: {path}")
    return files, directories


def verify_extraction(manifest_path: Path) -> dict[str, Any]:
    """Verify archive binding, exact inventory, member sizes, and member SHA256s."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"cannot read extraction manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ExtractionError("unsupported extraction manifest schema")
    destination = Path(str(manifest.get("destination", "")))
    if manifest_path.resolve() != (destination / MANIFEST_NAME).resolve():
        raise ExtractionError("manifest is not located in its declared destination")
    archive_record = manifest.get("archive")
    if not isinstance(archive_record, dict):
        raise ExtractionError("manifest lacks archive record")
    archive = Path(str(archive_record.get("path", "")))
    if not archive.is_file():
        raise ExtractionError(f"source archive is missing: {archive}")
    if archive.stat().st_size != archive_record.get("size_bytes"):
        raise ExtractionError(f"source archive size changed: {archive}")
    if _hash_file(archive) != archive_record.get("sha256"):
        raise ExtractionError(f"source archive SHA256 changed: {archive}")

    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != manifest.get("member_count"):
        raise ExtractionError("manifest has an invalid member inventory")
    expected_files = {entry["path"] for entry in members if entry.get("kind") == "file"}
    expected_directories = {
        entry["path"].rstrip("/") for entry in members if entry.get("kind") == "directory"
    }
    actual_files, actual_directories = _filesystem_inventory(destination)
    expected_files_with_manifest = expected_files | {MANIFEST_NAME}
    if actual_files != expected_files_with_manifest:
        raise ExtractionError(
            f"extracted file inventory differs: missing={sorted(expected_files_with_manifest - actual_files)}, "
            f"extra={sorted(actual_files - expected_files_with_manifest)}"
        )
    if not expected_directories.issubset(actual_directories):
        raise ExtractionError(
            f"extracted directories missing: {sorted(expected_directories - actual_directories)}"
        )
    for index, entry in enumerate(members, start=1):
        if entry.get("kind") != "file":
            continue
        target = destination / Path(*PurePosixPath(entry["path"]).parts)
        if target.stat().st_size != entry.get("size_bytes"):
            raise ExtractionError(f"member size changed: {target}")
        if _hash_file(target) != entry.get("sha256"):
            raise ExtractionError(f"member SHA256 changed: {target}")
        if index % 50 == 0:
            print(f"Verified {index}/{len(members)} members", flush=True)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely extract a provenance-bound ZIP archive.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--dest", type=Path, required=True)
    extract.add_argument("--max-members", type=int, default=DEFAULT_MAX_MEMBERS)
    extract.add_argument(
        "--max-uncompressed-gb",
        type=float,
        default=DEFAULT_MAX_UNCOMPRESSED_BYTES / 1024**3,
    )
    extract.add_argument(
        "--max-compression-ratio", type=float, default=DEFAULT_MAX_COMPRESSION_RATIO
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            manifest = verify_extraction(args.manifest.resolve())
            print(
                f"Verified {manifest['file_count']} files from "
                f"{manifest['archive']['source_id']}"
            )
            return 0
        maximum = int(args.max_uncompressed_gb * 1024**3)
        manifest = extract_zip(
            args.archive,
            args.dest,
            max_members=args.max_members,
            max_uncompressed_bytes=maximum,
            max_compression_ratio=args.max_compression_ratio,
        )
        print(
            f"Extracted {manifest['file_count']} files ({manifest['total_uncompressed_bytes']} bytes) "
            f"to {manifest['destination']}"
        )
        return 0
    except ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
