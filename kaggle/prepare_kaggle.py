"""Merge the paired code/data exports into a writable, hash-verified workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

MARKERS = {kind: f"VARIFUSE_{kind.upper()}_MANIFEST.json" for kind in ("code", "data")}


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or ".." in path.parts
        or value == "."
    ):
        raise ValueError(f"Unsafe package path: {value}")
    return path


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Package:
    def __init__(self, path: Path, kind: str):
        self.path, self.kind = path, kind
        self.archive = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None
        marker = MARKERS[kind]
        if self.archive:
            entries = [
                name for name in self.archive.namelist() if PurePosixPath(name).name == marker
            ]
            if len(entries) != 1:
                raise ValueError(f"Expected one {marker} in {path}")
            self.prefix = str(PurePosixPath(entries[0]).parent)
            self.raw = self.archive.read(entries[0])
        else:
            self.prefix = ""
            self.raw = (path / marker).read_bytes()
        self.manifest = json.loads(self.raw)
        if self.manifest.get("kind") != kind or self.manifest.get("format_version") != 1:
            raise ValueError(f"Invalid {kind} manifest")
        paths = [record["path"] for record in self.manifest["files"]]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate package entries")
        for value in paths:
            relative_path(value)

    def close(self) -> None:
        if self.archive:
            self.archive.close()

    def open(self, record: dict):
        relative = relative_path(record["path"])
        if self.archive:
            name = str(PurePosixPath(self.prefix) / relative)
            info = self.archive.getinfo(name)
            if info.file_size != record["size_bytes"] or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Invalid ZIP member: {name}")
            return self.archive.open(info)
        source = self.path.joinpath(*relative.parts).resolve()
        if not source.is_relative_to(self.path.resolve()) or not source.is_file():
            raise ValueError(f"Missing or unsafe package source: {relative}")
        return source.open("rb")


def discover(input_root: Path, kind: str) -> Package:
    expanded = list(input_root.rglob(MARKERS[kind]))
    if len(expanded) == 1:
        return Package(expanded[0].parent, kind)
    if len(expanded) > 1:
        raise ValueError(f"Multiple {kind} packages mounted; attach one matching export")
    archives = list(input_root.rglob(f"VariFuse_Kaggle_{kind.title()}.zip"))
    if len(archives) != 1:
        raise ValueError(f"Attach the {kind} ZIP or its extracted dataset; found {len(archives)}")
    return Package(archives[0], kind)


def prepare(code: Package, data: Package, work_root: Path) -> dict:
    if code.manifest["pair_id"] != data.manifest["pair_id"]:
        raise ValueError("Code and data ZIPs belong to different exports")
    overlap = {r["path"] for r in code.manifest["files"]} & {
        r["path"] for r in data.manifest["files"]
    }
    if overlap:
        raise ValueError(f"Code/data paths overlap: {sorted(overlap)}")
    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    needed = sum(
        r["size_bytes"]
        for p in (code, data)
        for r in p.manifest["files"]
        if not (work_root / r["path"]).exists()
    )
    if shutil.disk_usage(work_root).free < needed:
        raise RuntimeError(f"Insufficient space: need {needed / 2**30:.2f} GiB for package files")
    copied = verified = 0
    for package in (code, data):
        for record in package.manifest["files"]:
            target = work_root.joinpath(*relative_path(record["path"]).parts).resolve()
            if not target.is_relative_to(work_root):
                raise ValueError("Destination escapes workspace")
            if target.exists():
                if (
                    not target.is_file()
                    or target.stat().st_size != record["size_bytes"]
                    or sha(target) != record["sha256"]
                ):
                    raise ValueError(
                        f"Existing file differs from export: {target}. Use a fresh workspace."
                    )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".import.tmp")
                digest, size = hashlib.sha256(), 0
                with package.open(record) as source, temporary.open("wb") as destination:
                    while block := source.read(8 << 20):
                        size += len(block)
                        if size > record["size_bytes"]:
                            raise ValueError(f"Unexpected source size: {target}")
                        digest.update(block)
                        destination.write(block)
                if size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Package checksum failed: {target}")
                temporary.replace(target)
                copied += 1
            verified += 1
            if verified % 500 == 0 or record["size_bytes"] > 1 << 30:
                print(f"Verified {verified} files: {record['path']}", flush=True)
        (work_root / MARKERS[package.kind]).write_bytes(package.raw)
    report = {
        "status": "passed",
        "pair_id": code.manifest["pair_id"],
        "verified_files": verified,
        "copied_files": copied,
        "workspace": str(work_root),
    }
    (work_root / "kaggle_import_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--work-root", type=Path, default=Path("/kaggle/working/VariFuse_2"))
    args = parser.parse_args()
    code, data = discover(args.input_root, "code"), discover(args.input_root, "data")
    try:
        prepare(code, data, args.work_root)
    finally:
        code.close()
        data.close()


if __name__ == "__main__":
    main()
