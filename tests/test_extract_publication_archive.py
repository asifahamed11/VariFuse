from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "extract_publication_archive.py"
SPEC = importlib.util.spec_from_file_location("extract_publication_archive", MODULE_PATH)
assert SPEC and SPEC.loader
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def _write_provenance(archive: Path) -> None:
    sidecar = archive.with_name(archive.name + extractor.PROVENANCE_SUFFIX)
    sidecar.write_text(
        json.dumps(
            {
                "source_id": "test_zip_v1",
                "provider": "Test archive",
                "version": "1",
                "publisher_checksum": {"algorithm": "md5", "value": "test"},
                "observed_checksums": {
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )


def _make_valid_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("dataset/", b"")
        bundle.writestr("dataset/a.csv", b"variant,score\nA1G,0.5\n")
        bundle.writestr("dataset/nested/b.txt", b"publication fixture\n")


def test_safe_atomic_extraction_and_exact_verification(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_valid_zip(archive)
    _write_provenance(archive)
    destination = tmp_path / "extracted"

    manifest = extractor.extract_zip(archive, destination)

    manifest_path = destination / extractor.MANIFEST_NAME
    assert manifest_path.is_file()
    assert manifest["file_count"] == 2
    assert (destination / "dataset" / "a.csv").read_text(encoding="utf-8").startswith(
        "variant,score"
    )
    assert extractor.verify_extraction(manifest_path)["archive"]["source_id"] == "test_zip_v1"
    assert not list(tmp_path.glob("extracted.partial-*"))


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/absolute.txt", "C:/drive.txt"])
def test_zip_slip_paths_are_rejected_before_extraction(tmp_path, unsafe_name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(unsafe_name, b"unsafe")
    _write_provenance(archive)
    destination = tmp_path / "extracted"

    with pytest.raises(extractor.ExtractionError, match="unsafe|absolute"):
        extractor.extract_zip(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_symbolic_link_member_is_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("dataset/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "target")
    _write_provenance(archive)

    with pytest.raises(extractor.ExtractionError, match="symbolic links"):
        extractor.extract_zip(archive, tmp_path / "extracted")


def test_case_colliding_members_are_rejected(tmp_path):
    archive = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dataset/A.csv", b"a")
        bundle.writestr("dataset/a.csv", b"b")
    _write_provenance(archive)
    with pytest.raises(extractor.ExtractionError, match="case-colliding"):
        extractor.extract_zip(archive, tmp_path / "extracted")


def test_verifier_detects_tampered_or_extra_members(tmp_path):
    archive = tmp_path / "dataset.zip"
    _make_valid_zip(archive)
    _write_provenance(archive)
    destination = tmp_path / "extracted"
    extractor.extract_zip(archive, destination)
    manifest_path = destination / extractor.MANIFEST_NAME
    (destination / "dataset" / "a.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(extractor.ExtractionError, match="size changed|SHA256 changed"):
        extractor.verify_extraction(manifest_path)

    destination = tmp_path / "second"
    extractor.extract_zip(archive, destination)
    manifest_path = destination / extractor.MANIFEST_NAME
    (destination / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(extractor.ExtractionError, match="inventory differs"):
        extractor.verify_extraction(manifest_path)
