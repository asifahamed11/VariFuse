from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "acquire_publication_data.py"
SPEC = importlib.util.spec_from_file_location("acquire_publication_data", MODULE_PATH)
assert SPEC and SPEC.loader
acquire = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acquire
SPEC.loader.exec_module(acquire)


def _source(url: str, payload: bytes, *, version: str = "test-1") -> dict:
    return {
        "provider": "Test archive",
        "version": version,
        "release_date": "2025-01-02",
        "description": "Local deterministic fixture.",
        "filename": "fixture.bin",
        "relative_directory": "fixture/test-1",
        "urls": [url],
        "size_bytes": len(payload),
        "checksum": {
            "algorithm": "md5",
            "value": hashlib.md5(payload).hexdigest(),
            "provenance": "test fixture",
        },
        "checksum_policy": "verify_publisher_checksum_and_record_sha256",
        "license": {
            "name": "CC0 1.0",
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "note": "Test data.",
        },
        "documentation_url": "https://example.org/docs/test-1",
        "citation_url": "https://example.org/cite/test-1",
    }


def _catalog(source: dict) -> dict:
    return {
        "schema_version": 1,
        "catalog_version": "test-1",
        "generated_date": "2025-01-02",
        "default_destination": "data/test",
        "profiles": {"test_profile": ["test_source"]},
        "sources": {"test_source": source},
    }


@contextmanager
def _range_server(payload: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/fixture.bin":
                self.send_error(404)
                return
            range_header = self.headers.get("Range")
            if range_header:
                start = int(range_header.removeprefix("bytes=").split("-", maxsplit=1)[0])
                body = payload[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"test-etag"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/fixture.bin"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_checked_in_catalog_is_strict_and_pinned():
    catalog = acquire.load_catalog(ROOT / "tools" / "publication_data_catalog.json")
    assert catalog["schema_version"] == 1
    assert "publication_core" in catalog["profiles"]
    assert "clinvar_vcv_2024_06" in catalog["sources"]
    assert "proteingym_dms_substitutions_v1_3" in catalog["sources"]
    assert "mavedb_bulk_2026_06_24" in catalog["sources"]
    assert all(
        "latest" not in url.lower()
        for source in catalog["sources"].values()
        for url in source["urls"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "latest"),
        ("urls", ["https://example.org/releases/dataset-latest.zip"]),
    ],
)
def test_catalog_rejects_ambiguous_releases(field, value):
    payload = b"fixture"
    source = _source("https://example.org/releases/test-1/fixture.bin", payload)
    source[field] = value
    with pytest.raises(acquire.CatalogError, match="ambiguous|moving 'latest'"):
        acquire.validate_catalog(_catalog(source))


def test_profile_resolution_is_stable_and_deduplicated():
    payload = b"fixture"
    catalog = _catalog(_source("https://example.org/test-1/fixture.bin", payload))
    acquire.validate_catalog(catalog)
    assert acquire.resolve_source_ids(
        catalog, ["test_source"], ["test_profile"]
    ) == ["test_source"]


def test_resumable_download_records_provenance_and_verifies(tmp_path):
    payload = (b"publication-grade-fixture-" * 4096) + b"done"
    with _range_server(payload) as url:
        catalog = _catalog(_source(url, payload))
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        loaded = acquire.load_catalog(catalog_path)
        destination = tmp_path / "downloads"
        plan = acquire.make_plan(loaded, ["test_source"], destination)
        target = plan[0].destination
        target.parent.mkdir(parents=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(payload[:777])

        records = acquire.acquire_plan(
            plan,
            loaded,
            catalog_path,
            destination,
            maximum_bytes=1024**2,
            allow_large=False,
            chunk_bytes=1024,
        )

    assert target.read_bytes() == payload
    assert records[0]["status"] == "downloaded"
    provenance = json.loads(
        target.with_name(target.name + acquire.PROVENANCE_SUFFIX).read_text(encoding="utf-8")
    )
    assert provenance["resumed"] is True
    assert provenance["observed_checksums"]["sha256"] == hashlib.sha256(payload).hexdigest()
    verified = acquire.verify_manifest(destination / acquire.MANIFEST_NAME)
    assert verified == [
        {
            "source_id": "test_source",
            "path": str(target),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "ok": True,
        }
    ]


def test_large_files_require_two_explicit_safety_controls(tmp_path):
    item = acquire.PlannedFile(
        source_id="large_test",
        provider="Test",
        version="1",
        release_date="2025-01-02",
        description="Synthetic plan entry.",
        destination=tmp_path / "large.bin",
        size_bytes=acquire.DEFAULT_LARGE_THRESHOLD + 1,
        checksum=None,
        license={"name": "CC0"},
        urls=("https://example.org/large-1.bin",),
    )
    with pytest.raises(acquire.AcquisitionError, match="max-file-gb"):
        acquire._guard_file_sizes([item], acquire.DEFAULT_LARGE_THRESHOLD, allow_large=True)
    with pytest.raises(acquire.AcquisitionError, match="safety gate"):
        acquire._guard_file_sizes(
            [item], acquire.DEFAULT_LARGE_THRESHOLD + 1, allow_large=False
        )
    acquire._guard_file_sizes([item], acquire.DEFAULT_LARGE_THRESHOLD + 1, allow_large=True)


def test_no_checksum_archive_requires_explicit_recorded_sha256_policy():
    payload = b"fixture"
    source = _source("https://example.org/test-1/fixture.bin", payload)
    source["checksum"] = None
    source["checksum_policy"] = "trust_remote"
    with pytest.raises(acquire.CatalogError, match="recorded-SHA256 policy"):
        acquire.validate_catalog(_catalog(source))
