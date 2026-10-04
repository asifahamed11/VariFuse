#!/usr/bin/env python3
"""Acquire pinned public datasets with resumable, publication-grade provenance.

This module intentionally uses only the Python standard library.  Catalog entries
must identify a fixed release; moving ``latest``/``current`` URLs are rejected.
Large files require both an explicit size ceiling and ``--allow-large``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


CATALOG_PATH = Path(__file__).with_name("publication_data_catalog.json")
DEFAULT_LARGE_THRESHOLD = 1024**3
DEFAULT_MAX_FILE_BYTES = DEFAULT_LARGE_THRESHOLD
DEFAULT_CHUNK_BYTES = 8 * 1024**2
DEFAULT_TIMEOUT_SECONDS = 120
MANIFEST_NAME = "acquisition_manifest.json"
PROVENANCE_SUFFIX = ".provenance.json"
USER_AGENT = "VariFuse-publication-data/1.0"
AMBIGUOUS_VERSION_WORDS = {"", "current", "latest", "rolling", "unknown", "unversioned"}
HEX_LENGTHS = {"md5": 32, "sha256": 64}


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot proceed safely."""


class CatalogError(AcquisitionError):
    """Raised when a catalog violates the reproducibility contract."""


@dataclass(frozen=True)
class PlannedFile:
    source_id: str
    provider: str
    version: str
    release_date: str
    description: str
    destination: Path
    size_bytes: int
    checksum: Mapping[str, str] | None
    license: Mapping[str, str]
    urls: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        payload = _json_bytes(value)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _catalog_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DEFAULT_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{where} must be a JSON object")
    return value


def _require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where} must be a non-empty string")
    return value.strip()


def _validate_url(url: str, where: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise CatalogError(f"{where} must use HTTPS (HTTP is allowed only for loopback tests)")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise CatalogError(f"{where} uses insecure HTTP for a non-loopback host")
    if not parsed.netloc:
        raise CatalogError(f"{where} has no host")
    if re.search(r"(^|[-_/])latest($|[-_/.])", parsed.path, flags=re.IGNORECASE):
        raise CatalogError(f"{where} is a moving 'latest' URL; pin an immutable release")


def _validate_source(source_id: str, raw: Any) -> None:
    source = _require_mapping(raw, f"sources.{source_id}")
    for field in (
        "provider",
        "version",
        "release_date",
        "description",
        "filename",
        "relative_directory",
        "checksum_policy",
        "documentation_url",
        "citation_url",
    ):
        _require_text(source.get(field), f"sources.{source_id}.{field}")

    version = str(source["version"]).strip().lower()
    if version in AMBIGUOUS_VERSION_WORDS:
        raise CatalogError(f"sources.{source_id}.version is ambiguous: {source['version']!r}")
    try:
        date.fromisoformat(str(source["release_date"]))
    except ValueError as exc:
        raise CatalogError(
            f"sources.{source_id}.release_date must be ISO YYYY-MM-DD"
        ) from exc

    filename = str(source["filename"])
    if Path(filename).name != filename or filename in {".", ".."}:
        raise CatalogError(f"sources.{source_id}.filename must be a plain filename")
    relative = PurePosixPath(str(source["relative_directory"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError(f"sources.{source_id}.relative_directory escapes the destination")

    size = source.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise CatalogError(f"sources.{source_id}.size_bytes must be a positive integer")

    urls = source.get("urls")
    if not isinstance(urls, list) or not urls:
        raise CatalogError(f"sources.{source_id}.urls must be a non-empty list")
    for index, raw_url in enumerate(urls):
        url = _require_text(raw_url, f"sources.{source_id}.urls[{index}]")
        _validate_url(url, f"sources.{source_id}.urls[{index}]")

    checksum = source.get("checksum")
    policy = str(source["checksum_policy"])
    if checksum is None:
        if policy != "publisher_has_no_archive_checksum; verify_size_and_record_sha256":
            raise CatalogError(
                f"sources.{source_id} has no publisher checksum without the explicit "
                "recorded-SHA256 policy"
            )
    else:
        checksum_map = _require_mapping(checksum, f"sources.{source_id}.checksum")
        algorithm = _require_text(
            checksum_map.get("algorithm"), f"sources.{source_id}.checksum.algorithm"
        ).lower()
        value = _require_text(
            checksum_map.get("value"), f"sources.{source_id}.checksum.value"
        ).lower()
        _require_text(
            checksum_map.get("provenance"), f"sources.{source_id}.checksum.provenance"
        )
        if algorithm not in HEX_LENGTHS:
            raise CatalogError(f"sources.{source_id} uses unsupported checksum {algorithm!r}")
        if not re.fullmatch(rf"[0-9a-f]{{{HEX_LENGTHS[algorithm]}}}", value):
            raise CatalogError(f"sources.{source_id} has an invalid {algorithm} checksum")
        if policy != "verify_publisher_checksum_and_record_sha256":
            raise CatalogError(f"sources.{source_id} has a checksum but an incompatible policy")

    license_info = _require_mapping(source.get("license"), f"sources.{source_id}.license")
    _require_text(license_info.get("name"), f"sources.{source_id}.license.name")
    license_url = _require_text(license_info.get("url"), f"sources.{source_id}.license.url")
    _require_text(license_info.get("note"), f"sources.{source_id}.license.note")
    _validate_url(license_url, f"sources.{source_id}.license.url")
    _validate_url(str(source["documentation_url"]), f"sources.{source_id}.documentation_url")
    _validate_url(str(source["citation_url"]), f"sources.{source_id}.citation_url")


def validate_catalog(catalog: Mapping[str, Any]) -> None:
    """Validate the complete acquisition catalog and reject moving releases."""

    if catalog.get("schema_version") != 1:
        raise CatalogError("catalog schema_version must equal 1")
    _require_text(catalog.get("catalog_version"), "catalog_version")
    _require_text(catalog.get("generated_date"), "generated_date")
    default_destination = _require_text(
        catalog.get("default_destination"), "default_destination"
    )
    default_path = PurePosixPath(default_destination)
    if default_path.is_absolute() or ".." in default_path.parts:
        raise CatalogError("default_destination must remain inside the project")

    sources = _require_mapping(catalog.get("sources"), "sources")
    if not sources:
        raise CatalogError("catalog contains no sources")
    targets: set[str] = set()
    for source_id, source in sources.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", str(source_id)):
            raise CatalogError(f"invalid source id: {source_id!r}")
        _validate_source(str(source_id), source)
        source_map = _require_mapping(source, f"sources.{source_id}")
        target = f"{source_map['relative_directory']}/{source_map['filename']}"
        if target in targets:
            raise CatalogError(f"multiple sources resolve to the same target: {target}")
        targets.add(target)

    profiles = _require_mapping(catalog.get("profiles"), "profiles")
    for profile, members in profiles.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", str(profile)):
            raise CatalogError(f"invalid profile id: {profile!r}")
        if not isinstance(members, list) or not members:
            raise CatalogError(f"profiles.{profile} must be a non-empty list")
        if len(members) != len(set(members)):
            raise CatalogError(f"profiles.{profile} contains duplicate source ids")
        unknown = sorted(set(members) - set(sources))
        if unknown:
            raise CatalogError(f"profiles.{profile} refers to unknown sources: {unknown}")


def load_catalog(path: Path | str = CATALOG_PATH) -> dict[str, Any]:
    """Load and strictly validate a JSON catalog."""

    catalog_path = Path(path)
    try:
        with catalog_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot load catalog {catalog_path}: {exc}") from exc
    catalog = dict(_require_mapping(raw, "catalog"))
    validate_catalog(catalog)
    return catalog


def resolve_source_ids(
    catalog: Mapping[str, Any],
    source_ids: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
) -> list[str]:
    """Resolve explicit sources and profiles in stable, duplicate-free order."""

    source_ids = list(source_ids or [])
    profiles = list(profiles or [])
    if not source_ids and not profiles:
        raise AcquisitionError("select at least one --source or --profile")
    sources = _require_mapping(catalog.get("sources"), "sources")
    profile_map = _require_mapping(catalog.get("profiles"), "profiles")

    expanded: list[str] = []
    for profile in profiles:
        if profile not in profile_map:
            raise AcquisitionError(f"unknown profile {profile!r}")
        expanded.extend(str(item) for item in profile_map[profile])
    expanded.extend(source_ids)

    resolved: list[str] = []
    seen: set[str] = set()
    for source_id in expanded:
        if source_id not in sources:
            raise AcquisitionError(f"unknown source {source_id!r}")
        if source_id not in seen:
            seen.add(source_id)
            resolved.append(source_id)
    return resolved


def make_plan(
    catalog: Mapping[str, Any], source_ids: Sequence[str], destination_root: Path
) -> list[PlannedFile]:
    """Build a deterministic download plan without touching the network."""

    sources = _require_mapping(catalog["sources"], "sources")
    plan: list[PlannedFile] = []
    root = destination_root.resolve()
    for source_id in source_ids:
        source = _require_mapping(sources[source_id], f"sources.{source_id}")
        relative = Path(str(source["relative_directory"])) / str(source["filename"])
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise AcquisitionError(f"source {source_id!r} escapes destination root") from exc
        raw_checksum = source.get("checksum")
        checksum = dict(raw_checksum) if isinstance(raw_checksum, dict) else None
        plan.append(
            PlannedFile(
                source_id=source_id,
                provider=str(source["provider"]),
                version=str(source["version"]),
                release_date=str(source["release_date"]),
                description=str(source["description"]),
                destination=target,
                size_bytes=int(source["size_bytes"]),
                checksum=checksum,
                license=dict(_require_mapping(source["license"], f"sources.{source_id}.license")),
                urls=tuple(str(url) for url in source["urls"]),
            )
        )
    return plan


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def plan_as_records(plan: Sequence[PlannedFile]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": item.source_id,
            "provider": item.provider,
            "version": item.version,
            "release_date": item.release_date,
            "description": item.description,
            "destination": str(item.destination),
            "size_bytes": item.size_bytes,
            "size_human": human_bytes(item.size_bytes),
            "large_file": item.size_bytes > DEFAULT_LARGE_THRESHOLD,
            "publisher_checksum": item.checksum,
            "license": item.license,
            "urls": list(item.urls),
        }
        for item in plan
    ]


def _hash_file(path: Path, algorithms: Iterable[str]) -> dict[str, str]:
    names = tuple(dict.fromkeys(str(name).lower() for name in algorithms))
    hashers = {name: hashlib.new(name) for name in names}
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DEFAULT_CHUNK_BYTES), b""):
            for digest in hashers.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in hashers.items()}


def _provenance_path(path: Path) -> Path:
    return path.with_name(path.name + PROVENANCE_SUFFIX)


def _verified_existing(
    item: PlannedFile, catalog_digest: str
) -> tuple[dict[str, str], dict[str, Any]] | None:
    path = item.destination
    if not path.exists():
        return None
    if not path.is_file():
        raise AcquisitionError(f"destination exists but is not a file: {path}")
    if path.stat().st_size != item.size_bytes:
        raise AcquisitionError(
            f"existing file has the wrong size: {path} "
            f"({path.stat().st_size} != {item.size_bytes}); move it aside before retrying"
        )

    algorithms = ["sha256"]
    if item.checksum:
        algorithms.append(item.checksum["algorithm"])
    hashes = _hash_file(path, algorithms)
    if item.checksum:
        algorithm = item.checksum["algorithm"]
        if hashes[algorithm] != item.checksum["value"]:
            raise AcquisitionError(
                f"existing file fails publisher {algorithm} checksum: {path}; move it aside"
            )
    else:
        sidecar = _provenance_path(path)
        if not sidecar.exists():
            raise AcquisitionError(
                f"existing file has no publisher checksum and no provenance sidecar: {path}; "
                "move it aside so it can be acquired and fingerprinted"
            )
        try:
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcquisitionError(f"invalid provenance sidecar for {path}: {exc}") from exc
        recorded = provenance.get("observed_checksums", {}).get("sha256")
        if not recorded or recorded != hashes["sha256"]:
            raise AcquisitionError(f"existing file fails recorded SHA256: {path}")

    provenance_record = {
        "source_id": item.source_id,
        "catalog_sha256": catalog_digest,
        "verified_at_utc": _utc_now(),
        "acquisition_mode": "verified_existing",
        "observed_size_bytes": item.size_bytes,
        "observed_checksums": hashes,
    }
    return hashes, provenance_record


def _header_subset(headers: Mapping[str, str]) -> dict[str, str]:
    wanted = ("Content-Length", "Content-Range", "ETag", "Last-Modified", "Content-Type")
    return {name: str(headers[name]) for name in wanted if headers.get(name) is not None}


def _stream_response(
    response: BinaryIO,
    partial: Path,
    resume_at: int,
    expected_size: int,
    maximum_bytes: int,
    chunk_bytes: int,
) -> tuple[int, bool]:
    status = int(getattr(response, "status", 200) or 200)
    append = resume_at > 0 and status == 206
    if resume_at > 0 and status == 206:
        content_range = response.headers.get("Content-Range", "")  # type: ignore[attr-defined]
        if not content_range.startswith(f"bytes {resume_at}-"):
            raise AcquisitionError(
                f"server returned an invalid Content-Range for resume: {content_range!r}"
            )
    elif resume_at > 0 and status != 200:
        raise AcquisitionError(f"server returned HTTP {status} for a resumed request")

    bytes_written = resume_at if append else 0
    mode = "ab" if append else "wb"
    with partial.open(mode) as handle:
        while True:
            block = response.read(chunk_bytes)
            if not block:
                break
            handle.write(block)
            bytes_written += len(block)
            if bytes_written > expected_size:
                raise AcquisitionError(
                    f"download exceeded catalog size ({bytes_written} > {expected_size})"
                )
            if bytes_written > maximum_bytes:
                raise AcquisitionError(
                    f"download exceeded configured ceiling ({human_bytes(maximum_bytes)})"
                )
        handle.flush()
        os.fsync(handle.fileno())
    return bytes_written, append


def _download_to_partial(
    item: PlannedFile,
    maximum_bytes: int,
    timeout_seconds: int,
    chunk_bytes: int,
) -> tuple[str, dict[str, str], bool]:
    partial = item.destination.with_name(item.destination.name + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > item.size_bytes:
        raise AcquisitionError(
            f"partial file exceeds catalog size: {partial}; move it aside before retrying"
        )
    if partial_size == item.size_bytes:
        return "completed-partial", {}, True

    errors: list[str] = []
    for url in item.urls:
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if partial_size:
            headers["Range"] = f"bytes={partial_size}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_headers = _header_subset(response.headers)
                bytes_written, resumed = _stream_response(
                    response,
                    partial,
                    partial_size,
                    item.size_bytes,
                    maximum_bytes,
                    chunk_bytes,
                )
            if bytes_written != item.size_bytes:
                raise AcquisitionError(
                    f"incomplete response from {url}: {bytes_written} != {item.size_bytes}; "
                    "the .part file was kept for resume"
                )
            return url, response_headers, resumed
        except (HTTPError, URLError, TimeoutError, OSError, AcquisitionError) as exc:
            errors.append(f"{url}: {exc}")
            partial_size = partial.stat().st_size if partial.exists() else 0
            if partial_size > item.size_bytes:
                break
    raise AcquisitionError("all source URLs failed:\n  " + "\n  ".join(errors))


def _verify_downloaded_partial(item: PlannedFile) -> dict[str, str]:
    partial = item.destination.with_name(item.destination.name + ".part")
    if not partial.exists() or partial.stat().st_size != item.size_bytes:
        observed = partial.stat().st_size if partial.exists() else 0
        raise AcquisitionError(
            f"cannot finalize incomplete file {partial}: {observed} != {item.size_bytes}"
        )
    algorithms = ["sha256"]
    if item.checksum:
        algorithms.append(item.checksum["algorithm"])
    hashes = _hash_file(partial, algorithms)
    if item.checksum:
        algorithm = item.checksum["algorithm"]
        expected = item.checksum["value"]
        if hashes[algorithm] != expected:
            raise AcquisitionError(
                f"publisher checksum mismatch for {partial}: "
                f"{hashes[algorithm]} != {expected}; the .part file was retained"
            )
    return hashes


def _source_catalog_record(catalog: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    return _require_mapping(
        _require_mapping(catalog["sources"], "sources")[source_id], f"sources.{source_id}"
    )


def acquire_one(
    item: PlannedFile,
    catalog: Mapping[str, Any],
    catalog_digest: str,
    maximum_bytes: int,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Acquire or verify one planned file, then atomically finalize it."""

    existing = _verified_existing(item, catalog_digest)
    if existing is not None:
        hashes, minimal = existing
        sidecar_path = _provenance_path(item.destination)
        if not sidecar_path.exists():
            source = _source_catalog_record(catalog, item.source_id)
            provenance = {
                **minimal,
                "schema_version": 1,
                "provider": item.provider,
                "version": item.version,
                "release_date": item.release_date,
                "path": str(item.destination),
                "publisher_checksum": item.checksum,
                "checksum_policy": source["checksum_policy"],
                "license": item.license,
                "documentation_url": source["documentation_url"],
                "citation_url": source["citation_url"],
            }
            _atomic_write_json(sidecar_path, provenance)
        return {
            "source_id": item.source_id,
            "path": str(item.destination),
            "size_bytes": item.size_bytes,
            "sha256": hashes["sha256"],
            "status": "verified_existing",
            "provenance": str(sidecar_path),
        }

    used_url, response_headers, resumed = _download_to_partial(
        item,
        maximum_bytes=maximum_bytes,
        timeout_seconds=timeout_seconds,
        chunk_bytes=chunk_bytes,
    )
    hashes = _verify_downloaded_partial(item)
    partial = item.destination.with_name(item.destination.name + ".part")
    os.replace(partial, item.destination)

    source = _source_catalog_record(catalog, item.source_id)
    provenance = {
        "schema_version": 1,
        "source_id": item.source_id,
        "provider": item.provider,
        "version": item.version,
        "release_date": item.release_date,
        "description": item.description,
        "path": str(item.destination),
        "selected_url": used_url,
        "catalog_urls": list(item.urls),
        "fetched_at_utc": _utc_now(),
        "resumed": resumed,
        "response_headers": response_headers,
        "catalog_sha256": catalog_digest,
        "expected_size_bytes": item.size_bytes,
        "observed_size_bytes": item.destination.stat().st_size,
        "publisher_checksum": item.checksum,
        "checksum_policy": source["checksum_policy"],
        "observed_checksums": hashes,
        "license": item.license,
        "documentation_url": source["documentation_url"],
        "citation_url": source["citation_url"],
    }
    sidecar = _provenance_path(item.destination)
    _atomic_write_json(sidecar, provenance)
    return {
        "source_id": item.source_id,
        "path": str(item.destination),
        "size_bytes": item.size_bytes,
        "sha256": hashes["sha256"],
        "status": "downloaded",
        "provenance": str(sidecar),
    }


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise AcquisitionError(f"cannot find an existing ancestor for {path}")
        candidate = parent
    return candidate


def _remaining_download_bytes(plan: Sequence[PlannedFile]) -> int:
    total = 0
    for item in plan:
        if item.destination.exists() and item.destination.is_file():
            continue
        partial = item.destination.with_name(item.destination.name + ".part")
        partial_size = partial.stat().st_size if partial.exists() else 0
        total += max(item.size_bytes - partial_size, 0)
    return total


def ensure_disk_capacity(destination_root: Path, required_bytes: int) -> dict[str, int]:
    """Require download bytes plus a conservative working-space reserve."""

    ancestor = _existing_ancestor(destination_root)
    usage = shutil.disk_usage(ancestor)
    reserve = max(256 * 1024**2, int(required_bytes * 0.10))
    required_with_reserve = required_bytes + reserve
    if usage.free < required_with_reserve:
        raise AcquisitionError(
            f"insufficient disk space on {ancestor}: need {human_bytes(required_with_reserve)}, "
            f"have {human_bytes(usage.free)}"
        )
    return {
        "free_bytes": usage.free,
        "download_bytes": required_bytes,
        "reserve_bytes": reserve,
        "required_with_reserve_bytes": required_with_reserve,
    }


def _guard_file_sizes(
    plan: Sequence[PlannedFile], maximum_bytes: int, allow_large: bool
) -> None:
    if maximum_bytes <= 0:
        raise AcquisitionError("--max-file-gb must be positive")
    too_large = [item for item in plan if item.size_bytes > maximum_bytes]
    if too_large:
        details = ", ".join(
            f"{item.source_id}={human_bytes(item.size_bytes)}" for item in too_large
        )
        raise AcquisitionError(
            f"selected files exceed --max-file-gb ({human_bytes(maximum_bytes)}): {details}"
        )
    gated = [item for item in plan if item.size_bytes > DEFAULT_LARGE_THRESHOLD]
    if gated and not allow_large:
        details = ", ".join(
            f"{item.source_id}={human_bytes(item.size_bytes)}" for item in gated
        )
        raise AcquisitionError(
            "large-file safety gate blocked: "
            f"{details}. Re-run with --allow-large and an adequate --max-file-gb."
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "files": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read acquisition manifest {path}: {exc}") from exc
    manifest = dict(_require_mapping(raw, "acquisition manifest"))
    if manifest.get("schema_version") != 1:
        raise AcquisitionError(f"unsupported acquisition manifest schema in {path}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise AcquisitionError(f"acquisition manifest has invalid files object: {path}")
    return manifest


def acquire_plan(
    plan: Sequence[PlannedFile],
    catalog: Mapping[str, Any],
    catalog_path: Path,
    destination_root: Path,
    maximum_bytes: int,
    allow_large: bool,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> list[dict[str, Any]]:
    """Acquire a selected plan and update one deterministic manifest."""

    _guard_file_sizes(plan, maximum_bytes, allow_large)
    disk = ensure_disk_capacity(destination_root, _remaining_download_bytes(plan))
    print(
        f"Disk preflight: {human_bytes(disk['free_bytes'])} free; "
        f"{human_bytes(disk['download_bytes'])} remaining"
    )
    catalog_digest = _catalog_hash(catalog_path)
    manifest_path = destination_root / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    manifest["catalog_version"] = catalog["catalog_version"]
    manifest["catalog_sha256"] = catalog_digest
    manifest["updated_at_utc"] = _utc_now()
    manifest["destination_root"] = str(destination_root.resolve())
    files = manifest["files"]

    results: list[dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        print(
            f"[{index}/{len(plan)}] {item.source_id}: {human_bytes(item.size_bytes)} -> "
            f"{item.destination}"
        )
        record = acquire_one(
            item,
            catalog,
            catalog_digest,
            maximum_bytes=maximum_bytes,
            timeout_seconds=timeout_seconds,
            chunk_bytes=chunk_bytes,
        )
        files[item.source_id] = record
        manifest["updated_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        results.append(record)
        print(f"  {record['status']}; SHA256={record['sha256']}")
    return results


def verify_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Re-hash every file recorded in an acquisition manifest."""

    manifest = _load_manifest(manifest_path)
    failures: list[str] = []
    results: list[dict[str, Any]] = []
    for source_id, raw in sorted(manifest["files"].items()):
        record = _require_mapping(raw, f"manifest.files.{source_id}")
        path = Path(_require_text(record.get("path"), f"manifest.files.{source_id}.path"))
        expected_size = record.get("size_bytes")
        expected_sha256 = _require_text(
            record.get("sha256"), f"manifest.files.{source_id}.sha256"
        )
        if not path.is_file():
            failures.append(f"{source_id}: missing {path}")
            continue
        size = path.stat().st_size
        sha256 = _hash_file(path, ["sha256"])["sha256"]
        ok = size == expected_size and sha256 == expected_sha256
        results.append(
            {
                "source_id": source_id,
                "path": str(path),
                "size_bytes": size,
                "sha256": sha256,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(
                f"{source_id}: expected size/SHA256 {expected_size}/{expected_sha256}, "
                f"observed {size}/{sha256}"
            )
    if failures:
        raise AcquisitionError("manifest verification failed:\n  " + "\n  ".join(failures))
    return results


def _default_destination(catalog: Mapping[str, Any], catalog_path: Path) -> Path:
    project_root = catalog_path.resolve().parents[1]
    return project_root / str(catalog["default_destination"])


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", action="append", default=[], help="pinned source id")
    parser.add_argument("--profile", action="append", default=[], help="catalog profile id")
    parser.add_argument("--dest", type=Path, help="destination root (default: catalog setting)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire fixed public releases with resume, checksums, and provenance."
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list profiles and pinned sources")
    list_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    plan_parser = subparsers.add_parser("plan", help="dry-run a selected acquisition")
    _add_selection_arguments(plan_parser)
    plan_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    fetch_parser = subparsers.add_parser("fetch", help="download selected fixed releases")
    _add_selection_arguments(fetch_parser)
    fetch_parser.add_argument(
        "--dry-run", action="store_true", help="show the plan without network or filesystem writes"
    )
    fetch_parser.add_argument(
        "--max-file-gb",
        type=float,
        default=DEFAULT_MAX_FILE_BYTES / 1024**3,
        help="hard ceiling for each file; default 1 GiB",
    )
    fetch_parser.add_argument(
        "--allow-large",
        action="store_true",
        help="explicitly authorize individual files larger than 1 GiB",
    )
    fetch_parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout seconds"
    )
    fetch_parser.add_argument(
        "--chunk-mib", type=int, default=8, help="streaming chunk size in MiB"
    )

    verify_parser = subparsers.add_parser("verify", help="verify an acquisition manifest")
    verify_parser.add_argument("--manifest", type=Path, help="manifest path")
    verify_parser.add_argument("--dest", type=Path, help="destination root containing manifest")
    verify_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _print_catalog(catalog: Mapping[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print(f"Catalog {catalog['catalog_version']} ({catalog['generated_date']})")
    print("Profiles:")
    for profile, members in sorted(catalog["profiles"].items()):
        total = sum(int(catalog["sources"][member]["size_bytes"]) for member in members)
        print(f"  {profile:<36} {len(members):>2} files  {human_bytes(total):>10}")
    print("Sources:")
    for source_id, source in sorted(catalog["sources"].items()):
        marker = " [LARGE]" if int(source["size_bytes"]) > DEFAULT_LARGE_THRESHOLD else ""
        print(
            f"  {source_id:<43} {source['version']:<20} "
            f"{human_bytes(int(source['size_bytes'])):>10}{marker}"
        )


def _print_plan(plan: Sequence[PlannedFile], as_json: bool = False) -> None:
    records = plan_as_records(plan)
    if as_json:
        print(json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print(f"Selected {len(plan)} fixed files; total compressed size {human_bytes(sum(i.size_bytes for i in plan))}")
    for record in records:
        marker = " [LARGE: explicit authorization required]" if record["large_file"] else ""
        print(
            f"  {record['source_id']}\n"
            f"    release: {record['version']} ({record['release_date']})\n"
            f"    size:    {record['size_human']}{marker}\n"
            f"    target:  {record['destination']}\n"
            f"    license: {record['license']['name']} ({record['license']['url']})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        catalog_path = args.catalog.resolve()
        catalog = load_catalog(catalog_path)
        if args.command == "list":
            _print_catalog(catalog, args.json)
            return 0

        if args.command == "verify":
            if args.manifest and args.dest:
                raise AcquisitionError("use either --manifest or --dest, not both")
            destination = args.dest or _default_destination(catalog, catalog_path)
            manifest_path = args.manifest or (destination / MANIFEST_NAME)
            results = verify_manifest(manifest_path)
            if args.json:
                print(json.dumps(results, indent=2, sort_keys=True))
            else:
                print(f"Verified {len(results)} files from {manifest_path}")
            return 0

        source_ids = resolve_source_ids(catalog, args.source, args.profile)
        destination = (args.dest or _default_destination(catalog, catalog_path)).resolve()
        plan = make_plan(catalog, source_ids, destination)
        if args.command == "plan" or args.dry_run:
            _print_plan(plan, getattr(args, "json", False))
            return 0

        maximum_bytes = int(args.max_file_gb * 1024**3)
        if args.timeout <= 0 or args.chunk_mib <= 0:
            raise AcquisitionError("--timeout and --chunk-mib must be positive")
        acquire_plan(
            plan,
            catalog,
            catalog_path,
            destination,
            maximum_bytes=maximum_bytes,
            allow_large=args.allow_large,
            timeout_seconds=args.timeout,
            chunk_bytes=args.chunk_mib * 1024**2,
        )
        return 0
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
