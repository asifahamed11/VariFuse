#!/usr/bin/env python3
"""Stream an official ClinVar variant_summary archive into a GRCh37 TSV.

The raw gzip is never expanded in memory.  The output keeps every upstream
column and every GRCh37 row, allowing downstream stages to apply their own
predeclared label/review filters.  A transformation manifest cryptographically
binds the derived TSV to the acquired raw archive.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO


CHUNK_BYTES = 8 * 1024**2
PROVENANCE_SUFFIX = ".provenance.json"
TRANSFORM_SUFFIX = ".transformation.json"
ASSEMBLY_ALIASES = {"GRCH37", "GRCH37.P13", "HG19"}
REQUIRED_COLUMNS = {
    "Assembly",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "ClinicalSignificance",
    "ReviewStatus",
    "VariationID",
    "LastEvaluated",
}


class SnapshotError(RuntimeError):
    """Raised when a snapshot cannot be transformed reproducibly."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path, algorithms: Iterable[str] = ("sha256",)) -> dict[str, str]:
    names = tuple(dict.fromkeys(name.lower() for name in algorithms))
    hashers = {name: hashlib.new(name) for name in names}
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            for digest in hashers.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in hashers.items()}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normalise_assembly(value: str) -> str:
    return value.upper().replace(" ", "")


def _open_input(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _open_output(path: Path, *, gzip_output: bool) -> TextIO:
    if gzip_output:
        return gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6)
    return path.open("w", encoding="utf-8", newline="")


def _load_source_provenance(
    input_path: Path, input_sha256: str, require_provenance: bool
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    sidecar = input_path.with_name(input_path.name + PROVENANCE_SUFFIX)
    if not sidecar.exists():
        if require_provenance:
            raise SnapshotError(
                f"raw input lacks acquisition provenance: {sidecar}. "
                "Acquire it with tools/acquire_publication_data.py or explicitly use "
                "--allow-unprovenanced-input."
            )
        return None, None, None
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read source provenance {sidecar}: {exc}") from exc
    recorded = provenance.get("observed_checksums", {}).get("sha256")
    if recorded != input_sha256:
        raise SnapshotError(
            f"source provenance SHA256 mismatch for {input_path}: "
            f"recorded={recorded!r}, observed={input_sha256}"
        )
    return provenance, sidecar, _hash_file(sidecar)["sha256"]


def _validate_release(release: str, provenance: dict[str, Any] | None) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", release):
        raise SnapshotError("release must be an explicit YYYY-MM value")
    try:
        date.fromisoformat(release + "-01")
    except ValueError as exc:
        raise SnapshotError(f"invalid release month: {release}") from exc
    if provenance is not None and str(provenance.get("version")) != release:
        raise SnapshotError(
            f"declared release {release} differs from source provenance "
            f"{provenance.get('version')!r}"
        )


def _raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def transform_snapshot(
    input_path: Path,
    output_path: Path,
    release: str,
    *,
    require_provenance: bool = True,
    overwrite: bool = False,
    progress_every: int = 1_000_000,
) -> dict[str, Any]:
    """Write all GRCh37 records from one ClinVar variant_summary file."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise SnapshotError(f"raw ClinVar archive does not exist: {input_path}")
    if input_path == output_path:
        raise SnapshotError("input and output paths must differ")
    if output_path.exists() and not overwrite:
        raise SnapshotError(f"output already exists: {output_path}; use --overwrite deliberately")
    manifest_path = output_path.with_name(output_path.name + TRANSFORM_SUFFIX)
    if manifest_path.exists() and not overwrite:
        raise SnapshotError(
            f"transformation manifest already exists: {manifest_path}; use --overwrite deliberately"
        )
    if progress_every < 0:
        raise SnapshotError("progress_every cannot be negative")

    input_size = input_path.stat().st_size
    input_sha256 = _hash_file(input_path)["sha256"]
    source_provenance, source_sidecar, source_sidecar_sha256 = _load_source_provenance(
        input_path, input_sha256, require_provenance
    )
    _validate_release(release, source_provenance)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise SnapshotError(
            f"temporary output already exists: {temporary}; inspect or move it before retrying"
        )

    _raise_csv_limit()
    rows_scanned = 0
    rows_written = 0
    header: list[str] = []
    try:
        gzip_output = output_path.name.lower().endswith(".gz")
        with _open_input(input_path) as source, _open_output(
            temporary, gzip_output=gzip_output
        ) as destination:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(
                destination,
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            try:
                header = next(reader)
            except StopIteration as exc:
                raise SnapshotError(f"empty ClinVar archive: {input_path}") from exc
            if not header:
                raise SnapshotError(f"empty ClinVar header: {input_path}")
            if len(header) != len(set(header)):
                duplicates = sorted({name for name in header if header.count(name) > 1})
                raise SnapshotError(f"duplicate ClinVar columns: {duplicates}")
            missing = sorted(REQUIRED_COLUMNS - set(header))
            if missing:
                raise SnapshotError(f"ClinVar archive lacks required columns: {missing}")
            assembly_index = header.index("Assembly")
            writer.writerow(header)

            for line_number, row in enumerate(reader, start=2):
                rows_scanned += 1
                if len(row) != len(header):
                    raise SnapshotError(
                        f"malformed row {line_number}: {len(row)} fields, expected {len(header)}"
                    )
                if _normalise_assembly(row[assembly_index]) in ASSEMBLY_ALIASES:
                    writer.writerow(row)
                    rows_written += 1
                if progress_every and rows_scanned % progress_every == 0:
                    print(
                        f"Scanned {rows_scanned:,} rows; retained {rows_written:,} GRCh37 rows",
                        flush=True,
                    )
            destination.flush()
            if hasattr(destination, "fileno"):
                try:
                    os.fsync(destination.fileno())
                except (AttributeError, OSError):
                    pass
        if rows_written == 0:
            raise SnapshotError("no GRCh37 records were found; refusing to publish an empty snapshot")
        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    output_hashes = _hash_file(output_path, ("sha256",))
    manifest = {
        "schema_version": 1,
        "transformation": "clinvar_variant_summary_grch37_stream_v1",
        "release": release,
        "created_at_utc": _utc_now(),
        "assembly_filter": sorted(ASSEMBLY_ALIASES),
        "input": {
            "path": str(input_path),
            "size_bytes": input_size,
            "sha256": input_sha256,
            "acquisition_provenance": str(source_sidecar) if source_sidecar else None,
            "acquisition_provenance_sha256": source_sidecar_sha256,
            "source_id": source_provenance.get("source_id") if source_provenance else None,
            "provider": source_provenance.get("provider") if source_provenance else None,
            "version": source_provenance.get("version") if source_provenance else None,
            "publisher_checksum": (
                source_provenance.get("publisher_checksum") if source_provenance else None
            ),
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": output_hashes["sha256"],
            "compression": "gzip" if output_path.name.lower().endswith(".gz") else "none",
            "columns": header,
            "column_count": len(header),
            "rows_scanned": rows_scanned,
            "rows_written": rows_written,
        },
        "policy": {
            "rows_preserved": "all rows whose Assembly is GRCh37/GRCh37.p13/hg19",
            "columns_preserved": "all upstream columns in original order",
            "label_filtering": "none; downstream stage applies prespecified review/significance rules",
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def verify_transformation(manifest_path: Path) -> dict[str, Any]:
    """Verify both raw input and derived output against a transform manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read transformation manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise SnapshotError("unsupported transformation manifest schema")
    for role in ("input", "output"):
        record = manifest.get(role)
        if not isinstance(record, dict):
            raise SnapshotError(f"transformation manifest lacks {role} record")
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise SnapshotError(f"{role} file is missing: {path}")
        observed_size = path.stat().st_size
        observed_hash = _hash_file(path)["sha256"]
        if observed_size != record.get("size_bytes") or observed_hash != record.get("sha256"):
            raise SnapshotError(
                f"{role} file fails size/SHA256 verification: {path}; "
                f"observed {observed_size}/{observed_hash}"
            )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream an acquired ClinVar variant_summary release into a GRCh37 TSV."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transform = subparsers.add_parser("transform", help="create a derived GRCh37 snapshot")
    transform.add_argument("--input", type=Path, required=True, help="raw variant_summary .gz")
    transform.add_argument("--output", type=Path, required=True, help="derived .tsv or .tsv.gz")
    transform.add_argument("--release", required=True, help="pinned release in YYYY-MM format")
    transform.add_argument("--overwrite", action="store_true", help="replace an existing output")
    transform.add_argument(
        "--allow-unprovenanced-input",
        action="store_true",
        help="allow a raw input not acquired by the publication downloader",
    )
    transform.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="report every N input rows; use 0 to disable",
    )

    verify = subparsers.add_parser("verify", help="verify a transformation manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            manifest = verify_transformation(args.manifest.resolve())
            print(
                f"Verified ClinVar {manifest['release']} transformation: "
                f"{manifest['output']['rows_written']:,} GRCh37 rows"
            )
            return 0
        manifest = transform_snapshot(
            args.input,
            args.output,
            args.release,
            require_provenance=not args.allow_unprovenanced_input,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
        print(
            f"Wrote {manifest['output']['rows_written']:,} of "
            f"{manifest['output']['rows_scanned']:,} rows to {manifest['output']['path']}"
        )
        print(f"SHA256={manifest['output']['sha256']}")
        return 0
    except SnapshotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
