#!/usr/bin/env python3
"""Create a streaming, submission-level ClinVar temporal audit.

ClinVar ``submission_summary`` releases are ordered by VariationID.  This tool
performs a merge of the two ordered streams, so memory use is bounded by the
largest set of submissions for one variation rather than total archive size.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO


CHUNK_BYTES = 8 * 1024**2
PROVENANCE_SUFFIX = ".provenance.json"
AUDIT_SUFFIX = ".audit.json"
SCV_PATTERN = re.compile(r"^(SCV\d+)\.(\d+)$")
REQUIRED_COLUMNS = {
    "#VariationID",
    "ClinicalSignificance",
    "DateLastEvaluated",
    "Description",
    "SubmittedPhenotypeInfo",
    "ReportedPhenotypeInfo",
    "ReviewStatus",
    "CollectionMethod",
    "OriginCounts",
    "Submitter",
    "SCV",
    "SubmittedGeneSymbol",
    "ExplanationOfInterpretation",
    "SomaticClinicalImpact",
    "Oncogenicity",
}
PAYLOAD_COLUMNS = (
    "ClinicalSignificance",
    "DateLastEvaluated",
    "Description",
    "SubmittedPhenotypeInfo",
    "ReportedPhenotypeInfo",
    "ReviewStatus",
    "CollectionMethod",
    "OriginCounts",
    "Submitter",
    "SubmittedGeneSymbol",
    "ExplanationOfInterpretation",
    "SomaticClinicalImpact",
    "Oncogenicity",
)
OUTPUT_COLUMNS = (
    "VariationID",
    "baseline_scv_count",
    "endpoint_scv_count",
    "new_scv_count",
    "updated_scv_count",
    "same_version_payload_change_count",
    "version_regression_count",
    "withdrawn_scv_count",
    "event_scv_count",
    "event_post_cutoff_scv_count",
    "event_latest_evaluation",
    "event_clinical_significances",
    "event_review_statuses",
    "event_submitter_count",
)
SCREEN_COLUMNS = (
    "VariationID",
    "candidate_class",
    "current_contributing_matching_scv_count",
    "current_contributing_opposing_scv_count",
    "current_contributing_ambiguous_scv_count",
    "current_contributing_unique_submitter_count",
    "current_contributing_matching_unique_submitter_count",
    "post_cutoff_new_matching_scv_count",
    "post_cutoff_updated_matching_scv_count",
    "post_cutoff_matching_event_scv_count",
    "current_contributing_matching_scv_ids",
    "current_contributing_opposing_scv_ids",
    "post_cutoff_matching_event_scv_ids",
    "post_cutoff_matching_event_dates",
    "scv_evidence_pass",
    "scv_evidence_reason",
)


class SubmissionAuditError(RuntimeError):
    """Raised when SCV temporal provenance cannot be established safely."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _validate_month(value: str, label: str) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise SubmissionAuditError(f"{label} must be an explicit YYYY-MM release")
    try:
        date.fromisoformat(value + "-01")
    except ValueError as exc:
        raise SubmissionAuditError(f"invalid {label}: {value}") from exc


def _verified_provenance(path: Path, release: str) -> dict[str, Any]:
    if not path.is_file():
        raise SubmissionAuditError(f"submission archive is missing: {path}")
    sha256 = _hash_file(path)
    sidecar = path.with_name(path.name + PROVENANCE_SUFFIX)
    if not sidecar.is_file():
        raise SubmissionAuditError(f"submission archive lacks acquisition provenance: {sidecar}")
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionAuditError(f"invalid acquisition provenance {sidecar}: {exc}") from exc
    recorded = provenance.get("observed_checksums", {}).get("sha256")
    if recorded != sha256:
        raise SubmissionAuditError(
            f"submission archive SHA256 differs from provenance: {path}; "
            f"recorded={recorded!r}, observed={sha256}"
        )
    if str(provenance.get("version")) != release:
        raise SubmissionAuditError(
            f"declared release {release} differs from provenance {provenance.get('version')!r}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
        "provenance_path": str(sidecar),
        "provenance_sha256": _hash_file(sidecar),
        "source_id": provenance.get("source_id"),
        "provider": provenance.get("provider"),
        "version": provenance.get("version"),
        "publisher_checksum": provenance.get("publisher_checksum"),
    }


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


@contextmanager
def _open_atomic_text_payload(
    path: Path, *, gzip_output: bool, compresslevel: int = 6
) -> Iterator[TextIO]:
    """Open a temporary output with deterministic gzip metadata and durable bytes."""

    if not gzip_output:
        with path.open("w", encoding="utf-8", newline="") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        return
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=compresslevel,
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            text_handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            try:
                yield text_handle
                text_handle.flush()
            finally:
                text_handle.detach()
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _new_stats(*, collect_distributions: bool = True) -> dict[str, Any]:
    return {
        "rows": 0,
        "variation_ids": 0,
        "header": [],
        "metadata_line_count": 0,
        "clinical_significance": Counter(),
        "review_status": Counter(),
        "collection_method": Counter(),
        "collect_distributions": collect_distributions,
    }


def _rows(path: Path, stats: dict[str, Any]) -> Iterator[dict[str, str]]:
    with _open_text(path) as handle:
        metadata_lines = 0
        header_line: str | None = None
        for line in handle:
            if "\t" in line:
                header_line = line
                break
            metadata_lines += 1
        if header_line is None:
            raise SubmissionAuditError(f"no tabular header found in {path}")
        header = next(csv.reader([header_line], delimiter="\t"))
        missing = sorted(REQUIRED_COLUMNS - set(header))
        if missing:
            raise SubmissionAuditError(f"{path} lacks required SCV columns: {missing}")
        if len(header) != len(set(header)):
            raise SubmissionAuditError(f"{path} has duplicate header columns")
        stats["header"] = header
        stats["metadata_line_count"] = metadata_lines
        # NCBI's tab-delimited export is not RFC-4180 CSV: literal double quotes
        # can occur inside unquoted free-text fields.  QUOTE_NONE prevents one
        # such quote from swallowing hundreds of otherwise valid physical rows.
        reader = csv.DictReader(
            itertools.chain([header_line], handle),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise SubmissionAuditError(
                    f"malformed SCV row {metadata_lines + reader.line_num} in {path}"
                )
            stats["rows"] += 1
            if stats["collect_distributions"]:
                stats["clinical_significance"][row["ClinicalSignificance"]] += 1
                stats["review_status"][row["ReviewStatus"]] += 1
                stats["collection_method"][row["CollectionMethod"]] += 1
            yield row


def _variation_groups(
    path: Path, stats: dict[str, Any]
) -> Iterator[tuple[int, list[dict[str, str]]]]:
    current_id: int | None = None
    current_rows: list[dict[str, str]] = []
    previous_id = -1
    for row in _rows(path, stats):
        raw_variation = row["#VariationID"]
        if not raw_variation.isdigit():
            raise SubmissionAuditError(f"invalid VariationID {raw_variation!r} in {path}")
        variation_id = int(raw_variation)
        if variation_id < previous_id:
            raise SubmissionAuditError(
                f"{path} is not ordered by VariationID: {variation_id} follows {previous_id}"
            )
        previous_id = variation_id
        if current_id is None:
            current_id = variation_id
        if variation_id != current_id:
            stats["variation_ids"] += 1
            yield current_id, current_rows
            current_id = variation_id
            current_rows = []
        current_rows.append(row)
    if current_id is not None:
        stats["variation_ids"] += 1
        yield current_id, current_rows


def _scv_key(row: dict[str, str]) -> tuple[str, int]:
    match = SCV_PATTERN.fullmatch(row["SCV"].strip())
    if not match:
        raise SubmissionAuditError(f"invalid versioned SCV accession: {row['SCV']!r}")
    return match.group(1), int(match.group(2))


def _scv_map(
    rows: list[dict[str, str]], variation_id: int
) -> dict[str, tuple[int, str, list[dict[str, str]]]]:
    grouped: dict[str, list[tuple[int, str, dict[str, str]]]] = {}
    for row in rows:
        accession, version = _scv_key(row)
        payload = "\x1f".join(row[column] for column in PAYLOAD_COLUMNS)
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        grouped.setdefault(accession, []).append((version, payload_hash, row))
    result: dict[str, tuple[int, str, list[dict[str, str]]]] = {}
    for accession, entries in grouped.items():
        versions = {entry[0] for entry in entries}
        if len(versions) != 1:
            raise SubmissionAuditError(
                f"SCV accession {accession} has multiple versions within "
                f"VariationID {variation_id}: {sorted(versions)}"
            )
        combined_payload = "\x1e".join(sorted(entry[1] for entry in entries))
        combined_hash = hashlib.sha256(combined_payload.encode("ascii")).hexdigest()
        result[accession] = (
            next(iter(versions)),
            combined_hash,
            [entry[2] for entry in entries],
        )
    return result


def _parse_evaluation_date(raw: str) -> date | None:
    value = raw.strip()
    if not value or value == "-":
        return None
    for pattern in ("%b %d, %Y", "%Y-%m-%d", "%b %Y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    raise SubmissionAuditError(f"unrecognized DateLastEvaluated value: {raw!r}")


def _compare_variation(
    variation_id: int,
    baseline_rows: list[dict[str, str]],
    endpoint_rows: list[dict[str, str]],
    cutoff: date,
) -> tuple[dict[str, Any] | None, Counter]:
    baseline = _scv_map(baseline_rows, variation_id)
    endpoint = _scv_map(endpoint_rows, variation_id)
    counts: Counter = Counter()
    event_endpoint_scvs: list[list[dict[str, str]]] = []
    for accession, (version, payload_hash, rows) in endpoint.items():
        old = baseline.get(accession)
        if old is None:
            status = "new_scv"
        elif version > old[0]:
            status = "updated_scv"
        elif version < old[0]:
            status = "version_regression"
        elif payload_hash != old[1]:
            status = "same_version_payload_change"
        else:
            status = "unchanged"
        counts[status] += 1
        if status != "unchanged":
            event_endpoint_scvs.append(rows)
    withdrawn = set(baseline) - set(endpoint)
    counts["withdrawn_scv"] += len(withdrawn)
    event_count = sum(counts[name] for name in counts if name != "unchanged")
    if event_count == 0:
        return None, counts

    event_endpoint_rows = [row for rows in event_endpoint_scvs for row in rows]
    dates_by_scv = [
        [
            parsed
            for row in rows
            if (parsed := _parse_evaluation_date(row["DateLastEvaluated"])) is not None
        ]
        for rows in event_endpoint_scvs
    ]
    evaluation_dates = [value for values in dates_by_scv for value in values]
    post_cutoff = sum(any(value > cutoff for value in values) for values in dates_by_scv)
    significances = sorted({row["ClinicalSignificance"] for row in event_endpoint_rows})
    reviews = sorted({row["ReviewStatus"] for row in event_endpoint_rows})
    submitters = {
        row["Submitter"].strip()
        for row in event_endpoint_rows
        if row["Submitter"].strip() not in {"", "-"}
    }
    record = {
        "VariationID": variation_id,
        "baseline_scv_count": len(baseline),
        "endpoint_scv_count": len(endpoint),
        "new_scv_count": counts["new_scv"],
        "updated_scv_count": counts["updated_scv"],
        "same_version_payload_change_count": counts["same_version_payload_change"],
        "version_regression_count": counts["version_regression"],
        "withdrawn_scv_count": counts["withdrawn_scv"],
        "event_scv_count": event_count,
        "event_post_cutoff_scv_count": post_cutoff,
        "event_latest_evaluation": max(evaluation_dates).isoformat() if evaluation_dates else "",
        "event_clinical_significances": "|".join(significances),
        "event_review_statuses": "|".join(reviews),
        "event_submitter_count": len(submitters),
    }
    return record, counts


def _counter_json(counter: Counter) -> dict[str, int]:
    return dict(sorted(((str(key), int(value)) for key, value in counter.items())))


def _stats_json(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": stats["rows"],
        "variation_ids": stats["variation_ids"],
        "header": stats["header"],
        "metadata_line_count": stats["metadata_line_count"],
        "clinical_significance_counts": _counter_json(stats["clinical_significance"]),
        "review_status_counts": _counter_json(stats["review_status"]),
        "collection_method_counts": _counter_json(stats["collection_method"]),
    }


def _normalise_candidate_class(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"1", "pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}:
        return "pathogenic"
    if raw in {"0", "benign", "likely benign", "benign/likely benign"}:
        return "benign"
    if "conflicting" in raw:
        raise SubmissionAuditError(f"unsupported/ambiguous candidate class: {value!r}")
    pathogenic = re.search(r"\bpathogenic\b", raw) is not None
    benign = re.search(r"\bbenign\b", raw) is not None
    if pathogenic and not benign:
        return "pathogenic"
    if benign and not pathogenic:
        return "benign"
    raise SubmissionAuditError(f"unsupported/ambiguous candidate class: {value!r}")


def _submission_class(value: str) -> str | None:
    raw = value.strip().lower()
    if "conflicting" in raw:
        return None
    pathogenic = re.search(r"\bpathogenic\b", raw) is not None
    benign = re.search(r"\bbenign\b", raw) is not None
    if pathogenic and not benign:
        return "pathogenic"
    if benign and not pathogenic:
        return "benign"
    return None


def _is_contributing(row: dict[str, str]) -> bool:
    field = "ContributesToAggregateClassification"
    if field not in row:
        return True
    return row[field].strip().lower() in {"yes", "true", "1"}


def _empty_screen_record(variation_id: int, candidate_class: str) -> dict[str, Any]:
    return {
        "VariationID": variation_id,
        "candidate_class": candidate_class,
        "current_contributing_matching_scv_count": 0,
        "current_contributing_opposing_scv_count": 0,
        "current_contributing_ambiguous_scv_count": 0,
        "current_contributing_unique_submitter_count": 0,
        "current_contributing_matching_unique_submitter_count": 0,
        "post_cutoff_new_matching_scv_count": 0,
        "post_cutoff_updated_matching_scv_count": 0,
        "post_cutoff_matching_event_scv_count": 0,
        "current_contributing_matching_scv_ids": "",
        "current_contributing_opposing_scv_ids": "",
        "post_cutoff_matching_event_scv_ids": "",
        "post_cutoff_matching_event_dates": "",
        "scv_evidence_pass": 0,
        "scv_evidence_reason": "candidate_missing_from_endpoint",
    }


def _screen_one_candidate(
    variation_id: int,
    candidate_class: str,
    baseline_rows: list[dict[str, str]],
    endpoint_rows: list[dict[str, str]],
    cutoff: date,
    min_matching_scvs: int,
    min_unique_submitters: int,
    require_no_opposition: bool,
    require_post_cutoff_event: bool,
) -> dict[str, Any]:
    if not endpoint_rows:
        return _empty_screen_record(variation_id, candidate_class)
    baseline = _scv_map(baseline_rows, variation_id)
    endpoint = _scv_map(endpoint_rows, variation_id)
    opposite = "benign" if candidate_class == "pathogenic" else "pathogenic"
    matching_ids: list[str] = []
    opposing_ids: list[str] = []
    ambiguous_count = 0
    all_submitters: set[str] = set()
    matching_submitters: set[str] = set()
    post_new: list[str] = []
    post_updated: list[str] = []
    post_dates: set[str] = set()

    for accession, (version, _payload_hash, rows) in endpoint.items():
        contributing_rows = [row for row in rows if _is_contributing(row)]
        if not contributing_rows:
            continue
        all_submitters.update(
            row["Submitter"].strip()
            for row in contributing_rows
            if row["Submitter"].strip() not in {"", "-"}
        )
        observed_classes = [
            _submission_class(row["ClinicalSignificance"]) for row in contributing_rows
        ]
        classes = {observed for observed in observed_classes if observed is not None}
        has_unresolved_class = any(observed is None for observed in observed_classes)
        full_accession = f"{accession}.{version}"
        if candidate_class in classes and opposite not in classes and not has_unresolved_class:
            bucket = "matching"
            matching_ids.append(full_accession)
            matching_submitters.update(
                row["Submitter"].strip()
                for row in contributing_rows
                if row["Submitter"].strip() not in {"", "-"}
            )
        elif opposite in classes and candidate_class not in classes and not has_unresolved_class:
            bucket = "opposing"
            opposing_ids.append(full_accession)
        else:
            bucket = "ambiguous"
            ambiguous_count += 1

        old = baseline.get(accession)
        event = "new" if old is None else "updated" if version > old[0] else None
        if bucket != "matching" or event is None:
            continue
        dates = [
            parsed
            for row in contributing_rows
            if _submission_class(row["ClinicalSignificance"]) == candidate_class
            if (parsed := _parse_evaluation_date(row["DateLastEvaluated"])) is not None
        ]
        qualifying = [value for value in dates if value > cutoff]
        if not qualifying:
            continue
        post_dates.update(value.isoformat() for value in qualifying)
        if event == "new":
            post_new.append(full_accession)
        else:
            post_updated.append(full_accession)

    post_ids = sorted(post_new + post_updated)
    reasons: list[str] = []
    if len(matching_ids) < min_matching_scvs:
        reasons.append("insufficient_current_contributing_matching_scvs")
    if len(matching_submitters) < min_unique_submitters:
        reasons.append("insufficient_unique_matching_submitters")
    if require_no_opposition and opposing_ids:
        reasons.append("opposing_current_contributing_scv")
    if require_no_opposition and ambiguous_count:
        reasons.append("ambiguous_current_contributing_scv")
    if require_post_cutoff_event and not post_ids:
        reasons.append("no_post_cutoff_new_or_updated_matching_scv")
    record = {
        "VariationID": variation_id,
        "candidate_class": candidate_class,
        "current_contributing_matching_scv_count": len(matching_ids),
        "current_contributing_opposing_scv_count": len(opposing_ids),
        "current_contributing_ambiguous_scv_count": ambiguous_count,
        "current_contributing_unique_submitter_count": len(all_submitters),
        "current_contributing_matching_unique_submitter_count": len(matching_submitters),
        "post_cutoff_new_matching_scv_count": len(post_new),
        "post_cutoff_updated_matching_scv_count": len(post_updated),
        "post_cutoff_matching_event_scv_count": len(post_ids),
        "current_contributing_matching_scv_ids": "|".join(sorted(matching_ids)),
        "current_contributing_opposing_scv_ids": "|".join(sorted(opposing_ids)),
        "post_cutoff_matching_event_scv_ids": "|".join(post_ids),
        "post_cutoff_matching_event_dates": "|".join(sorted(post_dates)),
        "scv_evidence_pass": int(not reasons),
        "scv_evidence_reason": "pass" if not reasons else ";".join(reasons),
    }
    return record


def screen_candidate_variations(
    baseline_path: Path,
    endpoint_path: Path,
    candidates: dict[int, Any],
    cutoff_date: str,
    *,
    min_matching_scvs: int = 1,
    min_unique_submitters: int = 1,
    require_no_opposition: bool = True,
    require_post_cutoff_event: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream SCV archives and screen a bounded set of candidate VariationIDs."""

    if not candidates:
        raise SubmissionAuditError("candidate set is empty")
    if min_matching_scvs < 1 or min_unique_submitters < 1:
        raise SubmissionAuditError("SCV and submitter thresholds must be at least one")
    try:
        cutoff = date.fromisoformat(cutoff_date)
    except ValueError as exc:
        raise SubmissionAuditError("cutoff date must be ISO YYYY-MM-DD") from exc
    normalised: dict[int, str] = {}
    for raw_id, raw_class in candidates.items():
        try:
            variation_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise SubmissionAuditError(f"invalid candidate VariationID: {raw_id!r}") from exc
        if variation_id <= 0:
            raise SubmissionAuditError(f"invalid candidate VariationID: {raw_id!r}")
        normalised[variation_id] = _normalise_candidate_class(raw_class)

    baseline_stats = _new_stats(collect_distributions=False)
    endpoint_stats = _new_stats(collect_distributions=False)
    baseline_groups = iter(_variation_groups(baseline_path, baseline_stats))
    endpoint_groups = iter(_variation_groups(endpoint_path, endpoint_stats))
    old = next(baseline_groups, None)
    new = next(endpoint_groups, None)
    records: dict[int, dict[str, Any]] = {}
    while old is not None or new is not None:
        if new is None or (old is not None and old[0] < new[0]):
            variation_id, baseline_rows = old
            endpoint_rows: list[dict[str, str]] = []
            old = next(baseline_groups, None)
        elif old is None or new[0] < old[0]:
            variation_id, endpoint_rows = new
            baseline_rows = []
            new = next(endpoint_groups, None)
        else:
            variation_id, baseline_rows = old
            _, endpoint_rows = new
            old = next(baseline_groups, None)
            new = next(endpoint_groups, None)
        candidate_class = normalised.get(variation_id)
        if candidate_class is not None:
            records[variation_id] = _screen_one_candidate(
                variation_id,
                candidate_class,
                baseline_rows,
                endpoint_rows,
                cutoff,
                min_matching_scvs,
                min_unique_submitters,
                require_no_opposition,
                require_post_cutoff_event,
            )
    for variation_id, candidate_class in normalised.items():
        records.setdefault(variation_id, _empty_screen_record(variation_id, candidate_class))
    scan_stats = {
        "baseline_rows_scanned": baseline_stats["rows"],
        "baseline_variation_ids_scanned": baseline_stats["variation_ids"],
        "endpoint_rows_scanned": endpoint_stats["rows"],
        "endpoint_variation_ids_scanned": endpoint_stats["variation_ids"],
    }
    return [records[key] for key in sorted(records)], scan_stats


def _read_candidates(path: Path, id_column: str, class_column: str) -> dict[int, str]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    lower_name = path.name.lower()
    delimiter = "," if lower_name.endswith((".csv", ".csv.gz")) else "\t"
    candidates: dict[int, str] = {}
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if (
            not reader.fieldnames
            or id_column not in reader.fieldnames
            or class_column not in reader.fieldnames
        ):
            raise SubmissionAuditError(
                f"candidate file must contain {id_column!r} and {class_column!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                variation_id = int(row[id_column])
            except (TypeError, ValueError) as exc:
                raise SubmissionAuditError(
                    f"invalid candidate VariationID at line {line_number}: {row.get(id_column)!r}"
                ) from exc
            candidate_class = _normalise_candidate_class(row[class_column])
            previous = candidates.get(variation_id)
            if previous is not None and previous != candidate_class:
                raise SubmissionAuditError(
                    f"candidate VariationID {variation_id} has conflicting classes"
                )
            candidates[variation_id] = candidate_class
    return candidates


def build_candidate_screen(
    baseline_path: Path,
    endpoint_path: Path,
    candidate_path: Path,
    output_path: Path,
    baseline_release: str,
    endpoint_release: str,
    cutoff_date: str,
    *,
    id_column: str = "VariationID",
    class_column: str = "ClinicalSignificance",
    min_matching_scvs: int = 1,
    min_unique_submitters: int = 1,
    require_no_opposition: bool = True,
    require_post_cutoff_event: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a provenance-bound, Stage09-ready candidate SCV evidence screen."""

    _validate_month(baseline_release, "baseline release")
    _validate_month(endpoint_release, "endpoint release")
    if baseline_release >= endpoint_release:
        raise SubmissionAuditError("baseline release must precede endpoint release")
    baseline_path = baseline_path.resolve()
    endpoint_path = endpoint_path.resolve()
    candidate_path = candidate_path.resolve()
    output_path = output_path.resolve()
    summary_path = output_path.with_name(output_path.name + AUDIT_SUFFIX)
    if not candidate_path.is_file():
        raise SubmissionAuditError(f"candidate file is missing: {candidate_path}")
    if (output_path.exists() or summary_path.exists()) and not overwrite:
        raise SubmissionAuditError("screen output already exists; use --overwrite deliberately")
    baseline_provenance = _verified_provenance(baseline_path, baseline_release)
    endpoint_provenance = _verified_provenance(endpoint_path, endpoint_release)
    candidates = _read_candidates(candidate_path, id_column, class_column)
    records, scan_stats = screen_candidate_variations(
        baseline_path,
        endpoint_path,
        candidates,
        cutoff_date,
        min_matching_scvs=min_matching_scvs,
        min_unique_submitters=min_unique_submitters,
        require_no_opposition=require_no_opposition,
        require_post_cutoff_event=require_post_cutoff_event,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    gzip_output = output_path.name.lower().endswith(".gz")
    try:
        with _open_atomic_text_payload(temporary, gzip_output=gzip_output) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SCREEN_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    reasons = Counter(record["scv_evidence_reason"] for record in records)
    summary = {
        "schema_version": 1,
        "audit_policy": "candidate_scv_contribution_screen_v1",
        "created_at_utc": _utc_now(),
        "baseline_release": baseline_release,
        "endpoint_release": endpoint_release,
        "cutoff_date": cutoff_date,
        "baseline": baseline_provenance,
        "endpoint": endpoint_provenance,
        "candidates": {
            "path": str(candidate_path),
            "size_bytes": candidate_path.stat().st_size,
            "sha256": _hash_file(candidate_path),
            "row_count": len(candidates),
            "id_column": id_column,
            "class_column": class_column,
        },
        "thresholds": {
            "min_matching_scvs": min_matching_scvs,
            "min_unique_submitters": min_unique_submitters,
            "require_no_opposition": require_no_opposition,
            "require_post_cutoff_event": require_post_cutoff_event,
        },
        "scan": scan_stats,
        "results": {
            "candidate_count": len(records),
            "pass_count": sum(record["scv_evidence_pass"] for record in records),
            "fail_count": sum(not record["scv_evidence_pass"] for record in records),
            "reason_counts": _counter_json(reasons),
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": _hash_file(output_path),
            "row_count": len(records),
            "columns": list(SCREEN_COLUMNS),
            "compression": "gzip" if gzip_output else "none",
        },
        "limitations": [
            "Aggregate ClinVar review status must be screened by the caller before this SCV audit.",
            "DateLastEvaluated is an assertion date, not proof of first public availability.",
            "An SCV moved between VariationIDs can appear new without VCV replacement-history XML.",
        ],
    }
    _atomic_json(summary_path, summary)
    return summary


def build_audit(
    baseline_path: Path,
    endpoint_path: Path,
    output_path: Path,
    baseline_release: str,
    endpoint_release: str,
    cutoff_date: str,
    *,
    overwrite: bool = False,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    """Merge two verified SCV streams and emit per-VariationID event evidence."""

    _validate_month(baseline_release, "baseline release")
    _validate_month(endpoint_release, "endpoint release")
    if baseline_release >= endpoint_release:
        raise SubmissionAuditError("baseline release must precede endpoint release")
    try:
        cutoff = date.fromisoformat(cutoff_date)
    except ValueError as exc:
        raise SubmissionAuditError("cutoff date must be ISO YYYY-MM-DD") from exc
    if progress_every < 0:
        raise SubmissionAuditError("progress_every cannot be negative")

    baseline_path = baseline_path.resolve()
    endpoint_path = endpoint_path.resolve()
    output_path = output_path.resolve()
    summary_path = output_path.with_name(output_path.name + AUDIT_SUFFIX)
    if (output_path.exists() or summary_path.exists()) and not overwrite:
        raise SubmissionAuditError("audit output already exists; use --overwrite deliberately")
    baseline_provenance = _verified_provenance(baseline_path, baseline_release)
    endpoint_provenance = _verified_provenance(endpoint_path, endpoint_release)

    baseline_stats = _new_stats()
    endpoint_stats = _new_stats()
    baseline_groups = iter(_variation_groups(baseline_path, baseline_stats))
    endpoint_groups = iter(_variation_groups(endpoint_path, endpoint_stats))
    old = next(baseline_groups, None)
    new = next(endpoint_groups, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    event_counts: Counter = Counter()
    audit_rows = 0
    post_cutoff_variations = 0
    try:
        with _open_atomic_text_payload(temporary, gzip_output=True) as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            while old is not None or new is not None:
                if new is None or (old is not None and old[0] < new[0]):
                    variation_id, baseline_rows = old
                    endpoint_rows: list[dict[str, str]] = []
                    old = next(baseline_groups, None)
                elif old is None or new[0] < old[0]:
                    variation_id, endpoint_rows = new
                    baseline_rows = []
                    new = next(endpoint_groups, None)
                else:
                    variation_id, baseline_rows = old
                    _, endpoint_rows = new
                    old = next(baseline_groups, None)
                    new = next(endpoint_groups, None)
                record, counts = _compare_variation(
                    variation_id, baseline_rows, endpoint_rows, cutoff
                )
                event_counts.update(counts)
                if record is not None:
                    writer.writerow(record)
                    audit_rows += 1
                    post_cutoff_variations += int(record["event_post_cutoff_scv_count"] > 0)
                    if progress_every and audit_rows % progress_every == 0:
                        print(f"Wrote {audit_rows:,} changed VariationIDs", flush=True)
        if audit_rows == 0:
            raise SubmissionAuditError("SCV audit produced no temporal events")
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    output_sha256 = _hash_file(output_path)
    summary = {
        "schema_version": 1,
        "audit_policy": "ordered_variation_scv_version_merge_v1",
        "created_at_utc": _utc_now(),
        "baseline_release": baseline_release,
        "endpoint_release": endpoint_release,
        "cutoff_date": cutoff.isoformat(),
        "baseline": {**baseline_provenance, "content": _stats_json(baseline_stats)},
        "endpoint": {**endpoint_provenance, "content": _stats_json(endpoint_stats)},
        "events": {
            "status_counts": _counter_json(event_counts),
            "changed_variation_ids": audit_rows,
            "changed_variation_ids_with_post_cutoff_evaluation": post_cutoff_variations,
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": output_sha256,
            "row_count": audit_rows,
            "columns": list(OUTPUT_COLUMNS),
            "compression": "gzip",
        },
        "limitations": [
            "The lightweight merge assumes each release is ordered by VariationID; this is checked.",
            "An SCV moved between VariationIDs is represented as withdrawal plus addition; VCV ReplacedList XML is needed to resolve merge history.",
            "DateLastEvaluated is submission metadata, not proof that the variant was first observed on that date.",
        ],
    }
    _atomic_json(summary_path, summary)
    return summary


def verify_audit(summary_path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionAuditError(f"cannot read audit summary {summary_path}: {exc}") from exc
    if summary.get("schema_version") != 1:
        raise SubmissionAuditError("unsupported SCV audit schema")
    roles = ["baseline", "endpoint", "output"]
    if "candidates" in summary:
        roles.append("candidates")
    for role in roles:
        record = summary.get(role)
        if not isinstance(record, dict):
            raise SubmissionAuditError(f"audit summary lacks {role} record")
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise SubmissionAuditError(f"audit {role} file is missing: {path}")
        observed_size = path.stat().st_size
        observed_hash = _hash_file(path)
        if observed_size != record.get("size_bytes") or observed_hash != record.get("sha256"):
            raise SubmissionAuditError(
                f"audit {role} file fails size/SHA256 verification: {path}"
            )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit SCV changes between two ClinVar releases.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--baseline", type=Path, required=True)
    audit.add_argument("--endpoint", type=Path, required=True)
    audit.add_argument("--baseline-release", required=True)
    audit.add_argument("--endpoint-release", required=True)
    audit.add_argument("--cutoff-date", required=True)
    audit.add_argument("--output", type=Path, required=True, help="output .tsv.gz")
    audit.add_argument("--overwrite", action="store_true")
    audit.add_argument("--progress-every", type=int, default=250_000)
    screen = subparsers.add_parser(
        "screen",
        help="screen candidate VariationIDs using current and post-cutoff SCV evidence",
    )
    screen.add_argument("--baseline", type=Path, required=True)
    screen.add_argument("--endpoint", type=Path, required=True)
    screen.add_argument("--candidates", type=Path, required=True)
    screen.add_argument("--baseline-release", required=True)
    screen.add_argument("--endpoint-release", required=True)
    screen.add_argument("--cutoff-date", required=True)
    screen.add_argument("--output", type=Path, required=True, help="output .tsv or .tsv.gz")
    screen.add_argument("--id-column", default="VariationID")
    screen.add_argument("--class-column", default="ClinicalSignificance")
    screen.add_argument("--min-matching-scvs", type=int, default=1)
    screen.add_argument("--min-unique-submitters", type=int, default=1)
    screen.add_argument(
        "--allow-opposing",
        action="store_true",
        help="do not fail candidates with opposing or ambiguous contributing SCVs",
    )
    screen.add_argument(
        "--allow-no-post-cutoff-event",
        action="store_true",
        help="do not require a post-cutoff new/version-updated matching SCV",
    )
    screen.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            summary = verify_audit(args.summary.resolve())
            if summary.get("audit_policy") == "candidate_scv_contribution_screen_v1":
                print(f"Verified {summary['results']['candidate_count']:,} candidates")
            else:
                print(
                    f"Verified {summary['events']['changed_variation_ids']:,} "
                    "changed VariationIDs"
                )
            return 0
        if args.command == "screen":
            summary = build_candidate_screen(
                args.baseline,
                args.endpoint,
                args.candidates,
                args.output,
                args.baseline_release,
                args.endpoint_release,
                args.cutoff_date,
                id_column=args.id_column,
                class_column=args.class_column,
                min_matching_scvs=args.min_matching_scvs,
                min_unique_submitters=args.min_unique_submitters,
                require_no_opposition=not args.allow_opposing,
                require_post_cutoff_event=not args.allow_no_post_cutoff_event,
                overwrite=args.overwrite,
            )
            print(
                f"Screened {summary['results']['candidate_count']:,} candidates: "
                f"{summary['results']['pass_count']:,} pass, "
                f"{summary['results']['fail_count']:,} fail"
            )
            print(f"Wrote candidate SCV evidence to {summary['output']['path']}")
            return 0
        summary = build_audit(
            args.baseline,
            args.endpoint,
            args.output,
            args.baseline_release,
            args.endpoint_release,
            args.cutoff_date,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
        print(
            f"Audited {summary['baseline']['content']['rows']:,} baseline and "
            f"{summary['endpoint']['content']['rows']:,} endpoint SCVs"
        )
        print(
            f"Wrote {summary['events']['changed_variation_ids']:,} changed VariationIDs to "
            f"{summary['output']['path']}"
        )
        return 0
    except SubmissionAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
