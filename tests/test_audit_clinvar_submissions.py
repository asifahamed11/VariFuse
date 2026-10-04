from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "audit_clinvar_submissions.py"
SPEC = importlib.util.spec_from_file_location("audit_clinvar_submissions", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


HEADER = [
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
]


def _row(variation: int, scv: str, significance: str, evaluated: str) -> list[str]:
    return [
        str(variation),
        significance,
        evaluated,
        "description",
        "submitted phenotype",
        "reported phenotype",
        "criteria provided, single submitter",
        "clinical testing",
        "germline:1",
        f"submitter-{scv}",
        scv,
        "GENE",
        "explanation",
        "-",
        "-",
    ]


def _write_archive(path: Path, rows: list[list[str]], release: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("##ClinVar submission fixture\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    sidecar = path.with_name(path.name + audit.PROVENANCE_SUFFIX)
    sidecar.write_text(
        json.dumps(
            {
                "source_id": f"clinvar_submission_{release.replace('-', '_')}",
                "provider": "NCBI ClinVar",
                "version": release,
                "publisher_checksum": None,
                "observed_checksums": {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )


def test_streaming_scv_merge_emits_versioned_temporal_events(tmp_path):
    baseline = tmp_path / "submission_summary_2024-06.txt.gz"
    endpoint = tmp_path / "submission_summary_2026-08.txt.gz"
    _write_archive(
        baseline,
        [
            _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024"),
            _row(2, "SCV000000002.1", "Uncertain significance", "Jan 01, 2024"),
            _row(4, "SCV000000004.1", "Pathogenic", "Jan 01, 2024"),
        ],
        "2024-06",
    )
    _write_archive(
        endpoint,
        [
            _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024"),
            _row(2, "SCV000000002.2", "Pathogenic", "Feb 10, 2025"),
            _row(3, "SCV000000003.1", "Benign", "Mar 01, 2026"),
        ],
        "2026-08",
    )
    output = tmp_path / "scv_events.tsv.gz"

    summary = audit.build_audit(
        baseline,
        endpoint,
        output,
        "2024-06",
        "2026-08",
        "2025-01-31",
        progress_every=0,
    )

    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert [record["VariationID"] for record in records] == ["2", "3", "4"]
    assert records[0]["updated_scv_count"] == "1"
    assert records[0]["event_post_cutoff_scv_count"] == "1"
    assert records[1]["new_scv_count"] == "1"
    assert records[2]["withdrawn_scv_count"] == "1"
    assert summary["events"]["status_counts"] == {
        "new_scv": 1,
        "unchanged": 1,
        "updated_scv": 1,
        "withdrawn_scv": 1,
    }
    summary_path = output.with_name(output.name + audit.AUDIT_SUFFIX)
    assert audit.verify_audit(summary_path)["events"]["changed_variation_ids"] == 3


def test_unsorted_variation_ids_fail_closed(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    _write_archive(
        baseline,
        [
            _row(2, "SCV000000002.1", "Benign", "Jan 01, 2024"),
            _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024"),
        ],
        "2024-06",
    )
    _write_archive(
        endpoint,
        [_row(2, "SCV000000002.2", "Pathogenic", "Jan 01, 2026")],
        "2026-08",
    )
    with pytest.raises(audit.SubmissionAuditError, match="not ordered"):
        audit.build_audit(
            baseline,
            endpoint,
            tmp_path / "events.tsv.gz",
            "2024-06",
            "2026-08",
            "2025-01-31",
            progress_every=0,
        )


def test_same_scv_can_have_multiple_rows_but_not_multiple_versions(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    duplicate = _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024")
    _write_archive(baseline, [duplicate, duplicate], "2024-06")
    endpoint_rows = [
        duplicate,
        _row(1, "SCV000000001.2", "Pathogenic", "Jan 01, 2026"),
    ]
    _write_archive(endpoint, endpoint_rows, "2026-08")
    with pytest.raises(audit.SubmissionAuditError, match="multiple versions"):
        audit.build_audit(
            baseline,
            endpoint,
            tmp_path / "events.tsv.gz",
            "2024-06",
            "2026-08",
            "2025-01-31",
            progress_every=0,
        )


def test_post_cutoff_count_is_by_scv_not_duplicate_phenotype_row(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    _write_archive(
        baseline,
        [_row(1, "SCV000000001.1", "Uncertain significance", "Jan 01, 2024")],
        "2024-06",
    )
    duplicate = _row(1, "SCV000000001.2", "Pathogenic", "Feb 10, 2025")
    _write_archive(endpoint, [duplicate, duplicate], "2026-08")
    output = tmp_path / "events.tsv.gz"

    audit.build_audit(
        baseline,
        endpoint,
        output,
        "2024-06",
        "2026-08",
        "2025-01-31",
        progress_every=0,
    )

    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        record = next(csv.DictReader(handle, delimiter="\t"))
    assert record["event_scv_count"] == "1"
    assert record["event_post_cutoff_scv_count"] == "1"


def test_missing_acquisition_provenance_is_rejected(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    _write_archive(
        baseline,
        [_row(1, "SCV000000001.1", "Benign", "Jan 01, 2024")],
        "2024-06",
    )
    _write_archive(
        endpoint,
        [_row(1, "SCV000000001.2", "Pathogenic", "Jan 01, 2026")],
        "2026-08",
    )
    baseline.with_name(baseline.name + audit.PROVENANCE_SUFFIX).unlink()
    with pytest.raises(audit.SubmissionAuditError, match="lacks acquisition provenance"):
        audit.build_audit(
            baseline,
            endpoint,
            tmp_path / "events.tsv.gz",
            "2024-06",
            "2026-08",
            "2025-01-31",
            progress_every=0,
        )


def test_candidate_screen_counts_current_and_temporal_scv_evidence(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    _write_archive(
        baseline,
        [
            _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024"),
            _row(2, "SCV000000002.1", "Uncertain significance", "Jan 01, 2024"),
        ],
        "2024-06",
    )
    _write_archive(
        endpoint,
        [
            _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024"),
            _row(2, "SCV000000002.2", "Pathogenic", "Feb 10, 2025"),
            _row(3, "SCV000000003.1", "Benign", "Mar 01, 2026"),
            _row(4, "SCV000000004.1", "Pathogenic", "Apr 01, 2026"),
            _row(4, "SCV000000005.1", "Benign", "Apr 02, 2026"),
        ],
        "2026-08",
    )

    records, scan = audit.screen_candidate_variations(
        baseline,
        endpoint,
        {1: "Benign", 2: "Pathogenic", 3: "Benign", 4: "Pathogenic", 5: "Benign"},
        "2025-01-31",
    )

    by_id = {record["VariationID"]: record for record in records}
    assert by_id[1]["scv_evidence_reason"] == "no_post_cutoff_new_or_updated_matching_scv"
    assert by_id[2]["scv_evidence_pass"] == 1
    assert by_id[2]["post_cutoff_updated_matching_scv_count"] == 1
    assert by_id[2]["post_cutoff_matching_event_scv_ids"] == "SCV000000002.2"
    assert by_id[3]["scv_evidence_pass"] == 1
    assert by_id[3]["post_cutoff_new_matching_scv_count"] == 1
    assert by_id[4]["scv_evidence_pass"] == 0
    assert by_id[4]["current_contributing_matching_scv_count"] == 1
    assert by_id[4]["current_contributing_opposing_scv_count"] == 1
    assert by_id[4]["scv_evidence_reason"] == "opposing_current_contributing_scv"
    assert by_id[5]["scv_evidence_reason"] == "candidate_missing_from_endpoint"
    assert scan["baseline_rows_scanned"] == 2
    assert scan["endpoint_rows_scanned"] == 5


def test_candidate_screen_cli_artifact_is_provenance_bound_and_verifiable(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    candidates = tmp_path / "candidates.csv.gz"
    output = tmp_path / "candidate_scv_screen.tsv.gz"
    _write_archive(
        baseline,
        [_row(2, "SCV000000002.1", "Uncertain significance", "Jan 01, 2024")],
        "2024-06",
    )
    _write_archive(
        endpoint,
        [_row(2, "SCV000000002.2", "Pathogenic", "Feb 10, 2025")],
        "2026-08",
    )
    with gzip.open(candidates, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["VariationID", "ClinicalSignificance", "ignored_description"])
        writer.writerow([2, "Pathogenic", "must not be copied"])

    summary = audit.build_candidate_screen(
        baseline,
        endpoint,
        candidates,
        output,
        "2024-06",
        "2026-08",
        "2025-01-31",
    )

    summary_path = output.with_name(output.name + audit.AUDIT_SUFFIX)
    verified = audit.verify_audit(summary_path)
    assert verified["results"] == summary["results"]
    assert verified["candidates"]["sha256"] == hashlib.sha256(candidates.read_bytes()).hexdigest()
    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        record = next(reader)
        assert reader.fieldnames == list(audit.SCREEN_COLUMNS)
    assert record["scv_evidence_pass"] == "1"
    assert "description" not in record

    first_output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    audit.build_candidate_screen(
        baseline,
        endpoint,
        candidates,
        output,
        "2024-06",
        "2026-08",
        "2025-01-31",
        overwrite=True,
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first_output_sha256


def test_literal_quote_in_unquoted_clinvar_tsv_does_not_merge_rows(tmp_path):
    archive = tmp_path / "submission.txt.gz"
    rows = [
        _row(1, "SCV000000001.1", "Benign", "Jan 01, 2024"),
        _row(2, "SCV000000002.1", "Pathogenic", "Jan 01, 2024"),
    ]
    rows[0][9] = 'The Shared Resource Centre Genome", Research Centre'
    with gzip.open(archive, "wt", encoding="utf-8", newline="") as handle:
        handle.write("##fixture\n")
        handle.write("\t".join(HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")

    stats = audit._new_stats()
    parsed = list(audit._rows(archive, stats))

    assert [row["#VariationID"] for row in parsed] == ["1", "2"]
    assert stats["rows"] == 2


def test_candidate_screen_ignores_noncontributing_endpoint_scv(tmp_path):
    baseline = tmp_path / "baseline.txt.gz"
    endpoint = tmp_path / "endpoint.txt.gz"
    extended_header = HEADER + ["ContributesToAggregateClassification"]
    with gzip.open(baseline, "wt", encoding="utf-8", newline="") as handle:
        handle.write("##fixture\n")
        handle.write("\t".join(extended_header) + "\n")
        handle.write(
            "\t".join(
                _row(9, "SCV000000009.1", "Uncertain significance", "Jan 01, 2024")
                + ["yes"]
            )
            + "\n"
        )
    with gzip.open(endpoint, "wt", encoding="utf-8", newline="") as handle:
        handle.write("##fixture\n")
        handle.write("\t".join(extended_header) + "\n")
        handle.write(
            "\t".join(
                _row(9, "SCV000000009.2", "Pathogenic", "Feb 10, 2025") + ["yes"]
            )
            + "\n"
        )
        handle.write(
            "\t".join(
                _row(9, "SCV000000010.1", "Benign", "Feb 11, 2025") + ["no"]
            )
            + "\n"
        )

    records, _ = audit.screen_candidate_variations(
        baseline,
        endpoint,
        {9: "Pathogenic"},
        "2025-01-31",
    )

    assert records[0]["current_contributing_matching_scv_count"] == 1
    assert records[0]["current_contributing_opposing_scv_count"] == 0
    assert records[0]["scv_evidence_pass"] == 1
