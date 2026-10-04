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
MODULE_PATH = ROOT / "tools" / "prepare_clinvar_snapshot.py"
SPEC = importlib.util.spec_from_file_location("prepare_clinvar_snapshot", MODULE_PATH)
assert SPEC and SPEC.loader
snapshot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snapshot
SPEC.loader.exec_module(snapshot)


HEADER = [
    "#AlleleID",
    "Type",
    "ClinicalSignificance",
    "LastEvaluated",
    "OriginSimple",
    "Assembly",
    "Chromosome",
    "ReviewStatus",
    "VariationID",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "ExtraAuditColumn",
]


def _write_raw(path: Path, rows: list[list[str]], header: list[str] | None = None) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header or HEADER)
        writer.writerows(rows)


def _write_provenance(path: Path, release: str = "2024-06") -> Path:
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + snapshot.PROVENANCE_SUFFIX)
    sidecar.write_text(
        json.dumps(
            {
                "source_id": "clinvar_variant_test",
                "provider": "NCBI ClinVar",
                "version": release,
                "observed_checksums": {"sha256": sha256},
                "publisher_checksum": None,
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _rows() -> list[list[str]]:
    common = ["Pathogenic", "2024-05-01", "germline", "1", "expert", "101", "10", "A", "G"]
    return [
        ["1", "single nucleotide variant", *common[:3], "GRCh37", *common[3:], "keep"],
        ["2", "single nucleotide variant", *common[:3], "GRCh38", *common[3:], "drop"],
        ["3", "single nucleotide variant", *common[:3], "GRCh37.p13", *common[3:], "keep2"],
    ]


@pytest.mark.parametrize("compressed_output", [False, True])
def test_stream_transform_preserves_all_columns_and_only_grch37(tmp_path, compressed_output):
    raw = tmp_path / "variant_summary_2024-06.txt.gz"
    _write_raw(raw, _rows())
    _write_provenance(raw)
    suffix = ".tsv.gz" if compressed_output else ".tsv"
    output = tmp_path / f"clinvar_2024-06_grch37{suffix}"

    manifest = snapshot.transform_snapshot(raw, output, "2024-06", progress_every=0)

    opener = gzip.open if compressed_output else open
    with opener(output, "rt", encoding="utf-8", newline="") as handle:
        observed = list(csv.reader(handle, delimiter="\t"))
    assert observed[0] == HEADER
    assert [row[-1] for row in observed[1:]] == ["keep", "keep2"]
    assert manifest["output"]["rows_scanned"] == 3
    assert manifest["output"]["rows_written"] == 2
    assert manifest["output"]["column_count"] == len(HEADER)
    transform_manifest = output.with_name(output.name + snapshot.TRANSFORM_SUFFIX)
    assert snapshot.verify_transformation(transform_manifest)["release"] == "2024-06"


def test_transform_requires_matching_acquisition_provenance(tmp_path):
    raw = tmp_path / "variant_summary_2024-06.txt.gz"
    _write_raw(raw, _rows())
    output = tmp_path / "clinvar_2024-06_grch37.tsv"
    with pytest.raises(snapshot.SnapshotError, match="lacks acquisition provenance"):
        snapshot.transform_snapshot(raw, output, "2024-06", progress_every=0)

    sidecar = _write_provenance(raw, release="2025-01")
    assert sidecar.exists()
    with pytest.raises(snapshot.SnapshotError, match="differs from source provenance"):
        snapshot.transform_snapshot(raw, output, "2024-06", progress_every=0)


def test_transform_rejects_missing_contract_columns_without_partial_output(tmp_path):
    raw = tmp_path / "variant_summary_2024-06.txt.gz"
    incomplete_header = [column for column in HEADER if column != "LastEvaluated"]
    _write_raw(raw, [], header=incomplete_header)
    _write_provenance(raw)
    output = tmp_path / "clinvar_2024-06_grch37.tsv"

    with pytest.raises(snapshot.SnapshotError, match="LastEvaluated"):
        snapshot.transform_snapshot(raw, output, "2024-06", progress_every=0)
    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


def test_verify_detects_modified_derived_snapshot(tmp_path):
    raw = tmp_path / "variant_summary_2024-06.txt.gz"
    _write_raw(raw, _rows())
    _write_provenance(raw)
    output = tmp_path / "clinvar_2024-06_grch37.tsv"
    snapshot.transform_snapshot(raw, output, "2024-06", progress_every=0)
    manifest = output.with_name(output.name + snapshot.TRANSFORM_SUFFIX)
    with output.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(snapshot.SnapshotError, match="output file fails"):
        snapshot.verify_transformation(manifest)
