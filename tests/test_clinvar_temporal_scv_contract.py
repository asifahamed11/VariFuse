from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

import config
import schema


stage09 = importlib.import_module("09_prepare_external_esm_dataset")


REVIEW_EXPERT = "reviewed by expert panel"
REVIEW_MULTIPLE = "criteria provided, multiple submitters, no conflicts"


def _identity_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["review"] = frame.get(
        "review", pd.Series(REVIEW_EXPERT, index=frame.index)
    ).fillna(REVIEW_EXPERT)
    frame["last_evaluated"] = frame.get(
        "last_evaluated", pd.Series("2025-02-01", index=frame.index)
    ).fillna("2025-02-01")
    return stage09._attach_clinvar_identity(frame)


def test_snapshot_identity_excludes_ambiguous_mappings_and_conflicts() -> None:
    frame = _identity_frame(
        [
            {"variant_id": "g1", "clinvar_variation_id": "101", schema.LABEL_COL: 1},
            {"variant_id": "g2", "clinvar_variation_id": "101", schema.LABEL_COL: 1},
            {"variant_id": "g3", "clinvar_variation_id": "201", schema.LABEL_COL: 0},
            {"variant_id": "g3", "clinvar_variation_id": "202", schema.LABEL_COL: 0},
            {"variant_id": "g4", "clinvar_variation_id": "301", schema.LABEL_COL: 0},
            {"variant_id": "g4", "clinvar_variation_id": "301", schema.LABEL_COL: 1},
            {"variant_id": "g5", "clinvar_variation_id": None, schema.LABEL_COL: 0},
            {"variant_id": "g5", "clinvar_variation_id": "401", schema.LABEL_COL: 0},
            {"variant_id": "g6", "clinvar_variation_id": "000501", schema.LABEL_COL: 1},
        ]
    )

    assert frame["CLINVAR_VARIATION_ID"].tolist() == ["401", "501"]
    audit = frame.attrs["clinvar_identity_audit"]
    assert audit["ambiguous_stable_variation_ids"] == 1
    assert audit["ambiguous_genomic_keys_with_multiple_stable_ids"] == 1
    assert audit["rows_excluded_for_identity_ambiguity_or_conflict"] == 6
    assert audit["genomic_fallback_rows_shadowed_by_stable_identity"] == 1


def test_clinvar_reader_never_substitutes_allele_id_for_variation_id(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "variant_summary.tsv"
    pd.DataFrame(
        {
            "Chromosome": ["1", "1"],
            "PositionVCF": [10, 20],
            "ReferenceAlleleVCF": ["A", "C"],
            "AlternateAlleleVCF": ["G", "T"],
            "ClinicalSignificance": ["Pathogenic", "Benign"],
            "ReviewStatus": [REVIEW_EXPERT, REVIEW_EXPERT],
            "Assembly": ["GRCh37", "GRCh37"],
            "LastEvaluated": ["2025-02-01", "2025-02-01"],
            "VariationID": ["123", None],
            "AlleleID": ["9001", "9002"],
        }
    ).to_csv(archive, sep="\t", index=False)

    observed = stage09._read_clinvar(archive).set_index("variant_id")

    assert observed.loc["GRCh37:1:10:A:G", "CLINVAR_VARIATION_ID"] == "123"
    assert pd.isna(
        observed.loc["GRCh37:1:20:C:T", "CLINVAR_VARIATION_ID"]
    )
    assert observed.loc[
        "GRCh37:1:20:C:T", "CLINVAR_IDENTITY_SOURCE"
    ] == "genomic_fallback_no_variation_id"


def test_temporal_matching_uses_stable_id_then_missing_id_genomic_fallback() -> None:
    baseline = _identity_frame(
        [
            {"variant_id": "g1", "clinvar_variation_id": "10", schema.LABEL_COL: 0},
            {"variant_id": "g2", "clinvar_variation_id": "20", schema.LABEL_COL: 0},
            {"variant_id": "g3", "clinvar_variation_id": None, schema.LABEL_COL: 1},
            {"variant_id": "g4", "clinvar_variation_id": "40", schema.LABEL_COL: 1},
        ]
    )
    endpoint = _identity_frame(
        [
            {"variant_id": "g9", "clinvar_variation_id": "10", schema.LABEL_COL: 0},
            {"variant_id": "g2", "clinvar_variation_id": "21", schema.LABEL_COL: 0},
            {"variant_id": "g3", "clinvar_variation_id": "30", schema.LABEL_COL: 1},
            {"variant_id": "g4", "clinvar_variation_id": None, schema.LABEL_COL: 1},
            {"variant_id": "g5", "clinvar_variation_id": "50", schema.LABEL_COL: 0},
        ]
    )

    filtered, audit = stage09._exclude_cross_snapshot_identity_ambiguity(
        baseline, endpoint
    )
    assert set(filtered["variant_id"]) == {"g3", "g4", "g5"}
    assert audit["stable_variation_ids_remapped_between_snapshots"] == 1
    assert audit["genomic_keys_linked_to_multiple_stable_ids_between_snapshots"] == 1
    assert audit["endpoint_rows_excluded_for_cross_snapshot_remapping"] == 2

    matched = stage09._match_clinvar_temporal_identities(baseline, filtered)
    source = dict(zip(matched["variant_id"], matched["TEMPORAL_MATCH_SOURCE"]))
    assert source == {
        "g3": "genomic_fallback_baseline_missing_variation_id",
        "g4": "genomic_fallback_endpoint_missing_variation_id",
        "g5": "absent_at_baseline",
    }


def _screen_record(
    variation_id: int,
    candidate_class: str,
    *,
    matching_submitters: int,
    tool_pass: int = 1,
    reason: str = "pass",
) -> dict[str, object]:
    return {
        "VariationID": variation_id,
        "candidate_class": candidate_class,
        "current_contributing_matching_scv_count": max(1, matching_submitters),
        "current_contributing_opposing_scv_count": int(not tool_pass),
        "current_contributing_ambiguous_scv_count": 0,
        "current_contributing_unique_submitter_count": matching_submitters,
        "current_contributing_matching_unique_submitter_count": matching_submitters,
        "post_cutoff_new_matching_scv_count": 1,
        "post_cutoff_updated_matching_scv_count": 0,
        "post_cutoff_matching_event_scv_count": 1,
        "current_contributing_matching_scv_ids": f"SCV{variation_id}.1",
        "current_contributing_opposing_scv_ids": "" if tool_pass else "SCV999.1",
        "post_cutoff_matching_event_scv_ids": f"SCV{variation_id}.1",
        "post_cutoff_matching_event_dates": "2025-02-01",
        "scv_evidence_pass": tool_pass,
        "scv_evidence_reason": reason,
    }


def test_scv_gate_enforces_review_specific_submitters_and_preserves_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _identity_frame(
        [
            {"variant_id": "g1", "clinvar_variation_id": "1", schema.LABEL_COL: 1},
            {
                "variant_id": "g2",
                "clinvar_variation_id": "2",
                schema.LABEL_COL: 0,
                "review": REVIEW_MULTIPLE,
            },
            {
                "variant_id": "g3",
                "clinvar_variation_id": "3",
                schema.LABEL_COL: 0,
                "review": REVIEW_MULTIPLE,
            },
            {"variant_id": "g4", "clinvar_variation_id": "4", schema.LABEL_COL: 1},
            {"variant_id": "g5", "clinvar_variation_id": None, schema.LABEL_COL: 1},
        ]
    )
    candidates["TEMPORAL_STATUS"] = (
        candidates["CLINVAR_VARIATION_ID"]
        .map({"1": "unchanged", "3": "unchanged"})
        .fillna("new_variant")
    )
    candidates["TEMPORAL_MATCH_SOURCE"] = "absent_at_baseline"
    records = [
        _screen_record(1, "pathogenic", matching_submitters=1),
        _screen_record(2, "benign", matching_submitters=1),
        _screen_record(3, "benign", matching_submitters=2),
        _screen_record(
            4,
            "pathogenic",
            matching_submitters=1,
            tool_pass=0,
            reason="opposing_current_contributing_scv",
        ),
    ]
    monkeypatch.setattr(stage09, "CLINVAR_REQUIRE_SCV_EVIDENCE", True)
    monkeypatch.setattr(stage09, "CLINVAR_SCV_MIN_MATCHING", 1)
    monkeypatch.setattr(stage09, "CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS", 1)
    monkeypatch.setattr(stage09, "CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM", 2)
    monkeypatch.setattr(
        stage09,
        "_run_scv_candidate_screen",
        lambda candidate_classes: (records, {"authenticated": True}),
    )

    retained, audit, summary = stage09._apply_clinvar_scv_evidence(candidates)

    assert retained["CLINVAR_VARIATION_ID"].tolist() == ["1", "3"]
    assert retained["TEMPORAL_ASSERTION_POLICY"].eq(
        "current_high_confidence_aggregate_with_post_cutoff_new_or_versioned_"
        "matching_scv_evidence"
    ).all()
    by_id = audit.set_index("CLINVAR_VARIATION_ID")
    assert by_id.loc["2", "SCV_EVIDENCE_REASON"] == (
        "insufficient_unique_matching_submitters_for_aggregate_review_status"
    )
    assert by_id.loc["4", "SCV_EVIDENCE_REASON"] == (
        "opposing_current_contributing_scv"
    )
    fallback = audit[audit["CLINVAR_VARIATION_ID"].isna()].iloc[0]
    assert fallback["SCV_EVIDENCE_REASON"] == (
        "missing_stable_variation_id_scv_screen_unavailable"
    )
    assert list(audit.columns) == list(stage09.CLINVAR_SCV_AUDIT_COLUMNS)
    assert summary["model_outputs_or_predictor_availability_used_for_selection"] is False
    assert summary["retained_temporal_status_counts"]["unchanged"] == 2


def test_load_temporal_clinvar_keeps_aggregate_unchanged_rows_for_scv_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline.tsv"
    endpoint_path = tmp_path / "endpoint.tsv"
    baseline_path.touch()
    endpoint_path.touch()
    baseline = _identity_frame(
        [{"variant_id": "g1", "clinvar_variation_id": "1", schema.LABEL_COL: 1}]
    )
    endpoint = _identity_frame(
        [
            {"variant_id": "g1", "clinvar_variation_id": "1", schema.LABEL_COL: 1},
            {"variant_id": "g2", "clinvar_variation_id": "2", schema.LABEL_COL: 0},
        ]
    )
    seen_statuses: list[str] = []

    def fake_gate(frame: pd.DataFrame):
        seen_statuses.extend(frame["TEMPORAL_STATUS"].tolist())
        return frame.copy(), pd.DataFrame(), {"status": "test"}

    monkeypatch.setattr(stage09, "CLINVAR_TRAIN_ARCHIVE", baseline_path)
    monkeypatch.setattr(stage09, "CLINVAR_EXTERNAL_ARCHIVE", endpoint_path)
    monkeypatch.setattr(stage09, "CLINVAR_REQUIRE_SCV_EVIDENCE", True)
    monkeypatch.setattr(stage09, "CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION", True)
    monkeypatch.setattr(stage09, "TRAIN_CUTOFF_DATE", "2024-06-30")
    monkeypatch.setattr(stage09, "_validate_external_temporal_configuration", lambda: {})
    monkeypatch.setattr(
        stage09,
        "_read_clinvar",
        lambda path: baseline.copy() if path == baseline_path else endpoint.copy(),
    )
    monkeypatch.setattr(stage09, "_apply_clinvar_scv_evidence", fake_gate)

    result, temporal_audit, _ = stage09.load_temporal_clinvar(return_audit=True)

    assert seen_statuses == ["unchanged", "new_variant"]
    assert result["TEMPORAL_STATUS"].tolist() == seen_statuses
    assert temporal_audit["aggregate_candidate_policy"] == (
        "all_current_high_confidence_aggregates_then_scv_temporal_gate"
    )


def test_submission_provenance_binds_exact_archive_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "submission_summary_2024-06.txt.gz"
    archive.write_bytes(b"pinned submission archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    provenance = archive.with_name(archive.name + ".provenance.json")
    provenance.write_text(
        json.dumps(
            {
                "provider": "NCBI ClinVar",
                "version": "2024-06",
                "source_id": "test",
                "publisher_checksum": "test",
                "observed_checksums": {"sha256": digest.upper()},
            }
        ),
        encoding="utf-8",
    )

    authenticated = config.validate_clinvar_submission_provenance(
        archive, "2024-06"
    )
    assert authenticated["archive_sha256"] == digest
    archive.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="differs from its pinned"):
        config.validate_clinvar_submission_provenance(archive, "2024-06")
