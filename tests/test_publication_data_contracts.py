from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import schema


stage01 = importlib.import_module("01_dbnsfp_processor")
stage07 = importlib.import_module("07_dataset_balancing")
stage09 = importlib.import_module("09_prepare_external_esm_dataset")
stage12 = importlib.import_module("12_external_validation")


def _dbnsfp_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {column: "." for column in stage01.DBNSFP_COLS}
    records: list[dict[str, object]] = []
    for row in rows:
        record = {**defaults, **row}
        if stage01.DBNSFP_GRCH37_CHROM_COLUMN not in row:
            record[stage01.DBNSFP_GRCH37_CHROM_COLUMN] = row.get("#chr", ".")
        if stage01.DBNSFP_GRCH37_POSITION_COLUMN not in row:
            record[stage01.DBNSFP_GRCH37_POSITION_COLUMN] = row.get("pos(1-based)", ".")
        records.append(record)
    return pd.DataFrame(records)


def _empty_cancer_resources() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    cgc = pd.DataFrame(columns=["genename", "ROLE_IN_CANCER", "TIER"])
    roles = {name: set() for name in ("cancer", "oncogene", "tsg", "tier1")}
    return cgc, roles


def test_source_release_status_is_explicit_and_month_conservative() -> None:
    status = config.source_release_status("snapshot", "2025-01", cutoff="2025-01-15")
    assert status["precision"] == "month"
    assert status["comparison_date"] == "2025-01-31"
    assert status["on_or_before_cutoff"] is False
    missing = config.source_release_status("undated", None, cutoff="2025-01-31")
    assert missing["comparison_date"] is None
    assert missing["on_or_before_cutoff"] is None


def test_clinical_label_task_excludes_ba1_and_somatic_only_labels() -> None:
    raw = _dbnsfp_rows(
        [
            {
                "#chr": "1",
                "pos(1-based)": 10,
                "ref": "A",
                "alt": "G",
                "aaref": "A",
                "aaalt": "V",
                "aapos": 1,
                "genename": "COMMON",
                "gnomAD4.1_joint_AF": 0.10,
            },
            {
                "#chr": "1",
                "pos(1-based)": 20,
                "ref": "C",
                "alt": "T",
                "aaref": "C",
                "aaalt": "Y",
                "aapos": 2,
                "genename": "CLINICAL_BENIGN",
                "gnomAD4.1_joint_AF": 0.0,
            },
            {
                "#chr": "12",
                "pos(1-based)": 30,
                "ref": "A",
                "alt": "C",
                "aaref": "G",
                "aaalt": "V",
                "aapos": 12,
                "genename": "KRAS",
                "gnomAD4.1_joint_AF": 0.0,
            },
            {
                "#chr": "2",
                "pos(1-based)": 40,
                "ref": "G",
                "alt": "A",
                "aaref": "G",
                "aaalt": "D",
                "aapos": 3,
                "genename": "CLINICAL_PATHOGENIC",
                "gnomAD4.1_joint_AF": 0.0,
            },
        ]
    )
    cosmic = pd.DataFrame(
        {
            "genename": ["KRAS"],
            "aapos": pd.array([12], dtype="Int64"),
            "aaref": ["G"],
            "aaalt": ["V"],
            "COSMIC_RECURRENCE": [100],
            "COSMIC_FREQUENCY": [0.1],
        }
    )
    cgc, roles = _empty_cancer_resources()
    identity = pd.DataFrame(
        {
            "CLINVAR_VARIATION_ID": ["20", "40"],
            "CLINVAR_SOURCE_GENE": ["CLINICAL_BENIGN", "CLINICAL_PATHOGENIC"],
            "CLINVAR_SOURCE_NAME": ["p.Cys2Tyr", "p.Gly3Asp"],
            "CLINVAR_REVIEW_STATUS": [
                "criteria provided, multiple submitters, no conflicts",
                "reviewed by expert panel",
            ],
        },
        index=["GRCh37:1:20:C:T", "GRCh37:2:40:G:A"],
    )
    output, _ = stage01._process_chunk(
        raw,
        {"GRCh37:2:40:G:A"},
        {"GRCh37:1:20:C:T"},
        set(),
        set(),
        cgc,
        roles,
        cosmic,
        set(),
        "clinical",
        identity,
    )
    assert output[["variant_id", schema.LABEL_COL]].values.tolist() == [
        ["GRCh37:1:20:C:T", 0],
        ["GRCh37:2:40:G:A", 1],
    ]
    assert output["EVIDENCE_SOURCE"].tolist() == [
        "clinvar_benign",
        "clinvar_pathogenic",
    ]
    assert output["CLINVAR_VARIATION_ID"].tolist() == ["20", "40"]


def test_dbnsfp_grch37_projection_never_uses_primary_coordinates() -> None:
    raw = _dbnsfp_rows(
        [
            {
                "#chr": "1",
                "pos(1-based)": 955677,
                stage01.DBNSFP_GRCH37_CHROM_COLUMN: "1",
                stage01.DBNSFP_GRCH37_POSITION_COLUMN: 891057,
                "ref": "A",
                "alt": "C",
                "aaref": "V",
                "aaalt": "G",
                "aapos": 243,
                "genename": "NOC2L",
            },
            {
                "#chr": "17",
                "pos(1-based)": 7674220,
                stage01.DBNSFP_GRCH37_CHROM_COLUMN: ".",
                stage01.DBNSFP_GRCH37_POSITION_COLUMN: ".",
                "ref": "C",
                "alt": "T",
                "aaref": "R",
                "aaalt": "W",
                "aapos": 248,
                "genename": "TP53",
            },
        ]
    )
    projected = stage01.project_dbnsfp_to_grch37(raw)
    assert stage01._variant_keys(projected).tolist() == ["GRCh37:1:891057:A:C"]
    assert projected["genename"].tolist() == ["NOC2L"]
    assert "GRCh37:1:955677:A:C" not in set(stage01._variant_keys(projected))


def test_dbnsfp_grch37_projection_requires_explicit_hg19_columns() -> None:
    raw = pd.DataFrame(
        {
            "#chr": ["1"],
            "pos(1-based)": [955677],
            "ref": ["A"],
            "alt": ["C"],
        }
    )
    with pytest.raises(KeyError, match="explicit hg19 columns"):
        stage01.project_dbnsfp_to_grch37(raw)


def test_external_dbnsfp_scan_uses_same_grch37_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "dbnsfp.tsv.gz"
    raw = _dbnsfp_rows(
        [
            {
                "#chr": "1",
                "pos(1-based)": 955677,
                stage01.DBNSFP_GRCH37_CHROM_COLUMN: "1",
                stage01.DBNSFP_GRCH37_POSITION_COLUMN: 891057,
                "ref": "A",
                "alt": "C",
                "aaref": "V",
                "aaalt": "G",
                "aapos": 243,
                "genename": "NOC2L",
                "Ensembl_transcriptid": "ENST00000327044",
            }
        ]
    )[stage09.DBNSFP_COLS]
    raw.to_csv(archive, sep="\t", index=False, compression="gzip")
    monkeypatch.setattr(stage09, "DBNSFP_FILE", archive)

    wrong = stage09.scan_dbnsfp_for_clinvar({"GRCh37:1:955677:A:C"})
    assert wrong.empty
    correct = stage09.scan_dbnsfp_for_clinvar({"GRCh37:1:891057:A:C"})
    assert correct["variant_id"].tolist() == ["GRCh37:1:891057:A:C"]
    assert (
        correct.attrs["dbnsfp_coordinate_projection"]["dbnsfp_projection_policy"]
        == config.DBNSFP_COORDINATE_POLICY
    )


def test_upstream_manifest_rejects_coordinate_contract_mismatch(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "somatic_variant_dbNSFP.csv"
    artifact.write_text("variant_id\nGRCh37:1:1:A:G\n", encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    manifest = config.build_run_manifest("01_dbnsfp_processor", outputs=[artifact])
    manifest["coordinate_contract"] = {
        **config.COORDINATE_CONTRACT,
        "dbnsfp_projection_policy": "unsafe_primary_coordinate_fallback",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="genome-assembly coordinate contract"):
        config.validate_upstream_manifest(
            manifest_path,
            "01_dbnsfp_processor",
            [artifact],
        )


def test_training_clinvar_identity_is_canonical_and_ambiguous_mappings_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "clinvar.tsv"
    rows = [
        ("1", 10, "A", "G", "000101", "Benign", "G1"),
        ("1", 10, "A", "G", "101", "Benign", "G1"),
        ("2", 20, "C", "T", "202", "Pathogenic", "G2"),
        ("3", 30, "G", "A", "202", "Pathogenic", "G3"),
        ("4", 40, "T", "C", "303", "Benign", "G4"),
        ("4", 40, "T", "C", "304", "Benign", "G4"),
        ("5", 50, "A", "C", "404", "Benign", "G5"),
        ("5", 50, "A", "C", "404", "Pathogenic", "G5"),
        ("6", 60, "G", "T", "505", "Pathogenic", "G6"),
    ]
    pd.DataFrame(
        {
            "Chromosome": [row[0] for row in rows],
            "PositionVCF": [row[1] for row in rows],
            "ReferenceAlleleVCF": [row[2] for row in rows],
            "AlternateAlleleVCF": [row[3] for row in rows],
            "VariationID": [row[4] for row in rows],
            "ClinicalSignificance": [row[5] for row in rows],
            "GeneSymbol": [row[6] for row in rows],
            "Name": [f"name-{index}" for index in range(len(rows))],
            "Assembly": ["GRCh37"] * len(rows),
            "ReviewStatus": ["reviewed by expert panel"] * len(rows),
        }
    ).to_csv(archive, sep="\t", index=False)
    monkeypatch.setattr(stage01, "CLINVAR_TRAIN_ARCHIVE", archive)
    pathogenic, benign, identity, audit, audit_rows = stage01._load_clinvar_training_labels(
        return_identity=True
    )
    assert pathogenic == {"GRCh37:6:60:G:T"}
    assert benign == {"GRCh37:1:10:A:G"}
    assert identity.loc["GRCh37:1:10:A:G", "CLINVAR_VARIATION_ID"] == "101"
    assert audit["ambiguous_stable_ids"] == 1
    assert audit["keys_with_multiple_stable_ids"] == 1
    assert audit["stable_id_label_conflicts"] == 1
    assert set(audit_rows["identity_status"]) == {"retained", "excluded"}


def test_stable_clinvar_id_deoverlap_precedes_coordinate_fallback(
    tmp_path: Path,
) -> None:
    internal = tmp_path / "internal.parquet"
    pd.DataFrame(
        {
            "chr": ["1"],
            "pos": [10],
            "ref": ["A"],
            "alt": ["G"],
            "genename": ["TRAIN_GENE"],
            "uniprot_id": ["P00001"],
            "aapos": [1],
            "aaref": ["A"],
            "aaalt": ["G"],
            "CLINVAR_VARIATION_ID": ["101"],
        }
    ).to_parquet(internal, index=False)
    external = pd.DataFrame(
        {
            "chr": ["2"],
            "pos": [20],
            "ref": ["C"],
            "alt": ["T"],
            "genename": ["EXTERNAL_GENE"],
            "uniprot_id": ["P99999"],
            "aa_pos": [2],
            "aa_ref": ["C"],
            "aa_alt": ["T"],
            "CLINVAR_VARIATION_ID": ["000101"],
        }
    )
    masks = stage12._deoverlap_masks(internal, external)
    assert masks["exact_variant_disjoint"].tolist() == [False]
    assert masks["gene_disjoint"].tolist() == [False]


def test_nonclinical_label_tasks_have_explicit_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage01, "LABEL_TASK", "legacy_mixed")
    monkeypatch.setattr(stage01, "ALLOW_LEGACY_MIXED_LABELS", False)
    with pytest.raises(RuntimeError, match="legacy_mixed"):
        stage01._validate_label_configuration()
    monkeypatch.setattr(stage01, "LABEL_TASK", "somatic")
    with pytest.raises(RuntimeError, match="somatic negative cohort"):
        stage01._validate_label_configuration()


def test_external_clinvar_selector_ranks_mane_after_mapping() -> None:
    frame = pd.DataFrame(
        {
            "variant_id": ["v1", "v1", "v1", "v2"],
            schema.ROW_ID_COL: ["v1-canonical", "v1-mane", "v1-plus", "v2"],
            schema.GENE_COL: ["G1", "G1", "G1", "G2"],
            "Ensembl_transcriptid": ["ENST-C", "ENST-M", "ENST-P", "ENST-2"],
            "VEP_canonical": ["YES", ".", ".", "YES"],
            "MANE": [".", "Select", "MANE Plus Clinical", "."],
            "HGVSp_snpEff": ["p.A1V", "p.A2V", "p.A3V", "p.C3Y"],
            "HGVSc_snpEff": ["c.1A>G", "c.2A>G", "c.3A>G", "c.3C>T"],
            "aapos": [1, 2, 3, 3],
            "aaref": ["A", "A", "A", "C"],
            "aaalt": ["V", "V", "V", "Y"],
            "uniprot_id": ["P1", "P1", "P1", "P2"],
            "PROTEIN_MAPPING_STATUS": [
                "mapped_transcript",
                "mapped_reference",
                "mapped_transcript",
                "mapped_transcript",
            ],
            "MAPPING_TRANSCRIPT_MATCH": [1, 0, 1, 1],
            "PRIMARY_MAPPING_ELIGIBLE": [1, 0, 1, 1],
            "MAPPING_CONFIDENCE": [
                "transcript_verified",
                "reference_only_sensitivity",
                "transcript_verified",
                "transcript_verified",
            ],
            "HAS_PROTEIN_MAPPING": [1, 1, 1, 1],
            schema.LABEL_COL: [1, 1, 1, 0],
        }
    )
    selected, audit = stage09._select_primary_clinvar_consequences(frame)
    assert selected["variant_id"].is_unique
    assert selected.set_index("variant_id").loc["v1", schema.ROW_ID_COL] == "v1-plus"
    assert not audit.loc[
        audit[schema.ROW_ID_COL].eq("v1-mane"), "PRIMARY_CONSEQUENCE_SELECTED"
    ].item()
    assert audit.loc[audit[schema.ROW_ID_COL].eq("v1-plus"), "PRIMARY_CONSEQUENCE_SELECTED"].item()


def test_clinvar_significance_containing_both_classes_is_excluded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clinvar.tsv"
    pd.DataFrame(
        {
            "Chromosome": ["1", "2", "3", "4"],
            "PositionVCF": [10, 20, 30, 40],
            "ReferenceAlleleVCF": ["A", "C", "G", "T"],
            "AlternateAlleleVCF": ["G", "T", "A", "C"],
            "Assembly": ["GRCh37"] * 4,
            "ReviewStatus": ["reviewed by expert panel"] * 4,
            "ClinicalSignificance": [
                "Pathogenic",
                "Benign",
                "Pathogenic; Benign",
                "Pathogenic; risk factor",
            ],
            "LastEvaluated": ["2026-01-01"] * 4,
            "VariationID": ["1", "2", "3", "4"],
        }
    ).to_csv(path, sep="\t", index=False)

    observed = stage09._read_clinvar(path)
    assert observed["variant_id"].tolist() == [
        "GRCh37:1:10:A:G",
        "GRCh37:2:20:C:T",
    ]
    assert observed[schema.LABEL_COL].tolist() == [1, 0]


def test_dms_sampling_is_label_independent_and_has_inverse_probability_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage09, "DMS_SAMPLING_POLICY", "hash_uniform")
    frame = pd.DataFrame(
        {
            schema.ROW_ID_COL: [f"row-{index}" for index in range(20)],
            "ASSAY_ID": ["assay"] * 20,
            "variant_id": [f"v-{index}" for index in range(20)],
            schema.LABEL_COL: [0] * 15 + [1] * 5,
        }
    )
    first = stage09._sample_dms_assay(frame, 5)
    relabelled = frame.copy()
    relabelled[schema.LABEL_COL] = 1 - relabelled[schema.LABEL_COL]
    second = stage09._sample_dms_assay(relabelled, 5)
    assert first[schema.ROW_ID_COL].tolist() == second[schema.ROW_ID_COL].tolist()
    assert np.allclose(first["DMS_SAMPLING_PROBABILITY"], 0.25)
    assert np.allclose(first["DMS_SAMPLE_WEIGHT"], 4.0)
    assert first["DMS_SAMPLING_POLICY"].eq("hash_uniform_without_label").all()


def test_strict_temporal_clinvar_requires_post_cutoff_last_evaluated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_path = tmp_path / "old.tsv"
    new_path = tmp_path / "new.tsv"
    old_path.touch()
    new_path.touch()
    old = pd.DataFrame({"variant_id": ["unchanged"], schema.LABEL_COL: [0]})
    current = pd.DataFrame(
        {
            "variant_id": ["unchanged", "old-evaluation", "future-evaluation"],
            schema.LABEL_COL: [0, 1, 1],
            "last_evaluated": ["2026-01-01", "2024-12-31", "2025-02-01"],
        }
    )
    monkeypatch.setattr(stage09, "CLINVAR_TRAIN_ARCHIVE", old_path)
    monkeypatch.setattr(stage09, "CLINVAR_EXTERNAL_ARCHIVE", new_path)
    monkeypatch.setattr(stage09, "CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION", True)
    monkeypatch.setattr(stage09, "CLINVAR_REQUIRE_SCV_EVIDENCE", False)
    monkeypatch.setattr(stage09, "TRAIN_CUTOFF_DATE", "2025-01-31")
    monkeypatch.setattr(stage09, "_validate_external_temporal_configuration", lambda: {})
    monkeypatch.setattr(
        stage09,
        "_read_clinvar",
        lambda path: old.copy() if path == old_path else current.copy(),
    )
    result = stage09.load_temporal_clinvar()
    assert result["variant_id"].tolist() == ["future-evaluation"]
    assert result["TEMPORAL_ASSERTION_POLICY"].eq("last_evaluated_after_cutoff").all()


def test_feature_coverage_audit_can_guard_large_class_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = stage07._build_feature_coverage_report(
        ["HAS_STRUCTURE"],
        Counter({0: 100, 1: 100}),
        Counter({("HAS_STRUCTURE", 0): 20, ("HAS_STRUCTURE", 1): 90}),
    )
    assert report["flagged"] == ["HAS_STRUCTURE"]
    monkeypatch.setattr(stage07, "FEATURE_COVERAGE_POLICY", "error")
    with pytest.raises(RuntimeError, match="ascertainment"):
        stage07._enforce_feature_coverage(report)


def test_clinvar_snapshot_provenance_binds_release_and_exact_bytes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "clinvar_2024-06_grch37.tsv"
    archive.write_bytes(b"header\nrow\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / (archive.name + ".transformation.json")
    payload = {
        "schema_version": 1,
        "transformation": "clinvar_variant_summary_grch37_stream_v1",
        "release": "2024-06",
        "input": {
            "sha256": "1" * 64,
            "acquisition_provenance_sha256": "2" * 64,
            "version": "2024-06",
            "provider": "NCBI ClinVar",
        },
        "output": {
            "size_bytes": archive.stat().st_size,
            "sha256": digest,
            "rows_written": 1,
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    verified = config.validate_clinvar_snapshot_provenance(archive, manifest, "2024-06")
    assert verified["status"] == "authenticated"
    assert verified["snapshot_sha256"] == digest

    archive.write_bytes(b"header\nchanged\n")
    with pytest.raises(RuntimeError, match="bytes differ"):
        config.validate_clinvar_snapshot_provenance(archive, manifest, "2024-06")


def test_proteingym_provenance_authenticates_exact_extracted_inventory(
    tmp_path: Path,
) -> None:
    extraction_root = tmp_path / "extracted"
    assays = extraction_root / "DMS_ProteinGym_substitutions"
    assays.mkdir(parents=True)
    assay = assays / "assay.csv"
    assay.write_bytes(b"mutant,DMS_score\nA1V,0.5\n")
    assay_digest = hashlib.sha256(assay.read_bytes()).hexdigest()
    extraction_manifest = extraction_root / "extraction_manifest.json"
    extraction_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extraction_policy": "safe_atomic_zip_v1",
                "archive": {
                    "version": "1.3",
                    "sha256": "3" * 64,
                    "acquisition_provenance_sha256": "4" * 64,
                    "publisher_checksum": {"algorithm": "md5", "value": "5" * 32},
                },
                "file_count": 1,
                "total_uncompressed_bytes": assay.stat().st_size,
                "members": [
                    {
                        "kind": "file",
                        "path": "DMS_ProteinGym_substitutions/assay.csv",
                        "size_bytes": assay.stat().st_size,
                        "sha256": assay_digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "DMS_substitutions.csv"
    metadata.write_bytes(b"DMS_id\nassay\n")
    metadata_digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    provenance = tmp_path / "DMS_substitutions.csv.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "ProteinGym",
                "version": "1.3",
                "observed_size_bytes": metadata.stat().st_size,
                "observed_checksums": {"sha256": metadata_digest},
                "publisher_checksum": {"algorithm": "md5", "value": "6" * 32},
            }
        ),
        encoding="utf-8",
    )

    verified = config.validate_proteingym_provenance(
        assays, metadata, extraction_manifest, provenance, "v1.3"
    )
    assert verified["status"] == "authenticated"
    assert verified["assay_files"] == 1

    (assays / "unexpected.csv").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="inventory differs"):
        config.validate_proteingym_provenance(
            assays, metadata, extraction_manifest, provenance, "v1.3"
        )
