from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import config
import schema


stage01 = importlib.import_module("01_dbnsfp_processor")
stage04 = importlib.import_module("04_feature_engineering")
stage08 = importlib.import_module("08_prepare_esm_dataset")
stage08b = importlib.import_module("08b_build_homology_groups")
stage09 = importlib.import_module("09_prepare_external_esm_dataset")
stage10 = importlib.import_module("10_extract_esm_features")


def _mapped_candidate(
    variant: str,
    row_id: str,
    transcript: str,
    *,
    mane: str = ".",
    canonical: str = ".",
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "variant_id": variant,
        schema.ROW_ID_COL: row_id,
        schema.GENE_COL: "GENE",
        "Ensembl_transcriptid": transcript,
        "MANE": mane,
        "VEP_canonical": canonical,
        "HGVSp_snpEff": "p.A1V",
        "HGVSc_snpEff": "c.1C>T",
        "uniprot_id": "P1",
        "HAS_PROTEIN_MAPPING": 1,
        "PROTEIN_MAPPING_STATUS": ("mapped_transcript" if eligible else "mapped_reference"),
        "MAPPING_TRANSCRIPT_MATCH": int(eligible),
        "PRIMARY_MAPPING_ELIGIBLE": int(eligible),
        "MAPPING_CONFIDENCE": ("transcript_verified" if eligible else "reference_only_sensitivity"),
        "UNIPROT_REVIEWED": 1,
        schema.LABEL_COL: 1,
    }


def test_output_manifest_is_exact_and_portable_across_output_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "pc_outputs"
    artifact = first_root / "04_feature_engineering" / "table.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"publication-artifact")
    manifest = tmp_path / "run_manifest.json"
    monkeypatch.setattr(config, "OUTPUT_DIR", first_root)
    config.write_run_manifest(
        manifest,
        "04_feature_engineering",
        extra={"upstream_validation": "passed"},
        outputs=[artifact],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact_id = "04_feature_engineering/table.bin"
    assert set(payload["outputs"]) == {artifact_id}
    assert payload["outputs"][artifact_id]["hash_policy"] == "full_sha256"
    assert (
        payload["outputs"][artifact_id]["sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )

    second_root = tmp_path / "kaggle_outputs"
    copied = second_root / artifact_id
    copied.parent.mkdir(parents=True)
    shutil.copyfile(artifact, copied)
    monkeypatch.setattr(config, "OUTPUT_DIR", second_root)
    config.validate_upstream_manifest(
        manifest,
        "04_feature_engineering",
        [copied],
    )
    copied.write_bytes(b"publication-artifacU")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        config.validate_upstream_manifest(
            manifest,
            "04_feature_engineering",
            [copied],
        )


def test_legacy_absolute_output_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs"
    artifact = output_root / "04_feature_engineering" / "table.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x")
    monkeypatch.setattr(config, "OUTPUT_DIR", output_root)
    payload = config.build_run_manifest(
        "04_feature_engineering",
        extra={"upstream_validation": "passed"},
        outputs=[artifact],
    )
    record = next(iter(payload["outputs"].values()))
    payload["outputs"] = {str(artifact.resolve()): record}
    manifest = tmp_path / "legacy.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy absolute"):
        config.validate_upstream_manifest(
            manifest,
            "04_feature_engineering",
            [artifact],
        )


def test_manifest_input_binding_survives_relocation_and_rejects_changed_bytes(
    tmp_path: Path,
) -> None:
    current_input = tmp_path / "current" / "run_manifest.json"
    current_input.parent.mkdir(parents=True)
    current_input.write_bytes(b"authenticated-stage09-manifest")
    producer_manifest = tmp_path / "stage10_manifest.json"
    producer_manifest.write_text(
        json.dumps(
            {
                "stage": "10_extract_esm_features:clinvar",
                "inputs": {
                    r"C:\old_machine\outputs\09_external\run_manifest.json": (
                        config.artifact_record(current_input)
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    payload = config.validate_manifest_input_bindings(
        producer_manifest,
        "10_extract_esm_features:clinvar",
        [current_input],
    )
    assert payload["stage"] == "10_extract_esm_features:clinvar"

    current_input.write_bytes(b"changed-stage09-manifest")
    with pytest.raises(RuntimeError, match="does not cryptographically bind"):
        config.validate_manifest_input_bindings(
            producer_manifest,
            "10_extract_esm_features:clinvar",
            [current_input],
        )


@pytest.mark.parametrize(
    "stage",
    [
        "11_train_and_evaluate",
        "12_external_validation",
        "14_tune_cross_attention",
    ],
)
def test_model_stage_manifests_reject_stale_common_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    required_sources = {
        f"{stage}.py",
        "common.py",
        "config.py",
        "schema.py",
        "table_io.py",
    }
    for name in required_sources:
        (source_dir / name).write_text("VALUE = 1\n", encoding="utf-8")

    output_root = tmp_path / "outputs"
    artifact = output_root / stage / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model-result")
    monkeypatch.setattr(config, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_root)
    manifest = tmp_path / f"{stage}_manifest.json"
    manifest.write_text(
        json.dumps(config.build_run_manifest(stage, outputs=[artifact])),
        encoding="utf-8",
    )

    (source_dir / "common.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="common.py"):
        config.validate_upstream_manifest(manifest, stage, [artifact])


def test_primary_selection_is_post_mapping_and_supports_mane_plus_clinical() -> None:
    frame = pd.DataFrame(
        [
            _mapped_candidate("v1", "mane-ineligible", "ENST1", mane="MANE Select", eligible=False),
            _mapped_candidate("v1", "canonical", "ENST2", canonical="YES"),
            _mapped_candidate("v1", "plus", "ENST3", mane="MANE Plus Clinical"),
            _mapped_candidate("v2", "unmapped", "ENST4", eligible=False),
        ]
    )
    selected, audit = stage04.select_primary_mapped_consequences(frame)
    assert selected[["variant_id", schema.ROW_ID_COL]].values.tolist() == [["v1", "plus"]]
    assert (
        audit.loc[audit["variant_id"].eq("v2"), "TRANSCRIPT_SELECTION_OUTCOME"]
        .eq("ineligible_protein_or_transcript_mapping")
        .all()
    )
    assert selected["CONSEQUENCE_SELECTION_POLICY"].eq(config.TRANSCRIPT_SELECTION_POLICY).all()


def test_primary_selection_requires_clinvar_source_gene_concordance() -> None:
    non_source = _mapped_candidate("v1", "wrong-mane", "ENST1", mane="MANE Select")
    non_source[schema.GENE_COL] = "OVERLAPPING_GENE"
    non_source["CLINVAR_SOURCE_GENE"] = "SOURCE_GENE"
    source = _mapped_candidate("v1", "right-source", "ENST2")
    source[schema.GENE_COL] = "SOURCE_GENE"
    source["CLINVAR_SOURCE_GENE"] = "SOURCE_GENE"
    no_match = _mapped_candidate("v2", "wrong-only", "ENST3")
    no_match[schema.GENE_COL] = "OTHER_GENE"
    no_match["CLINVAR_SOURCE_GENE"] = "MISSING_SOURCE_GENE"

    selected, audit = stage04.select_primary_mapped_consequences(
        pd.DataFrame([non_source, source, no_match])
    )
    assert selected[["variant_id", schema.ROW_ID_COL]].values.tolist() == [["v1", "right-source"]]
    rejected = audit[audit["variant_id"].eq("v2")]
    assert (
        rejected["TRANSCRIPT_SELECTION_OUTCOME"].eq("ineligible_clinvar_source_gene_mismatch").all()
    )
    assert rejected["TRANSCRIPT_MAPPED_CANDIDATE_COUNT"].eq(1).all()
    assert rejected["SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT"].eq(0).all()


def test_variant_candidate_stream_never_splits_a_variant_across_chunks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "variant_id": ["v1", "v1", "v1", "v2", "v3", "v3"],
            schema.ROW_ID_COL: [f"r{index}" for index in range(6)],
        }
    ).to_csv(source, index=False)
    groups = list(stage04._iter_complete_variant_chunks(source, 2, None))
    assert [group["variant_id"].tolist() for group in groups] == [
        ["v1", "v1", "v1"],
        ["v2"],
        ["v3", "v3"],
    ]


def test_stage08_never_silently_accepts_missing_mapping_contract() -> None:
    frame = pd.DataFrame(
        {
            schema.ROW_ID_COL: ["r1"],
            schema.GENE_COL: ["G"],
            schema.LABEL_COL: [1],
            "aa_pos": [1],
            "aa_ref": ["A"],
            "aa_alt": ["V"],
            "protein_sequence": ["A"],
        }
    )
    with pytest.raises(KeyError, match="Transcript-mapped ESM preparation misses"):
        stage08._prepare_chunk(frame, require_transcript_mapping=True)
    ready, _ = stage08._prepare_chunk(frame, require_transcript_mapping=False)
    assert len(ready) == 1

    mapped = pd.DataFrame(
        [
            {
                **_mapped_candidate("v1", "r1", "ENST1"),
                "aa_pos": 1,
                "aa_ref": "A",
                "aa_alt": "V",
                "protein_sequence": "A",
                "PRIMARY_CONSEQUENCE_RANK": 1,
                "PRIMARY_CONSEQUENCE_SELECTED": True,
                "TRANSCRIPT_CANDIDATE_COUNT": 1,
                "TRANSCRIPT_MAPPED_CANDIDATE_COUNT": 1,
                "SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT": 1,
                "CLINVAR_SOURCE_GENE_MATCH": 1,
                "CONSEQUENCE_SELECTION_POLICY": "stale_policy",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="selection policy differs"):
        stage08._prepare_chunk(mapped, require_transcript_mapping=True)


def test_mmseqs_provenance_binds_sequence_set_policy_and_tsv(
    tmp_path: Path,
) -> None:
    sequence = "ACD"
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
    sequences = pd.DataFrame({"sequence_hash": [sequence_hash], "protein_sequence": [sequence]})
    sequence_set_hash = stage08b._sequence_set_sha256(sequences)
    cluster = tmp_path / "cluster.tsv"
    cluster.write_text(f"{sequence_hash}\t{sequence_hash}\n", encoding="utf-8")
    provenance = stage08b._provenance_payload(
        cluster,
        sequence_set_hash,
        1,
        config.HOMOLOGY_MIN_SEQUENCE_IDENTITY,
        config.HOMOLOGY_MIN_COVERAGE,
        "test-mmseqs",
    )
    provenance_file = tmp_path / "cluster.provenance.json"
    provenance_file.write_text(json.dumps(provenance), encoding="utf-8")
    stage08b._validate_cluster_provenance(
        provenance_file,
        cluster,
        sequence_set_hash,
        1,
        config.HOMOLOGY_MIN_SEQUENCE_IDENTITY,
        config.HOMOLOGY_MIN_COVERAGE,
    )
    cluster.write_text(f"{sequence_hash}\t{sequence_hash}\n#changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance does not match"):
        stage08b._validate_cluster_provenance(
            provenance_file,
            cluster,
            sequence_set_hash,
            1,
            config.HOMOLOGY_MIN_SEQUENCE_IDENTITY,
            config.HOMOLOGY_MIN_COVERAGE,
        )


def test_stage10_internal_never_falls_back_to_gene_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stage10, "STAGE08_OUT", tmp_path)
    tasks = stage10._task_definitions()
    assert tasks["internal"].input_table == tmp_path / "internal_esm_ready_homology.parquet"
    with pytest.raises(FileNotFoundError):
        stage10._validate_task_upstream(tasks["internal"])


def test_modern_contextual_scores_are_routed_as_forbidden_predictors() -> None:
    modern = set(schema.RAW_CONTEXTUAL_PREDICTOR_COLS)
    assert modern <= set(schema.PREDICTOR_COLS)
    assert modern <= set(stage01.DBNSFP_COLS)
    assert modern - {"CADD_phred"} <= set(stage01.TRANSCRIPT_COLUMNS)
    assert modern <= set(stage09.RAW_CONTEXTUAL_PREDICTOR_COLS)
    assert modern.isdisjoint(schema.MODEL_FEATURE_ALLOWLIST)


def test_stage10_reads_normalized_external_sequence_table(tmp_path: Path) -> None:
    prepared = tmp_path / "dms_esm_ready.csv"
    sequence_table = tmp_path / "dms_sequences.parquet"
    pd.DataFrame(
        {
            schema.ROW_ID_COL: ["row-1", "row-2"],
            "sequence_hash": ["hash-a", "hash-a"],
            "aa_pos": [1, 2],
            "aa_ref": ["A", "C"],
            "aa_alt": ["V", "D"],
        }
    ).to_csv(prepared, index=False)
    pd.DataFrame({"sequence_hash": ["hash-a"], "protein_sequence": ["AC"]}).to_parquet(
        sequence_table, index=False
    )
    task = stage10.ExtractionTask(
        "dms",
        prepared,
        tmp_path / "output.parquet",
        tmp_path / "embedding.npy",
        tmp_path / "status.parquet",
        tmp_path / "manifest.json",
        tmp_path / "cache",
        sequence_table,
    )

    frame, sampling = stage10._read_task_frame(task)

    assert frame["protein_sequence"].tolist() == ["AC", "AC"]
    assert sampling["selected_rows"] == 2


def test_clinvar_name_fallback_recovers_only_standard_missense() -> None:
    frame = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "gene_symbol": ["GENE1", "GENE2", "GENE3"],
            "name": [
                "NM_000001.3(GENE1):c.12C>T (p.Arg4Gly)",
                "NM_000002.1(GENE2):c.30C>T (p.Gln10Ter)",
                "genomic variant without a protein consequence",
            ],
        }
    )

    parsed = stage09._parse_clinvar_name_consequences(frame)

    assert parsed["variant_id"].tolist() == ["v1"]
    assert parsed["Ensembl_transcriptid"].tolist() == ["NM_000001.3"]
    assert parsed[["aaref", "aapos", "aaalt"]].iloc[0].tolist() == ["R", 4, "G"]
    assert parsed["CLINVAR_CONSEQUENCE_SOURCE"].eq("clinvar_name_refseq").all()
    assert parsed["TRANSCRIPT_NAMESPACE"].eq("RefSeq").all()


def test_uniprot_parser_authenticates_refseq_transcripts(tmp_path: Path) -> None:
    flat_file = tmp_path / "uniprot.txt"
    flat_file.write_text(
        "ID   TEST_HUMAN Reviewed; 4 AA.\n"
        "AC   P00001;\n"
        "GN   Name=GENE1;\n"
        "DR   RefSeq; NP_000001.1; NM_000001.3.\n"
        "SQ   SEQUENCE   4 AA;\n"
        "     ARGV\n"
        "//\n",
        encoding="utf-8",
    )

    records = stage04.UniProtParser(flat_file, tmp_path / "cache").parse()

    assert records["P00001"].transcript_ids >= {"NP_000001", "NM_000001"}


def test_clinvar_reader_orders_mixed_date_formats_correctly(tmp_path: Path) -> None:
    archive = tmp_path / "variant_summary.tsv"
    rows = pd.DataFrame(
        {
            "Chromosome": ["1", "1"],
            "PositionVCF": [100, 100],
            "ReferenceAlleleVCF": ["A", "A"],
            "AlternateAlleleVCF": ["G", "G"],
            "ClinicalSignificance": ["Pathogenic", "Pathogenic"],
            "ReviewStatus": [
                "reviewed by expert panel",
                "reviewed by expert panel",
            ],
            "Assembly": ["GRCh37", "GRCh37"],
            "LastEvaluated": ["2025-02-01", "Mar 1, 2025"],
            "VariationID": [1, 1],
            "Name": [
                "NM_000001.1(GENE1):c.3A>G (p.Ala1Gly)",
                "NM_000001.1(GENE1):c.3A>G (p.Ala1Gly)",
            ],
            "GeneSymbol": ["GENE1", "GENE1"],
        }
    )
    rows.to_csv(archive, sep="\t", index=False)

    selected = stage09._read_clinvar(archive)

    assert len(selected) == 1
    assert selected["last_evaluated"].iloc[0] == "Mar 1, 2025"
