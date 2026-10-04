from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
import config
import schema


stage01 = importlib.import_module("01_dbnsfp_processor")
stage03 = importlib.import_module("03_remove_duplicates")
stage04 = importlib.import_module("04_feature_engineering")
stage08 = importlib.import_module("08_prepare_esm_dataset")
stage09 = importlib.import_module("09_prepare_external_esm_dataset")
stage10 = importlib.import_module("10_extract_esm_features")
stage05 = importlib.import_module("05_remove_leakage")
stage06 = importlib.import_module("06_clean_and_finalize")
stage07 = importlib.import_module("07_dataset_balancing")
stage12 = importlib.import_module("12_external_validation")


def _dbnsfp_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {column: "." for column in stage01.DBNSFP_COLS}
    records = []
    for values in rows:
        record = dict(defaults)
        record.update(values)
        if stage01.DBNSFP_GRCH37_CHROM_COLUMN not in values:
            record[stage01.DBNSFP_GRCH37_CHROM_COLUMN] = values.get("#chr", ".")
        if stage01.DBNSFP_GRCH37_POSITION_COLUMN not in values:
            record[stage01.DBNSFP_GRCH37_POSITION_COLUMN] = values.get("pos(1-based)", ".")
        records.append(record)
    return pd.DataFrame(records)


def _empty_resources() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    cgc = pd.DataFrame(columns=["genename", "ROLE_IN_CANCER", "TIER"])
    roles = {name: set() for name in ("cancer", "oncogene", "tsg", "tier1")}
    return cgc, roles


def test_ba1_assignment_remains_positional_after_filter_and_merge() -> None:
    raw = _dbnsfp_frame(
        [
            {
                "#chr": "1",
                "pos(1-based)": 10,
                "ref": "A",
                "alt": "G",
                "aaref": "A",
                "aaalt": "A",
                "aapos": 1,
                "genename": "BAD",
                "gnomAD4.1_joint_AF": 0.9,
            },
            {
                "#chr": "1",
                "pos(1-based)": 20,
                "ref": "A",
                "alt": "G",
                "aaref": "A",
                "aaalt": "V",
                "aapos": 2,
                "genename": "HIGH",
                "gnomAD4.1_joint_AF": 0.10,
            },
            {
                "#chr": "1",
                "pos(1-based)": 30,
                "ref": "C",
                "alt": "T",
                "aaref": "C",
                "aaalt": "Y",
                "aapos": 3,
                "genename": "LOW",
                "gnomAD4.1_joint_AF": 0.001,
            },
        ]
    )
    cgc, roles = _empty_resources()
    output, _ = stage01._process_chunk(
        raw,
        set(),
        set(),
        set(),
        set(),
        cgc,
        roles,
        pd.DataFrame(),
        set(),
        label_task="legacy_mixed",
    )
    assert output["variant_id"].tolist() == ["GRCh37:1:20:A:G"]
    assert output["HAS_BA1_EVIDENCE"].tolist() == [1]
    assert output["EVIDENCE_SOURCES"].tolist() == ["gnomad_ba1"]
    assert not output["HAS_BA1_EVIDENCE"].isna().any()


def test_training_clinvar_loader_enforces_snapshot_review_and_conflict_rules(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "clinvar.tsv"
    pd.DataFrame(
        {
            "Chromosome": ["1", "1", "2", "3", "4"],
            "PositionVCF": [10, 10, 20, 30, 40],
            "ReferenceAlleleVCF": ["A", "A", "C", "G", "T"],
            "AlternateAlleleVCF": ["G", "G", "T", "A", "C"],
            "VariationID": [10, 10, 20, 30, 40],
            "GeneSymbol": ["G1", "G1", "G2", "G3", "G4"],
            "Name": ["n1", "n1", "n2", "n3", "n4"],
            "Assembly": ["GRCh37"] * 5,
            "ReviewStatus": [
                "reviewed by expert panel",
                "reviewed by expert panel",
                "criteria provided, multiple submitters, no conflicts",
                "criteria provided, single submitter",
                "practice guideline",
            ],
            "ClinicalSignificance": [
                "Pathogenic",
                "Benign",
                "Benign",
                "Pathogenic",
                "Pathogenic",
            ],
        }
    ).to_csv(archive, sep="\t", index=False)
    monkeypatch.setattr(stage01, "CLINVAR_TRAIN_ARCHIVE", archive)
    pathogenic, benign = stage01._load_clinvar_training_labels()
    assert pathogenic == {"GRCh37:4:40:T:C"}
    assert benign == {"GRCh37:2:20:C:T"}


def test_cosmic_support_requires_exact_amino_acid_change() -> None:
    raw = _dbnsfp_frame(
        [
            {
                "#chr": "12",
                "pos(1-based)": 100,
                "ref": "A",
                "alt": "C",
                "aaref": "G",
                "aaalt": "V",
                "aapos": 12,
                "genename": "KRAS",
                "gnomAD4.1_joint_AF": 0.0,
            },
            {
                "#chr": "12",
                "pos(1-based)": 101,
                "ref": "A",
                "alt": "G",
                "aaref": "G",
                "aaalt": "D",
                "aapos": 12,
                "genename": "KRAS",
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
            "COSMIC_RECURRENCE": [10],
            "COSMIC_FREQUENCY": [0.1],
        }
    )
    cgc, roles = _empty_resources()
    output, _ = stage01._process_chunk(
        raw,
        set(),
        set(),
        set(),
        set(),
        cgc,
        roles,
        cosmic,
        set(),
        label_task="legacy_mixed",
    )
    assert output["aaalt"].tolist() == ["V"]
    assert output["HAS_HOTSPOT_SUPPORT"].tolist() == [1]


def test_stage03_preserves_transcript_candidates_for_post_mapping_selection(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "input.csv"
    frame = pd.DataFrame(
        [
            {
                "variant_id": "v1",
                schema.ROW_ID_COL: "row-canonical",
                schema.LABEL_COL: 0,
                "Ensembl_transcriptid": "ENST2",
                "HGVSp_snpEff": "p.A1V",
                "VEP_canonical": "YES",
                "MANE": ".",
            },
            {
                "variant_id": "v1",
                schema.ROW_ID_COL: "row-mane",
                schema.LABEL_COL: 0,
                "Ensembl_transcriptid": "ENST1",
                "HGVSp_snpEff": "p.A1V",
                "VEP_canonical": ".",
                "MANE": "Select",
            },
            {
                "variant_id": "v2",
                schema.ROW_ID_COL: "row-v2",
                schema.LABEL_COL: 1,
                "Ensembl_transcriptid": "ENST3",
                "HGVSp_snpEff": "p.C2Y",
                "VEP_canonical": "YES",
                "MANE": ".",
            },
        ]
    )
    frame.to_csv(source, index=False)
    monkeypatch.setattr(stage03, "INPUT_FILE", source)
    monkeypatch.setattr(stage03, "OUTPUT_FILE", tmp_path / "output.csv")
    monkeypatch.setattr(stage03, "CONFLICT_FILE", tmp_path / "conflicts.csv")
    monkeypatch.setattr(stage03, "MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr(stage03, "DATABASE_FILE", tmp_path / "dedup.sqlite")
    stage03.remove_duplicates(chunksize=2, validate_upstream=False)
    result = pd.read_csv(stage03.OUTPUT_FILE)
    assert len(result) == 3
    assert set(result.loc[result["variant_id"].eq("v1"), schema.ROW_ID_COL]) == {
        "row-canonical",
        "row-mane",
    }
    manifest = json.loads(stage03.MANIFEST_FILE.read_text(encoding="utf-8"))
    assert manifest["extra"]["tie_breaking"][:2] == [
        "mane_select",
        "mane_plus_clinical",
    ]


def test_alphafold_index_keeps_all_fragments(tmp_path: Path) -> None:
    for fragment in (1, 2, 14):
        (tmp_path / f"AF-P12345-F{fragment}-model_v6.pdb").write_text("HEADER\n", encoding="utf-8")
    index = stage04.AlphaFoldStructureIndex(tmp_path, tmp_path / "cache")
    fragments = index.get_fragments("P12345")
    assert [item.fragment for item in fragments] == [1, 2, 14]
    assert [item.offset for item in fragments] == [0, 200, 2600]


def test_stage08_writes_compact_parquet_and_sequence_table(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "stage7.parquet"
    sequences = tmp_path / "protein_sequences.parquet"
    pd.DataFrame(
        {
            schema.ROW_ID_COL: ["r0", "r1"],
            "variant_id": ["v0", "v1"],
            "CLINVAR_VARIATION_ID": ["100", "101"],
            schema.GENE_COL: ["G0", "G1"],
            "aapos": [2, 2],
            "aaref": ["C", "K"],
            "aaalt": ["A", "A"],
            "uniprot_id": ["P0", "P1"],
            "Ensembl_transcriptid": ["ENST0", "ENST1"],
            "PROTEIN_MAPPING_STATUS": ["mapped_transcript", "mapped_transcript"],
            "MAPPING_TRANSCRIPT_MATCH": [1, 1],
            "PRIMARY_MAPPING_ELIGIBLE": [1, 1],
            "MAPPING_CONFIDENCE": ["transcript_verified", "transcript_verified"],
            "PRIMARY_CONSEQUENCE_RANK": [1, 1],
            "PRIMARY_CONSEQUENCE_SELECTED": [True, True],
            "TRANSCRIPT_CANDIDATE_COUNT": [1, 1],
            "TRANSCRIPT_MAPPED_CANDIDATE_COUNT": [1, 1],
            "SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT": [1, 1],
            "CLINVAR_SOURCE_GENE_MATCH": [1, 1],
            "CONSEQUENCE_SELECTION_POLICY": [
                config.TRANSCRIPT_SELECTION_POLICY,
                config.TRANSCRIPT_SELECTION_POLICY,
            ],
            schema.LABEL_COL: [0, 1],
        }
    ).to_parquet(source, index=False)
    pd.DataFrame(
        {
            "uniprot_id": ["P0", "P1"],
            "protein_sequence": ["ACD", "MKT"],
        }
    ).to_parquet(sequences, index=False)
    monkeypatch.setattr(stage08, "INPUT_FILE", source)
    monkeypatch.setattr(stage08, "SOURCE_SEQUENCE_FILE", sequences)
    monkeypatch.setattr(stage08, "OUTPUT_FILE", tmp_path / "ready.parquet")
    monkeypatch.setattr(stage08, "SEQUENCE_FILE", tmp_path / "sequences.parquet")
    monkeypatch.setattr(stage08, "LOSS_REPORT_FILE", tmp_path / "loss.csv")
    monkeypatch.setattr(stage08, "GROUPED_LOSS_FILE", tmp_path / "groups.csv")
    monkeypatch.setattr(stage08, "SUMMARY_FILE", tmp_path / "summary.json")
    monkeypatch.setattr(stage08, "MANIFEST_FILE", tmp_path / "manifest.json")
    stage08.prepare_esm_dataset(chunksize=1, validate_upstream=False)
    ready = pd.read_parquet(stage08.OUTPUT_FILE)
    sequence_table = pd.read_parquet(stage08.SEQUENCE_FILE)
    assert len(ready) == 2
    assert "protein_sequence" not in ready
    assert set(sequence_table["protein_sequence"]) == {"ACD", "MKT"}
    assert json.loads(stage08.SUMMARY_FILE.read_text())["storage_format"] == "parquet_zstd"


def test_internal_sampling_is_deterministic_stratified_and_gene_diverse(
    tmp_path: Path,
) -> None:
    path = tmp_path / "internal.parquet"
    rows = 1000
    frame = pd.DataFrame(
        {
            schema.ROW_ID_COL: [f"row-{index}" for index in range(rows)],
            schema.LABEL_COL: [0] * 950 + [1] * 50,
            schema.GENE_COL: [f"G{index % 100}" for index in range(rows)],
        }
    )
    frame.to_parquet(path, index=False, row_group_size=73)
    first, first_info = stage10._select_internal_rows(path, 100)
    second, second_info = stage10._select_internal_rows(path, 100)
    assert np.array_equal(first, second)
    assert first_info == second_info
    selected = frame.iloc[first]
    assert len(selected) == 100
    assert selected[schema.LABEL_COL].value_counts().to_dict() == {0: 95, 1: 5}
    assert selected[schema.GENE_COL].nunique() > 50


def test_stage10_constructs_cuda_model_directly_in_fp16(monkeypatch) -> None:
    calls: list[object] = []

    class FakeAlphabet:
        prepend_bos = True

    class FakeModel:
        def eval(self):
            calls.append("eval")
            return self

        def half(self):
            calls.append("half")
            return self

        def to(self, device):
            calls.append(("to", device))
            return self

    class FakePretrained:
        @staticmethod
        def load_model_and_alphabet(model_name):
            calls.append(("load", model_name, torch.get_default_dtype()))
            return FakeModel(), FakeAlphabet()

    class FakeEsm:
        pretrained = FakePretrained()

    monkeypatch.setattr(stage10, "DEVICE", "cuda")
    monkeypatch.setattr(stage10, "ESM_USE_FP16", True)
    monkeypatch.setattr(stage10, "data_parallel", lambda model, device: model)
    original_dtype = torch.get_default_dtype()
    model, alphabet = stage10._load_esm_model(FakeEsm())
    assert isinstance(model, FakeModel)
    assert isinstance(alphabet, FakeAlphabet)
    assert calls == [
        ("load", stage10.ESM_MODEL_NAME, torch.float16),
        "eval",
        "half",
        ("to", "cuda"),
    ]
    assert torch.get_default_dtype() == original_dtype


def test_stage10_masked_results_do_not_retain_full_gpu_outputs(monkeypatch) -> None:
    class FakeModel:
        def __call__(self, tokens, repr_layers):
            batch = tokens.shape[0]
            length = tokens.shape[1]
            return {
                "logits": torch.randn(batch, length, 33),
                "representations": {stage10.ESM_LAYER: torch.randn(batch, length, 1280)},
            }

    def batch_converter(_):
        return None, None, torch.ones((1, 9), dtype=torch.long)

    monkeypatch.setattr(stage10, "DEVICE", "cpu")
    monkeypatch.setattr(stage10, "ESM_USE_FP16", False)
    item = stage10.ContextItem("ACDEFGH", 1, 7, [])
    results = stage10._masked_log_probabilities(item, [2, 5], FakeModel(), batch_converter, 32)
    for probabilities, embedding in results.values():
        assert probabilities.device.type == "cpu"
        assert embedding.device.type == "cpu"
        assert embedding.shape == (1280,)
        assert embedding.untyped_storage().nbytes() == (
            embedding.numel() * embedding.element_size()
        )


def test_stage10_masked_marginal_skips_redundant_unmasked_forward(
    monkeypatch,
) -> None:
    item = stage10.ContextItem("ACD", 1, 3, [(0, 2, "C", "D")])
    embedding = torch.arange(stage10.ESM_EMBED_DIM, dtype=torch.float32)
    log_probabilities = torch.zeros(32, dtype=torch.float32)
    log_probabilities[5] = 2.5
    log_probabilities[4] = -0.5

    def unexpected_unmasked_forward(*args, **kwargs):
        raise AssertionError("masked-marginal must not run the unmasked forward")

    monkeypatch.setattr(stage10, "_forward_batch", unexpected_unmasked_forward)
    monkeypatch.setattr(
        stage10,
        "_masked_log_probabilities",
        lambda *args, **kwargs: {2: (log_probabilities, embedding)},
    )

    class FakeAlphabet:
        mask_idx = 31

        @staticmethod
        def get_idx(amino_acid: str) -> int:
            return {"C": 4, "D": 5}[amino_acid]

    embeddings = np.full((1, stage10.ESM_EMBED_DIM), np.nan, dtype=np.float32)
    wt_scores = np.full(1, np.nan, dtype=np.float32)
    masked_scores = np.full(1, np.nan, dtype=np.float32)
    statuses = pd.DataFrame(
        {"extraction_status": ["pending"], "error_type": [""]}
    )

    stage10._process_batch(
        [item],
        object(),
        FakeAlphabet(),
        object(),
        FakeAlphabet.mask_idx,
        "masked-marginal",
        embeddings,
        wt_scores,
        masked_scores,
        statuses,
    )

    np.testing.assert_array_equal(embeddings[0], embedding.numpy())
    assert np.isnan(wt_scores[0])
    assert masked_scores[0] == 3.0
    assert statuses.loc[0, "extraction_status"] == "success"


def test_stage10_batches_masked_positions_across_contexts(monkeypatch) -> None:
    items = [
        stage10.ContextItem("ACD", 1, 3, [(0, 2, "C", "D")]),
        stage10.ContextItem("EFG", 1, 3, [(1, 2, "F", "G")]),
    ]
    observed_batch_sizes: list[int] = []

    class FakeModel:
        def __call__(self, tokens, repr_layers):
            observed_batch_sizes.append(len(tokens))
            batch, length = tokens.shape
            return {
                "logits": torch.arange(
                    batch * length * 32, dtype=torch.float32
                ).reshape(batch, length, 32),
                "representations": {
                    stage10.ESM_LAYER: torch.ones(
                        batch, length, stage10.ESM_EMBED_DIM
                    )
                },
            }

    def batch_converter(data):
        return None, None, torch.ones((len(data), 5), dtype=torch.long)

    class FakeAlphabet:
        mask_idx = 31

        @staticmethod
        def get_idx(amino_acid: str) -> int:
            return {"C": 4, "D": 5, "F": 6, "G": 7}[amino_acid]

    monkeypatch.setattr(stage10, "DEVICE", "cpu")
    monkeypatch.setattr(stage10, "ESM_USE_FP16", False)
    embeddings = np.full((2, stage10.ESM_EMBED_DIM), np.nan, dtype=np.float32)
    wt_scores = np.full(2, np.nan, dtype=np.float32)
    masked_scores = np.full(2, np.nan, dtype=np.float32)
    statuses = pd.DataFrame(
        {"extraction_status": ["pending", "pending"], "error_type": ["", ""]}
    )

    stage10._process_batch(
        items,
        FakeModel(),
        FakeAlphabet(),
        batch_converter,
        FakeAlphabet.mask_idx,
        "masked-marginal",
        embeddings,
        wt_scores,
        masked_scores,
        statuses,
    )

    assert observed_batch_sizes == [2]
    assert np.isfinite(embeddings).all()
    assert np.isfinite(masked_scores).all()
    assert statuses["extraction_status"].eq("success").all()


def test_dms_uses_proteingym_target_sequence() -> None:
    raw = pd.DataFrame(
        {"mutant": ["A1V", "C2Y"], "DMS_score": [-1.0, 1.0], "DMS_score_bin": [0, 1]}
    )
    metadata = pd.Series({"UniProt_ID": "PTEST_HUMAN", "target_seq": "ACDE", "target_gene": pd.NA})
    result = stage09._normalize_dms_frame(raw, "assay", metadata, {"PTEST": "GENE"})
    assert result["protein_sequence"].eq("ACDE").all()
    assert result["uniprot_id"].eq("PTEST").all()
    assert result[schema.GENE_COL].eq("GENE").all()
    assert set(result[schema.LABEL_COL]) == {0, 1}


def test_dms_accepts_split_mutation_columns_and_requires_assay_cutoff() -> None:
    raw = pd.DataFrame(
        {
            "genename": ["TP53", "TP53"],
            "aapos": [1, 2],
            "aaref": ["A", "C"],
            "aaalt": ["V", "Y"],
            "dms_score": [0.1, 0.9],
        }
    )
    metadata = pd.Series(
        {
            "UniProt_ID": "PTEST_HUMAN",
            "target_seq": "ACDE",
            "DMS_binarization_cutoff": 0.5,
        }
    )
    result = stage09._normalize_dms_frame(raw, "split", metadata, {})
    assert result[["aaref", "aapos", "aaalt"]].values.tolist() == [
        ["A", 1, "V"],
        ["C", 2, "Y"],
    ]
    assert result[schema.LABEL_COL].tolist() == [1, 0]
    assert stage09._normalize_dms_frame(raw, "legacy", None, {}).empty


def test_parquet_contract_across_stages_05_to_07(tmp_path: Path, monkeypatch) -> None:
    stage4 = tmp_path / "stage4.parquet"
    rows = pd.DataFrame(
        {
            schema.ROW_ID_COL: ["r0", "r1"],
            "variant_id": ["v0", "v1"],
            "transcript_variant_id": ["t0", "t1"],
            "CLINVAR_VARIATION_ID": ["100", "101"],
            "chr": ["1", "2"],
            "pos": [10, 20],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            schema.GENE_COL: ["G0", "G1"],
            "Ensembl_transcriptid": ["ENST0", "ENST1"],
            "HGVSp_snpEff": ["p.A1V", "p.C2Y"],
            "HGVSc_snpEff": ["c.1A>G", "c.2C>T"],
            "aapos": [1, 2],
            "aaref": ["A", "C"],
            "aaalt": ["V", "Y"],
            "uniprot_id": ["P0", "P1"],
            "PROTEIN_MAPPING_STATUS": ["mapped_transcript", "mapped_transcript"],
            "MAPPING_TRANSCRIPT_MATCH": [1, 1],
            "PRIMARY_MAPPING_ELIGIBLE": [1, 1],
            "MAPPING_CONFIDENCE": ["transcript_verified", "transcript_verified"],
            "PRIMARY_CONSEQUENCE_RANK": [1, 1],
            "PRIMARY_CONSEQUENCE_SELECTED": [True, True],
            "TRANSCRIPT_CANDIDATE_COUNT": [1, 1],
            "TRANSCRIPT_MAPPED_CANDIDATE_COUNT": [1, 1],
            "SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT": [1, 1],
            "CLINVAR_SOURCE_GENE_MATCH": [1, 1],
            "CONSEQUENCE_SELECTION_POLICY": [
                config.TRANSCRIPT_SELECTION_POLICY,
                config.TRANSCRIPT_SELECTION_POLICY,
            ],
            schema.LABEL_COL: [0, 1],
            "GERP++_RS": [np.nan, 3.0],
            "REVEL_score": [0.1, 0.9],
        }
    )
    rows.to_parquet(stage4, index=False)
    monkeypatch.setattr(stage05, "INPUT_FILE", stage4)
    monkeypatch.setattr(stage05, "OUTPUT_FILE", tmp_path / "stage5.parquet")
    monkeypatch.setattr(stage05, "SIDECAR_FILE", tmp_path / "sidecar.parquet")
    monkeypatch.setattr(stage05, "MANIFEST_FILE", tmp_path / "stage5.json")
    stage05.remove_leakage(chunksize=1, validate_upstream=False)
    assert "REVEL_score" not in pd.read_parquet(stage05.OUTPUT_FILE).columns

    monkeypatch.setattr(stage06, "INPUT_FILE", stage05.OUTPUT_FILE)
    monkeypatch.setattr(stage06, "OUTPUT_FILE", tmp_path / "stage6.parquet")
    monkeypatch.setattr(stage06, "REPORT_FILE", tmp_path / "clean.json")
    monkeypatch.setattr(stage06, "MANIFEST_FILE", tmp_path / "stage6.json")
    stage06.clean_dataset(chunksize=1, validate_upstream=False)
    cleaned = pd.read_parquet(stage06.OUTPUT_FILE)
    assert cleaned["GERP++_RS__missing"].tolist() == [1, 0]

    monkeypatch.setattr(stage07, "INPUT_FILE", stage06.OUTPUT_FILE)
    monkeypatch.setattr(stage07, "OUTPUT_FILE", tmp_path / "stage7.parquet")
    monkeypatch.setattr(stage07, "DISTRIBUTION_FILE", tmp_path / "distribution.csv")
    monkeypatch.setattr(stage07, "GENE_DISTRIBUTION_FILE", tmp_path / "genes.csv")
    monkeypatch.setattr(stage07, "MANIFEST_FILE", tmp_path / "stage7.json")
    stage07.preserve_natural_prevalence(chunksize=1, validate_upstream=False)
    assert len(pd.read_parquet(stage07.OUTPUT_FILE)) == 2


def test_stage04_streaming_identifiers_survive_numeric_to_sex_chromosome_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stage03.csv"
    pd.DataFrame(
        {
            "variant_id": ["1:10:A:G", "1:10:A:G", "X:20:C:T", "X:20:C:T"],
            "chr": [1, 1, "X", "X"],
            "CLINVAR_VARIATION_ID": [100, 100, 200, 200],
            "candidate": [0, 1, 0, 1],
        }
    ).to_csv(source, index=False)

    chunks = list(stage04._iter_complete_variant_chunks(source, chunksize=2, max_rows=None))
    assert chunks
    assert all(str(chunk["chr"].dtype) == "string" for chunk in chunks)
    assert all(
        str(chunk["CLINVAR_VARIATION_ID"].dtype) == "string" for chunk in chunks
    )

    output = tmp_path / "stage04.parquet"
    with stage04.AtomicParquetWriter(output) as writer:
        for chunk in chunks:
            writer.write(chunk)
    restored = pd.read_parquet(output)
    assert restored["chr"].tolist() == ["1", "1", "X", "X"]
    assert restored["CLINVAR_VARIATION_ID"].tolist() == ["100", "100", "200", "200"]


def test_external_deoverlap_uses_full_internal_universe(tmp_path: Path) -> None:
    internal = tmp_path / "stage7.parquet"
    pd.DataFrame(
        {
            "chr": ["1", "2"],
            "pos": [10, 20],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "CLINVAR_VARIATION_ID": ["100", "101"],
            schema.GENE_COL: ["SHARED", "OTHER"],
            "uniprot_id": ["P0", "P1"],
            "aapos": [1, 2],
            "aaref": ["A", "C"],
            "aaalt": ["V", "Y"],
        }
    ).to_parquet(internal, index=False)
    external = pd.DataFrame(
        {
            "chr": ["1", "3", "4"],
            "pos": [10, 30, 40],
            "ref": ["A", "G", "T"],
            "alt": ["G", "A", "C"],
            schema.GENE_COL: ["SHARED", "SHARED", "NEW"],
            "uniprot_id": ["PX", "PY", "PZ"],
            "aa_pos": [9, 9, 9],
            "aa_ref": ["A", "A", "A"],
            "aa_alt": ["V", "V", "V"],
        }
    )
    masks = stage12._deoverlap_masks(internal, external)
    assert masks["exact_variant_disjoint"].tolist() == [False, True, True]
    assert masks["gene_disjoint"].tolist() == [False, False, True]


def test_shared_tabular_schema_and_float32_preprocessing() -> None:
    features = ["GERP++_RS", "esm_variant_score", "esm_variant_score__missing"]
    assert schema.select_tabular_features(features) == ["GERP++_RS"]
    values = np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    original = values.copy()
    processor = common.ArrayPreprocessor.fit(values)
    transformed = processor.transform(values)
    assert np.allclose(values, original, equal_nan=True)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
