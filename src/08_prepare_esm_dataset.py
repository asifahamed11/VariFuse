from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    AUDIT_SAMPLE_MAX_ROWS,
    ESM_WINDOW_SIZE,
    LORA_MAX_RESIDUES,
    REQUIRE_TRANSCRIPT_MAPPING,
    STAGE04_OUT,
    STAGE07_OUT,
    STAGE08_OUT,
    TRANSCRIPT_SELECTION_POLICY,
    ensure_directories,
    json_default,
    validate_upstream_manifest,
    write_run_manifest,
)
from table_io import iter_table, table_columns
from schema import (
    GENE_COL,
    LABEL_COL,
    REQUIRED_MAPPING_COLS,
    ROW_ID_COL,
    STANDARD_AA,
    TRANSCRIPT_SELECTION_COLS,
    add_mutation_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage08_esm_dataset")

INPUT_FILE = STAGE07_OUT / "Final_Dataset_Natural_Prevalence.parquet"
SOURCE_SEQUENCE_FILE = STAGE04_OUT / "protein_sequences.parquet"
OUTPUT_FILE = STAGE08_OUT / "internal_esm_ready.parquet"
SEQUENCE_FILE = STAGE08_OUT / "internal_sequences.parquet"
LOSS_REPORT_FILE = STAGE08_OUT / "esm_mapping_loss_sample.csv"
GROUPED_LOSS_FILE = STAGE08_OUT / "esm_mapping_groups.csv"
SUMMARY_FILE = STAGE08_OUT / "esm_mapping_summary.json"
MANIFEST_FILE = STAGE08_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE07_OUT / "run_manifest.json"
SEQUENCE_UPSTREAM_MANIFEST = STAGE04_OUT / "run_manifest.json"
CHUNK_SIZE = 50000
WINDOW_SIZE = min(ESM_WINDOW_SIZE, LORA_MAX_RESIDUES)


def _length_band(length: int) -> str:
    if length <= 300:
        return "001_300"
    if length <= 600:
        return "301_600"
    if length <= WINDOW_SIZE:
        return f"601_{WINDOW_SIZE}"
    if length <= 2000:
        return f"{WINDOW_SIZE + 1}_2000"
    return "over_2000"


@lru_cache(maxsize=65536)
def _sequence_sha256(sequence: str) -> str:
    """sha256 of a protein sequence, memoised per-process.

    CHANGELOG 2026-09 (performance): the same protein sequence recurs many times
    inside one chunk (one row per variant), so hashing it once saves hundreds of
    thousands of redundant sha256 calls per pipeline run.  The cache is bounded
    and holds only short string keys, so it cannot grow without bound.
    """
    return hashlib.sha256(sequence.encode()).hexdigest()


def _prepare_chunk(
    chunk: pd.DataFrame,
    *,
    require_transcript_mapping: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = chunk.rename(
        columns={"aapos": "aa_pos", "aaref": "aa_ref", "aaalt": "aa_alt"}
    ).copy()
    position_numeric = pd.to_numeric(frame["aa_pos"], errors="coerce")
    integral = position_numeric.notna() & position_numeric.mod(1).eq(0)
    positions = position_numeric.fillna(-1).astype(int)
    references = frame["aa_ref"].astype("string").str.upper()
    alternates = frame["aa_alt"].astype("string").str.upper()
    sequences = frame["protein_sequence"].astype("string")
    standard = set(STANDARD_AA)
    valid_amino_acids = (
        references.isin(standard)
        & alternates.isin(standard)
        & references.ne(alternates)
    )
    has_sequence = sequences.notna() & sequences.str.len().gt(0)
    primary_mapping = pd.Series(True, index=frame.index)
    if require_transcript_mapping:
        required_mapping = {
            *REQUIRED_MAPPING_COLS,
            *TRANSCRIPT_SELECTION_COLS,
        }
        missing_mapping = required_mapping - set(frame.columns)
        if missing_mapping:
            raise KeyError(
                "Transcript-mapped ESM preparation misses "
                f"{sorted(missing_mapping)}"
            )
        policy_values = (
            frame["CONSEQUENCE_SELECTION_POLICY"]
            .astype("string")
            .fillna("")
            .str.strip()
        )
        policies = set(policy_values)
        if policies != {TRANSCRIPT_SELECTION_POLICY}:
            raise RuntimeError(
                "Transcript selection policy differs from the configured "
                f"publication contract: {sorted(policies)}"
            )
        selected_flags = pd.to_numeric(
            frame["PRIMARY_CONSEQUENCE_SELECTED"], errors="coerce"
        ).fillna(0)
        selected_ranks = pd.to_numeric(
            frame["PRIMARY_CONSEQUENCE_RANK"], errors="coerce"
        )
        if not selected_flags.eq(1).all() or not selected_ranks.eq(1).all():
            raise RuntimeError(
                "Stage 08 received unselected transcript candidates; Stage 04 must "
                "produce exactly one primary consequence per variant"
            )
        primary_mapping = (
            pd.to_numeric(frame["PRIMARY_MAPPING_ELIGIBLE"], errors="coerce")
            .fillna(0)
            .eq(1)
        )
    lengths = sequences.str.len().fillna(0).astype(int)
    in_bounds = integral & positions.ge(1) & positions.le(lengths)
    observed = np.asarray(
        [
            sequence[position - 1]
            if isinstance(sequence, str) and 1 <= position <= len(sequence)
            else ""
            for sequence, position in zip(sequences.astype(object), positions)
        ],
        dtype=object,
    )
    reference_matches = pd.Series(
        in_bounds.to_numpy()
        & (observed == references.astype(object).to_numpy()),
        index=frame.index,
    )
    reasons = np.full(len(frame), "kept", dtype=object)
    reasons[~integral.to_numpy()] = "invalid_position"
    reasons[(integral & ~valid_amino_acids).to_numpy()] = "invalid_substitution"
    reasons[
        (integral & valid_amino_acids & has_sequence & ~primary_mapping).to_numpy()
    ] = "reference_only_mapping"
    reasons[(integral & valid_amino_acids & ~has_sequence).to_numpy()] = "unmapped_sequence"
    reasons[
        (
            integral
            & valid_amino_acids
            & has_sequence
            & ~in_bounds
        ).to_numpy()
    ] = "position_out_of_bounds"
    reasons[
        (
            integral
            & valid_amino_acids
            & has_sequence
            & in_bounds
            & ~reference_matches
        ).to_numpy()
    ] = "reference_mismatch"
    audit = pd.DataFrame(
        {
            ROW_ID_COL: frame[ROW_ID_COL],
            GENE_COL: frame[GENE_COL],
            LABEL_COL: frame[LABEL_COL],
            "source": frame["EXT_SOURCE"] if "EXT_SOURCE" in frame else "internal",
            "sequence_length": lengths,
            "length_band": [_length_band(length) for length in lengths],
            "mapping_status": frame.get("PROTEIN_MAPPING_STATUS", "unknown"),
            "outcome": reasons,
        }
    )
    keep = reasons == "kept"
    ready = frame.loc[keep].copy()
    ready["aa_pos"] = positions.loc[keep].to_numpy(dtype=int)
    ready["aa_ref"] = references.loc[keep].astype(str).to_numpy()
    ready["aa_alt"] = alternates.loc[keep].astype(str).to_numpy()
    window_sequences: list[str] = []
    window_positions: list[int] = []
    window_starts: list[int] = []
    window_ends: list[int] = []
    sequence_hashes: list[str] = []
    for sequence, position in zip(
        ready["protein_sequence"].astype(str), ready["aa_pos"].astype(int)
    ):
        start_zero = max(0, position - 1 - WINDOW_SIZE // 2)
        end_zero = min(len(sequence), start_zero + WINDOW_SIZE)
        if end_zero == len(sequence):
            start_zero = max(0, end_zero - WINDOW_SIZE)
        window = sequence[start_zero:end_zero]
        window_sequences.append(window)
        window_positions.append(position - start_zero)
        window_starts.append(start_zero + 1)
        window_ends.append(end_zero)
        sequence_hashes.append(_sequence_sha256(sequence))
    ready["sequence_hash"] = sequence_hashes
    ready["mutation_window"] = window_sequences
    ready["window_aa_pos"] = np.asarray(window_positions, dtype=np.int32)
    ready["window_start"] = np.asarray(window_starts, dtype=np.int32)
    ready["window_end"] = np.asarray(window_ends, dtype=np.int32)
    ready["split_group"] = ready[GENE_COL].astype(str)
    ready = add_mutation_features(ready)
    return ready, audit


def prepare_esm_dataset(
    chunksize: int = CHUNK_SIZE, *, validate_upstream: bool = True
) -> None:
    """Prepare frozen and LoRA ESM inputs."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if not SOURCE_SEQUENCE_FILE.exists():
        raise FileNotFoundError(SOURCE_SEQUENCE_FILE)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST,
            "07_dataset_balancing",
            [INPUT_FILE],
        )
        validate_upstream_manifest(
            SEQUENCE_UPSTREAM_MANIFEST,
            "04_feature_engineering",
            [SOURCE_SEQUENCE_FILE],
        )
    ensure_directories(STAGE08_OUT)
    header = table_columns(INPUT_FILE)
    required = {
        ROW_ID_COL,
        "variant_id",
        "CLINVAR_VARIATION_ID",
        GENE_COL,
        "aapos",
        "aaref",
        "aaalt",
        "uniprot_id",
        "Ensembl_transcriptid",
        *REQUIRED_MAPPING_COLS,
        *TRANSCRIPT_SELECTION_COLS,
        LABEL_COL,
    }
    missing = required - set(header)
    if missing:
        raise KeyError(f"Stage 08 input misses {sorted(missing)}")
    sequence_frame = pd.read_parquet(SOURCE_SEQUENCE_FILE)
    sequence_map = dict(
        zip(sequence_frame["uniprot_id"].astype(str), sequence_frame["protein_sequence"])
    )
    output_temporary = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    sequence_temporary = SEQUENCE_FILE.with_suffix(SEQUENCE_FILE.suffix + ".tmp")
    report_temporary = LOSS_REPORT_FILE.with_suffix(LOSS_REPORT_FILE.suffix + ".tmp")
    output_temporary.unlink(missing_ok=True)
    sequence_temporary.unlink(missing_ok=True)
    report_temporary.unlink(missing_ok=True)
    total_input = 0
    total_output = 0
    outcome_counts = Counter()
    class_input = Counter()
    class_output = Counter()
    report_groups = Counter()
    output_writer: pq.ParquetWriter | None = None
    sequence_writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    sequence_schema: pa.Schema | None = None
    seen_sequences: set[str] = set()
    seen_variants: set[str] = set()
    audit_samples: list[pd.DataFrame] = []
    sampled_rows = 0
    try:
        for chunk_number, chunk in enumerate(iter_table(INPUT_FILE, chunksize), 1):
            if chunk.empty:
                continue
            variant_ids = chunk["variant_id"].astype(str)
            duplicated = variant_ids.duplicated().any() or bool(
                set(variant_ids) & seen_variants
            )
            if duplicated:
                raise RuntimeError(
                    "Stage 08 input is not genomic-variant-unique after transcript "
                    "selection"
                )
            seen_variants.update(variant_ids)
            chunk["protein_sequence"] = chunk["uniprot_id"].astype(str).map(sequence_map)
            ready, audit = _prepare_chunk(
                chunk,
                require_transcript_mapping=REQUIRE_TRANSCRIPT_MAPPING,
            )
            total_input += len(chunk)
            total_output += len(ready)
            class_input.update(chunk[LABEL_COL].value_counts().to_dict())
            class_output.update(ready[LABEL_COL].value_counts().to_dict())
            outcome_counts.update(audit["outcome"].value_counts().to_dict())
            grouped = audit.groupby(
                [GENE_COL, LABEL_COL, "source", "length_band", "outcome"],
                dropna=False,
            ).size()
            report_groups.update(grouped.to_dict())
            if not ready.empty:
                sequences = ready[["sequence_hash", "protein_sequence"]].drop_duplicates(
                    "sequence_hash"
                )
                sequences = sequences[
                    ~sequences["sequence_hash"].astype(str).isin(seen_sequences)
                ].copy()
                if not sequences.empty:
                    seen_sequences.update(sequences["sequence_hash"].astype(str))
                    sequence_table = pa.Table.from_pandas(
                        sequences.reset_index(drop=True), preserve_index=False
                    )
                    if sequence_writer is None:
                        sequence_schema = sequence_table.schema
                        sequence_writer = pq.ParquetWriter(
                            sequence_temporary,
                            sequence_schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    else:
                        sequence_table = sequence_table.cast(sequence_schema, safe=False)
                    sequence_writer.write_table(sequence_table)
                compact = ready.drop(
                    columns=[
                        "protein_sequence",
                        "mutation_window",
                        "window_aa_pos",
                        "window_start",
                        "window_end",
                    ],
                    errors="ignore",
                )
                output_table = pa.Table.from_pandas(
                    compact.reset_index(drop=True), preserve_index=False
                )
                if output_writer is None:
                    output_schema = output_table.schema
                    output_writer = pq.ParquetWriter(
                        output_temporary,
                        output_schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                else:
                    output_table = output_table.cast(output_schema, safe=False)
                output_writer.write_table(output_table)
            if sampled_rows < AUDIT_SAMPLE_MAX_ROWS:
                remaining = AUDIT_SAMPLE_MAX_ROWS - sampled_rows
                sample = audit[audit["outcome"].ne("kept")].head(remaining).copy()
                if not sample.empty:
                    audit_samples.append(sample)
                    sampled_rows += len(sample)
            logger.info(
                "Processed chunk %d; ESM-ready rows %d",
                chunk_number,
                total_output,
            )
        if output_writer is None or sequence_writer is None or total_output == 0:
            raise RuntimeError("No rows survived ESM preparation")
        if set(class_output) != {0, 1}:
            raise RuntimeError(f"ESM output classes are invalid: {dict(class_output)}")
        output_writer.close()
        output_writer = None
        sequence_writer.close()
        sequence_writer = None
        audit_sample = (
            pd.concat(audit_samples, ignore_index=True)
            if audit_samples
            else pd.DataFrame(
                columns=[
                    ROW_ID_COL,
                    GENE_COL,
                    LABEL_COL,
                    "source",
                    "sequence_length",
                    "length_band",
                    "mapping_status",
                    "outcome",
                ]
            )
        )
        audit_sample.to_csv(report_temporary, index=False)
        output_temporary.replace(OUTPUT_FILE)
        sequence_temporary.replace(SEQUENCE_FILE)
        report_temporary.replace(LOSS_REPORT_FILE)
    except BaseException:
        if output_writer is not None:
            output_writer.close()
        if sequence_writer is not None:
            sequence_writer.close()
        output_temporary.unlink(missing_ok=True)
        sequence_temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
        raise
    grouped_loss = pd.DataFrame(
        [
            {
                GENE_COL: key[0],
                LABEL_COL: int(key[1]),
                "source": key[2],
                "length_band": key[3],
                "outcome": key[4],
                "count": count,
            }
            for key, count in report_groups.items()
        ]
    )
    if not grouped_loss.empty:
        grouped_loss = grouped_loss.sort_values(
            [GENE_COL, LABEL_COL, "source", "length_band", "outcome"],
            kind="mergesort",
        )
    grouped_temporary = GROUPED_LOSS_FILE.with_suffix(GROUPED_LOSS_FILE.suffix + ".tmp")
    grouped_loss.to_csv(grouped_temporary, index=False)
    grouped_temporary.replace(GROUPED_LOSS_FILE)
    summary = {
        "input_rows": total_input,
        "output_rows": total_output,
        "coverage": total_output / total_input,
        "window_size": WINDOW_SIZE,
        "input_class_counts": dict(class_input),
        "output_class_counts": dict(class_output),
        "outcomes": dict(outcome_counts),
        "storage_format": "parquet_zstd",
        "sequence_table": str(SEQUENCE_FILE),
        "unique_sequences": len(seen_sequences),
        "grouped_loss_file": str(GROUPED_LOSS_FILE),
        "loss_sample_file": str(LOSS_REPORT_FILE),
        "loss_sample_rows": sampled_rows,
        "require_transcript_mapping": REQUIRE_TRANSCRIPT_MAPPING,
        "transcript_selection_policy": TRANSCRIPT_SELECTION_POLICY,
        "variant_unique_input": True,
        "variant_unique_output": True,
        "upstream_validation": (
            "passed" if validate_upstream else "explicitly_skipped_nonpublication"
        ),
    }
    summary_temporary = SUMMARY_FILE.with_suffix(SUMMARY_FILE.suffix + ".tmp")
    summary_temporary.write_text(
        json.dumps(summary, indent=2, default=json_default), encoding="utf-8"
    )
    summary_temporary.replace(SUMMARY_FILE)
    write_run_manifest(
        MANIFEST_FILE,
        "08_prepare_esm_dataset",
        [
            INPUT_FILE,
            UPSTREAM_MANIFEST,
            SOURCE_SEQUENCE_FILE,
            SEQUENCE_UPSTREAM_MANIFEST,
        ],
        summary,
        outputs=[
            OUTPUT_FILE,
            SEQUENCE_FILE,
            LOSS_REPORT_FILE,
            GROUPED_LOSS_FILE,
            SUMMARY_FILE,
        ],
    )
    logger.info("Prepared %d of %d rows for ESM", total_output, total_input)


if __name__ == "__main__":
    prepare_esm_dataset(validate_upstream=True)
