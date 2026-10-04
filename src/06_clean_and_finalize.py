from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

from config import (
    STAGE05_OUT,
    STAGE06_OUT,
    ensure_directories,
    json_default,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import (
    BASE_FEATURE_ALLOWLIST,
    GENE_COL,
    LABEL_COL,
    REQUIRED_MAPPING_COLS,
    ROW_ID_COL,
    STANDARD_AA,
    TRANSCRIPT_SELECTION_COLS,
)
from table_io import AtomicParquetWriter, iter_table, table_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage06_clean")

INPUT_FILE = STAGE05_OUT / "somatic_variant_predictor_free.parquet"
OUTPUT_FILE = STAGE06_OUT / "somatic_variant_cleaned.parquet"
REPORT_FILE = STAGE06_OUT / "cleaning_report.json"
MANIFEST_FILE = STAGE06_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE05_OUT / "run_manifest.json"
CHUNK_SIZE = 50000

SENTINELS = {
    "SASA": {-1.0},
    "RELATIVE_SASA": {-1.0},
    "PLDDT_SCORE": {-1.0},
    "DISTANCE_TO_ACTIVE_SITE": {-1.0, 999.0},
}


def _validate_mutations(chunk: pd.DataFrame, chunk_number: int) -> None:
    reference = chunk["aaref"].astype("string").str.upper()
    alternate = chunk["aaalt"].astype("string").str.upper()
    position = pd.to_numeric(chunk["aapos"], errors="coerce")
    standard = set(STANDARD_AA)
    invalid = (
        ~reference.isin(standard)
        | ~alternate.isin(standard)
        | reference.eq(alternate)
        | position.isna()
        | position.lt(1)
        | position.mod(1).ne(0)
    )
    if invalid.any():
        raise ValueError(
            f"Chunk {chunk_number} has {int(invalid.sum())} invalid missense rows"
        )


def clean_dataset(
    chunksize: int = CHUNK_SIZE, *, validate_upstream: bool = True
) -> None:
    """Validate and finalize the modelling dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST,
            "05_remove_leakage",
            [INPUT_FILE],
        )
    ensure_directories(STAGE06_OUT)
    header = table_columns(INPUT_FILE)
    required = {
        ROW_ID_COL,
        "variant_id",
        "CLINVAR_VARIATION_ID",
        GENE_COL,
        "aapos",
        "aaref",
        "aaalt",
        "Ensembl_transcriptid",
        *REQUIRED_MAPPING_COLS,
        *TRANSCRIPT_SELECTION_COLS,
        LABEL_COL,
    }
    missing = required - set(header)
    if missing:
        raise KeyError(f"Stage 06 input misses {sorted(missing)}")
    indicator_bases = [
        column
        for column in BASE_FEATURE_ALLOWLIST
        if column in header and not column.endswith("__missing")
    ]
    output_columns = [
        *header,
        *[
            f"{column}__missing"
            for column in indicator_bases
            if f"{column}__missing" not in header
        ],
    ]
    id_database = STAGE06_OUT / "row_ids.sqlite.tmp"
    id_database.unlink(missing_ok=True)
    id_connection = sqlite3.connect(id_database)
    id_connection.execute("CREATE TABLE row_ids (row_id TEXT PRIMARY KEY)")
    total = 0
    class_counts = Counter()
    source_counts = Counter()
    gene_counts = Counter()
    sentinel_counts = Counter()
    nonfinite_counts = Counter()
    structure_counts = Counter()
    try:
        with AtomicParquetWriter(OUTPUT_FILE) as output_writer:
            for chunk_number, chunk in enumerate(iter_table(INPUT_FILE, chunksize), 1):
                if chunk[ROW_ID_COL].isna().any():
                    raise ValueError(f"Chunk {chunk_number} has missing row identifiers")
                identifiers = chunk[ROW_ID_COL].astype(str)
                before = id_connection.total_changes
                id_connection.executemany(
                    "INSERT OR IGNORE INTO row_ids(row_id) VALUES (?)",
                    ((value,) for value in identifiers),
                )
                inserted = id_connection.total_changes - before
                if inserted != len(identifiers):
                    duplicate_count = len(identifiers) - inserted
                    raise ValueError(
                        f"Chunk {chunk_number} has {duplicate_count} duplicate row identifiers"
                    )
                if chunk[GENE_COL].isna().any() or chunk[GENE_COL].astype(str).str.strip().eq("").any():
                    raise ValueError(f"Chunk {chunk_number} has missing gene groups")
                invalid_labels = ~chunk[LABEL_COL].isin([0, 1])
                if invalid_labels.any():
                    raise ValueError(
                        f"Chunk {chunk_number} has {int(invalid_labels.sum())} invalid labels"
                    )
                _validate_mutations(chunk, chunk_number)
                for column, sentinels in SENTINELS.items():
                    if column in chunk:
                        numeric = pd.to_numeric(chunk[column], errors="coerce")
                        mask = numeric.isin(sentinels)
                        sentinel_counts[column] += int(mask.sum())
                        chunk[column] = numeric.mask(mask)
                for column in indicator_bases:
                    numeric = pd.to_numeric(chunk[column], errors="coerce")
                    infinite = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
                    nonfinite_counts[column] += int(infinite.sum())
                    if infinite.any():
                        numeric.iloc[np.flatnonzero(infinite)] = np.nan
                    chunk[column] = numeric
                    chunk[f"{column}__missing"] = numeric.isna().astype("int8")
                if {"HAS_STRUCTURE", "SASA", "PLDDT_SCORE"} <= set(chunk.columns):
                    chunk["HAS_STRUCTURE"] = chunk["SASA"].notna().astype("int8")
                    chunk["LOW_CONFIDENCE_STRUCTURE"] = (
                        chunk["HAS_STRUCTURE"].eq(1) & chunk["PLDDT_SCORE"].lt(70)
                    ).astype("int8")
                    structure_counts.update(chunk["HAS_STRUCTURE"].value_counts().to_dict())
                chunk = chunk.reindex(columns=output_columns)
                output_writer.write(chunk)
                total += len(chunk)
                class_counts.update(chunk[LABEL_COL].value_counts().to_dict())
                gene_counts.update(chunk[GENE_COL].value_counts().to_dict())
                if "EXT_SOURCE" in chunk:
                    source_counts.update(chunk["EXT_SOURCE"].value_counts().to_dict())
            if total == 0:
                raise RuntimeError("Stage 06 input contains no rows")
            if set(class_counts) != {0, 1}:
                raise RuntimeError(f"Stage 06 classes are invalid: {dict(class_counts)}")
    finally:
        id_connection.close()
        id_database.unlink(missing_ok=True)
    report = {
        "rows": total,
        "rows_removed": 0,
        "unique_row_ids": total,
        "unique_genes": len(gene_counts),
        "class_counts": dict(class_counts),
        "source_counts": dict(source_counts),
        "structure_coverage": {str(key): value for key, value in structure_counts.items()},
        "sentinels_converted": dict(sentinel_counts),
        "infinities_converted": dict(nonfinite_counts),
        "missingness_indicators": indicator_bases,
        "imputation": "deferred_to_training_folds",
        "upstream_validation": (
            "passed" if validate_upstream else "explicitly_skipped_nonpublication"
        ),
    }
    report_temporary = REPORT_FILE.with_suffix(REPORT_FILE.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, default=json_default), encoding="utf-8"
    )
    report_temporary.replace(REPORT_FILE)
    write_run_manifest(
        MANIFEST_FILE,
        "06_clean_and_finalize",
        [INPUT_FILE, UPSTREAM_MANIFEST],
        report,
        outputs=[OUTPUT_FILE, REPORT_FILE],
    )
    logger.info("Validated %d rows without coverage filtering", total)


if __name__ == "__main__":
    clean_dataset(validate_upstream=True)
