from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import (
    STAGE02_OUT,
    STAGE03_OUT,
    ensure_directories,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import LABEL_COL, ROW_ID_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage03_duplicates")

INPUT_FILE = STAGE02_OUT / "somatic_variant_missingness_preserved.csv"
OUTPUT_FILE = STAGE03_OUT / "somatic_variant_transcript_deduplicated.csv"
CONFLICT_FILE = STAGE03_OUT / "conflicting_variant_labels.csv"
MANIFEST_FILE = STAGE03_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE02_OUT / "run_manifest.json"
DATABASE_FILE = STAGE03_OUT / "deduplication.tmp.sqlite"
CHUNK_SIZE = 100000

ENTITY_COLUMNS = (
    "variant_id",
    "Ensembl_transcriptid",
    "genename",
    "aapos",
    "aaref",
    "aaalt",
)


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _entity_keys(
    chunk: pd.DataFrame, entity_columns: tuple[str, ...] = ENTITY_COLUMNS
) -> pd.Series:
    values = []
    for column in entity_columns:
        series = chunk[column].astype("string").fillna("missing").str.strip()
        values.append(series)
    key = values[0]
    for series in values[1:]:
        key = key + "|" + series
    return key


def _stream_query_to_csv(
    connection: sqlite3.Connection,
    query: str,
    output: Path,
    columns: list[str],
    chunksize: int,
) -> tuple[int, dict[int, int]]:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    total = 0
    class_counts = {0: 0, 1: 0}
    first = True
    for chunk in pd.read_sql_query(query, connection, chunksize=chunksize):
        chunk = chunk.reindex(columns=columns)
        total += len(chunk)
        if LABEL_COL in chunk:
            for label, count in chunk[LABEL_COL].value_counts().items():
                class_counts[int(label)] = class_counts.get(int(label), 0) + int(count)
        chunk.to_csv(
            temporary,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False
    if first:
        pd.DataFrame(columns=columns).to_csv(temporary, index=False)
    return total, class_counts


def remove_duplicates(
    chunksize: int = CHUNK_SIZE,
    *,
    validate_upstream: bool = True,
    preserve_transcript_candidates: bool = True,
) -> None:
    """Deduplicate records; publication mode retains transcript alternatives."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST, "02_remove_missing_values", [INPUT_FILE]
        )
    ensure_directories(STAGE03_OUT)
    columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
    requested_entity = (
        ENTITY_COLUMNS if preserve_transcript_candidates else ("variant_id",)
    )
    entity_columns = tuple(column for column in requested_entity if column in columns)
    required = {LABEL_COL, ROW_ID_COL, "variant_id", "Ensembl_transcriptid"}
    if validate_upstream:
        required.update(ENTITY_COLUMNS)
        if not preserve_transcript_candidates:
            raise RuntimeError(
                "Publication Stage 03 must preserve transcript candidates for "
                "post-mapping variant selection"
            )
    missing = required - set(columns)
    if missing:
        raise KeyError(f"Stage 03 input misses {sorted(missing)}")
    DATABASE_FILE.unlink(missing_ok=True)
    connection = sqlite3.connect(DATABASE_FILE)
    total_input = 0
    ingest_order = 0
    try:
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
        for chunk_number, chunk in enumerate(
            pd.read_csv(INPUT_FILE, chunksize=chunksize, low_memory=False), 1
        ):
            if chunk.empty:
                continue
            chunk["_dedup_key"] = _entity_keys(chunk, entity_columns)
            chunk["_ingest_order"] = range(ingest_order, ingest_order + len(chunk))
            ingest_order += len(chunk)
            total_input += len(chunk)
            sql_chunksize = max(1, min(1000, 900 // len(chunk.columns)))
            chunk.to_sql(
                "records",
                connection,
                if_exists="replace" if chunk_number == 1 else "append",
                index=False,
                chunksize=sql_chunksize,
                method="multi",
            )
            if chunk_number % 10 == 0:
                logger.info("Ingested %d rows", total_input)
        if total_input == 0:
            raise RuntimeError("Stage 03 input contains no rows")
        connection.execute(
            f"CREATE INDEX idx_records_variant ON records({_quoted('variant_id')})"
        )
        connection.execute(
            "CREATE INDEX idx_records_entity ON records(_dedup_key)"
        )
        connection.execute(
            f"CREATE TABLE conflicts AS "
            f"SELECT {_quoted('variant_id')} AS variant_id "
            f"FROM records GROUP BY {_quoted('variant_id')} "
            f"HAVING COUNT(DISTINCT CAST({_quoted(LABEL_COL)} AS INTEGER)) > 1"
        )
        connection.execute("CREATE UNIQUE INDEX idx_conflicts ON conflicts(variant_id)")
        connection.commit()
        selected_columns = ", ".join(_quoted(column) for column in columns)
        transcript = _quoted("Ensembl_transcriptid")
        protein_hgvs = _quoted("HGVSp_snpEff")
        canonical = _quoted("VEP_canonical") if "VEP_canonical" in columns else None
        mane = _quoted("MANE") if "MANE" in columns else None
        row_id = _quoted(ROW_ID_COL)
        variant_id = _quoted("variant_id")
        ranked_query = (
            f"SELECT {selected_columns} FROM ("
            f"SELECT {selected_columns}, "
            f"ROW_NUMBER() OVER (PARTITION BY _dedup_key ORDER BY "
            + (
                f"CASE WHEN UPPER(REPLACE(TRIM(COALESCE({mane}, '')), '_', ' ')) "
                f"IN ('SELECT', 'MANE SELECT') THEN 0 ELSE 1 END, "
                if mane is not None
                else ""
            )
            + (
                f"CASE WHEN UPPER(REPLACE(TRIM(COALESCE({mane}, '')), '_', ' ')) "
                f"IN ('PLUS CLINICAL', 'MANE PLUS CLINICAL') THEN 0 ELSE 1 END, "
                if mane is not None
                else ""
            )
            + (
                f"CASE WHEN UPPER(COALESCE({canonical}, '')) IN ('YES', 'Y', '1', 'TRUE') THEN 0 ELSE 1 END, "
                if canonical is not None
                else ""
            )
            + f"CASE WHEN {transcript} IS NULL OR TRIM({transcript}) = '' THEN 1 ELSE 0 END, "
            + f"CASE WHEN {protein_hgvs} IS NULL OR TRIM({protein_hgvs}) = '' THEN 1 ELSE 0 END, "
            + f"{row_id}, _ingest_order) AS rank "
            + "FROM records r WHERE NOT EXISTS ("
            + f"SELECT 1 FROM conflicts c WHERE c.variant_id = r.{variant_id})"
            + f") WHERE rank = 1 ORDER BY {variant_id}, {transcript}, {row_id}"
        )
        conflict_query = (
            f"SELECT {selected_columns} FROM records r "
            f"WHERE EXISTS (SELECT 1 FROM conflicts c "
            f"WHERE c.variant_id = r.{variant_id}) "
            f"ORDER BY {variant_id}, {row_id}, _ingest_order"
        )
        total_output, class_counts = _stream_query_to_csv(
            connection, ranked_query, OUTPUT_FILE, columns, chunksize
        )
        conflict_rows, _ = _stream_query_to_csv(
            connection, conflict_query, CONFLICT_FILE, columns, chunksize
        )
        if total_output == 0:
            raise RuntimeError("Stage 03 removed every row")
        if set(label for label, count in class_counts.items() if count > 0) != {0, 1}:
            raise RuntimeError(f"Stage 03 classes are invalid: {class_counts}")
        duplicate_rows = total_input - conflict_rows - total_output
        if duplicate_rows < 0:
            raise RuntimeError("Stage 03 row accounting failed")
        OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp").replace(OUTPUT_FILE)
        CONFLICT_FILE.with_suffix(CONFLICT_FILE.suffix + ".tmp").replace(
            CONFLICT_FILE
        )
        write_run_manifest(
            MANIFEST_FILE,
            "03_remove_duplicates",
            [INPUT_FILE, *([UPSTREAM_MANIFEST] if UPSTREAM_MANIFEST.exists() else [])],
            {
                "entity": list(entity_columns),
                "input_rows": total_input,
                "output_rows": total_output,
                "duplicates_removed": duplicate_rows,
                "conflicting_rows_excluded": conflict_rows,
                "class_counts": class_counts,
                "upstream_validation": (
                    "passed"
                    if validate_upstream
                    else "explicitly_skipped_nonpublication"
                ),
                "tie_breaking": [
                    "mane_select",
                    "mane_plus_clinical",
                    "vep_canonical",
                    "transcript_present",
                    "protein_hgvs_present",
                    ROW_ID_COL,
                    "input_order",
                ],
            },
            outputs=[OUTPUT_FILE, CONFLICT_FILE],
        )
    finally:
        connection.close()
        DATABASE_FILE.unlink(missing_ok=True)
        Path(str(DATABASE_FILE) + "-journal").unlink(missing_ok=True)
        OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp").unlink(missing_ok=True)
        CONFLICT_FILE.with_suffix(CONFLICT_FILE.suffix + ".tmp").unlink(
            missing_ok=True
        )
    logger.info(
        "Retained %d rows; excluded %d conflicting rows",
        total_output,
        conflict_rows,
    )


if __name__ == "__main__":
    remove_duplicates(
        validate_upstream=True, preserve_transcript_candidates=True
    )
