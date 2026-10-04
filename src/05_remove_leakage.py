from __future__ import annotations

import logging

from config import (
    STAGE04_OUT,
    STAGE05_OUT,
    ensure_directories,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import (
    BASE_FEATURE_ALLOWLIST,
    GENE_COL,
    ID_COLS,
    LABEL_COL,
    LABEL_PROXY_COLS,
    MODEL_FEATURE_ALLOWLIST,
    PREDICTOR_COLS,
    REQUIRED_MAPPING_COLS,
    ROW_ID_COL,
    TRANSCRIPT_SELECTION_COLS,
)
from table_io import AtomicParquetWriter, iter_table, table_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage05_leakage")

INPUT_FILE = STAGE04_OUT / "somatic_variant_structural_functional.parquet"
OUTPUT_FILE = STAGE05_OUT / "somatic_variant_predictor_free.parquet"
SIDECAR_FILE = STAGE05_OUT / "somatic_variant_audit_sidecar.parquet"
MANIFEST_FILE = STAGE05_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE04_OUT / "run_manifest.json"
CHUNK_SIZE = 50000

REQUIRED_CONTEXT = (
    ROW_ID_COL,
    "variant_id",
    "transcript_variant_id",
    "CLINVAR_VARIATION_ID",
    "chr",
    "pos",
    "ref",
    "alt",
    GENE_COL,
    "Ensembl_transcriptid",
    "HGVSp_snpEff",
    "HGVSc_snpEff",
    "aapos",
    "aaref",
    "aaalt",
    *REQUIRED_MAPPING_COLS,
    *TRANSCRIPT_SELECTION_COLS,
    LABEL_COL,
)

OPTIONAL_CONTEXT = (
    "variant_type",
    "STRUCTURE_MAPPING_STATUS",
)


def _ordered_subset(order: list[str], selected: set[str]) -> list[str]:
    return [column for column in order if column in selected]


def remove_leakage(
    chunksize: int = CHUNK_SIZE, *, validate_upstream: bool = True
) -> None:
    """Create a predictor-free modelling table."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST,
            "04_feature_engineering",
            [INPUT_FILE],
        )
    ensure_directories(STAGE05_OUT)
    columns = table_columns(INPUT_FILE)
    missing = set(REQUIRED_CONTEXT) - set(columns)
    if missing:
        raise KeyError(f"Stage 05 input misses {sorted(missing)}")
    forbidden = set(PREDICTOR_COLS) | set(LABEL_PROXY_COLS)
    retained_set = (
        set(REQUIRED_CONTEXT)
        | set(OPTIONAL_CONTEXT)
        | set(ID_COLS)
        | set(BASE_FEATURE_ALLOWLIST)
    ) - forbidden
    retained_columns = _ordered_subset(columns, retained_set)
    audit_set = set(REQUIRED_CONTEXT) | forbidden | {
        "DOMAIN_NAME",
        "Interpro_domain",
        "ROLE_IN_CANCER",
        "TIER",
        "PROTEIN_MAPPING_STATUS",
        "STRUCTURE_MAPPING_STATUS",
        "UNIPROT_REVIEWED",
        "MAPPING_TRANSCRIPT_MATCH",
    }
    sidecar_columns = _ordered_subset(columns, audit_set)
    leaked = forbidden & set(retained_columns)
    if leaked:
        raise RuntimeError(f"Forbidden columns retained: {sorted(leaked)}")
    feature_columns = [
        column for column in MODEL_FEATURE_ALLOWLIST if column in retained_columns
    ]
    if not feature_columns:
        raise RuntimeError("Stage 05 found no approved model features")
    total = 0
    class_counts = {0: 0, 1: 0}
    with (
        AtomicParquetWriter(OUTPUT_FILE) as output_writer,
        AtomicParquetWriter(SIDECAR_FILE) as sidecar_writer,
    ):
        for chunk_number, chunk in enumerate(iter_table(INPUT_FILE, chunksize), 1):
            if chunk.empty:
                continue
            invalid = ~chunk[LABEL_COL].isin([0, 1])
            if invalid.any():
                raise ValueError(
                    f"Chunk {chunk_number} has {int(invalid.sum())} invalid labels"
                )
            model_chunk = chunk[retained_columns]
            audit_chunk = chunk[sidecar_columns]
            output_writer.write(model_chunk)
            sidecar_writer.write(audit_chunk)
            total += len(chunk)
            for label, count in chunk[LABEL_COL].value_counts().items():
                class_counts[int(label)] = class_counts.get(int(label), 0) + int(count)
        if total == 0:
            raise RuntimeError("Stage 05 input contains no rows")
        if set(label for label, count in class_counts.items() if count > 0) != {0, 1}:
            raise RuntimeError(f"Stage 05 classes are invalid: {class_counts}")
        output_columns = set(retained_columns)
        remaining_forbidden = output_columns & forbidden
        if remaining_forbidden:
            raise RuntimeError(
                f"Leakage removal failed for {sorted(remaining_forbidden)}"
            )
    write_run_manifest(
        MANIFEST_FILE,
        "05_remove_leakage",
        [INPUT_FILE, UPSTREAM_MANIFEST],
        {
            "rows": total,
            "class_counts": class_counts,
            "retained_columns": retained_columns,
            "approved_features": feature_columns,
            "dropped_predictors": sorted(set(PREDICTOR_COLS) & set(columns)),
            "dropped_label_proxies": sorted(set(LABEL_PROXY_COLS) & set(columns)),
            "sidecar_columns": sidecar_columns,
            "upstream_validation": (
                "passed" if validate_upstream else "explicitly_skipped_nonpublication"
            ),
        },
        outputs=[OUTPUT_FILE, SIDECAR_FILE],
    )
    logger.info("Saved %d predictor-free rows", total)


if __name__ == "__main__":
    remove_leakage(validate_upstream=True)
