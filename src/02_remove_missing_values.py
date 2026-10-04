from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from config import (
    STAGE01_OUT,
    STAGE02_OUT,
    ensure_directories,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import BASE_FEATURE_ALLOWLIST, LABEL_COL, ROW_ID_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage02_missingness")

INPUT_FILE = STAGE01_OUT / "somatic_variant_dbNSFP.csv"
OUTPUT_FILE = STAGE02_OUT / "somatic_variant_missingness_preserved.csv"
REPORT_FILE = STAGE02_OUT / "missingness_report.csv"
MANIFEST_FILE = STAGE02_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE01_OUT / "run_manifest.json"
CHUNK_SIZE = 100000


def _update_missingness(
    chunk: pd.DataFrame,
    missing_counts: defaultdict[tuple[str, str], int],
    totals: defaultdict[str, int],
) -> None:
    totals["all"] += len(chunk)
    for column in chunk.columns:
        missing_counts[(column, "all")] += int(chunk[column].isna().sum())
    for label in (0, 1):
        subset = chunk.loc[chunk[LABEL_COL].eq(label)]
        totals[str(label)] += len(subset)
        for column in chunk.columns:
            missing_counts[(column, str(label))] += int(subset[column].isna().sum())


def _build_report(
    columns: list[str],
    missing_counts: defaultdict[tuple[str, str], int],
    totals: defaultdict[str, int],
) -> pd.DataFrame:
    rows = []
    for column in columns:
        for label in ("all", "0", "1"):
            total = totals[label]
            missing = missing_counts[(column, label)]
            rows.append(
                {
                    "column": column,
                    "class": label,
                    "rows": total,
                    "missing": missing,
                    "missing_fraction": missing / total if total else None,
                }
            )
    return pd.DataFrame(rows)


def preserve_missing_values(
    chunksize: int = CHUNK_SIZE, *, validate_upstream: bool = True
) -> None:
    """Preserve rows and audit missingness."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST, "01_dbnsfp_processor", [INPUT_FILE]
        )
    ensure_directories(STAGE02_OUT)
    header = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
    required = {LABEL_COL, ROW_ID_COL}
    missing_required = required - set(header)
    if missing_required:
        raise KeyError(f"Stage 02 input misses {sorted(missing_required)}")
    indicator_bases = [
        column for column in BASE_FEATURE_ALLOWLIST if column in header
    ]
    output_columns = [
        *header,
        *[
            f"{column}__missing"
            for column in indicator_bases
            if f"{column}__missing" not in header
        ],
    ]
    missing_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    totals: defaultdict[str, int] = defaultdict(int)
    class_counts: defaultdict[int, int] = defaultdict(int)
    temporary = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    report_temporary = REPORT_FILE.with_suffix(REPORT_FILE.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    report_temporary.unlink(missing_ok=True)
    first_chunk = True
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(INPUT_FILE, chunksize=chunksize, low_memory=False), 1
        ):
            if chunk.empty:
                continue
            invalid_labels = ~chunk[LABEL_COL].isin([0, 1])
            if invalid_labels.any():
                raise ValueError(
                    f"Chunk {chunk_number} has {int(invalid_labels.sum())} invalid labels"
                )
            _update_missingness(chunk, missing_counts, totals)
            for label, count in chunk[LABEL_COL].value_counts().items():
                class_counts[int(label)] += int(count)
            for column in indicator_bases:
                indicator = f"{column}__missing"
                if indicator not in chunk.columns:
                    chunk[indicator] = chunk[column].isna().astype("int8")
            chunk = chunk.reindex(columns=output_columns)
            chunk.to_csv(
                temporary,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            if chunk_number % 10 == 0:
                logger.info("Processed %d chunks", chunk_number)
        if first_chunk or totals["all"] == 0:
            raise RuntimeError("Stage 02 input contains no rows")
        if set(class_counts) != {0, 1}:
            raise RuntimeError(f"Stage 02 classes are invalid: {dict(class_counts)}")
        report = _build_report(header, missing_counts, totals)
        report.to_csv(report_temporary, index=False)
        output_header = pd.read_csv(temporary, nrows=0).columns.tolist()
        if output_header != output_columns:
            raise RuntimeError("Stage 02 output schema changed unexpectedly")
        temporary.replace(OUTPUT_FILE)
        report_temporary.replace(REPORT_FILE)
    except BaseException:
        temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
        raise
    write_run_manifest(
        MANIFEST_FILE,
        "02_remove_missing_values",
        [INPUT_FILE, *([UPSTREAM_MANIFEST] if UPSTREAM_MANIFEST.exists() else [])],
        {
            "input_rows": totals["all"],
            "output_rows": totals["all"],
            "rows_removed": 0,
            "missingness_indicators": indicator_bases,
            "class_counts": dict(class_counts),
            "imputation": "deferred_to_training_folds",
            "upstream_validation": (
                "passed" if validate_upstream else "explicitly_skipped_nonpublication"
            ),
        },
        outputs=[OUTPUT_FILE, REPORT_FILE],
    )
    logger.info("Preserved %d rows without global imputation", totals["all"])


if __name__ == "__main__":
    preserve_missing_values(validate_upstream=True)
