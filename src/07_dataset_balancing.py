from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from config import (
    FEATURE_COVERAGE_POLICY,
    FEATURE_COVERAGE_WARN_GAP,
    STAGE06_OUT,
    STAGE07_OUT,
    ensure_directories,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import (
    GENE_COL,
    LABEL_COL,
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
logger = logging.getLogger("stage07_cohort_distribution")

INPUT_FILE = STAGE06_OUT / "somatic_variant_cleaned.parquet"
# Legacy filename retained because Stage 08 and existing manifests depend on it.
# The class fraction is not interpreted as natural/population prevalence.
OUTPUT_FILE = STAGE07_OUT / "Final_Dataset_Natural_Prevalence.parquet"
DISTRIBUTION_FILE = STAGE07_OUT / "class_distribution.csv"
GENE_DISTRIBUTION_FILE = STAGE07_OUT / "gene_class_distribution.csv"
MANIFEST_FILE = STAGE07_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE06_OUT / "run_manifest.json"
CHUNK_SIZE = 50000

FEATURE_COVERAGE_COLUMNS = (
    "GERP++_RS",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "SASA",
    "RELATIVE_SASA",
    "PLDDT_SCORE",
    "HAS_PROTEIN_MAPPING",
    "STRUCTURE_FILE_AVAILABLE",
    "HAS_STRUCTURE",
    "HAS_DOMAIN_ANNOTATION",
    "HAS_ACTIVE_SITE_ANNOTATION",
    "HAS_BINDING_SITE_ANNOTATION",
    "HAS_TRANSMEMBRANE_ANNOTATION",
)


def _available_count(frame: pd.DataFrame, column: str) -> int:
    values = frame[column]
    if column.startswith("HAS_") or column == "STRUCTURE_FILE_AVAILABLE":
        return int(pd.to_numeric(values, errors="coerce").fillna(0).gt(0).sum())
    return int(values.notna().sum())


def _build_feature_coverage_report(
    columns: list[str],
    class_counts: Counter,
    coverage_counts: Counter,
) -> dict[str, object]:
    report: dict[str, object] = {
        "policy": FEATURE_COVERAGE_POLICY,
        "gap_threshold": FEATURE_COVERAGE_WARN_GAP,
        "features": {},
        "flagged": [],
    }
    features: dict[str, object] = {}
    flagged: list[str] = []
    for column in FEATURE_COVERAGE_COLUMNS:
        if column not in columns:
            continue
        by_class = {
            str(label): (
                coverage_counts[(column, label)] / class_counts[label]
                if class_counts[label]
                else 0.0
            )
            for label in (0, 1)
        }
        gap = abs(by_class["1"] - by_class["0"])
        features[column] = {"by_class": by_class, "absolute_gap": gap}
        if gap >= FEATURE_COVERAGE_WARN_GAP:
            flagged.append(column)
    report["features"] = features
    report["flagged"] = flagged
    return report


def _enforce_feature_coverage(report: dict[str, object]) -> None:
    flagged = list(report.get("flagged", []))
    if not flagged or FEATURE_COVERAGE_POLICY == "off":
        return
    message = (
        "Class-dependent feature coverage exceeds "
        f"{FEATURE_COVERAGE_WARN_GAP:.3f}: {flagged}. This can encode ascertainment "
        "rather than variant biology; run coverage-only and matched-cohort ablations."
    )
    if FEATURE_COVERAGE_POLICY == "error":
        raise RuntimeError(message)
    logger.warning(message)


def preserve_observed_cohort_distribution(
    chunksize: int = CHUNK_SIZE, *, validate_upstream: bool = True
) -> None:
    """Preserve the observed labelled-cohort distribution without resampling.

    The resulting class fraction is an ascertainment property of the assembled
    evidence cohort, not population or clinical prevalence.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST,
            "06_clean_and_finalize",
            [INPUT_FILE],
        )
    ensure_directories(STAGE07_OUT)
    header = table_columns(INPUT_FILE)
    required = {
        LABEL_COL,
        GENE_COL,
        ROW_ID_COL,
        "CLINVAR_VARIATION_ID",
        "Ensembl_transcriptid",
        *REQUIRED_MAPPING_COLS,
        *TRANSCRIPT_SELECTION_COLS,
    }
    missing = required - set(header)
    if missing:
        raise KeyError(f"Stage 07 input misses {sorted(missing)}")
    total = 0
    class_counts = Counter()
    source_counts = Counter()
    gene_class_counts = Counter()
    coverage_counts = Counter()
    coverage_report: dict[str, object] = {}
    with AtomicParquetWriter(OUTPUT_FILE) as output_writer:
        for chunk_number, chunk in enumerate(iter_table(INPUT_FILE, chunksize), 1):
            invalid = ~chunk[LABEL_COL].isin([0, 1])
            if invalid.any():
                raise ValueError(
                    f"Chunk {chunk_number} has {int(invalid.sum())} invalid labels"
                )
            output_writer.write(chunk)
            total += len(chunk)
            class_counts.update(chunk[LABEL_COL].value_counts().to_dict())
            for label in (0, 1):
                subset = chunk.loc[chunk[LABEL_COL].eq(label)]
                for column in FEATURE_COVERAGE_COLUMNS:
                    if column in subset:
                        coverage_counts[(column, label)] += _available_count(
                            subset, column
                        )
            grouped = chunk.groupby([GENE_COL, LABEL_COL], dropna=False).size()
            gene_class_counts.update(grouped.to_dict())
            if "EXT_SOURCE" in chunk:
                source_counts.update(chunk["EXT_SOURCE"].value_counts().to_dict())
        if total == 0:
            raise RuntimeError("Stage 07 input contains no rows")
        if set(class_counts) != {0, 1}:
            raise RuntimeError(f"Stage 07 classes are invalid: {dict(class_counts)}")
        coverage_report = _build_feature_coverage_report(
            header, class_counts, coverage_counts
        )
        _enforce_feature_coverage(coverage_report)
    distribution = pd.DataFrame(
        [
            {
                "label": label,
                "count": class_counts[label],
                "fraction": class_counts[label] / total,
            }
            for label in (0, 1)
        ]
    )
    distribution_temporary = DISTRIBUTION_FILE.with_suffix(
        DISTRIBUTION_FILE.suffix + ".tmp"
    )
    distribution.to_csv(distribution_temporary, index=False)
    distribution_temporary.replace(DISTRIBUTION_FILE)
    gene_distribution = pd.DataFrame(
        [
            {GENE_COL: gene, LABEL_COL: label, "count": count}
            for (gene, label), count in sorted(
                gene_class_counts.items(), key=lambda item: (str(item[0][0]), item[0][1])
            )
        ]
    )
    gene_temporary = GENE_DISTRIBUTION_FILE.with_suffix(
        GENE_DISTRIBUTION_FILE.suffix + ".tmp"
    )
    gene_distribution.to_csv(gene_temporary, index=False)
    gene_temporary.replace(GENE_DISTRIBUTION_FILE)
    positives = class_counts[1]
    negatives = class_counts[0]
    write_run_manifest(
        MANIFEST_FILE,
        "07_dataset_balancing",
        [INPUT_FILE, UPSTREAM_MANIFEST],
        {
            "policy": "observed_labelled_cohort_distribution_preserved",
            "prevalence_interpretation": (
                "label_ascertainment_fraction_not_population_prevalence"
            ),
            "rows": total,
            "rows_removed": 0,
            "class_counts": dict(class_counts),
            "source_counts": dict(source_counts),
            "suggested_positive_weight": negatives / positives,
            "training_policy": "fold_specific_class_weights",
            "feature_coverage_audit": coverage_report,
            "upstream_validation": (
                "passed" if validate_upstream else "explicitly_skipped_nonpublication"
            ),
        },
        outputs=[OUTPUT_FILE, DISTRIBUTION_FILE, GENE_DISTRIBUTION_FILE],
    )
    logger.info(
        "Preserved %d rows at %.4f observed labelled-cohort positive fraction",
        total,
        positives / total,
    )


def preserve_natural_prevalence(
    chunksize: int = CHUNK_SIZE, *, validate_upstream: bool = True
) -> None:
    """Backward-compatible alias for the corrected cohort-distribution policy."""
    logger.warning(
        "preserve_natural_prevalence is a legacy name; the value is an observed "
        "labelled-cohort fraction, not natural population prevalence"
    )
    preserve_observed_cohort_distribution(
        chunksize, validate_upstream=validate_upstream
    )


if __name__ == "__main__":
    preserve_observed_cohort_distribution(validate_upstream=True)
