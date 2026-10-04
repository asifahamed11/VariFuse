from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import pickle
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import (
    ALPHAFOLD_DIR,
    AUDIT_SAMPLE_MAX_ROWS,
    REQUIRE_TRANSCRIPT_MAPPING,
    STAGE03_OUT,
    STAGE04_OUT,
    TRANSCRIPT_SELECTION_POLICY,
    UNIPROT_FILE,
    ensure_directories,
    file_sha256,
    json_default,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import (
    LABEL_COL,
    REQUIRED_MAPPING_COLS,
    ROW_ID_COL,
    TRANSCRIPT_SELECTION_COLS,
)
from table_io import AtomicParquetWriter, iter_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage04_features")

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.SeqUtils import seq1

    BIOPYTHON_AVAILABLE = True
except ImportError:
    MMCIFParser = None
    PDBParser = None
    ShrakeRupley = None
    seq1 = None
    BIOPYTHON_AVAILABLE = False

INPUT_FILE = STAGE03_OUT / "somatic_variant_transcript_deduplicated.csv"
OUTPUT_FILE = STAGE04_OUT / "somatic_variant_structural_functional.parquet"
SEQUENCE_FILE = STAGE04_OUT / "protein_sequences.parquet"
MAPPING_FILE = STAGE04_OUT / "protein_mapping_sample.parquet"
TRANSCRIPT_SELECTION_FILE = STAGE04_OUT / "transcript_selection_audit.parquet"
COVERAGE_FILE = STAGE04_OUT / "feature_coverage.json"
MANIFEST_FILE = STAGE04_OUT / "run_manifest.json"
UPSTREAM_MANIFEST = STAGE03_OUT / "run_manifest.json"
CACHE_DIR = STAGE04_OUT / "cache"
CHUNK_SIZE = 50000
CACHE_VERSION = "v7_refseq_local_structure_environment"
CPU_COUNT = max(1, min(os.cpu_count() or 1, 8))

MAX_SASA = {
    "ALA": 121.0,
    "ARG": 265.0,
    "ASN": 187.0,
    "ASP": 187.0,
    "CYS": 148.0,
    "GLN": 214.0,
    "GLU": 214.0,
    "GLY": 97.0,
    "HIS": 216.0,
    "ILE": 195.0,
    "LEU": 191.0,
    "LYS": 230.0,
    "MET": 203.0,
    "PHE": 228.0,
    "PRO": 154.0,
    "SER": 143.0,
    "THR": 163.0,
    "TRP": 264.0,
    "TYR": 255.0,
    "VAL": 165.0,
}

HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "CYS", "ILE", "LEU", "MET", "PHE", "PRO", "TRP", "TYR", "VAL"}
)
CHARGED_RESIDUES = frozenset({"ARG", "ASP", "GLU", "HIS", "LYS"})
LOCAL_STRUCTURE_FEATURES = (
    "local_contact_count_8a",
    "local_contact_count_12a",
    "local_long_range_contact_count_8a",
    "local_mean_plddt_8a",
    "local_min_plddt_8a",
    "local_confident_contact_fraction_8a",
    "local_mean_distance_8a",
    "local_hydrophobic_fraction_8a",
    "local_charged_fraction_8a",
)


@dataclass
class ProteinFeatures:
    """Store one canonical UniProt record."""

    entry_name: str = ""
    accession: str = ""
    gene_name: str = ""
    reviewed: bool = False
    sequence: str = ""
    transcript_ids: set[str] = field(default_factory=set)
    domains: list[dict[str, Any]] = field(default_factory=list)
    active_sites: list[dict[str, Any]] = field(default_factory=list)
    binding_sites: list[dict[str, Any]] = field(default_factory=list)
    transmembrane: list[dict[str, Any]] = field(default_factory=list)
    signal_peptide: bool = False


class UniProtParser:
    """Parse and cache reviewed protein annotations."""

    def __init__(self, path: Path, cache_dir: Path = CACHE_DIR):
        self.path = Path(path)
        self.cache_dir = Path(cache_dir)
        self.gene_pattern = re.compile(r"Name=([^;{\s]+)")
        self.coordinate_pattern = re.compile(r"[<>]?(\d+)(?:\.\.[<>]?(\d+))?")

    def _cache_path(self) -> Path:
        digest = file_sha256(self.path)
        if digest is None:
            raise FileNotFoundError(self.path)
        key = hashlib.sha256(f"{self.path.name}|{digest}|{CACHE_VERSION}".encode()).hexdigest()[:20]
        return self.cache_dir / f"uniprot_{key}.pkl"

    def parse(self, force: bool = False) -> dict[str, ProteinFeatures]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path()
        if cache_path.exists() and not force:
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, dict):
                    logger.info("Loaded %d UniProt records", len(cached))
                    return cached
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as error:
                logger.warning("UniProt cache failed: %s", type(error).__name__)
        records: dict[str, ProteinFeatures] = {}
        current: ProteinFeatures | None = None
        sequence_lines: list[str] = []
        in_sequence = False
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("ID"):
                    parts = line.split()
                    current = ProteinFeatures(
                        entry_name=parts[1] if len(parts) > 1 else "",
                        reviewed="Reviewed;" in line,
                    )
                    sequence_lines = []
                    in_sequence = False
                elif line.startswith("AC") and current is not None:
                    if not current.accession:
                        accessions = [
                            value.strip() for value in line[5:].split(";") if value.strip()
                        ]
                        if accessions:
                            current.accession = accessions[0]
                elif line.startswith("GN") and current is not None:
                    match = self.gene_pattern.search(line)
                    if match and not current.gene_name:
                        current.gene_name = match.group(1)
                elif line.startswith("DR   Ensembl;") and current is not None:
                    fields = [value.strip() for value in line.split(";")]
                    for value in fields[1:3]:
                        if value.startswith(("ENST", "ENSP")):
                            current.transcript_ids.add(value.split(".")[0])
                elif line.startswith("DR   RefSeq;") and current is not None:
                    fields = [value.strip().rstrip(".") for value in line.split(";")]
                    for value in fields[1:3]:
                        if value.startswith(("NM_", "NR_", "NP_", "XM_", "XP_")):
                            current.transcript_ids.add(value.split(".")[0])
                elif line.startswith("FT") and current is not None:
                    self._parse_feature(line, current)
                elif line.startswith("SQ") and current is not None:
                    in_sequence = True
                    sequence_lines = []
                elif line.startswith("//"):
                    if current is not None and current.accession:
                        current.sequence = "".join(sequence_lines)
                        records[current.accession] = current
                    current = None
                    sequence_lines = []
                    in_sequence = False
                elif in_sequence and current is not None:
                    sequence_lines.append(re.sub(r"[^A-Za-z]", "", line).upper())
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
        logger.info("Parsed %d UniProt records", len(records))
        return records

    def _parse_feature(self, line: str, record: ProteinFeatures) -> None:
        parts = line.split(None, 3)
        if len(parts) < 3:
            return
        feature_type = parts[1]
        if feature_type not in {"DOMAIN", "ACT_SITE", "BINDING", "TRANSMEM", "SIGNAL"}:
            return
        match = self.coordinate_pattern.fullmatch(parts[2])
        if not match:
            return
        start = int(match.group(1))
        end = int(match.group(2) or start)
        description = parts[3].strip() if len(parts) > 3 else ""
        if feature_type == "DOMAIN":
            record.domains.append({"start": start, "end": end, "name": description})
        elif feature_type == "ACT_SITE":
            record.active_sites.append({"position": start, "description": description})
        elif feature_type == "BINDING":
            record.binding_sites.append({"position": start, "description": description})
        elif feature_type == "TRANSMEM":
            record.transmembrane.append({"start": start, "end": end})
        elif feature_type == "SIGNAL":
            record.signal_peptide = True


@dataclass(frozen=True)
class StructureFragment:
    path: Path
    fragment: int
    offset: int


class AlphaFoldStructureIndex:
    """Index supported AlphaFold structure files."""

    def __init__(self, directory: Path, cache_dir: Path = CACHE_DIR):
        self.directory = Path(directory)
        self.cache_dir = Path(cache_dir)
        self.index: dict[str, list[StructureFragment]] = {}
        self._build()

    def _candidate_files(self) -> list[Path]:
        patterns = (
            "AF-*-model_v*.pdb",
            "AF-*-model_v*.cif",
            "AF-*-model_v*.pdb.gz",
            "AF-*-model_v*.cif.gz",
        )
        files: list[Path] = []
        for pattern in patterns:
            files.extend(self.directory.glob(pattern))
        return sorted(set(files))

    def _signature(self, files: list[Path]) -> str:
        digest = hashlib.sha256(CACHE_VERSION.encode())
        for path in files:
            stat = path.stat()
            digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def _build(self) -> None:
        if not self.directory.exists():
            logger.warning("AlphaFold directory is missing")
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        files = self._candidate_files()
        signature = self._signature(files)
        cache_path = self.cache_dir / "alphafold_index.pkl"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if cached.get("signature") == signature:
                    self.index = {
                        key: [
                            StructureFragment(
                                self.directory / item["path"],
                                int(item["fragment"]),
                                int(item["offset"]),
                            )
                            for item in values
                        ]
                        for key, values in cached["index"].items()
                    }
                    logger.info("Loaded %d AlphaFold structures", len(self.index))
                    return
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError) as error:
                logger.warning("AlphaFold cache failed: %s", type(error).__name__)
        selected: dict[tuple[str, int], tuple[tuple[int, int], Path]] = {}
        pattern = re.compile(
            r"^AF-(?P<accession>.+?)-F(?P<fragment>\d+)-model_v(?P<version>\d+)\.(?P<format>pdb|cif)(?:\.gz)?$"
        )
        for path in files:
            match = pattern.match(path.name)
            if not match:
                continue
            accession = match.group("accession")
            fragment = int(match.group("fragment"))
            version = int(match.group("version"))
            is_pdb = int(match.group("format") == "pdb")
            priority = (version, is_pdb)
            key = (accession, fragment)
            if key not in selected or priority > selected[key][0]:
                selected[key] = (priority, path)
        grouped: dict[str, list[StructureFragment]] = defaultdict(list)
        for (accession, fragment), (_, path) in selected.items():
            grouped[accession].append(StructureFragment(path, fragment, 200 * (fragment - 1)))
        self.index = {
            accession: sorted(fragments, key=lambda item: item.fragment)
            for accession, fragments in grouped.items()
        }
        payload = {
            "signature": signature,
            "index": {
                key: [
                    {
                        "path": item.path.name,
                        "fragment": item.fragment,
                        "offset": item.offset,
                    }
                    for item in values
                ]
                for key, values in self.index.items()
            },
        }
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
        logger.info("Indexed %d AlphaFold structures", len(self.index))

    def get_fragments(self, accession: str) -> list[StructureFragment]:
        return self.index.get(accession, [])

    def available(self, accession: str) -> bool:
        return accession in self.index


@dataclass
class AnnotationResources:
    """Share mapping and structure resources."""

    records: dict[str, ProteinFeatures]
    structures: AlphaFoldStructureIndex
    gene_candidates: dict[str, list[ProteinFeatures]]
    mapping_counts: Counter[str] = field(default_factory=Counter)
    structure_counts: Counter[str] = field(default_factory=Counter)
    structure_parse_counts: Counter[str] = field(default_factory=Counter)
    structure_failures: dict[str, dict[str, str]] = field(default_factory=dict)


def build_annotation_resources(
    uniprot_file: Path = UNIPROT_FILE,
    alphafold_dir: Path = ALPHAFOLD_DIR,
    force_reparse: bool = False,
) -> AnnotationResources:
    """Build reusable annotation resources."""
    records = UniProtParser(Path(uniprot_file)).parse(force=force_reparse)
    structures = AlphaFoldStructureIndex(Path(alphafold_dir))
    candidates: dict[str, list[ProteinFeatures]] = defaultdict(list)
    for record in records.values():
        if record.gene_name:
            candidates[record.gene_name].append(record)
    for gene in candidates:
        candidates[gene].sort(key=lambda record: record.accession)
    return AnnotationResources(records, structures, dict(candidates))


def _choose_protein(
    row: pd.Series, resources: AnnotationResources
) -> tuple[ProteinFeatures | None, str, bool]:
    gene = str(row.get("genename", "")).strip()
    candidates = resources.gene_candidates.get(gene, [])
    if not candidates:
        return None, "gene_unmapped", False
    try:
        position = int(row.get("aapos", row.get("aa_pos")))
    except (TypeError, ValueError):
        return None, "position_invalid", False
    reference = str(row.get("aaref", row.get("aa_ref", ""))).upper()
    matched = [
        record
        for record in candidates
        if record.sequence
        and 1 <= position <= len(record.sequence)
        and record.sequence[position - 1] == reference
    ]
    if not matched:
        return None, "reference_mismatch", False
    transcript = str(row.get("Ensembl_transcriptid", "")).split(".")[0]
    accession_hint = str(row.get("uniprot_id_hint", "")).split("-")[0]

    def priority(record: ProteinFeatures) -> tuple[int, int, int, int, str]:
        hint_match = int(bool(accession_hint) and record.accession == accession_hint)
        transcript_match = int(bool(transcript) and transcript in record.transcript_ids)
        return (
            -hint_match,
            -transcript_match,
            -int(record.reviewed),
            -int(resources.structures.available(record.accession)),
            record.accession,
        )

    selected = sorted(matched, key=priority)[0]
    transcript_match = bool(transcript) and transcript in selected.transcript_ids
    status = "mapped_transcript" if transcript_match else "mapped_reference"
    return selected, status, transcript_match


def map_variants_to_proteins(dataset: pd.DataFrame, resources: AnnotationResources) -> pd.DataFrame:
    """Map variants using reference-compatible proteins."""
    result = dataset.copy()
    accessions: list[Any] = []
    sequences: list[Any] = []
    statuses: list[str] = []
    transcript_matches: list[int] = []
    reviewed: list[int] = []
    for _, row in result.iterrows():
        record, status, transcript_match = _choose_protein(row, resources)
        resources.mapping_counts[status] += 1
        statuses.append(status)
        transcript_matches.append(int(transcript_match))
        if record is None:
            accessions.append(pd.NA)
            sequences.append(pd.NA)
            reviewed.append(0)
        else:
            accessions.append(record.accession)
            sequences.append(record.sequence)
            reviewed.append(int(record.reviewed))
    result["uniprot_id"] = pd.array(accessions, dtype="string")
    result["protein_sequence"] = pd.array(sequences, dtype="string")
    result["PROTEIN_MAPPING_STATUS"] = statuses
    result["HAS_PROTEIN_MAPPING"] = result["uniprot_id"].notna().astype("int8")
    result["MAPPING_TRANSCRIPT_MATCH"] = np.asarray(transcript_matches, dtype=np.int8)
    result["UNIPROT_REVIEWED"] = np.asarray(reviewed, dtype=np.int8)
    result["PRIMARY_MAPPING_ELIGIBLE"] = result["MAPPING_TRANSCRIPT_MATCH"].astype(np.int8)
    result["MAPPING_CONFIDENCE"] = np.select(
        [
            result["PROTEIN_MAPPING_STATUS"].eq("mapped_transcript"),
            result["PROTEIN_MAPPING_STATUS"].eq("mapped_reference"),
        ],
        ["transcript_verified", "reference_only_sensitivity"],
        default="unmapped",
    )
    return result


def _clinvar_source_gene_matches(frame: pd.DataFrame) -> pd.Series:
    """Match dbNSFP consequence genes to the label source without inference."""
    if "CLINVAR_SOURCE_GENE" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)

    def matches(source: Any, annotated: Any) -> bool:
        gene = str(annotated).strip().upper()
        if not gene or gene in {".", "NAN", "<NA>"} or pd.isna(source):
            return False
        tokens = {
            token.strip().upper()
            for token in re.split(r"[;,|]+", str(source))
            if token.strip() and token.strip() not in {".", "-"}
        }
        return gene in tokens

    return pd.Series(
        [
            matches(source, annotated)
            for source, annotated in zip(frame["CLINVAR_SOURCE_GENE"], frame["genename"])
        ],
        index=frame.index,
        dtype=bool,
    )


def select_primary_mapped_consequences(
    frame: pd.DataFrame,
    *,
    require_transcript_mapping: bool = REQUIRE_TRANSCRIPT_MAPPING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one consequence per genomic variant only after protein mapping.

    The primary contract first restricts candidates to a reference-compatible,
    transcript-verified UniProt mapping and, when ClinVar source-gene provenance
    is present, the asserted source gene. MANE Select, MANE Plus Clinical and VEP
    canonical status rank only within that biologically eligible set. Variants
    without an eligible source-gene mapping are audited and excluded rather than
    silently falling back to another gene or protein isoform.
    """
    if frame.empty:
        return frame.copy(), frame.copy()
    required = {
        "variant_id",
        ROW_ID_COL,
        "genename",
        "Ensembl_transcriptid",
        *REQUIRED_MAPPING_COLS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Mapped consequence selection misses {sorted(missing)}")
    for column in ("variant_id", ROW_ID_COL, "genename"):
        values = frame[column].astype("string").fillna("").str.strip()
        if values.eq("").any():
            raise ValueError(f"Mapped consequence candidates contain empty {column}")
    if frame[ROW_ID_COL].astype(str).duplicated().any():
        raise ValueError("Mapped consequence candidates contain duplicate row IDs")
    if LABEL_COL in frame:
        conflicting = frame.groupby("variant_id", sort=False)[LABEL_COL].nunique()
        if conflicting.gt(1).any():
            raise ValueError("A genomic variant has conflicting candidate labels")
    ranked = frame.copy()
    transcript_verified = (
        pd.to_numeric(ranked["PRIMARY_MAPPING_ELIGIBLE"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(ranked["MAPPING_TRANSCRIPT_MATCH"], errors="coerce").fillna(0).eq(1)
        & ranked["PROTEIN_MAPPING_STATUS"].astype("string").eq("mapped_transcript")
        & ranked["uniprot_id"].astype("string").fillna("").str.strip().ne("")
    ).fillna(False)
    protein_mapped = (
        pd.to_numeric(
            ranked.get("HAS_PROTEIN_MAPPING", pd.Series(0, index=ranked.index)),
            errors="coerce",
        )
        .fillna(0)
        .eq(1)
        & ranked["uniprot_id"].astype("string").fillna("").str.strip().ne("")
    ).fillna(False)
    mapping_eligible = transcript_verified if require_transcript_mapping else protein_mapped
    source_gene_required = "CLINVAR_SOURCE_GENE" in ranked
    source_gene_match = _clinvar_source_gene_matches(ranked)
    eligible = mapping_eligible & source_gene_match
    mane = (
        ranked.get("MANE", pd.Series("", index=ranked.index))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace("_", " ", regex=False)
    )
    canonical = (
        ranked.get("VEP_canonical", pd.Series("", index=ranked.index))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )
    ranked["_mapping_rank"] = (~eligible).astype(np.int8)
    ranked["_mane_rank"] = np.select(
        [
            mane.isin({"SELECT", "MANE SELECT"}),
            mane.isin({"PLUS CLINICAL", "MANE PLUS CLINICAL"}),
        ],
        [0, 1],
        default=2,
    ).astype(np.int8)
    ranked["_canonical_rank"] = (~canonical.isin({"YES", "Y", "1", "TRUE"})).astype(np.int8)
    completeness_columns = [
        column
        for column in (
            "Ensembl_transcriptid",
            "HGVSp_snpEff",
            "HGVSc_snpEff",
            "uniprot_id",
        )
        if column in ranked
    ]
    ranked["_completeness_rank"] = -sum(
        ranked[column].astype("string").fillna("").str.strip().ne("").astype(int)
        for column in completeness_columns
    )
    ranked["_reviewed_rank"] = (
        -pd.to_numeric(
            ranked.get("UNIPROT_REVIEWED", pd.Series(0, index=ranked.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    ranked["_stable_tie"] = ranked[ROW_ID_COL].astype(str)
    ranked["TRANSCRIPT_CANDIDATE_COUNT"] = (
        ranked.groupby("variant_id", sort=False)["variant_id"].transform("size").astype(np.int32)
    )
    ranked["SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT"] = (
        eligible.astype(int)
        .groupby(ranked["variant_id"], sort=False)
        .transform("sum")
        .astype(np.int32)
    )
    ranked["TRANSCRIPT_MAPPED_CANDIDATE_COUNT"] = (
        mapping_eligible.astype(int)
        .groupby(ranked["variant_id"], sort=False)
        .transform("sum")
        .astype(np.int32)
    )
    ranked["CLINVAR_SOURCE_GENE_MATCH"] = source_gene_match.astype(np.int8)
    ranked = ranked.sort_values(
        [
            "variant_id",
            "_mapping_rank",
            "_mane_rank",
            "_canonical_rank",
            "_completeness_rank",
            "_reviewed_rank",
            "_stable_tie",
        ],
        kind="mergesort",
    )
    ranked["PRIMARY_CONSEQUENCE_RANK"] = (
        ranked.groupby("variant_id", sort=False).cumcount() + 1
    ).astype(np.int32)
    ranked["PRIMARY_CONSEQUENCE_SELECTED"] = ranked["PRIMARY_CONSEQUENCE_RANK"].eq(1) & ranked[
        "_mapping_rank"
    ].eq(0)
    ranked["CONSEQUENCE_SELECTION_POLICY"] = TRANSCRIPT_SELECTION_POLICY
    ranked["TRANSCRIPT_SELECTION_OUTCOME"] = np.select(
        [
            ranked["PRIMARY_CONSEQUENCE_SELECTED"],
            source_gene_required & ~source_gene_match,
            ranked["_mapping_rank"].ne(0),
        ],
        [
            "selected",
            "ineligible_clinvar_source_gene_mismatch",
            "ineligible_protein_or_transcript_mapping",
        ],
        default="lower_priority_eligible_candidate",
    )
    private = [
        "_mapping_rank",
        "_mane_rank",
        "_canonical_rank",
        "_completeness_rank",
        "_reviewed_rank",
        "_stable_tie",
    ]
    audit = ranked.drop(columns=private).reset_index(drop=True)
    selected = audit.loc[audit["PRIMARY_CONSEQUENCE_SELECTED"]].copy()
    selected = selected.drop(columns=["TRANSCRIPT_SELECTION_OUTCOME"])
    if selected["variant_id"].duplicated().any():
        raise RuntimeError("Primary mapped consequence selection is not variant-unique")
    if require_transcript_mapping and not selected.empty:
        invalid = ~selected["PRIMARY_MAPPING_ELIGIBLE"].astype(int).eq(1)
        if invalid.any():
            raise RuntimeError("Primary selection retained a non-transcript mapping")
    return selected.reset_index(drop=True), audit


def _iter_complete_variant_chunks(
    input_path: Path,
    chunksize: int,
    max_rows: int | None,
) -> Any:
    """Yield sorted chunks without splitting a variant's transcript candidates."""
    carry: pd.DataFrame | None = None
    for chunk in iter_table(input_path, chunksize, max_rows=max_rows):
        if chunk.empty:
            continue
        # CSV dtype inference is performed independently for every chunk.  A
        # chromosome-only numeric chunk is therefore inferred as int64 while a
        # later chunk containing X/Y is inferred as object.  Concatenating the
        # boundary carry can then create a mixed Python int/str column which
        # PyArrow cannot append to the schema established by the first output
        # batch.  These are identifiers, not numeric model features, so bind
        # them to their canonical storage type before any cross-chunk concat.
        chunk = chunk.copy()
        for identifier in ("chr", "CLINVAR_VARIATION_ID"):
            if identifier in chunk.columns:
                chunk[identifier] = chunk[identifier].astype("string")
        combined = (
            pd.concat([carry, chunk], ignore_index=True)
            if carry is not None and not carry.empty
            else chunk.reset_index(drop=True)
        )
        if "variant_id" not in combined:
            raise KeyError("Stage 04 input misses variant_id")
        keys = combined["variant_id"].astype(str)
        if not keys.is_monotonic_increasing:
            raise RuntimeError(
                "Stage 04 requires Stage 03 variant-sorted candidates so transcript "
                "selection remains correct across streaming chunk boundaries"
            )
        boundary = keys.iloc[-1]
        complete = combined.loc[keys.ne(boundary)].copy()
        carry = combined.loc[keys.eq(boundary)].copy()
        if not complete.empty:
            yield complete.reset_index(drop=True)
    if carry is not None and not carry.empty:
        yield carry.reset_index(drop=True)


def annotate_functional_features(
    dataset: pd.DataFrame, resources: AnnotationResources
) -> pd.DataFrame:
    """Annotate functional sites with missingness flags."""
    result = dataset.copy()
    columns: dict[str, list[Any]] = {
        "IS_IN_DOMAIN": [],
        "HAS_DOMAIN_ANNOTATION": [],
        "DOMAIN_NAME": [],
        "DISTANCE_TO_ACTIVE_SITE": [],
        "HAS_ACTIVE_SITE_ANNOTATION": [],
        "IS_ACTIVE_SITE": [],
        "HAS_BINDING_SITE_ANNOTATION": [],
        "IS_BINDING_SITE": [],
        "HAS_TRANSMEMBRANE_ANNOTATION": [],
        "IS_TRANSMEMBRANE": [],
    }
    for _, row in result.iterrows():
        accession = row.get("uniprot_id")
        record = resources.records.get(str(accession)) if pd.notna(accession) else None
        try:
            position = int(row.get("aapos", row.get("aa_pos")))
        except (TypeError, ValueError):
            position = -1
        if record is None or position < 1:
            columns["IS_IN_DOMAIN"].append(0)
            columns["HAS_DOMAIN_ANNOTATION"].append(0)
            columns["DOMAIN_NAME"].append(pd.NA)
            columns["DISTANCE_TO_ACTIVE_SITE"].append(np.nan)
            columns["HAS_ACTIVE_SITE_ANNOTATION"].append(0)
            columns["IS_ACTIVE_SITE"].append(0)
            columns["HAS_BINDING_SITE_ANNOTATION"].append(0)
            columns["IS_BINDING_SITE"].append(0)
            columns["HAS_TRANSMEMBRANE_ANNOTATION"].append(0)
            columns["IS_TRANSMEMBRANE"].append(0)
            continue
        domains = [
            domain for domain in record.domains if domain["start"] <= position <= domain["end"]
        ]
        active_positions = [site["position"] for site in record.active_sites]
        binding_positions = [site["position"] for site in record.binding_sites]
        in_transmembrane = any(
            region["start"] <= position <= region["end"] for region in record.transmembrane
        )
        columns["IS_IN_DOMAIN"].append(int(bool(domains)))
        columns["HAS_DOMAIN_ANNOTATION"].append(int(bool(record.domains)))
        columns["DOMAIN_NAME"].append(domains[0]["name"] if domains else pd.NA)
        columns["DISTANCE_TO_ACTIVE_SITE"].append(
            min(abs(position - site) for site in active_positions) if active_positions else np.nan
        )
        columns["HAS_ACTIVE_SITE_ANNOTATION"].append(int(bool(active_positions)))
        columns["IS_ACTIVE_SITE"].append(int(position in active_positions))
        columns["HAS_BINDING_SITE_ANNOTATION"].append(int(bool(binding_positions)))
        columns["IS_BINDING_SITE"].append(int(position in binding_positions))
        columns["HAS_TRANSMEMBRANE_ANNOTATION"].append(int(bool(record.transmembrane)))
        columns["IS_TRANSMEMBRANE"].append(int(in_transmembrane))
    for name, values in columns.items():
        result[name] = values
    result["DISTANCE_TO_ACTIVE_SITE"] = pd.to_numeric(
        result["DISTANCE_TO_ACTIVE_SITE"], errors="coerce"
    ).clip(upper=100)
    return result


def _structure_parser(path: Path) -> Any:
    name = path.name.lower()
    if name.endswith((".cif", ".cif.gz")):
        if MMCIFParser is None:
            return None
        return MMCIFParser(QUIET=True)
    if PDBParser is None:
        return None
    return PDBParser(QUIET=True)


def _read_structure_features(
    accession: str,
    fragment: StructureFragment,
    positions: set[int],
    expected_sequence: str | None = None,
) -> tuple[str, dict[int, dict[str, float]], str, str]:
    path = fragment.path
    if not BIOPYTHON_AVAILABLE:
        return accession, {}, "biopython_unavailable", "Biopython is not installed"
    parser = _structure_parser(path)
    if parser is None:
        return accession, {}, "unsupported_format", path.name
    try:
        if path.name.lower().endswith(".gz"):
            with gzip.open(path, "rt") as handle:
                structure = parser.get_structure(accession, handle)
        else:
            structure = parser.get_structure(accession, str(path))
        residues: dict[int, Any] = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " " and residue.id[1] not in residues:
                        residues[int(residue.id[1])] = residue
        eligible_positions = set()
        mismatches = 0
        for position in positions:
            residue = residues.get(position - fragment.offset)
            if residue is None:
                continue
            if expected_sequence is not None and (
                not 1 <= position <= len(expected_sequence)
                or seq1(str(residue.get_resname()).upper()) != expected_sequence[position - 1]
            ):
                mismatches += 1
                continue
            eligible_positions.add(position)
        if not eligible_positions:
            status = "reference_mismatch" if mismatches else "residues_missing"
            return accession, {}, status, ""
        ShrakeRupley().compute(structure, level="R")
        residue_geometry: dict[int, tuple[np.ndarray, float, str]] = {}
        for residue_position, residue_value in residues.items():
            if "CA" not in residue_value:
                continue
            coordinate = np.asarray(residue_value["CA"].get_coord(), dtype=np.float32)
            if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
                continue
            neighbour_atom_scores = [
                float(atom.get_bfactor()) for atom in residue_value.get_atoms()
            ]
            neighbour_plddt = (
                float(np.mean(neighbour_atom_scores)) if neighbour_atom_scores else np.nan
            )
            residue_geometry[residue_position] = (
                coordinate,
                neighbour_plddt,
                str(residue_value.get_resname()).upper(),
            )

        geometry_positions = list(residue_geometry)
        geometry_values = list(residue_geometry.values())
        coordinates = np.asarray([item[0] for item in geometry_values], dtype=np.float32)
        spatial_index = cKDTree(coordinates) if geometry_values else None

        features: dict[int, dict[str, float]] = {}
        for position in sorted(eligible_positions):
            local_position = position - fragment.offset
            residue = residues.get(local_position)
            if residue is None:
                continue
            sasa = float(getattr(residue, "sasa", np.nan))
            denominator = MAX_SASA.get(residue.get_resname())
            relative = sasa / denominator if denominator and np.isfinite(sasa) else np.nan
            atom_scores = [float(atom.get_bfactor()) for atom in residue.get_atoms()]
            plddt = float(np.mean(atom_scores)) if atom_scores else np.nan
            local_environment = {name: np.nan for name in LOCAL_STRUCTURE_FEATURES}
            target_geometry = residue_geometry.get(local_position)
            if target_geometry is not None and spatial_index is not None:
                target_coordinate = target_geometry[0]
                neighbours: list[tuple[int, float, float, str]] = []
                # A radius query avoids scanning the full protein for every
                # variant and never materializes a quadratic distance matrix.
                candidate_indices = sorted(
                    spatial_index.query_ball_point(target_coordinate, 12.0 + 1e-5)
                )
                for neighbour_index in candidate_indices:
                    neighbour_position = geometry_positions[neighbour_index]
                    neighbour_coordinate, neighbour_plddt, neighbour_name = (
                        geometry_values[neighbour_index]
                    )
                    if neighbour_position == local_position:
                        continue
                    distance = float(np.linalg.norm(neighbour_coordinate - target_coordinate))
                    if np.isfinite(distance) and distance <= 12.0:
                        neighbours.append(
                            (
                                neighbour_position,
                                distance,
                                neighbour_plddt,
                                neighbour_name,
                            )
                        )
                within_eight = [item for item in neighbours if item[1] <= 8.0]
                within_twelve = [item for item in neighbours if item[1] <= 12.0]
                local_environment["local_contact_count_8a"] = float(len(within_eight))
                local_environment["local_contact_count_12a"] = float(len(within_twelve))
                local_environment["local_long_range_contact_count_8a"] = float(
                    sum(
                        abs(neighbour_position - local_position) >= 12
                        for neighbour_position, *_ in within_eight
                    )
                )
                if within_eight:
                    distances = np.asarray([item[1] for item in within_eight], dtype=np.float32)
                    neighbour_confidence = np.asarray(
                        [item[2] for item in within_eight], dtype=np.float32
                    )
                    finite_confidence = neighbour_confidence[np.isfinite(neighbour_confidence)]
                    local_environment["local_mean_distance_8a"] = float(distances.mean())
                    local_environment["local_hydrophobic_fraction_8a"] = float(
                        np.mean([item[3] in HYDROPHOBIC_RESIDUES for item in within_eight])
                    )
                    local_environment["local_charged_fraction_8a"] = float(
                        np.mean([item[3] in CHARGED_RESIDUES for item in within_eight])
                    )
                    if finite_confidence.size:
                        local_environment["local_mean_plddt_8a"] = float(finite_confidence.mean())
                        local_environment["local_min_plddt_8a"] = float(finite_confidence.min())
                        local_environment["local_confident_contact_fraction_8a"] = float(
                            np.mean(finite_confidence >= 70.0)
                        )
            features[position] = {
                "sasa": sasa,
                "relative_sasa": relative,
                "plddt": plddt,
                "fragment": float(fragment.fragment),
                **local_environment,
            }
        status = "ok" if features else "residues_missing"
        return accession, features, status, ""
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        return accession, {}, f"parser_error:{type(error).__name__}", str(error)
    except Exception as error:
        return accession, {}, f"unexpected_error:{type(error).__name__}", str(error)


def annotate_structural_features(
    dataset: pd.DataFrame,
    resources: AnnotationResources,
    workers: int = CPU_COUNT,
) -> pd.DataFrame:
    """Annotate AlphaFold features by mapped accession."""
    result = dataset.copy()
    requests: dict[str, set[int]] = defaultdict(set)
    for _, row in result.iterrows():
        accession = row.get("uniprot_id")
        if pd.isna(accession):
            continue
        try:
            position = int(row.get("aapos", row.get("aa_pos")))
        except (TypeError, ValueError):
            continue
        if resources.structures.available(str(accession)):
            requests[str(accession)].add(position)
    collected: dict[tuple[str, int], dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, CPU_COUNT))) as executor:
        futures = {}
        for accession, positions in requests.items():
            for fragment in resources.structures.get_fragments(accession):
                future = executor.submit(
                    _read_structure_features,
                    accession,
                    fragment,
                    positions,
                    resources.records[accession].sequence,
                )
                futures[future] = (accession, fragment.fragment)
        for future in as_completed(futures):
            accession, features, status, message = future.result()
            resources.structure_parse_counts[status] += 1
            if status != "ok":
                resources.structure_failures[accession] = {
                    "status": status,
                    "message": message,
                }
            for position, values in features.items():
                key = (accession, position)
                previous = collected.get(key)
                candidate_priority = (
                    float(values["plddt"]) if np.isfinite(values["plddt"]) else -np.inf,
                    -int(values.get("fragment", 0)),
                )
                previous_priority = (
                    (
                        float(previous["plddt"]) if np.isfinite(previous["plddt"]) else -np.inf,
                        -int(previous.get("fragment", 0)),
                    )
                    if previous is not None
                    else None
                )
                if previous_priority is None or candidate_priority > previous_priority:
                    collected[key] = values
    sasa_values: list[float] = []
    relative_values: list[float] = []
    plddt_values: list[float] = []
    local_feature_values: dict[str, list[float]] = {name: [] for name in LOCAL_STRUCTURE_FEATURES}
    file_available: list[int] = []
    has_structure: list[int] = []
    statuses: list[str] = []
    for _, row in result.iterrows():
        accession = row.get("uniprot_id")
        try:
            position = int(row.get("aapos", row.get("aa_pos")))
        except (TypeError, ValueError):
            position = -1
        available = pd.notna(accession) and resources.structures.available(str(accession))
        values = collected.get((str(accession), position)) if available else None
        file_available.append(int(available))
        if values is None:
            sasa_values.append(np.nan)
            relative_values.append(np.nan)
            plddt_values.append(np.nan)
            for name in LOCAL_STRUCTURE_FEATURES:
                local_feature_values[name].append(np.nan)
            has_structure.append(0)
            statuses.append("residue_unavailable" if available else "structure_unavailable")
        else:
            sasa_values.append(values["sasa"])
            relative_values.append(values["relative_sasa"])
            plddt_values.append(values["plddt"])
            for name in LOCAL_STRUCTURE_FEATURES:
                local_feature_values[name].append(float(values.get(name, np.nan)))
            has_structure.append(1)
            statuses.append("mapped")
    result["SASA"] = np.asarray(sasa_values, dtype=np.float32)
    result["RELATIVE_SASA"] = np.asarray(relative_values, dtype=np.float32)
    result["PLDDT_SCORE"] = np.asarray(plddt_values, dtype=np.float32)
    for name, values in local_feature_values.items():
        result[name.upper()] = np.asarray(values, dtype=np.float32)
    result["STRUCTURE_FILE_AVAILABLE"] = np.asarray(file_available, dtype=np.int8)
    result["HAS_STRUCTURE"] = np.asarray(has_structure, dtype=np.int8)
    result["LOW_CONFIDENCE_STRUCTURE"] = (
        result["HAS_STRUCTURE"].eq(1) & result["PLDDT_SCORE"].lt(70)
    ).astype("int8")
    result["STRUCTURE_MAPPING_STATUS"] = statuses
    resources.structure_counts.update(statuses)
    return result


def annotate_variants(
    dataset: pd.DataFrame,
    uniprot_file: str | Path = UNIPROT_FILE,
    alphafold_dir: str | Path = ALPHAFOLD_DIR,
    n_workers: int | None = None,
    force_reparse: bool = False,
    resources: AnnotationResources | None = None,
) -> pd.DataFrame:
    """Apply the shared protein annotation transformer."""
    required = {"genename"}
    if not (
        {"aapos", "aaref"} <= set(dataset.columns) or {"aa_pos", "aa_ref"} <= set(dataset.columns)
    ):
        required |= {"aapos", "aaref"}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"Annotation input misses {sorted(missing)}")
    if resources is None:
        resources = build_annotation_resources(
            Path(uniprot_file), Path(alphafold_dir), force_reparse
        )
    result = map_variants_to_proteins(dataset, resources)
    result = annotate_functional_features(result, resources)
    result = annotate_structural_features(result, resources, workers=n_workers or CPU_COUNT)
    return result


def _write_coverage(resources: AnnotationResources, rows: int) -> None:
    payload = {
        "rows": rows,
        "mapping_status": dict(resources.mapping_counts),
        "structure_status": dict(resources.structure_counts),
        "structure_parse_status": dict(resources.structure_parse_counts),
        "structure_failures": resources.structure_failures,
    }
    temporary = COVERAGE_FILE.with_suffix(COVERAGE_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    temporary.replace(COVERAGE_FILE)


def enrich_dataset(
    input_csv: Path = INPUT_FILE,
    output_csv: Path = OUTPUT_FILE,
    uniprot_file: Path = UNIPROT_FILE,
    alphafold_dir: Path = ALPHAFOLD_DIR,
    workers: int = CPU_COUNT,
    chunksize: int = CHUNK_SIZE,
    max_rows: int | None = None,
    force_reparse: bool = False,
    validate_upstream: bool = True,
) -> None:
    """Map every candidate, then retain one eligible consequence per variant."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST,
            "03_remove_duplicates",
            [input_csv],
        )
    ensure_directories(STAGE04_OUT, output_csv.parent)
    resources = build_annotation_resources(Path(uniprot_file), Path(alphafold_dir), force_reparse)
    candidate_rows = 0
    total = 0
    class_counts = Counter()
    candidate_class_counts = Counter()
    selection_outcomes = Counter()
    variants_without_eligible_mapping = 0
    sampled_mapping_rows = 0
    seen_sequences: set[str] = set()
    with (
        AtomicParquetWriter(output_csv) as output_writer,
        AtomicParquetWriter(SEQUENCE_FILE) as sequence_writer,
        AtomicParquetWriter(MAPPING_FILE) as mapping_writer,
        AtomicParquetWriter(TRANSCRIPT_SELECTION_FILE) as selection_writer,
    ):
        for chunk_number, chunk in enumerate(
            _iter_complete_variant_chunks(input_csv, chunksize, max_rows), 1
        ):
            enriched = annotate_variants(
                chunk,
                resources=resources,
                n_workers=workers,
            )
            if len(enriched) != len(chunk):
                raise RuntimeError("Feature annotation changed row count")
            selected, selection_audit = select_primary_mapped_consequences(
                enriched,
                require_transcript_mapping=REQUIRE_TRANSCRIPT_MAPPING,
            )
            candidate_rows += len(enriched)
            candidate_class_counts.update(enriched[LABEL_COL].value_counts().to_dict())
            selection_outcomes.update(
                selection_audit["TRANSCRIPT_SELECTION_OUTCOME"].value_counts().to_dict()
            )
            variants_without_eligible_mapping += int(
                selection_audit.groupby("variant_id", sort=False)[
                    "SOURCE_GENE_CONCORDANT_MAPPED_CANDIDATE_COUNT"
                ]
                .first()
                .eq(0)
                .sum()
            )
            for label, count in selected[LABEL_COL].value_counts().items():
                class_counts[int(label)] += int(count)
            sequences = selected.loc[
                selected["protein_sequence"].notna(),
                ["uniprot_id", "protein_sequence"],
            ].drop_duplicates("uniprot_id")
            sequences = sequences[~sequences["uniprot_id"].astype(str).isin(seen_sequences)]
            if not sequences.empty:
                seen_sequences.update(sequences["uniprot_id"].astype(str))
                sequence_writer.write(sequences)
            output_writer.write(selected.drop(columns=["protein_sequence"], errors="ignore"))
            mapping_columns = [
                column
                for column in (
                    ROW_ID_COL,
                    "variant_id",
                    "genename",
                    "Ensembl_transcriptid",
                    "MANE",
                    "VEP_canonical",
                    "HGVSp_snpEff",
                    "HGVSc_snpEff",
                    "uniprot_id",
                    "CLINVAR_VARIATION_ID",
                    "CLINVAR_SOURCE_GENE",
                    "HAS_PROTEIN_MAPPING",
                    "PROTEIN_MAPPING_STATUS",
                    "MAPPING_TRANSCRIPT_MATCH",
                    "PRIMARY_MAPPING_ELIGIBLE",
                    "MAPPING_CONFIDENCE",
                    "UNIPROT_REVIEWED",
                    *TRANSCRIPT_SELECTION_COLS,
                    "TRANSCRIPT_SELECTION_OUTCOME",
                )
                if column in selection_audit.columns
            ]
            selection_writer.write(selection_audit[mapping_columns])
            if sampled_mapping_rows < AUDIT_SAMPLE_MAX_ROWS:
                sample = selection_audit[mapping_columns].head(
                    AUDIT_SAMPLE_MAX_ROWS - sampled_mapping_rows
                )
                mapping_writer.write(sample)
                sampled_mapping_rows += len(sample)
            total += len(selected)
            logger.info(
                "Annotated candidate group %d; selected %d of %d rows",
                chunk_number,
                total,
                candidate_rows,
            )
        if total == 0:
            raise RuntimeError("No variant has an eligible primary protein mapping")
        if set(class_counts) != {0, 1}:
            raise RuntimeError(f"Stage 04 classes are invalid: {dict(class_counts)}")
    _write_coverage(resources, candidate_rows)
    write_run_manifest(
        MANIFEST_FILE,
        "04_feature_engineering",
        [
            input_csv,
            UPSTREAM_MANIFEST,
            Path(uniprot_file),
            Path(alphafold_dir),
        ],
        {
            "alphafold_directory": str(Path(alphafold_dir)),
            "rows": total,
            "candidate_rows": candidate_rows,
            "class_counts": dict(class_counts),
            "candidate_class_counts": dict(candidate_class_counts),
            "mapping_status": dict(resources.mapping_counts),
            "require_transcript_mapping": REQUIRE_TRANSCRIPT_MAPPING,
            "transcript_selection_policy": TRANSCRIPT_SELECTION_POLICY,
            "selection_outcomes": dict(selection_outcomes),
            "variants_without_eligible_mapping": variants_without_eligible_mapping,
            "variant_unique_output": True,
            "upstream_validation": (
                "passed" if validate_upstream else "explicitly_skipped_nonpublication"
            ),
            "structure_status": dict(resources.structure_counts),
            "structure_parse_status": dict(resources.structure_parse_counts),
            "biopython_available": BIOPYTHON_AVAILABLE,
            "storage_format": "parquet_zstd",
            "sequence_table": str(SEQUENCE_FILE),
            "unique_sequences": len(seen_sequences),
            "mapping_sample_rows": sampled_mapping_rows,
            "transcript_selection_audit": str(TRANSCRIPT_SELECTION_FILE),
        },
        outputs=[
            output_csv,
            SEQUENCE_FILE,
            MAPPING_FILE,
            TRANSCRIPT_SELECTION_FILE,
            COVERAGE_FILE,
        ],
    )
    logger.info("Saved %d enriched rows", total)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--uniprot", type=Path, default=UNIPROT_FILE)
    parser.add_argument("--alphafold", type=Path, default=ALPHAFOLD_DIR)
    parser.add_argument("--workers", type=int, default=CPU_COUNT)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument(
        "--unsafe-skip-upstream-validation",
        action="store_true",
        help="Only for isolated development fixtures; never use for publication runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = build_args()
    enrich_dataset(
        input_csv=arguments.input,
        output_csv=arguments.output,
        uniprot_file=arguments.uniprot,
        alphafold_dir=arguments.alphafold,
        workers=arguments.workers,
        chunksize=arguments.chunksize,
        max_rows=arguments.max_rows,
        force_reparse=arguments.force_reparse,
        validate_upstream=not arguments.unsafe_skip_upstream_validation,
    )
