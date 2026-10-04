from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    HOMOLOGY_CLUSTER_MODE,
    HOMOLOGY_COVERAGE_MODE,
    HOMOLOGY_MIN_COVERAGE,
    HOMOLOGY_MIN_SEQUENCE_IDENTITY,
    STAGE08_OUT,
    ensure_directories,
    file_sha256,
    json_default,
    validate_upstream_manifest,
    write_run_manifest,
)
from schema import GENE_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage08b_homology")

INPUT_FILE = STAGE08_OUT / "internal_esm_ready.parquet"
SEQUENCE_FILE = STAGE08_OUT / "internal_sequences.parquet"
OUTPUT_FILE = STAGE08_OUT / "internal_esm_ready_homology.parquet"
GROUP_FILE = STAGE08_OUT / "homology_groups.csv"
SUMMARY_FILE = STAGE08_OUT / "homology_summary.json"
MANIFEST_FILE = STAGE08_OUT / "homology_run_manifest.json"
UPSTREAM_MANIFEST = STAGE08_OUT / "run_manifest.json"
CLUSTER_FILE = STAGE08_OUT / "mmseqs_cluster.tsv"
CLUSTER_PROVENANCE_FILE = STAGE08_OUT / "mmseqs_cluster.provenance.json"
PROVENANCE_SCHEMA_VERSION = 1


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def _write_fasta(sequences: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in sequences.itertuples(index=False):
            sequence_hash = str(row.sequence_hash)
            sequence = str(row.protein_sequence).strip().upper()
            if not sequence:
                raise ValueError(f"Empty protein sequence for {sequence_hash}")
            handle.write(f">{sequence_hash}\n{sequence}\n")


def _sequence_set_sha256(sequences: pd.DataFrame) -> str:
    """Bind the exact sequence universe represented by MMseqs identifiers."""
    required = {"sequence_hash", "protein_sequence"}
    missing = required - set(sequences.columns)
    if missing:
        raise KeyError(f"Sequence table misses {sorted(missing)}")
    if sequences["sequence_hash"].astype(str).duplicated().any():
        raise ValueError("Sequence table contains duplicate sequence hashes")
    records: list[tuple[str, str]] = []
    for row in sequences.itertuples(index=False):
        sequence_hash = str(row.sequence_hash)
        sequence = str(row.protein_sequence).strip().upper()
        observed_hash = hashlib.sha256(sequence.encode()).hexdigest()
        if not sequence or sequence_hash != observed_hash:
            raise ValueError(
                f"Sequence hash/content mismatch for {sequence_hash or '<empty>'}"
            )
        records.append((sequence_hash, sequence))
    digest = hashlib.sha256()
    for sequence_hash, sequence in sorted(records):
        digest.update(sequence_hash.encode())
        digest.update(b"\0")
        digest.update(str(len(sequence)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _mmseqs_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to record the MMseqs2 version") from error
    version = (result.stdout or result.stderr).strip().splitlines()
    if not version:
        raise RuntimeError("MMseqs2 returned an empty version string")
    return version[0]


def _provenance_payload(
    cluster_tsv: Path,
    sequence_set_sha256: str,
    sequence_count: int,
    min_identity: float,
    coverage: float,
    mmseqs_version: str,
) -> dict[str, Any]:
    cluster_hash = file_sha256(cluster_tsv, force=True)
    if cluster_hash is None:
        raise RuntimeError(f"Unable to hash MMseqs cluster TSV: {cluster_tsv}")
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "sequence_set_sha256": sequence_set_sha256,
        "sequence_count": int(sequence_count),
        "cluster_tsv_sha256": cluster_hash,
        "min_sequence_identity": float(min_identity),
        "minimum_coverage": float(coverage),
        "coverage_mode": HOMOLOGY_COVERAGE_MODE,
        "cluster_mode": HOMOLOGY_CLUSTER_MODE,
        "mmseqs_version": mmseqs_version,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def _validate_cluster_provenance(
    provenance_file: Path,
    cluster_tsv: Path,
    sequence_set_sha256: str,
    sequence_count: int,
    min_identity: float,
    coverage: float,
) -> dict[str, Any]:
    if not provenance_file.is_file():
        raise FileNotFoundError(
            "Precomputed MMseqs TSV requires its provenance JSON: "
            f"{provenance_file}"
        )
    try:
        payload = json.loads(provenance_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid MMseqs provenance: {provenance_file}") from error
    expected = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "sequence_set_sha256": sequence_set_sha256,
        "sequence_count": int(sequence_count),
        "cluster_tsv_sha256": file_sha256(cluster_tsv, force=True),
        "min_sequence_identity": float(min_identity),
        "minimum_coverage": float(coverage),
        "coverage_mode": HOMOLOGY_COVERAGE_MODE,
        "cluster_mode": HOMOLOGY_CLUSTER_MODE,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "MMseqs TSV provenance does not match the current sequence universe "
            f"or clustering policy: {mismatches}"
        )
    if not str(payload.get("mmseqs_version", "")).strip():
        raise RuntimeError("MMseqs provenance lacks a tool version")
    return payload


def _run_mmseqs(
    sequences: pd.DataFrame,
    executable: str,
    min_identity: float,
    coverage: float,
    threads: int,
) -> Path:
    ensure_directories(STAGE08_OUT)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="mmseqs_", dir=str(STAGE08_OUT.resolve()))
    )
    fasta = temporary_root / "sequences.fasta"
    prefix = temporary_root / "clusters"
    work = temporary_root / "work"
    persistent = CLUSTER_FILE
    try:
        _write_fasta(sequences, fasta)
        command = [
            executable,
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(work),
            "--min-seq-id",
            str(min_identity),
            "-c",
            str(coverage),
            "--cov-mode",
            str(HOMOLOGY_COVERAGE_MODE),
            "--cluster-mode",
            str(HOMOLOGY_CLUSTER_MODE),
            "--threads",
            str(threads),
        ]
        logger.info("Running MMseqs2 on %d unique sequences", len(sequences))
        subprocess.run(command, check=True)
        cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
        if not cluster_tsv.exists():
            raise RuntimeError(f"MMseqs2 did not create {cluster_tsv}")
        shutil.copy2(cluster_tsv, persistent)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return persistent


def _read_clusters(path: Path, sequence_hashes: set[str]) -> dict[str, str]:
    cluster_map: dict[str, str] = {}
    seen_members: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Invalid MMseqs cluster TSV line {line_number}")
            representative, member = fields[0], fields[1]
            if representative not in sequence_hashes or member not in sequence_hashes:
                raise ValueError(
                    "MMseqs TSV contains an identifier outside the current sequence "
                    f"universe at line {line_number}: {representative}, {member}"
                )
            seen_members.add(member)
            previous = cluster_map.setdefault(member, representative)
            if previous != representative:
                raise ValueError(f"Sequence {member} belongs to multiple clusters")
    missing_members = sequence_hashes - seen_members
    if missing_members:
        preview = sorted(missing_members)[:5]
        raise ValueError(
            f"MMseqs TSV omits {len(missing_members)} current sequences; examples: "
            f"{preview}"
        )
    invalid_representatives = [
        representative
        for representative in set(cluster_map.values())
        if cluster_map.get(representative) != representative
    ]
    if invalid_representatives:
        raise ValueError(
            "MMseqs representatives must occur as self-members: "
            f"{sorted(invalid_representatives)[:5]}"
        )
    return cluster_map


def _connected_split_groups(
    rows: pd.DataFrame, cluster_map: dict[str, str]
) -> pd.DataFrame:
    required = {"sequence_hash", GENE_COL}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"Homology input misses {sorted(missing)}")
    mapping = rows[["sequence_hash", GENE_COL]].drop_duplicates().copy()
    mapping["homology_cluster"] = mapping["sequence_hash"].astype(str).map(cluster_map)
    if mapping["homology_cluster"].isna().any():
        raise ValueError("Some model rows lack a homology cluster")
    union_find = UnionFind()
    for row in mapping.itertuples(index=False):
        union_find.union(
            f"gene:{getattr(row, GENE_COL)}",
            f"cluster:{row.homology_cluster}",
        )
    component_ids: dict[str, str] = {}
    split_groups: list[str] = []
    for row in mapping.itertuples(index=False):
        root = union_find.find(f"gene:{getattr(row, GENE_COL)}")
        component_ids.setdefault(
            root, "HC_" + hashlib.sha256(root.encode()).hexdigest()[:16]
        )
        split_groups.append(component_ids[root])
    mapping["split_group"] = split_groups
    return mapping


def build_homology_groups(
    cluster_tsv: Path | None = None,
    cluster_provenance: Path | None = None,
    executable: str = "mmseqs",
    min_identity: float = HOMOLOGY_MIN_SEQUENCE_IDENTITY,
    coverage: float = HOMOLOGY_MIN_COVERAGE,
    threads: int = 1,
    *,
    validate_upstream: bool = True,
) -> None:
    """Create connected gene/homology groups and a new immutable Stage 08 table."""
    for path in (INPUT_FILE, SEQUENCE_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
    if validate_upstream:
        validate_upstream_manifest(
            UPSTREAM_MANIFEST,
            "08_prepare_esm_dataset",
            [INPUT_FILE, SEQUENCE_FILE],
        )
    if not 0.0 < min_identity <= 1.0 or not 0.0 < coverage <= 1.0:
        raise ValueError("Identity and coverage must be in (0, 1]")
    if threads < 1:
        raise ValueError("MMseqs2 threads must be at least 1")
    ensure_directories(STAGE08_OUT)
    rows = pd.read_parquet(INPUT_FILE)
    sequences = pd.read_parquet(SEQUENCE_FILE)
    sequence_set_hash = _sequence_set_sha256(sequences)
    sequence_hashes = set(sequences["sequence_hash"].astype(str))
    generated_cluster_tsv = cluster_tsv is None
    if cluster_tsv is None:
        resolved_executable = shutil.which(executable)
        if resolved_executable is None:
            raise RuntimeError(
                "MMseqs2 is required for publication splits. Install `mmseqs`, "
                "or pass --cluster-tsv with a precomputed easy-cluster TSV."
            )
        cluster_tsv = _run_mmseqs(
            sequences, resolved_executable, min_identity, coverage, threads
        )
        cluster_provenance = CLUSTER_PROVENANCE_FILE
        provenance = _provenance_payload(
            cluster_tsv,
            sequence_set_hash,
            len(sequence_hashes),
            min_identity,
            coverage,
            _mmseqs_version(resolved_executable),
        )
        _write_json_atomic(cluster_provenance, provenance)
    cluster_tsv = Path(cluster_tsv).resolve()
    if not cluster_tsv.exists():
        raise FileNotFoundError(cluster_tsv)
    if cluster_provenance is None:
        cluster_provenance = cluster_tsv.with_suffix(".provenance.json")
    cluster_provenance = Path(cluster_provenance).resolve()
    provenance = _validate_cluster_provenance(
        cluster_provenance,
        cluster_tsv,
        sequence_set_hash,
        len(sequence_hashes),
        min_identity,
        coverage,
    )
    cluster_map = _read_clusters(cluster_tsv, sequence_hashes)
    mapping = _connected_split_groups(rows, cluster_map)
    output = rows.drop(columns=["split_group", "homology_cluster"], errors="ignore")
    output = output.merge(mapping, on=["sequence_hash", GENE_COL], how="left", validate="many_to_one")
    if output["split_group"].isna().any() or len(output) != len(rows):
        raise RuntimeError("Homology grouping changed or failed to map model rows")
    output_temporary = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    group_temporary = GROUP_FILE.with_suffix(GROUP_FILE.suffix + ".tmp")
    output.to_parquet(output_temporary, index=False, compression="zstd")
    mapping.sort_values(["split_group", GENE_COL, "sequence_hash"]).to_csv(
        group_temporary, index=False
    )
    output_temporary.replace(OUTPUT_FILE)
    group_temporary.replace(GROUP_FILE)
    summary = {
        "rows": len(output),
        "genes": int(output[GENE_COL].astype(str).nunique()),
        "unique_sequences": len(sequence_hashes),
        "homology_clusters": int(mapping["homology_cluster"].nunique()),
        "connected_split_groups": int(mapping["split_group"].nunique()),
        "min_sequence_identity": min_identity,
        "minimum_coverage": coverage,
        "coverage_mode": HOMOLOGY_COVERAGE_MODE,
        "cluster_mode": HOMOLOGY_CLUSTER_MODE,
        "cluster_tsv": str(cluster_tsv),
        "cluster_tsv_sha256": provenance["cluster_tsv_sha256"],
        "cluster_provenance": str(cluster_provenance),
        "sequence_set_sha256": sequence_set_hash,
        "member_universe_validation": "exact",
        "mmseqs_version": provenance["mmseqs_version"],
        "mmseqs_threads": threads,
        "upstream_validation": (
            "passed" if validate_upstream else "explicitly_skipped_nonpublication"
        ),
    }
    temporary_summary = SUMMARY_FILE.with_suffix(SUMMARY_FILE.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, default=json_default), encoding="utf-8"
    )
    temporary_summary.replace(SUMMARY_FILE)
    write_run_manifest(
        MANIFEST_FILE,
        "08b_build_homology_groups",
        [
            INPUT_FILE,
            SEQUENCE_FILE,
            UPSTREAM_MANIFEST,
            cluster_tsv,
            cluster_provenance,
        ],
        summary,
        outputs=[
            OUTPUT_FILE,
            GROUP_FILE,
            SUMMARY_FILE,
            *([cluster_tsv, cluster_provenance] if generated_cluster_tsv else []),
        ],
    )
    logger.info(
        "Created %d connected gene/homology split groups",
        summary["connected_split_groups"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    cluster_default = os.environ.get("HOMOLOGY_CLUSTER_TSV")
    parser.add_argument(
        "--cluster-tsv",
        type=Path,
        default=Path(cluster_default) if cluster_default else None,
    )
    parser.add_argument("--cluster-provenance", type=Path)
    parser.add_argument(
        "--mmseqs", default=os.environ.get("MMSEQS_EXECUTABLE", "mmseqs")
    )
    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=HOMOLOGY_MIN_SEQUENCE_IDENTITY,
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=HOMOLOGY_MIN_COVERAGE,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("MMSEQS_THREADS", "1")),
        help="MMseqs2 CPU threads; default 1 to limit memory on Windows",
    )
    arguments = parser.parse_args()
    build_homology_groups(
        cluster_tsv=arguments.cluster_tsv,
        cluster_provenance=arguments.cluster_provenance,
        executable=arguments.mmseqs,
        min_identity=arguments.min_seq_id,
        coverage=arguments.coverage,
        threads=arguments.threads,
    )


if __name__ == "__main__":
    main()
