"""Stage 04 - structural and functional feature engineering (AlphaFold, sequence, FCGR)."""

import pandas as pd
from config import DATA_DIR, STAGE03_OUT, STAGE04_OUT
import numpy as np
from pathlib import Path
import gzip
import re
import pickle
import hashlib
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try importing optional GPU acceleration
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # Fallback tqdm
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_postfix(self, **kwargs):
            pass


try:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: Biopython not available. Install with: pip install biopython")

# Detect optimal worker count for AMD Ryzen 7 7700
CPU_COUNT = min(mp.cpu_count(), 16)  # Cap at 16 for Ryzen 7 7700
IO_WORKERS = min(CPU_COUNT * 2, 32)  # IO-bound tasks can use more threads


@dataclass
class ProteinFeatures:
    """Memory-efficient protein feature storage using dataclass"""

    gene_name: str = ""
    uniprot_id: str = ""
    domains: List[Dict] = None
    active_sites: List[Dict] = None
    binding_sites: List[Dict] = None
    transmembrane: List[Dict] = None
    signal_peptide: bool = False

    def __post_init__(self):
        if self.domains is None:
            self.domains = []
        if self.active_sites is None:
            self.active_sites = []
        if self.binding_sites is None:
            self.binding_sites = []
        if self.transmembrane is None:
            self.transmembrane = []


class OptimizedUniProtParser:

    CACHE_VERSION = "v2"  # Increment when changing parsing logic

    def __init__(self, uniprot_file: str, cache_dir: str = None):
        self.uniprot_file = Path(uniprot_file)
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.uniprot_file.parent / ".cache"
        )
        self.cache_dir.mkdir(exist_ok=True)

        # Pre-compiled regex patterns for speed
        self.gene_pattern = re.compile(r"Name=([^;{\s]+)")
        self.coord_pattern = re.compile(r"(\d+)(?:\.\.(\d+))?")

        self.protein_features: Dict[str, ProteinFeatures] = {}

    def _get_cache_path(self) -> Path:
        """Generate cache filename based on file hash"""
        # Use file size + modification time for quick hash (faster than full content hash)
        stat = self.uniprot_file.stat()
        cache_key = f"{self.uniprot_file.name}_{stat.st_size}_{stat.st_mtime}_{self.CACHE_VERSION}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"uniprot_cache_{cache_hash}.pkl"

    def _load_from_cache(self) -> bool:
        """Try to load parsed features from cache"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                print(f"Loading cached UniProt features from {cache_path.name}...")
                with open(cache_path, "rb") as f:
                    self.protein_features = pickle.load(f)
                print(f"  Loaded {len(self.protein_features):,} proteins from cache")
                return True
            except Exception as e:
                print(f"  Cache load failed: {e}, re-parsing...")
        return False

    def _save_to_cache(self):
        """Save parsed features to cache"""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.protein_features, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved cache to {cache_path.name}")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")

    def parse(self, force_reparse: bool = False) -> Dict[str, ProteinFeatures]:

        if not force_reparse and self._load_from_cache():
            return self.protein_features

        print(f"Parsing UniProt file: {self.uniprot_file.name}...")

        current_protein = None
        current_features = None
        count = 0

        # Pre-allocate with estimated size (human proteome ~20k proteins)
        self.protein_features = {}

        with open(self.uniprot_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n\r")

                if line.startswith("ID"):
                    parts = line.split(None, 2)
                    if len(parts) > 1:
                        current_protein = parts[1]
                        current_features = ProteinFeatures()
                        self.protein_features[current_protein] = current_features
                        count += 1
                        if count % 5000 == 0:
                            print(f"  Parsed {count:,} proteins...", end="\r")

                elif line.startswith("GN") and current_features:
                    match = self.gene_pattern.search(line)
                    if match:
                        current_features.gene_name = match.group(1)

                elif line.startswith("AC") and current_features:
                    parts = line.split(None, 2)
                    if len(parts) > 1:
                        current_features.uniprot_id = parts[1].rstrip(";")

                elif line.startswith("FT") and current_features:
                    self._parse_feature_line(line, current_features)

        print(f"\nParsed {count:,} proteins with functional annotations")
        self._save_to_cache()
        return self.protein_features

    def _parse_feature_line(self, line: str, features: ProteinFeatures):
        """Parse individual FT line with optimized regex"""
        parts = line.split(None, 3)
        if len(parts) < 3:
            return

        feature_type = parts[1]
        coords = parts[2]

        # Skip irrelevant feature types early
        if feature_type not in ("DOMAIN", "ACT_SITE", "BINDING", "TRANSMEM", "SIGNAL"):
            return

        try:
            match = self.coord_pattern.match(coords)
            if not match:
                return

            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            description = parts[3] if len(parts) > 3 else ""

            if feature_type == "DOMAIN":
                features.domains.append(
                    {"start": start, "end": end, "name": description[:50]}
                )
            elif feature_type == "ACT_SITE":
                features.active_sites.append(
                    {"position": start, "description": description}
                )
            elif feature_type == "BINDING":
                features.binding_sites.append(
                    {"position": start, "description": description}
                )
            elif feature_type == "TRANSMEM":
                features.transmembrane.append({"start": start, "end": end})
            elif feature_type == "SIGNAL":
                features.signal_peptide = True

        except (ValueError, AttributeError):
            pass


class AlphaFoldStructureIndex:

    def __init__(self, alphafold_dir: str):
        self.alphafold_dir = Path(alphafold_dir)
        self.cache_file = self.alphafold_dir / ".pdb_index.pkl"
        self.index: Dict[str, Path] = {}
        self._build_index()

    def _build_index(self):
        """Build or load PDB file index"""
        # Try loading cached index
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cached = pickle.load(f)
                    if cached.get("dir_mtime") == os.path.getmtime(self.alphafold_dir):
                        self.index = cached["index"]
                        print(f"Loaded PDB index: {len(self.index):,} structures")
                        return
            except Exception:
                pass

        print(f"Building AlphaFold structure index for {self.alphafold_dir}...")

        # Index all PDB and CIF files
        pdb_files = list(self.alphafold_dir.glob("AF-*-model_v*.pdb"))
        cif_files = list(self.alphafold_dir.glob("AF-*-model_v*.cif"))

        # Prefer PDB over CIF, prefer F1 fragment
        file_priority = {}

        for pdb_file in tqdm(pdb_files + cif_files, desc="Indexing structures"):
            # Parse filename: AF-{UniProtID}-F{fragment}-model_v{version}.{ext}
            name = pdb_file.stem
            parts = name.split("-")
            if len(parts) >= 3:
                uniprot_id = parts[1]
                fragment = parts[2] if len(parts) > 2 else "F1"
                is_pdb = pdb_file.suffix == ".pdb"

                # Priority: PDB > CIF, F1 > F2 > ...
                priority = (is_pdb, fragment == "F1", fragment)

                if (
                    uniprot_id not in file_priority
                    or priority > file_priority[uniprot_id][0]
                ):
                    file_priority[uniprot_id] = (priority, pdb_file)

        self.index = {uid: path.name for uid, (_, path) in file_priority.items()}

        # Cache the index
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(
                    {
                        "index": self.index,
                        "dir_mtime": os.path.getmtime(self.alphafold_dir),
                    },
                    f,
                )
        except Exception:
            pass

        print(f"Indexed {len(self.index):,} unique structures")

    def get_file(self, uniprot_id: str) -> Optional[Path]:
        """O(1) lookup for PDB file"""
        filename = self.index.get(uniprot_id)
        if filename:
            # Handle backward compatibility if the cache has absolute/relative paths
            if isinstance(filename, Path):
                filename = filename.name
            return self.alphafold_dir / filename
        return None

    def get_available_ids(self) -> set:
        """Return set of all available UniProt IDs"""
        return set(self.index.keys())


# Thread-local storage for PDB parser (thread-safe, no subprocess spawning)
import threading

_thread_local = threading.local()


def _get_parser():
    """Get or create a thread-local PDB parser"""
    if not hasattr(_thread_local, "parser"):
        _thread_local.parser = PDBParser(QUIET=True) if BIOPYTHON_AVAILABLE else None
    return _thread_local.parser


# Keep module-level parser for backward compatibility
_global_parser = None


def _init_worker():
    """Initialize worker process with PDB parser (kept for compatibility)"""
    global _global_parser
    if BIOPYTHON_AVAILABLE:
        _global_parser = PDBParser(QUIET=True)


def _process_structure_batch(
    args: Tuple[str, Path, List[Tuple[int, int]]],
) -> List[Tuple[int, Dict]]:

    uniprot_id, pdb_path, residue_batch = args
    results = []

    # Use thread-local parser (for ThreadPoolExecutor) or global parser (for ProcessPoolExecutor)
    parser = _get_parser()
    if not BIOPYTHON_AVAILABLE or parser is None:
        return [
            (idx, {"sasa": -1, "relative_sasa": -1, "plddt": -1})
            for idx, _ in residue_batch
        ]

    try:
        # Parse structure once for all residues
        if str(pdb_path).endswith(".gz"):
            with gzip.open(pdb_path, "rt") as f:
                structure = parser.get_structure(uniprot_id, f)
        else:
            structure = parser.get_structure(uniprot_id, str(pdb_path))

        # Compute SASA once for entire structure
        sr = ShrakeRupley()
        sr.compute(structure, level="R")

        # Build residue lookup for O(1) access
        residue_map = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_num = residue.id[1]
                    residue_map[res_num] = residue

        # Max SASA values for relative calculation
        max_sasa = {
            "ALA": 121,
            "ARG": 265,
            "ASN": 187,
            "ASP": 187,
            "CYS": 148,
            "GLN": 214,
            "GLU": 214,
            "GLY": 97,
            "HIS": 216,
            "ILE": 195,
            "LEU": 191,
            "LYS": 230,
            "MET": 203,
            "PHE": 228,
            "PRO": 154,
            "SER": 143,
            "THR": 163,
            "TRP": 264,
            "TYR": 255,
            "VAL": 165,
        }

        # Process each residue
        for row_idx, position in residue_batch:
            if position in residue_map:
                residue = residue_map[position]
                sasa = getattr(residue, "sasa", -1)

                # Get residue type and calculate relative SASA
                res_name = residue.get_resname()
                max_val = max_sasa.get(res_name, 200)
                relative_sasa = sasa / max_val if sasa > 0 else -1

                # Get pLDDT from B-factor
                atoms = list(residue.get_atoms())
                plddt = atoms[0].get_bfactor() if atoms else -1

                results.append(
                    (
                        row_idx,
                        {
                            "sasa": round(sasa, 2) if sasa > 0 else -1,
                            "relative_sasa": (
                                round(relative_sasa, 3) if relative_sasa > 0 else -1
                            ),
                            "plddt": round(plddt, 2) if plddt > 0 else -1,
                        },
                    )
                )
            else:
                results.append(
                    (row_idx, {"sasa": -1, "relative_sasa": -1, "plddt": -1})
                )

    except Exception:
        # Return defaults for all residues on error
        results = [
            (idx, {"sasa": -1, "relative_sasa": -1, "plddt": -1})
            for idx, _ in residue_batch
        ]

    return results


def annotate_with_uniprot_vectorized(
    dataset: pd.DataFrame,
    uniprot_features: Dict[str, ProteinFeatures],
    gene_column: str = "genename",
    pos_column: str = "aapos",
) -> pd.DataFrame:

    print("Annotating variants with UniProt features (vectorized)...")

    # Build gene -> protein mapping
    gene_to_protein: Dict[str, ProteinFeatures] = {}
    for prot_id, features in uniprot_features.items():
        if features.gene_name:
            gene_to_protein[features.gene_name] = features

    n = len(dataset)

    # Pre-allocate output arrays
    is_in_domain = np.zeros(n, dtype=np.int8)
    domain_names = np.empty(n, dtype=object)
    domain_names.fill("")
    distance_to_active = np.full(n, 999, dtype=np.int16)
    is_active_site = np.zeros(n, dtype=np.int8)
    is_binding_site = np.zeros(n, dtype=np.int8)
    is_transmembrane = np.zeros(n, dtype=np.int8)

    # Get relevant columns as numpy arrays for speed
    genes = dataset[gene_column].values
    positions = pd.to_numeric(dataset[pos_column], errors="coerce").values

    # Process by gene (better cache locality)
    unique_genes = pd.unique(genes[pd.notna(genes)])
    matched_genes = set(unique_genes) & set(gene_to_protein.keys())

    print(f"  Found {len(matched_genes):,} / {len(unique_genes):,} genes in UniProt")

    annotated_count = 0
    for gene in tqdm(matched_genes, desc="  Annotating genes"):
        features = gene_to_protein[gene]

        # Find all rows for this gene
        gene_mask = (genes == gene) & pd.notna(positions)
        gene_indices = np.where(gene_mask)[0]
        gene_positions = positions[gene_mask].astype(int)

        for i, (idx, pos) in enumerate(zip(gene_indices, gene_positions)):
            # Domain check
            for domain in features.domains:
                if domain["start"] <= pos <= domain["end"]:
                    is_in_domain[idx] = 1
                    domain_names[idx] = domain["name"]
                    break

            # Active site distance
            min_distance = 999
            for active_site in features.active_sites:
                if active_site["position"] == pos:
                    is_active_site[idx] = 1
                    min_distance = 0
                    break
                distance = abs(active_site["position"] - pos)
                min_distance = min(min_distance, distance)
            distance_to_active[idx] = min_distance

            # Binding site check
            for binding_site in features.binding_sites:
                if binding_site["position"] == pos:
                    is_binding_site[idx] = 1
                    break

            # Transmembrane check
            for tm in features.transmembrane:
                if tm["start"] <= pos <= tm["end"]:
                    is_transmembrane[idx] = 1
                    break

            annotated_count += 1

    print(f"\n  Annotated {annotated_count:,} variants with UniProt features")

    # Add to dataset
    dataset["IS_IN_DOMAIN"] = is_in_domain
    dataset["DOMAIN_NAME"] = domain_names
    dataset["DISTANCE_TO_ACTIVE_SITE"] = np.clip(distance_to_active, 0, 100)
    dataset["IS_ACTIVE_SITE"] = is_active_site
    dataset["IS_BINDING_SITE"] = is_binding_site
    dataset["IS_TRANSMEMBRANE"] = is_transmembrane

    return dataset


def add_structural_features_parallel(
    dataset: pd.DataFrame,
    alphafold_index: AlphaFoldStructureIndex,
    uniprot_features: Dict[str, ProteinFeatures],
    gene_column: str = "genename",
    pos_column: str = "aapos",
    n_workers: int = None,
) -> pd.DataFrame:
    """Add AlphaFold structural features using ThreadPoolExecutor.

    Uses threads instead of processes to avoid Windows memory issues
    where each spawned process re-imports pandas/numpy and exhausts
    virtual memory (paging file too small error).
    """

    if not BIOPYTHON_AVAILABLE:
        print("Warning: Biopython not available. Skipping structural features.")
        dataset["SASA"] = -1
        dataset["RELATIVE_SASA"] = -1
        dataset["PLDDT_SCORE"] = -1
        dataset["SECONDARY_STRUCTURE"] = "U"
        return dataset

    # Use fewer workers for threads since they share memory and GIL
    n_workers = min(n_workers or CPU_COUNT, 8)
    print(f"Adding AlphaFold structural features ({n_workers} threads)...")

    # Build gene -> UniProt ID mapping
    gene_to_uniprot = {}
    for features in uniprot_features.values():
        if features.gene_name and features.uniprot_id:
            gene_to_uniprot[features.gene_name] = features.uniprot_id

    n = len(dataset)
    genes = dataset[gene_column].values
    positions = pd.to_numeric(dataset[pos_column], errors="coerce").values

    # Pre-allocate result arrays
    sasa_values = np.full(n, -1.0, dtype=np.float32)
    relative_sasa_values = np.full(n, -1.0, dtype=np.float32)
    plddt_values = np.full(n, -1.0, dtype=np.float32)

    # Group rows by structure file for batch processing
    structure_batches: Dict[str, Tuple[Path, List[Tuple[int, int]]]] = {}
    available_ids = alphafold_index.get_available_ids()

    for idx in range(n):
        gene = genes[idx]
        pos = positions[idx]

        if pd.isna(pos) or pd.isna(gene) or gene not in gene_to_uniprot:
            continue

        uniprot_id = gene_to_uniprot[gene]
        if uniprot_id not in available_ids:
            continue

        pdb_path = alphafold_index.get_file(uniprot_id)
        if pdb_path is None:
            continue

        if uniprot_id not in structure_batches:
            structure_batches[uniprot_id] = (pdb_path, [])
        structure_batches[uniprot_id][1].append((idx, int(pos)))

    total_variants = sum(len(b[1]) for b in structure_batches.values())
    print(
        f"  Processing {len(structure_batches):,} structures for {total_variants:,} variants"
    )

    # Prepare work items
    work_items = [
        (uid, path, residues) for uid, (path, residues) in structure_batches.items()
    ]

    # Process with ThreadPoolExecutor (no subprocess spawning = no memory explosion)
    all_results = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_structure_batch, item): item[0]
            for item in work_items
        }

        with tqdm(total=len(futures), desc="  Processing structures") as pbar:
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    results = future.result(timeout=300)  # 5 min timeout per structure
                    all_results.extend(results)
                except Exception:
                    failed_count += 1
                pbar.update(1)

    # Collect results
    for row_idx, features in all_results:
        sasa_values[row_idx] = features["sasa"]
        relative_sasa_values[row_idx] = features["relative_sasa"]
        plddt_values[row_idx] = features["plddt"]

    successful = np.sum(sasa_values > -1)
    print(
        f"\n  Successfully annotated {successful:,} / {n:,} variants with structural features"
    )
    if failed_count > 0:
        print(f"  {failed_count} structures failed during processing")

    dataset["SASA"] = sasa_values
    dataset["RELATIVE_SASA"] = relative_sasa_values
    dataset["PLDDT_SCORE"] = plddt_values
    dataset["SECONDARY_STRUCTURE"] = "U"  # DSSP requires additional setup

    return dataset


def build_smart_gene_mapping_optimized(
    uniprot_features: Dict[str, ProteinFeatures],
    alphafold_index: AlphaFoldStructureIndex,
) -> Dict[str, ProteinFeatures]:

    print("\nBuilding smart gene-to-UniProt mapping...")

    available_ids = alphafold_index.get_available_ids()
    print(f"  Available structures: {len(available_ids):,}")

    # Group proteins by gene, prefer those with structures
    gene_candidates: Dict[str, List[Tuple[ProteinFeatures, bool]]] = defaultdict(list)

    for prot_id, features in uniprot_features.items():
        if features.gene_name:
            has_structure = features.uniprot_id in available_ids
            gene_candidates[features.gene_name].append((features, has_structure))

    # Select best protein for each gene
    optimized_features = {}
    genes_with_structure = 0

    for gene, candidates in gene_candidates.items():
        # Sort: prefer those with structures
        candidates.sort(key=lambda x: (x[1], x[0].uniprot_id), reverse=True)
        best = candidates[0][0]
        optimized_features[best.uniprot_id] = best
        if candidates[0][1]:
            genes_with_structure += 1

    print(
        f"  Mapped {len(optimized_features):,} genes ({genes_with_structure:,} with structures)"
    )

    return optimized_features


def enrich_dataset_with_structure_function_optimized(
    input_csv: str = str(
        STAGE03_OUT / "somatic_variant_dbNSFP_Removes_missing_values_Deduplication.csv"
    ),
    uniprot_file: str = str(DATA_DIR / "uniprotkb_proteome_UP000005640_2026_01_07.txt"),
    alphafold_dir: str = str(DATA_DIR / "UP000005640_9606_HUMAN_v6"),
    output_csv: str = str(
        STAGE04_OUT
        / "somatic_variant_dbNSFP_Removes_missing_values_Deduplication_Structural_Functional.csv"
    ),
    n_workers: int = None,
    max_rows: int = None,
    force_reparse: bool = False,
):

    print("STRUCTURAL & FUNCTIONAL ENRICHMENT PIPELINE - OPTIMIZED")
    print(f"CPU Workers: {n_workers or CPU_COUNT}")
    print(
        f"GPU Acceleration: {'cuDF available' if CUDF_AVAILABLE else 'Not available'}"
    )

    # Load dataset
    print("\nLoading base dataset...")
    if CUDF_AVAILABLE:
        try:
            dataset = cudf.read_csv(input_csv)
            if max_rows:
                dataset = dataset.head(max_rows)
            dataset = dataset.to_pandas()  # Convert back for compatibility
            print(f"  Loaded with cuDF acceleration: {len(dataset):,} variants")
        except Exception:
            dataset = pd.read_csv(input_csv)
            if max_rows:
                dataset = dataset.head(max_rows)
            print(f"  Loaded with pandas: {len(dataset):,} variants")
    else:
        dataset = pd.read_csv(input_csv)
        if max_rows:
            dataset = dataset.head(max_rows)
        print(f"  Loaded {len(dataset):,} variants")

    # Parse UniProt with caching
    print("\nSTEP 1: UniProt Functional Annotation")
    parser = OptimizedUniProtParser(uniprot_file)
    uniprot_features = parser.parse(force_reparse=force_reparse)

    # Build AlphaFold index
    print("\nSTEP 2: Building AlphaFold Structure Index")
    alphafold_index = AlphaFoldStructureIndex(alphafold_dir)

    # Smart mapping
    optimized_features = build_smart_gene_mapping_optimized(
        uniprot_features, alphafold_index
    )

    # Annotate with UniProt features (vectorized)
    print("\nSTEP 3: UniProt Annotation")
    dataset = annotate_with_uniprot_vectorized(dataset, optimized_features)

    # Add structural features (parallel)
    print("\nSTEP 4: AlphaFold Structural Features")
    dataset = add_structural_features_parallel(
        dataset, alphafold_index, optimized_features, n_workers=n_workers
    )

    # Save results
    print("\nSaving enriched dataset...")
    dataset.to_csv(output_csv, index=False)
    print(f"  Saved to: {output_csv}")

    # Summary statistics
    print("\nENRICHMENT SUMMARY")

    print(f"Total variants: {len(dataset):,}")

    if "IS_IN_DOMAIN" in dataset.columns:
        print(f"In functional domain: {dataset['IS_IN_DOMAIN'].sum():,}")
    if "IS_ACTIVE_SITE" in dataset.columns:
        print(f"At active site: {dataset['IS_ACTIVE_SITE'].sum():,}")
    if "IS_BINDING_SITE" in dataset.columns:
        print(f"At binding site: {dataset['IS_BINDING_SITE'].sum():,}")
    if "SASA" in dataset.columns:
        structural = dataset[dataset["SASA"] > -1]
        print(
            f"With structural features: {len(structural):,} ({100*len(structural)/len(dataset):.1f}%)"
        )

    return dataset


# Alternative GPU-accelerated annotation (if cuDF available)
def annotate_with_cudf(dataset_path: str, output_path: str) -> None:

    if not CUDF_AVAILABLE:
        print("cuDF not available. Install with: conda install -c rapidsai cudf")
        return

    print("Using GPU-accelerated cuDF for dataframe operations...")
    gdf = cudf.read_csv(dataset_path)

    # GPU-accelerated operations would go here
    # (filtering, merging, aggregations, etc.)

    gdf.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Optimized Bioinformatics Feature Engineering Pipeline"
    )
    arg_parser.add_argument(
        "--input",
        "-i",
        default=str(
            STAGE03_OUT
            / "somatic_variant_dbNSFP_Removes_missing_values_Deduplication.csv"
        ),
        help="Input CSV file path",
    )
    arg_parser.add_argument(
        "--uniprot",
        "-u",
        default=str(DATA_DIR / "uniprotkb_proteome_UP000005640_2026_01_07.txt"),
        help="UniProt flat file path",
    )
    arg_parser.add_argument(
        "--alphafold",
        "-a",
        default=str(DATA_DIR / "UP000005640_9606_HUMAN_v6"),
        help="AlphaFold structures directory",
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        default=str(
            STAGE04_OUT
            / "somatic_variant_dbNSFP_Removes_missing_values_Deduplication_Structural_Functional.csv"
        ),
        help="Output CSV file path",
    )
    arg_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {CPU_COUNT})",
    )
    arg_parser.add_argument(
        "--max-rows", "-n", type=int, default=None, help="Limit rows for testing"
    )
    arg_parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Force re-parse UniProt file (ignore cache)",
    )

    args = arg_parser.parse_args()

    enriched_dataset = enrich_dataset_with_structure_function_optimized(
        input_csv=args.input,
        uniprot_file=args.uniprot,
        alphafold_dir=args.alphafold,
        output_csv=args.output,
        n_workers=args.workers,
        max_rows=args.max_rows,
        force_reparse=args.force_reparse,
    )
