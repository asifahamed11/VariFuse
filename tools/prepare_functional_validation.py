"""Prepare a structure-annotated DMS sensitivity cohort using frozen covariates.

Selection never uses DMS outcomes. Existing DMS outcomes were already inspected:
this is exploratory functional evidence, NOT a newly untouched confirmation.
Only sequence-identical residue covariates are transferred; nucleotide-specific
conservation values are deliberately not transferred between variants.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from research_evidence import dump_json, extension_dir, sha256

KEYS = ['sequence_hash', 'aa_pos', 'aa_ref']
STRUCTURE = ['SASA', 'RELATIVE_SASA', 'PLDDT_SCORE', 'LOCAL_CONTACT_COUNT_8A',
             'LOCAL_CONTACT_COUNT_12A', 'LOCAL_LONG_RANGE_CONTACT_COUNT_8A',
             'LOCAL_MEAN_PLDDT_8A', 'LOCAL_MIN_PLDDT_8A', 'LOCAL_CONFIDENT_CONTACT_FRACTION_8A',
             'LOCAL_MEAN_DISTANCE_8A', 'LOCAL_HYDROPHOBIC_FRACTION_8A', 'LOCAL_CHARGED_FRACTION_8A']
FUNCTION = ['IS_IN_DOMAIN', 'DISTANCE_TO_ACTIVE_SITE', 'IS_ACTIVE_SITE', 'IS_BINDING_SITE', 'IS_TRANSMEMBRANE',
            'HAS_DOMAIN_ANNOTATION', 'HAS_ACTIVE_SITE_ANNOTATION', 'HAS_BINDING_SITE_ANNOTATION', 'HAS_TRANSMEMBRANE_ANNOTATION']
FLAGS = ['HAS_STRUCTURE', 'STRUCTURE_FILE_AVAILABLE', 'LOW_CONFIDENCE_STRUCTURE']


def residue_covariates(source: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    source = source.loc[source.HAS_STRUCTURE.eq(1)].copy()
    columns = [*KEYS, *STRUCTURE, *FUNCTION, *FLAGS, 'genename', 'uniprot_id']
    source = source[columns]
    if source[KEYS].isna().any().any():
        raise ValueError('Covariate residue identity is incomplete')
    features = [c for c in columns if c not in KEYS]
    counts = source.groupby(KEYS, dropna=False)[features].nunique(dropna=False)
    ambiguous = counts.gt(1).any(axis=1)
    valid_keys = counts.loc[~ambiguous].reset_index()[KEYS]
    source = source.drop_duplicates(KEYS).merge(valid_keys, on=KEYS, validate='one_to_one')
    return source, int(ambiguous.sum())


def prepare(source_root: Path, output: Path) -> None:
    stages = source_root/'10_esm_features'
    columns = [*KEYS, *STRUCTURE, *FUNCTION, *FLAGS, 'genename', 'uniprot_id']
    paths = [stages/'internal_with_esm.parquet', stages/'clinvar_with_esm.parquet', stages/'dms_with_esm.parquet']
    covariates = pd.concat([pd.read_parquet(p, columns=columns) for p in paths[:2]], ignore_index=True)
    residues, ambiguous = residue_covariates(covariates)
    # Lightweight cohort definition uses only identity/availability. Labels and
    # DMS scores are attached after the inclusion mask has been frozen.
    identity = pd.read_parquet(paths[2], columns=[*KEYS, 'aa_alt', 'row_id', 'ASSAY_ID'])
    identity['source_embedding_index'] = np.arange(len(identity))
    if identity.row_id.duplicated().any():
        raise ValueError('DMS row IDs are not unique')
    annotated = identity.merge(residues, on=KEYS, how='inner', validate='many_to_one')
    train = pd.read_parquet(paths[0], columns=[*KEYS, 'aa_alt', 'genename'])
    protein_variant = [*KEYS, 'aa_alt']
    seen = pd.MultiIndex.from_frame(train[protein_variant])
    novel = ~pd.MultiIndex.from_frame(annotated[protein_variant]).isin(seen)
    exact_disjoint = annotated.loc[novel].copy()
    heldout = exact_disjoint.loc[~exact_disjoint.genename.astype(str).isin(set(train.genename.astype(str)))].copy()
    heldout = heldout.sort_values('source_embedding_index')
    # Existing local covariates imply ascertainment toward already annotated
    # residues. Report this explicitly rather than calling it proteome coverage.
    heldout[[*KEYS, 'aa_alt', 'row_id', 'ASSAY_ID', 'genename', 'uniprot_id', 'source_embedding_index']].to_parquet(
        output/'functional_cohort_membership.parquet', index=False)
    full = pd.read_parquet(paths[2])
    result = full.iloc[heldout.source_embedding_index.to_numpy(int)].copy().reset_index(drop=True)
    for column in [*STRUCTURE, *FUNCTION, *FLAGS, 'genename', 'uniprot_id']:
        result[column] = heldout[column].to_numpy()
        missing = column + '__missing'
        if missing in result:
            result[missing] = result[column].isna().astype(np.int8)
    result['source_embedding_index'] = heldout.source_embedding_index.to_numpy(int)
    result['EXTERNAL_FEATURE_PROFILE'] = 'exploratory_sequence_identical_residue_structure_transfer'
    # Conservation remains missing: aa position alone cannot identify which
    # nucleotide supplied a conservation annotation.
    for column in ['GERP++_RS','phyloP100way_vertebrate','phastCons100way_vertebrate']:
        result[column] = np.nan
        result[column+'__missing'] = np.int8(1)
    result.to_parquet(output/'functional_with_structure.parquet', index=False)
    used_path = stages/'dms_esm_embeddings.npy'
    embedding = np.load(used_path, mmap_mode='r')
    if len(embedding) != len(full):
        raise ValueError('DMS embedding/table row mismatch')
    np.save(output/'functional_esm_embeddings.npy', np.asarray(embedding[result.source_embedding_index.to_numpy(int)]))
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'prepared' if len(result) else 'no_eligible_rows',
        'scientific_role': 'exploratory_structure_transfer_sensitivity_analysis',
        'untouched': False, 'label_blind_selection': True,
        'training_gene_disjoint': True, 'training_protein_variant_disjoint': True,
        'homology_disjoint': 'not_established_do_not_claim',
        'sequence_contract': 'identical_full_sequence_hash_and_reference_amino_acid_position',
        'conservation_transferred': False, 'new_esm_extraction': False,
        'limitations': ['Previously evaluated DMS labels; not an untouched cohort',
                       'Availability-selected residues; not all possible residues or proteins',
                       'Human reference covariates; protein/gene-disjointness is not homology-disjointness',
                       'Later annotation releases do not reconstruct a historical deployment'],
        'total_dms_rows': len(identity), 'rows_matching_structure': len(annotated),
        'ambiguous_residue_covariate_keys_excluded': ambiguous,
        'exact_protein_variant_disjoint_rows': len(exact_disjoint),
        'gene_disjoint_rows': len(result), 'assays': int(result.ASSAY_ID.nunique()),
        'genes': int(result.genename.nunique()),
        'source_sha256': {str(p.resolve()): sha256(p) for p in paths},
        'source_embedding_sha256': sha256(used_path),
        'code_sha256': sha256(Path(__file__)),
        'outputs': {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name!='cohort_manifest.json'},
    }
    dump_json(output/'cohort_manifest.json', manifest)
    dump_json(output/'untouched_cohort_requirements.json', {
        'status': 'awaiting_new_independent_data', 'untouched_cohort_obtained': False,
        'required': ['documented source and release/access dates',
                     'model and analysis protocol locked before outcomes examined',
                     'exact variant, gene and homology overlap reports as applicable',
                     'sequence/reference allele checks and annotation coverage',
                     'no parameter, threshold or subgroup selection using cohort outcomes'],
        'existing_data_cannot_be_relabelled_untouched': True})
    print(f'Prepared {len(result)} gene-disjoint functional rows across {result.ASSAY_ID.nunique()} assays.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT/'outputs')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/15_research_extensions')
    args = parser.parse_args()
    # Validate schemas before any write; no silent imputation of absent source columns.
    for name in ['internal','clinvar']:
        available = set(pq.read_schema(args.source_root/f'10_esm_features/{name}_with_esm.parquet').names)
        if set([*KEYS,*STRUCTURE,*FUNCTION,*FLAGS,'genename','uniprot_id']) - available:
            raise ValueError(f'Incomplete structural covariate schema: {name}')
    output = extension_dir(args.source_root,args.output_root)/'functional_validation'
    output.mkdir(parents=True,exist_ok=True)
    prepare(args.source_root,output)


if __name__ == '__main__':
    main()
