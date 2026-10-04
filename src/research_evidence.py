"""Post-run research diagnostics; these never certify prospective evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def extension_dir(source: Path, destination: Path) -> Path:
    """Forbid overwriting the audited stages or the source root itself."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or destination in source.parents:
        raise ValueError('Research destination must not be the original output root or its parent')
    if source in destination.parents:
        relative = destination.relative_to(source)
        if relative.parts[0] != '15_research_extensions':
            raise ValueError('Inside the audited outputs, only 15_research_extensions is writable')
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def align_rows(frame: pd.DataFrame, ids, key: str = 'row_id') -> pd.DataFrame:
    ids = np.asarray(ids).astype(str)
    if frame[key].isna().any() or frame[key].astype(str).duplicated().any():
        raise ValueError(f'{key} must uniquely identify input rows')
    if len(set(ids)) != len(ids):
        raise ValueError('Prediction row identifiers are not unique')
    frame = frame.copy()
    frame[key] = frame[key].astype(str)
    missing = set(ids) - set(frame[key])
    if missing:
        raise ValueError(f'Unmapped prediction rows: {len(missing)}')
    return frame.set_index(key).loc[ids].reset_index()


def rank_metrics(y, score) -> dict:
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not len(y):
        raise ValueError('Labels and predictions must be nonempty aligned vectors')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError('Invalid labels or nonfinite predictions')
    two_class = len(np.unique(y)) == 2
    return {'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'auroc': float(roc_auc_score(y, score)) if two_class else None,
            'auprc': float(average_precision_score(y, score)) if two_class else None}


def gene_metrics(y, score, groups) -> pd.DataFrame:
    data = pd.DataFrame({'y': y, 'score': score, 'group': np.asarray(groups).astype(str)})
    return pd.DataFrame([{'group': name, **rank_metrics(g.y, g.score)}
                         for name, g in data.groupby('group', sort=True)])


def paired_group_bootstrap(y, a, b, groups, *, iterations=1000, seed=1701) -> dict:
    """Paired percentile intervals conditional on fixed predictions; no p-values."""
    if iterations < 20:
        raise ValueError('At least 20 bootstrap replicates required')
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    ma, mb = rank_metrics(y, a), rank_metrics(y, b)
    groups = np.asarray(groups).astype(str)
    if groups.shape != y.shape:
        raise ValueError('Groups and labels are misaligned')
    levels, codes = np.unique(groups, return_inverse=True)
    if len(levels) < 2:
        raise ValueError('At least two groups are needed')
    # Zero-weight rows need not be copied for repeated resampling of whole genes.
    rng = np.random.default_rng(seed)
    sampled = {'auprc': [], 'auroc': []}
    for _ in range(iterations):
        counts = np.bincount(rng.integers(0, len(levels), len(levels)), minlength=len(levels))
        weights = counts[codes]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        for metric, function in [('auprc', average_precision_score), ('auroc', roc_auc_score)]:
            sampled[metric].append(float(function(y, b, sample_weight=weights)
                                         - function(y, a, sample_weight=weights)))
    return {'direction': 'second_minus_first', 'groups': len(levels), 'requested_replicates': iterations,
            'inference': 'exploratory_paired_cluster_percentile_interval_fixed_predictions',
            'training_uncertainty_included': False, 'multiplicity_adjusted': False,
            'metrics': {k: {'difference': None if ma[k] is None else mb[k] - ma[k],
                            'valid_replicates': len(v),
                            'ci95': np.quantile(v, [.025, .975]).tolist() if len(v) >= 20 else None}
                        for k, v in sampled.items()}}


def subgroup_metrics(y, models: dict, strata: dict) -> list[dict]:
    rows = []
    for name, values in strata.items():
        values = np.asarray(values).astype(str)
        for level in sorted(set(values)):
            mask = values == level
            for model, probabilities in models.items():
                rows.append({'stratifier': name, 'stratum': level, 'model': model,
                             **rank_metrics(np.asarray(y)[mask], np.asarray(probabilities)[mask])})
    return rows


def proteingym_aggregate(assays: pd.DataFrame, metadata: pd.DataFrame, metric: str) -> dict:
    """Protein-within-functional-category means, then equal category weights.

    Implements the official hierarchy on the *declared evaluated subset*;
    does not equate a filtered single-mutant cohort with the full leaderboard.
    Undefined assay correlations are excluded explicitly, never set to zero.
    """
    required = ['DMS_id', 'UniProt_ID', 'coarse_selection_type']
    if metadata.DMS_id.duplicated().any() or metadata[required].isna().any().any():
        raise ValueError('ProteinGym metadata must map every assay uniquely to protein/category')
    if assays.duplicated(['model', 'DMS_id']).any():
        raise ValueError('Duplicate model/assay metrics')
    table = assays.merge(metadata[required], on='DMS_id', how='left', validate='many_to_one')
    if table[required].isna().any().any():
        raise ValueError('Assay metadata are missing; refusing silent cohort shrinkage')
    result = {}
    for model, group in table.groupby('model', sort=True):
        valid = group[np.isfinite(group[metric])]
        proteins = valid.groupby(['coarse_selection_type', 'UniProt_ID'])[metric].mean()
        categories = proteins.groupby(level=0).mean()
        result[model] = {
            'score': float(categories.mean()) if len(categories) else None,
            'assay_macro': float(valid[metric].mean()) if len(valid) else None,
            'valid_assays': len(valid), 'excluded_undefined_assays': len(group) - len(valid),
            'proteins': int(valid.UniProt_ID.nunique()),
            'category_means': {str(k): float(v) for k, v in categories.items()},
            'full_leaderboard_comparable': False,
        }
    return result


def dms_assay_metrics(frame: pd.DataFrame, models: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for assay, positions in frame.groupby('ASSAY_ID', sort=True).indices.items():
        observed = frame.iloc[positions]['DMS_SCORE'].to_numpy(float)
        y = frame.iloc[positions]['LABEL_PATHOGENIC'].to_numpy(int)
        for model, scores in models.items():
            score = np.asarray(scores)[positions]
            valid = np.isfinite(observed) & np.isfinite(score)
            o, s = observed[valid], score[valid]
            rho = (float(spearmanr(o, -s).statistic)
                   if len(o) > 1 and np.ptp(o) > 0 and np.ptp(s) > 0 else np.nan)
            rows.append({'DMS_id': assay, 'model': model, 'n': int(valid.sum()),
                         'excluded_rows': int((~valid).sum()), 'spearman': rho,
                         'auroc': float(roc_auc_score(y[valid], s))
                         if len(np.unique(y[valid])) == 2 else np.nan})
    return pd.DataFrame(rows)


def attach_public_scores(frame: pd.DataFrame, path: Path, *, direction: str) -> np.ndarray:
    """Join public per-variant scores; never compare unrelated leaderboard rows.

    Input columns: DMS_id, mutant (A123V), score. Direction must be declared.
    """
    scores = pd.read_csv(path, usecols=['DMS_id', 'mutant', 'score'])
    if scores.duplicated(['DMS_id', 'mutant']).any():
        raise ValueError('Duplicate public score keys')
    key = pd.DataFrame({'DMS_id': frame.ASSAY_ID.astype(str),
                        'mutant': frame.aa_ref.astype(str) + frame.aa_pos.astype(int).astype(str)
                        + frame.aa_alt.astype(str)})
    merged = key.merge(scores, on=['DMS_id', 'mutant'], how='left', validate='many_to_one')
    values = pd.to_numeric(merged.score, errors='coerce').to_numpy(float)
    if direction not in ['higher_is_functional', 'higher_is_pathogenic']:
        raise ValueError('Explicit public score orientation required')
    return -values if direction == 'higher_is_functional' else values
