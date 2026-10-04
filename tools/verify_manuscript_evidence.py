"""Check the public metric package and paired comparison directions without fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'publication/evidence'


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8'))


def main():
    manifest = read('evidence_manifest.json')
    for name, expected in manifest['output_sha256'].items():
        path = EVIDENCE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Compact evidence checksum differs: {name}')
    metrics = read('control_metrics.json')
    if any(len(metrics[scope]) != 8 for scope in ['internal', 'clinical']):
        raise ValueError('Expected eight complete controls in each cohort')
    original = read('original_verified_evidence.json')
    checked = 0
    for scope, filename in [('internal', 'control_internal_intervals.json'),
                            ('clinical', 'control_clinical_intervals.json')]:
        for name, interval in read(filename).items():
            second, first = name.split('_minus_')
            reference = metrics[scope].get(first)
            if reference is None:
                reference = original['metrics']['clinvar_exact_variant_disjoint'][first]
            for metric, value in interval['metrics'].items():
                observed = metrics[scope][second][metric] - reference[metric]
                if abs(observed - value['difference']) > 1e-12:
                    raise ValueError(f'Comparison direction or metric mismatch: {name}')
                if value['valid_replicates'] != 1000 or value['ci95'][0] > value['ci95'][1]:
                    raise ValueError(f'Invalid recorded interval: {name}')
                checked += 1
    if any(value['n'] != 48197 for value in metrics['internal'].values()):
        raise ValueError('Control internal evaluation units differ')
    if any(value['n'] != 14650 for value in metrics['clinical'].values()):
        raise ValueError('Control clinical evaluation units differ')
    print(f'Verified {len(manifest["output_sha256"])} compact file checksums and '
          f'{checked} paired metric differences; eight complete model records per cohort.')


if __name__ == '__main__':
    main()
