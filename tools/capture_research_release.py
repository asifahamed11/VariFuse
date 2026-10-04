"""Capture a compact, checksummed code release for the research extensions."""
# ruff: noqa: E402
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from research_evidence import dump_json, extension_dir, sha256


CODE = [
    'src/research_evidence.py', 'src/research_models.py',
    'tools/analyze_research_evidence.py', 'tools/run_research_controls.py',
    'tools/summarize_research_controls.py', 'tests/test_control_summary_compat.py',
    'tools/prepare_functional_validation.py', 'tools/score_functional_validation.py',
    'tools/capture_research_release.py', 'tests/test_research_extensions.py',
    'tools/analyze_manuscript_controls.py', 'tests/test_manuscript_control_analysis.py',
    'tools/collect_manuscript_evidence.py', 'tools/build_manuscript_figures.py',
    'tools/prepare_manuscript_references.py', 'tools/verify_manuscript_evidence.py',
    'LICENSE', 'CITATION.cff', '.gitattributes',
    'README.md', 'REPRODUCIBILITY.md', 'RESEARCH_EXTENSIONS_BN.md', 'Q1_READINESS_BN.md',
    'requirements-publication.txt', 'pyproject.toml',
]


def main() -> None:
    output = extension_dir(ROOT / 'outputs', ROOT / 'outputs/15_research_extensions') / 'release'
    output.mkdir(parents=True, exist_ok=True)
    missing = [name for name in CODE if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Release source files missing: {missing}')
    packages = sorted(f'{item.metadata["Name"]}=={item.version}'
                      for item in importlib.metadata.distributions() if item.metadata['Name'])
    (output / 'resolved_packages.txt').write_text('\n'.join(packages) + '\n', encoding='utf-8')
    archive = output / 'varifuse_research_extensions_0.1.0.zip'
    with zipfile.ZipFile(archive.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in CODE:
            bundle.write(ROOT / name, arcname=name)
    archive.with_suffix('.zip.tmp').replace(archive)
    evidence = sorted((ROOT / 'outputs/15_research_extensions').rglob('*'))
    evidence_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in evidence
                       if path.is_file() and 'release' not in path.parts and path.suffix not in ['.pkl', '.npy']}
    dump_json(output / 'release_manifest.json', {
        'release': 'varifuse-research-extensions-0.1.0',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'git_repository_present': (ROOT / '.git').exists(),
        'code_license': 'MIT_existing_repository_license_retained_with_owner_authorization',
        'archive': {'path': archive.name, 'sha256': sha256(archive)},
        'environment': {'path': 'resolved_packages.txt',
                        'sha256': sha256(output / 'resolved_packages.txt')},
        'source_sha256': {name: sha256(ROOT / name) for name in CODE},
        'evidence_sha256': evidence_hashes,
        'excluded_from_archive': ['datasets', 'model weights', 'original outputs', 'virtual environment'],
    })
    print('Captured compact research release:', output / 'release_manifest.json', flush=True)


if __name__ == '__main__':
    main()
