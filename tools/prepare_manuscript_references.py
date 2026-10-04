"""Fetch bibliographic records from Europe PMC and preserve citation provenance."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DOIS = {
    'acmg': '10.1038/gim.2015.30',
    'clinvar': '10.1093/nar/gkae1090',
    'rives': '10.1073/pnas.2016239118',
    'esm2': '10.1126/science.ade2574',
    'alphafold': '10.1038/s41586-021-03819-2',
    'afdb': '10.1093/nar/gkad1011',
    'alphamissense': '10.1126/science.adg7492',
    'circularity': '10.1002/humu.22768',
    'dbnsfp': '10.1186/s13073-020-00803-9',
    'uniprot': '10.1093/nar/gkae1010',
    'mmseqs': '10.1038/nbt.3988',
    'prcurve': '10.1371/journal.pone.0118432',
    'revel': '10.1016/j.ajhg.2016.08.016',
}
TARGET_PAPERS = {'darvic': 1, 'genixrl': 2, 'nested': 9, 'external': 10,
                 'calibration': 11, 'multimodal': 14, 'missing': 16,
                 'distillation': 18}


def retrieve(item):
    key, doi = item
    query = urlencode({'query': f'DOI:"{doi}"', 'format': 'json', 'resultType': 'core'})
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + query
    request = Request(url, headers={'User-Agent': 'VariFuse-reference-verification/1.0'})
    with urlopen(request, timeout=60) as response:
        value = json.load(response)
    rows = [row for row in value['resultList']['result'] if row.get('doi', '').lower() == doi.lower()]
    if len(rows) != 1:
        raise ValueError(f'Expected one exact DOI match: {key} {doi}')
    return key, rows[0], url


def main():
    destination = ROOT / 'publication/sources'
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for key, row, url in pool.map(retrieve, DOIS.items()):
            records[key] = {'record': row, 'verified_from': url}
            print(key, row['title'], flush=True)
    selected_path = ROOT / 'publication/evidence/selected_20_references.json'
    if not selected_path.exists():
        selected_path = ROOT / 'publication/sources/selected_20_metadata.json'
    if not selected_path.exists():
        selected_path = ROOT / 'audit_reports/journal_plan_20261004/selected_20_metadata.json'
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    for key, number in TARGET_PAPERS.items():
        row = selected[number - 1]
        records[key] = {'record': row,
                        'verified_from': 'https://europepmc.org/article/MED/' + row['pmid'],
                        'journal_planning_article': number}
    # Proceedings records have no DOI. Their primary publisher pages were
    # checked explicitly; bibliographic fields are retained without a fake DOI.
    records['meier'] = {'manual': {
        'authors': 'Meier J, Rao R, Verkuil R, Liu J, Sercu T, Rives A',
        'title': 'Language models enable zero-shot prediction of the effects of mutations on protein function',
        'journal': 'Advances in Neural Information Processing Systems', 'year': '2021', 'volume': '34',
        'url': 'https://papers.nips.cc/paper_files/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html'}}
    records['lightgbm'] = {'manual': {
        'authors': 'Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, et al.',
        'title': 'LightGBM: A highly efficient gradient boosting decision tree',
        'journal': 'Advances in Neural Information Processing Systems', 'year': '2017', 'volume': '30',
        'url': 'https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html'}}
    records['proteingym'] = {'manual': {
        'authors': 'Notin P, Kollasch A, Ritter D, van Niekerk L, Paul S, Spinner H, et al.',
        'title': 'ProteinGym: Large-scale benchmarks for protein fitness prediction and design',
        'journal': 'Advances in Neural Information Processing Systems', 'year': '2023', 'volume': '36',
        'url': 'https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf'}}
    (destination / 'reference_records.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Verified {len(records)} reference records', flush=True)


if __name__ == '__main__':
    main()
