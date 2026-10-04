# Publication data acquisition

`tools/acquire_publication_data.py` is the local, Kaggle-independent acquisition entrypoint.
It downloads only releases pinned in `tools/publication_data_catalog.json`, resumes `.part`
files, checks publisher MD5 values when available, calculates SHA256 for every artifact, and
writes both per-file provenance sidecars and a collection manifest.

No archive is extracted automatically. This avoids zip/tar path traversal and prevents a
small compressed file from unexpectedly consuming large amounts of disk.

## Quick start

Run these commands from the project root:

```powershell
python tools/acquire_publication_data.py list
python tools/acquire_publication_data.py plan --profile publication_core
python tools/acquire_publication_data.py fetch --profile proteingym_core_v1_3
python tools/acquire_publication_data.py fetch --profile clinical_temporal_tsv
python tools/acquire_publication_data.py verify
```

`plan`, or `fetch --dry-run`, performs no download. A specific file can be selected without
the rest of its profile:

```powershell
python tools/acquire_publication_data.py fetch --source proteingym_dms_metadata_v1_3
```

The default destination is `data/publication_sources/`. An interrupted download is retained
as `NAME.part`; repeating the same command resumes it when the server supports HTTP ranges.

### Build the project's GRCh37 snapshot files

The checked-in pipeline expects uncompressed `clinvar_RELEASE_grch37.tsv` files. Create them
from the preserved official gzip archives with the streaming transformer:

```powershell
python tools/acquire_publication_data.py fetch `
  --source clinvar_variant_2024_06 --source clinvar_variant_2026_08

python tools/prepare_clinvar_snapshot.py transform `
  --input data/publication_sources/clinvar/2024-06/tab_delimited/variant_summary_2024-06.txt.gz `
  --output data/external/clinvar_2024-06_grch37.tsv `
  --release 2024-06

python tools/prepare_clinvar_snapshot.py transform `
  --input data/publication_sources/clinvar/2026-08/tab_delimited/variant_summary_2026-08.txt.gz `
  --output data/external/clinvar_2026-08_grch37.tsv `
  --release 2026-08

python tools/prepare_clinvar_snapshot.py verify `
  --manifest data/external/clinvar_2024-06_grch37.tsv.transformation.json
python tools/prepare_clinvar_snapshot.py verify `
  --manifest data/external/clinvar_2026-08_grch37.tsv.transformation.json
```

The transformer reads one row at a time, verifies the acquired raw SHA256 and gzip stream,
retains every upstream column in its original order, and writes only GRCh37/GRCh37.p13/hg19
rows. It deliberately performs no significance/review/variant-type filtering; those
prespecified cohort rules remain downstream. Output is atomic and receives a transformation
manifest binding its SHA256 to the raw archive and acquisition provenance.

For this snapshot pair, configure the pipeline with `CLINVAR_TRAIN_RELEASE=2024-06`,
`CLINVAR_EXTERNAL_RELEASE=2026-08`, and the two generated paths through
`CLINVAR_TRAIN_ARCHIVE`/`CLINVAR_EXTERNAL_ARCHIVE` (or make those the corresponding defaults
in `src/config.py`).

### Audit submission-level SCV changes

The paired `submission_summary` releases provide versioned SCV evidence that is stronger than
comparing aggregate variant rows alone:

```powershell
python tools/acquire_publication_data.py fetch `
  --source clinvar_submission_2024_06 --source clinvar_submission_2026_08

python tools/audit_clinvar_submissions.py audit `
  --baseline data/publication_sources/clinvar/2024-06/tab_delimited/submission_summary_2024-06.txt.gz `
  --endpoint data/publication_sources/clinvar/2026-08/tab_delimited/submission_summary_2026-08.txt.gz `
  --baseline-release 2024-06 --endpoint-release 2026-08 `
  --cutoff-date 2024-06-30 `
  --output data/external/clinvar_scv_events_2024-06_to_2026-08.tsv.gz

python tools/audit_clinvar_submissions.py verify `
  --summary data/external/clinvar_scv_events_2024-06_to_2026-08.tsv.gz.audit.json
```

The audit performs an ordered merge by Variation ID with memory bounded by one variant's SCVs.
It distinguishes new SCVs, higher SCV versions, unexpected same-version payload changes,
version regressions, and withdrawals; it also reports which event submissions have a
post-cutoff evaluation date. Join this audit to the aggregate temporal cohort by VariationID.
It is provenance evidence, not a label source: current high-confidence germline significance,
conflict resolution, exact allele matching, and homology exclusion remain separate downstream
contracts. An SCV moved between Variation IDs appears as a withdrawal plus addition; resolving
ClinVar merge/replacement history fully requires the VCV XML `ReplacedList`.
The shown `2024-06-30` cutoff is the frozen `TRAIN_CUTOFF_DATE`; the cohort build must reject
an SCV audit manifest whose `cutoff_date` differs from that configured training cutoff.

For a publication cohort, apply the stricter candidate-level screen after Stage 09 has fixed
its high-confidence aggregate labels. The candidate file may be TSV/CSV, optionally gzip
compressed, and must contain integer `VariationID` plus binary `ClinicalSignificance` by
default:

```powershell
python tools/audit_clinvar_submissions.py screen `
  --baseline data/publication_sources/clinvar/2024-06/tab_delimited/submission_summary_2024-06.txt.gz `
  --endpoint data/publication_sources/clinvar/2026-08/tab_delimited/submission_summary_2026-08.txt.gz `
  --candidates data/external/clinvar_temporal_candidates.tsv.gz `
  --baseline-release 2024-06 --endpoint-release 2026-08 `
  --cutoff-date 2024-06-30 `
  --output data/external/clinvar_candidate_scv_screen_2024-06_to_2026-08.tsv.gz

python tools/audit_clinvar_submissions.py verify `
  --summary data/external/clinvar_candidate_scv_screen_2024-06_to_2026-08.tsv.gz.audit.json
```

By default a candidate passes only when the endpoint has at least one currently contributing
matching SCV from at least one unique matching submitter, has no currently contributing
opposing or unresolved SCV, and has a matching SCV that is either new or version-increased
relative to the baseline and whose `DateLastEvaluated` is strictly after the cutoff. Current
contribution is taken from `ContributesToAggregateClassification=yes`. The screen retains only
Variation/SCV identifiers, normalized binary class, qualifying dates, counts, and a stable
pass/fail reason; it does not copy submitter names, phenotypes, explanations, or descriptions.
The adjacent audit manifest binds both raw archives, the candidate table, policy thresholds,
and output by SHA256. The Python API for direct Stage 09 integration is
`screen_candidate_variations(...)` in `tools/audit_clinvar_submissions.py`.

Files larger than 1 GiB require two deliberate controls. For example:

```powershell
# Two complete ClinVar VCV snapshots: about 9.13 GiB compressed in total.
python tools/acquire_publication_data.py fetch --profile clinical_temporal_vcv `
  --allow-large --max-file-gb 6

# Pinned CC0 MaveDB quarterly bulk release: about 1.78 GiB compressed.
python tools/acquire_publication_data.py fetch --profile mavedb_bulk_2026_06 `
  --allow-large --max-file-gb 2

# ProteinGym comparator predictions include one 1.78 GiB archive.
python tools/acquire_publication_data.py fetch --profile proteingym_comparators_v1_3 `
  --allow-large --max-file-gb 2
```

The downloader checks free disk space before starting and reserves the larger of 10% of the
remaining transfer or 256 MiB. `--max-file-gb` is a hard per-file ceiling, not a prediction.

## Pinned releases

| Profile | Fixed release | Compressed size | Purpose |
|---|---:|---:|---|
| `clinical_temporal_tsv` | ClinVar 2024-06 and 2026-08 | 1.27 GiB | Practical SCV/variant snapshot comparison |
| `clinical_temporal_vcv` | ClinVar VCV XML 2024-06 and 2026-08 | 9.13 GiB | Complete submission/evidence history |
| `proteingym_core_v1_3` | ProteinGym 1.3, Zenodo 15293562 | 77.59 MiB | All 217 processed substitution assays, metadata, official folds, structures, and clinical benchmark |
| `proteingym_comparators_v1_3` | ProteinGym 1.3 | 2.04 GiB | Published zero-shot/supervised comparator predictions |
| `mavedb_bulk_2026_06` | MaveDB 2026-06-24, Zenodo 20840937 | 1.78 GiB | Quarterly CC0 metadata, scores, and counts |

All source URLs, byte sizes, release dates, publisher checksums, license links, citations, and
fallback archive locations are stored in the catalog. Moving `latest`/`current` URLs and
ambiguous versions are rejected by validation.

### ClinVar

ClinVar releases complete VCV XML monthly and retains the monthly releases indefinitely. A
VCV record contains its underlying submitted classifications (SCVs), whereas
`submission_summary` provides the practical submission-level table. NCBI does not publish
checksum sidecars for the archived monthly TSV slices in this catalog, so those downloads are
validated by exact archived byte size and assigned a SHA256 in the local provenance record.
The VCV archives have NCBI-provided MD5 checksums and are validated against them as well.

Authoritative documentation:

- [ClinVar FTP/XML guide](https://www.ncbi.nlm.nih.gov/clinvar/docs/ftp_primer/)
- [ClinVar identifiers and SCV/VCV paths](https://www.ncbi.nlm.nih.gov/clinvar/docs/identifiers/)
- [ClinVar release cycle](https://www.ncbi.nlm.nih.gov/clinvar/docs/release_cycle/)
- [ClinVar downloads](https://www.ncbi.nlm.nih.gov/clinvar/docs/downloads/)
- [NCBI data-usage policy](https://www.ncbi.nlm.nih.gov/home/about/policies/)

NCBI places no restriction on molecular-data reuse, but its policy notes that original
submitters may claim rights in contributed material. Preserve ClinVar and submitter attribution.

### ProteinGym

ProteinGym 1.3 is pinned to immutable Zenodo record `15293562`; no moving download link is
used. The core profile includes `DMS_ProteinGym_substitutions.zip`, which is the complete
processed substitution benchmark rather than the project's earlier 102,654-row sample. It
also includes assay metadata, official supervised folds, published structures, and the
clinical substitution benchmark.

Authoritative documentation:

- [Official ProteinGym repository and download table](https://github.com/OATML-Markslab/ProteinGym)
- [ProteinGym 1.3 Zenodo record](https://doi.org/10.5281/zenodo.15293562)
- [ProteinGym benchmark paper](https://papers.nips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)

The ProteinGym repository is MIT licensed. Comparator archives still require citation of the
underlying methods, and training-set overlap must be disclosed rather than treated as an
independent comparison.

Safely extract the complete DMS archive before pointing the pipeline at it:

```powershell
python tools/extract_publication_archive.py extract `
  --archive data/publication_sources/proteingym/1.3/DMS_ProteinGym_substitutions.zip `
  --dest data/publication_sources/proteingym/1.3/extracted `
  --max-uncompressed-gb 2

python tools/extract_publication_archive.py verify `
  --manifest data/publication_sources/proteingym/1.3/extracted/extraction_manifest.json
```

The extractor rejects absolute/parent paths, Windows-unsafe names, case collisions, symlinks,
special files, encryption, excessive member counts, and excessive expansion ratios. It hashes
all 217 assay CSV files and atomically publishes the destination only after writing an exact
inventory. Set `PROTEINGYM_DIR` to
`data/publication_sources/proteingym/1.3/extracted/DMS_ProteinGym_substitutions` and
`PROTEINGYM_METADATA` to
`data/publication_sources/proteingym/1.3/DMS_substitutions.csv`.

### MaveDB

The catalog pins quarterly Zenodo record `20840937`, not the concept DOI that always resolves
to the newest version. Its archive contains `main.json` plus score/count CSV files. MaveDB
includes only CC0 score sets in this bulk release; non-CC0 score sets must not be silently
mixed into the publication cohort.

Authoritative documentation:

- [MaveDB bulk-download specification](https://www.mavedb.org/docs/mavedb/finding-data/downloading.html)
- [MaveDB API quickstart](https://www.mavedb.org/docs/mavedb/programmatic-access/api-quickstart.html)
- [MaveDB license metadata](https://www.mavedb.org/docs/mavedb/submitting-data/metadata-guide.html)
- [Pinned MaveDB Zenodo release](https://doi.org/10.5281/zenodo.20840937)

## Temporal ClinVar cohort contract

The two snapshots should be used as a real time split, not merely concatenated:

1. Build the development/training eligibility state only from the 2024-06 snapshot.
2. Match submissions using stable SCV accession plus version, and variants using Variation ID
   (or VCV accession/version from XML). Do not use row order or a protein-change string alone.
3. A held-out temporal candidate must have a genuinely new or version-increased matching SCV
   after baseline. The aggregate label may be unchanged, but `LastEvaluated` alone never proves
   new evidence and is not sufficient.
4. Restrict the primary clinical endpoint to germline missense variants with an unambiguous
   protein consequence and prespecified high-confidence review states. Keep conflicts, VUS,
   somatic, pharmacogenomic, risk-factor, and association records in separate analyses.
5. Collapse multiple SCVs only with a prespecified consensus rule. Retain contributing SCV
   accessions/versions, review status, assertion dates, conflict state, and unique-submitter
   counts in the derived audit. Submitter names, conditions, and long text remain in the
   provenance-bound raw source and are not copied into the model artifact.
6. Remove exact variants used by model development and report gene/protein-homology overlap.
   A strict secondary endpoint should exclude the development homology groups entirely.
7. Freeze the resulting row IDs and labels before model scoring. Predictor availability or
   model outputs must never influence cohort inclusion or sampling.

For the large VCV files, use streaming `gzip` + XML `iterparse`; do not expand a 4--6 GiB gzip
to disk or load the complete XML tree into RAM.

## ProteinGym/MaveDB evaluation contract

- Preserve ProteinGym's continuous, direction-normalized `DMS_score`; binary labels are a
  secondary endpoint. Report assay-macro Spearman and uncertainty across assays/proteins.
- Hold out entire proteins or homology groups for any supervised DMS transfer experiment.
  Official random folds are useful as a benchmark comparator but are not a substitute for the
  project's stricter homology-disjoint endpoint.
- Reserve an untouched assay/protein set before DMS pretraining. Training on all 217 assays and
  then calling the same assays external validation is circular.
- MaveDB score columns are assay-specific. Interpret them from each score set's metadata and
  never pool raw numeric scores without within-assay direction and scale normalization.
- De-duplicate ProteinGym and MaveDB by publication, target sequence, assay, and variant before
  defining an external endpoint; many studies can appear in both resources.

## Provenance outputs

Each completed file receives `FILE.provenance.json` containing:

- fixed provider/version/release date and selected URL;
- catalog SHA256 and response headers;
- expected and observed sizes;
- publisher MD5 when provided and always an observed SHA256;
- retrieval timestamp, license, documentation, and citation links.

`data/publication_sources/acquisition_manifest.json` indexes all selected files. Run
`verify` before cohort construction and again before archiving a manuscript release. Files
without a publisher checksum are trusted only after their first acquisition fingerprint has
been recorded; an unrelated pre-existing file without a sidecar is rejected rather than
silently adopted.
