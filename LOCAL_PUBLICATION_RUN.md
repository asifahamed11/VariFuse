# Local publication runbook

This is the primary supported execution route for VariFuse. It is designed for
the current Windows workstation (GTX 1660 with 6 GB VRAM and about 15 GB system
RAM); Kaggle is not required. The commands below deliberately create a new run
namespace and preserve all legacy outputs.

The workflow strengthens reproducibility and the credibility of the experiment.
It cannot guarantee acceptance by a Q1 journal.

## 1. Create the environment

Open PowerShell and run:

```powershell
Set-Location 'I:\VariFuse_2'
python -m venv .venv-publication
.\.venv-publication\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt
```

The homology-disjoint protocol requires MMseqs2. This repository currently has
a local Windows binary. The `--local-publication` runner finds and binds it
automatically; this explicit check is still useful before a long run:

```powershell
$ProjectRoot = (Resolve-Path 'I:\VariFuse_2').Path
$env:MMSEQS_EXECUTABLE = Join-Path $ProjectRoot 'tools\mmseqs2\mmseqs\bin\mmseqs.exe'
& $env:MMSEQS_EXECUTABLE version
```

## 2. Verify immutable data inputs

Use the pinned downloader and provenance verifier documented in
[DATA_SOURCES.md](DATA_SOURCES.md). In particular, do not substitute a moving
`latest` URL for a declared release.

Fetch or verify the complete prespecified core: both variant and submission
ClinVar snapshots plus ProteinGym v1.3. These files are already present on the
current workstation, so `verify` is normally sufficient:

```powershell
python tools\acquire_publication_data.py fetch --profile publication_core

python tools\prepare_clinvar_snapshot.py transform `
  --input data\publication_sources\clinvar\2024-06\tab_delimited\variant_summary_2024-06.txt.gz `
  --output data\external\clinvar_2024-06_grch37.tsv `
  --release 2024-06

python tools\prepare_clinvar_snapshot.py transform `
  --input data\publication_sources\clinvar\2026-08\tab_delimited\variant_summary_2026-08.txt.gz `
  --output data\external\clinvar_2026-08_grch37.tsv `
  --release 2026-08

python tools\prepare_clinvar_snapshot.py verify `
  --manifest data\external\clinvar_2024-06_grch37.tsv.transformation.json
python tools\prepare_clinvar_snapshot.py verify `
  --manifest data\external\clinvar_2026-08_grch37.tsv.transformation.json
python tools\acquire_publication_data.py verify

python tools\extract_publication_archive.py verify `
  --manifest data\publication_sources\proteingym\1.3\extracted\extraction_manifest.json

python tools\audit_clinvar_submissions.py verify `
  --summary data\external\clinvar_scv_events_2024-06_to_2026-08.tsv.gz.audit.json
```

If a derived TSV already exists and its transformation manifest verifies, skip
that `transform` command. The transformer refuses replacement by default; do not
use `--overwrite` unless you have deliberately removed the dependent run and are
rebuilding the same source-to-output derivation.

The acquisition utility preserves compressed archives and never extracts them
automatically. The current ProteinGym archive has already been safely extracted
and authenticated as an exact 217-assay inventory. If rebuilding it, use the
bounded extractor command in the linked data guide; do not unzip it manually.
Stage 09 directly performs the candidate-level SCV screen using the frozen
`2024-06-30` cutoff and records its own audit artifact.

## 3. Freeze one run namespace

Choose the run ID before looking at external results. Do not reuse it for a
changed cutoff, source release, feature contract, split seed, or search space.
Run the remaining commands in this same PowerShell session; after opening a new
session, repeat the environment block before continuing the run.

```powershell
$RunId = 'reliability_temporal_v1'
$RunRoot = Join-Path $ProjectRoot "publication_runs\$RunId"
$env:VARIANT_PROJECT_ROOT = $ProjectRoot
$env:VARIANT_OUTPUT_DIR = Join-Path $RunRoot 'outputs'
$env:VARIANT_FIGURE_DIR = Join-Path $RunRoot 'figures'
$env:TORCH_HOME = Join-Path $ProjectRoot 'model_cache'

$env:VARIANT_LABEL_TASK = 'clinical'
$env:VARIANT_TRAIN_CUTOFF_DATE = '2024-06-30'
$env:CLINVAR_TRAIN_RELEASE = '2024-06'
$env:CLINVAR_EXTERNAL_RELEASE = '2026-08'
$env:CLINVAR_TRAIN_ARCHIVE = Join-Path $ProjectRoot 'data\external\clinvar_2024-06_grch37.tsv'
$env:CLINVAR_EXTERNAL_ARCHIVE = Join-Path $ProjectRoot 'data\external\clinvar_2026-08_grch37.tsv'
$env:CLINVAR_TRAIN_SUBMISSION_ARCHIVE = Join-Path $ProjectRoot 'data\publication_sources\clinvar\2024-06\tab_delimited\submission_summary_2024-06.txt.gz'
$env:CLINVAR_EXTERNAL_SUBMISSION_ARCHIVE = Join-Path $ProjectRoot 'data\publication_sources\clinvar\2026-08\tab_delimited\submission_summary_2026-08.txt.gz'
$env:CLINVAR_REQUIRE_POST_CUTOFF_EVALUATION = '1'
$env:CLINVAR_REQUIRE_SCV_EVIDENCE = '1'
$env:CLINVAR_SCV_MIN_MATCHING = '1'
$env:CLINVAR_SCV_MIN_UNIQUE_SUBMITTERS = '1'
$env:CLINVAR_SCV_MULTIPLE_SUBMITTER_MINIMUM = '2'
$env:PROTEINGYM_RELEASE = 'v1.3'
$env:PROTEINGYM_DIR = Join-Path $ProjectRoot 'data\publication_sources\proteingym\1.3\extracted\DMS_ProteinGym_substitutions'
$env:PROTEINGYM_METADATA = Join-Path $ProjectRoot 'data\publication_sources\proteingym\1.3\DMS_substitutions.csv'

$env:DMS_SAMPLING_POLICY = 'all'
$env:REQUIRE_EXTERNAL_CLINVAR = '1'
$env:REQUIRE_EXTERNAL_DMS = '1'
$env:REQUIRE_TRANSCRIPT_MAPPING = '1'
$env:REQUIRE_HOMOLOGY_GROUPS = '1'
$env:VARIANT_REPRODUCIBLE = '1'
$env:VARIANT_HASH_LARGE_FILES = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

$env:ESM_DEVICE = 'auto'
$env:ESM_USE_FP16 = '1'
$env:ESM_MAX_BATCH_PROTEINS = '1'
$env:ESM_MAX_BATCH_TOKENS = '1024'
$env:ESM_MAX_BATCH_ATTENTION = '1048576'
```

Capture the resolved environment alongside the run:

```powershell
New-Item -ItemType Directory -Force (Join-Path $RunRoot 'environment') | Out-Null
python tools\capture_environment.py `
  --output (Join-Path $RunRoot 'environment\resolved_environment.json')
```

Review the exact commands before doing any work:

```powershell
python run_pipeline.py --local-publication --run-id $RunId --dry-run
```

## 4. Execute in dependency order

The only valid order is:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 08b -> 09
   -> 10 (internal, ClinVar, DMS) -> 14 -> 11 -> 12 -> 13
```

Run the inexpensive/upstream stages first:

```powershell
python run_pipeline.py --local-publication --run-id $RunId `
  --stages 01 02 03 04 05 06 07 08 08b 09
```

Run each Stage 10 dataset separately. This makes progress and failures easier to
identify without changing the artifact namespace:

```powershell
python run_pipeline.py --local-publication --run-id $RunId `
  --stages 10 --stage10-dataset internal
python run_pipeline.py --local-publication --run-id $RunId `
  --stages 10 --stage10-dataset clinvar
python run_pipeline.py --local-publication --run-id $RunId `
  --stages 10 --stage10-dataset dms
```

Run the primary nested search plus prespecified fixed-configuration confirmation
seeds. The confirmation seeds must be chosen before their results are inspected:

```powershell
python run_pipeline.py --local-publication --run-id $RunId --stages 14 `
  --confirmation-split-seeds 1701 2903 4159 6841 7919
```

Stage 14 performs HPO only for the primary nested split plan. Confirmation runs
reuse the selected `reliability_residual` configuration without re-HPO and
evaluate raw ESM, fixed-C ESM+conservation logistic, and fixed primary-consensus
LightGBM references on each new split seed. They are a training-procedure
stability analysis; they never replace the primary nested OOF estimate.

Finish deployment fitting, untouched external validation, and verified figures:

```powershell
python run_pipeline.py --local-publication --run-id $RunId --stages 11 12 13
```

A one-command complete run is also supported as an alternative. It must use a
different fresh run ID; do not run it after the segmented commands above:

```powershell
python run_pipeline.py --local-publication `
  --run-id reliability_temporal_complete_v1 `
  --confirmation-split-seeds 1701 2903 4159 6841 7919
```

## 5. Resource expectations and recovery

- Stage 01 is dominated by compressed dbNSFP parsing and disk throughput.
- Stage 04 reads many AlphaFold structures and computes residue-level local
  neighborhoods; it can be quiet for long periods while CPU and disk are busy.
- Stage 10 is normally the longest local stage. The 650M-parameter ESM2 model has
  been smoke-tested on this GTX 1660 in FP16 with a 1,022-residue window, but a
  full all-assay DMS extraction can still take many hours or longer. Runtime is
  data-dependent, so a short exact estimate is not defensible.
- The local profile uses one protein per GPU batch and enforces a pre-load free
  VRAM check. Close browsers, games, and other CUDA applications first. If a
  driver-level crash occurs, reboot if necessary and repeat the same Stage 10
  dataset command.
- Stage 10 stores content-addressed context caches. A repeat with the same model,
  layer, scoring mode, FP16 setting, and row contract reuses successful entries.
  Do not delete `10_esm_features\*_cache` during recovery.
- Stage 14 persists its split plan, Optuna storage, and per-confirmation-repeat
  checkpoints. Repeat the identical command after an interruption. Fingerprint
  checks reject incompatible protocol changes rather than silently mixing them.
- The runner requires at least 15 GiB free near the project before local Stage
  10. The complete run needs substantially more for source data, AlphaFold files,
  normalized DMS tables, 1,280-dimensional embeddings, caches, models, and
  figures. Keep generous free space throughout.
- About 15 GB system RAM is workable because large arrays use streaming or
  memory mapping, but simultaneous memory-heavy applications can still cause
  paging. Monitor Task Manager and keep the Windows page file enabled.

Never use `--allow-existing-output` merely to start over. If an upstream contract
changes, choose a new run ID and rerun all dependent stages. Never copy selected
files between run IDs.

## 6. Understand the final artifacts

The main evidence is distributed across several bound artifacts:

- `outputs\14_tuning\architecture_selection.json`: authoritative primary
  internal nested outer-fold comparison.
- `outputs\14_tuning\nested_tuning_oof.npz`: row-aligned primary internal OOF
  predictions. This is not replaced by confirmation repeats.
- `outputs\14_tuning\confirmatory_repeated_cv_results.json`: per-seed and
  across-seed fixed-configuration stability results when confirmation seeds were
  requested.
- `outputs\11_train_and_evaluate\results.json`: deployment-fold training and
  post-selection descriptive results; not the primary internal estimate.
- `outputs\12_external_validation\external_validation.json`: exact-variant-
  disjoint temporal ClinVar, assay-macro DMS, uncertainty, contextual baselines,
  and missing-modality stress results.
- `outputs\09_prepare_external_esm\clinvar_scv_evidence_audit.csv`: stable-ID,
  SCV-version/date, submitter, opposition, and inclusion audit for every
  temporal ClinVar candidate.
- `outputs\12_external_validation\reliability_diagnostics.csv` and
  `figures\mechanistic_reliability_table.csv`: explicitly descriptive,
  post-selection gate/anchor/residual diagnostics; never primary evidence.
- `figures\figure_manifest.json`: generated-figure status plus the
  `completed` and `publication_complete` contract flags.

`publication_complete=true` means that required software and artifact checks
passed. It does not mean the proposed model won, the effect is clinically useful,
or a journal will accept the paper. Lead with Stage 14's nested OOF result and
Stage 12's prespecified external endpoints. If the reliability-residual model
does not beat the strongest simple comparator, report that negative result and
do not select a favorable seed, assay, modality mask, or subgroup after seeing
performance.

## 7. Final verification

```powershell
pytest -q
ruff check src tests run_pipeline.py tools
python -m compileall -q src tests run_pipeline.py tools
```

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) as the pre-submission audit and
reporting checklist.
