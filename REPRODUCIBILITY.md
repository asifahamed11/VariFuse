# VariFuse publication reproducibility protocol

This document is the pre-run and pre-submission audit for the local publication
workflow. Passing it means that the computational experiment is internally
consistent and auditable. It does not guarantee journal acceptance, biological
novelty, clinical utility, or superiority over the reference models.

The completed run's post-result controls are isolated under
`outputs/15_research_extensions`. Their protocol and exact stepwise commands are
documented in [RESEARCH_EXTENSIONS_BN.md](RESEARCH_EXTENSIONS_BN.md). Because the
primary/external outcomes had already been inspected, these additions are
exploratory unless evaluated on a newly acquired, prospectively locked cohort.

## 1. Freeze the scientific contract before external scoring

Record all of the following in a dated protocol file before looking at Stage 12
results:

- the clinical label and review-status rules;
- the baseline cutoff and exact ClinVar training/external releases;
- the reference genome, transcript namespaces, consequence-selection policy,
  and reference-amino-acid checks;
- dbNSFP, UniProt, AlphaFold, ESM2, ProteinGym, and MMseqs2 releases;
- the internal split unit, homology thresholds, outer/inner folds, all split and
  training seeds, and Optuna search budget;
- the proposed architecture and required reference baselines;
- calibration and locked-threshold partitions;
- the primary internal, temporal ClinVar, and DMS endpoints;
- all cohort exclusions, DMS sampling, missing-modality masks, confirmation
  split seeds, and stopping rules;
- the planned uncertainty intervals, paired tests, multiplicity correction, and
  minimum external-cohort adequacy rules.

The current primary model-selection protocol is
`fixed_nested_group_cv_v4_reliability_residual`. A changed cutoff, feature
contract, architecture, search space, seed plan, or endpoint requires a new run
ID. Never select a favorable seed, assay, mask level, or subgroup after seeing
its performance.

## 2. Verify and preserve source data

Use [DATA_SOURCES.md](DATA_SOURCES.md) and the fixed catalog in
`tools\publication_data_catalog.json`. The downloader resumes partial files,
checks declared publisher hashes when available, computes SHA256, and records
provenance. The ClinVar transformer streams official gzip files into atomic
GRCh37 TSVs and binds each derived file to its source hash.

Required principles:

- do not use moving `latest` or `current` URLs;
- preserve raw archives unchanged;
- retain acquisition and transformation manifests;
- verify every acquired file again before final analysis and archiving;
- record licenses and citations for ClinVar, ProteinGym, and any comparator;
- freeze the later ClinVar cohort before model scoring;
- use `DMS_SAMPLING_POLICY=all` when resources permit, otherwise only the
  deterministic, label-independent `hash_uniform` policy with retained sampling
  probability and inverse-probability weight.

The preferred local temporal pair is ClinVar 2024-06 for development and
2026-08 for external evaluation, with cutoff 2024-06-30. The exact release and
file hashes in the run manifests, not this prose, are the final provenance.

## 3. Capture the local environment

Follow [LOCAL_PUBLICATION_RUN.md](LOCAL_PUBLICATION_RUN.md). Use a fresh virtual
environment and a new namespace such as:

```powershell
Set-Location 'I:\VariFuse_2'
$RunId = 'reliability_temporal_v1'
$RunRoot = Join-Path (Resolve-Path .).Path "publication_runs\$RunId"
python tools\capture_environment.py `
  --output (Join-Path $RunRoot 'environment\resolved_environment.json')
```

Retain the resolved package list, Python executable, CUDA/cuDNN and driver
metadata, GPU model, MMseqs2 path/version, source-tree hash, and relevant
environment variables. A requirements file is an installation input; the
captured resolved environment is the execution record.

If Git is unavailable, `source_tree_sha256` is the code identity. Archive the
exact code tree with the paper supplement. Deterministic settings reduce but do
not eliminate numerical variation across GPU drivers and library builds.

## 4. Run only the valid dependency sequence

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 08b -> 09
   -> 10 (internal, ClinVar, DMS) -> 14 -> 11 -> 12 -> 13
```

The primary local entry point is:

```powershell
python run_pipeline.py --local-publication --run-id $RunId `
  --confirmation-split-seeds 1701 2903 4159 6841 7919
```

For the stronger segmented run, execute Stages 01-09, each Stage 10 dataset,
Stage 14 with prespecified confirmation seeds, and then Stages 11-13 exactly as
shown in [LOCAL_PUBLICATION_RUN.md](LOCAL_PUBLICATION_RUN.md).

Stage 14 must precede Stage 11. Stage 14's nested outer-fold OOF is the primary
internal estimate because every model choice occurs inside its training data.
Stage 11 fits deployment bundles with locked Stage 14 choices; its OOF is
post-selection and descriptive.

Confirmation repeats reuse the primary `reliability_residual` production
configuration on new homology-group split seeds without another HPO search. They
measure sensitivity to the split/training procedure. Across-seed summaries and
per-seed predictions are confirmatory-only and cannot replace the primary
`nested_tuning_oof.npz` estimate.

## 5. Resume without mixing experiments

- Stage 10 writes content-addressed context caches. After an interruption, rerun
  the same dataset command with the same run ID and ESM contract.
- Stage 14 persists the canonical split plan, Optuna storage, and
  per-confirmation-repeat checkpoints. Resume with the identical command and
  seed list.
- Do not delete caches merely because the final table was not completed; valid
  cached contexts are rechecked before reuse.
- Do not use `--allow-existing-output` for a changed experiment. That flag is an
  explicit recovery/replacement override, not permission to combine runs.
- Do not copy individual artifacts between run IDs.

Dependency invalidation rules:

| Change | Minimum required rerun |
|---|---|
| Raw source, label, cutoff, transcript, or Stage 01-09 code/contract | 01 through 13 |
| ESM model, layer, scoring, FP16, window, or Stage 10 row contract | 10, 14, 11, 12, 13 |
| Homology grouping or split plan | 08b, 14, 11, 12, 13 |
| Stage 14 architecture/search/confirmation protocol | 14, 11, 12, 13 |
| Deployment model serialization only | 11, 12, 13 |
| External evaluation or robustness logic | 12, 13 |
| Figure/report validation only | 13 into a new empty figure root |

## 6. Audit manifests and row alignment

Every completed stage must have a completed manifest. Manifest schema v2 binds
portable artifact IDs and exact SHA256 hashes; file existence alone is not proof
of completion.

Check all of the following:

- label task, cutoff, release declarations, source hashes, ESM contract,
  homology contract, and protocol version agree throughout the chain;
- Stage 08b records MMseqs2 version/settings and no connected homology group
  crosses a declared partition;
- Stage 09 records actual ClinVar/DMS source and selected counts, temporal and
  transcript-selection audits, stable VariationID remapping checks, the exact
  candidate-level SCV evidence audit, unique sequence-table storage, and a
  publication-safe DMS policy;
- Stage 10 table, embedding, and status artifacts have identical row IDs/order,
  finite successful rows, expected 1,280-dimensional embeddings, and exact
  manifest bindings for internal, ClinVar, and DMS datasets;
- Stage 14 binds `architecture_selection.json`, `nested_inner_splits.json`,
  `nested_tuning_oof.npz`, `nested_outer_results.csv`,
  `nested_outer_folds.csv`, `trials_history.csv`, and each required
  best-parameter file;
- Stage 14 includes `concatenation`, `gated_fusion`, `cross_attention`, and
  `reliability_residual`, plus raw ESM, ESM-score logistic, conservation
  logistic, ESM+conservation logistic, mutation logistic, availability-only
  logistic, ESM-embedding+mutation logistic, and LightGBM references;
- when confirmation seeds were requested, Stage 14 binds
  `confirmatory_seed_plan.json`, `confirmatory_repeated_cv_results.json`,
  `confirmatory_repeated_cv_predictions.npz`,
  `confirmatory_repeated_cv_folds.csv`, and every declared per-repeat checkpoint;
- confirmation row IDs and labels align with the primary Stage 10 universe, all
  requested seeds are represented, no repeat performs HPO, and the primary
  nested OOF remains unchanged;
- every Stage 11 fold bundle contains model, preprocessing, calibration,
  threshold, feature-name, and metadata artifacts, including the
  `reliability_residual` bundle;
- Stage 12 JSON, NPZ, summary CSV, prediction CSV, robustness records,
  reliability diagnostics, and manifests agree on row identities, counts,
  thresholds, model inventory, component semantics, and input hashes;
- external ClinVar rows are unique genomic variants, DMS `(ASSAY_ID, variant_id)`
  pairs are unique, and the DMS sequence-table binding is present;
- Stage 13's `figure_manifest.json` has `completed=true`,
  `publication_complete=true`, no failed required figure, and exact bindings for
  every reported source and figure; its mechanistic reliability table is marked
  descriptive/post-selection/non-confirmatory and is never used for inference.

## 7. Verify the reliability and robustness claims

The proposed architecture must retain these testable properties in its persisted
metadata and stress tests:

- a constrained monotonic ESM sequence anchor;
- separate local-structure, evolution, context, and pooled-ESM encoders;
- availability/missingness variables restricted to the reliability gate;
- a bounded residual correction;
- exact sequence-anchor fallback when both structure and conservation are
  unreliable;
- training-only modality dropout;
- prespecified missing-structure, missing-conservation, and joint-missing stress
  levels, including the complete-missing case;
- safe-fallback identity checks on masked units.

Stress-test results are descriptive sensitivity analyses, not extra independent
test cohorts. Stage 12/13 must fail closed if required robustness artifacts are
missing or inconsistent.

## 8. Required result reporting

For the primary nested OOF and an adequately sized external ClinVar cohort,
report cohort/positive/gene/homology-group counts; AUROC and AUPRC with group-
aware intervals; locked-threshold MCC, sensitivity, specificity, precision,
NPV, and F1; Brier score/skill, log loss, adaptive ECE, calibration slope and
intercept; paired group-bootstrap differences; paired group-randomization tests
with Holm correction; and prespecified risk-coverage results.

When the de-overlapped ClinVar cohort fails the prespecified adequacy rule, report
counts and descriptive point estimates only. Do not present inferential model
comparisons, confidence intervals, or broad novel-gene claims. The software guard
is a minimum protection, not evidence that a barely passing cohort is clinically
representative.

For DMS, lead with equal-weight assay-macro functional Spearman and AUROC with
assay-bootstrap intervals and paired assay effects. Pooled-row metrics and
calibration are descriptive. State whether all variants or deterministic hash
sampling was used and report every assay's coverage.

For confirmation repeats, report the prespecified seed list, fixed configuration,
per-seed paired effects, across-seed dispersion/instability, and incomplete
repeats. Do not average them with the primary estimate or treat rows repeated
across seeds as independent observations.

Report coverage and common-coverage results for every contextual predictor.
Established predictors may contain ClinVar or population-derived evidence, so
their release and circularity limitations must accompany any comparison.

## 9. Limitations that software cannot remove

- A label-temporal split with later dbNSFP, UniProt, or AlphaFold covariates is
  not a fully historical end-to-end deployment replay.
- ClinVar case-control data do not establish population calibration, clinical
  utility, or prospective benefit.
- An incomplete-modality external cohort cannot validate the full multimodal
  claim; robustness masking only characterizes degradation.
- A small de-overlapped gene/homology-disjoint cohort cannot support broad
  novel-gene generalization.
- DMS functional effects are not identical to clinical pathogenicity.
- Comparator methods may share ClinVar-derived evidence with the labels.
- Finite HPO budgets and a finite confirmation seed set leave optimization and
  training-procedure uncertainty.

Appropriate remedies require new evidence: larger untouched clinical cohorts,
historical covariates, prospective evaluation, complete external modalities,
protein/homology-held-out functional assays, and independent comparator sources.
The manuscript must state these limitations even when all software checks pass.

## 10. Final code verification

```powershell
pytest -q
ruff check src tests run_pipeline.py tools
python -m compileall -q src tests run_pipeline.py tools
```

Archive the test output, environment capture, acquisition/transformation
manifests, all run manifests, split plans, prediction tables, model bundles, and
figure manifest with the manuscript analysis release.
