# VariFuse

VariFuse is a publication-oriented pipeline for missense-variant pathogenicity
research. It combines frozen ESM2 sequence evidence, conservation, mutation
context, and local AlphaFold residue environments under strict transcript,
label, homology, temporal, and artifact-integrity contracts.

Local Windows execution is the primary supported workflow. Start with
[LOCAL_PUBLICATION_RUN.md](LOCAL_PUBLICATION_RUN.md) for exact commands tailored
to this workstation.

The author's local `outputs\` directory contains the preserved full run and is
not included in Git. Post-result analyses and new control protocols live under
`outputs\15_research_extensions\` and are exploratory. The public compact
[numerical evidence and figures](publication/README.md) distinguish the original
experiment from the completed later controls. Stepwise control commands are in
[RESEARCH_EXTENSIONS_BN.md](RESEARCH_EXTENSIONS_BN.md).

## Completed comparative evidence

Development contains 48,197 variants and uses five outer and three inner folds
over 3,718 connected gene/homology groups. The later clinical cohort contains
14,650 exact-variant-disjoint records; it is not fully homology-disjoint.

| Model and evidence phase | Development AP | Clinical AP |
|---|---:|---:|
| Original gated fusion | 0.9164 | 0.9625 |
| Original reliability residual | 0.9028 | 0.9574 |
| Later shared-input LightGBM | 0.9219 | 0.9640 |
| Later shared-input MLP | 0.9117 | 0.9593 |
| Later reliability control | 0.9010 | 0.9568 |

AP means average precision, not trapezoidal precision-recall integration.
The eight later controls have all 40 development-fold fits completed. Their
clinical scores use a previously inspected cohort and remain exploratory.
Matched LightGBM versus reliability-control development AP differs by 0.0209
(exploratory connected-group 95% interval 0.0166 to 0.0257). Its clinical AP
difference from original gated fusion is 0.0015 with an interval including zero.
The data support strong shared representations, not superiority of bounded
reliability fusion or clinical deployment readiness.

The original scalar-only LightGBM omits the ESM score and residue vector; the
later shared-input model includes both. Input matching does not equalize tuning
budgets: later neural controls inherit the reliability recipe, while tree and
linear models have their own documented selection procedures.

All 696,311 main ProteinGym rows lack structure and conservation. Original
reliability fusion therefore falls back to its sequence anchor on every row.
Assay-macro Spearman is 0.4291, identical to sequence ranking. This is fallback
behavior, not multimodal functional validation. The structure-equipped
19-variant sensitivity cohort is exploratory and underpowered.

## What the current protocol tests

- The primary training task is high-confidence ClinVar pathogenic versus
  high-confidence ClinVar benign missense variation.
- Development evidence is frozen at a declared baseline cutoff. The external
  ClinVar snapshot must be later, and exact development variants are removed.
- Internal model selection uses connected gene/MMseqs2 homology groups in fixed
  nested cross-validation. No connected group may cross an inner or outer
  partition.
- Stage 14 uses the same immutable partitions for every proposed architecture
  and reference baseline. Its nested outer-fold OOF predictions are the
  authoritative internal estimate.
- Stage 11 creates deployment fold bundles after Stage 14 has locked the model
  choices. Stage 11 OOF is post-selection and descriptive.
- External ClinVar is analyzed as unique genomic variants. DMS is a separate
  functional-transfer endpoint led by equal-weight assay-macro Spearman and
  AUROC, not by pooled-row clinical claims.
- Contextual predictors such as AlphaMissense, REVEL, CADD, PrimateAI, MetaRNN,
  BayesDel, SIFT, and PolyPhen are evaluation-only. They cannot create primary
  labels or enter VariFuse training features.

## Proposed reliability-residual model

The current proposed architecture is `reliability_residual`. Its sequence anchor
is a constrained monotonic transform of the ESM variant score. Separate
structure, evolution, biological-context, and pooled-ESM encoders can make only
a bounded residual correction through a reliability gate. Availability
indicators control the gate and do not become unrestricted predictive shortcuts.
When neither structure nor conservation is trustworthy, the prediction falls
back exactly to the sequence anchor.

The stored ESM vector is the layer-33 representation at the masked mutation
position, not a protein-mean vector. Some legacy class comments use the word
pooled for this fixed-size input; the extraction implementation defines the
representation. The attention baseline projects that vector into latent tokens.

The model includes training-only modality dropout. External validation therefore
reports prespecified structure/conservation masking stress levels and safe-
fallback behavior instead of hiding missing annotations behind imputation.
Stage 04 also computes residue-centered AlphaFold geometry, including contact
counts, long-range contacts, local pLDDT, distance, confidence, hydrophobicity,
and charge environment.

This is a testable methodological contribution, not proof of novelty or clinical
utility by itself. Comparisons against simple ESM, conservation, combined
logistic, LightGBM, concatenation, gated fusion, and legacy pooled-vector
attention baselines remain mandatory.

## Temporal ClinVar construction

Later ClinVar missense records are not required to appear in the fixed dbNSFP
snapshot. Stage 09 can parse standard RefSeq protein HGVS consequences from the
ClinVar `Name` field; Stage 04 maps RefSeq identifiers through UniProt and checks
the reference amino acid. Unsupported, ambiguous, non-standard, or mismatched
consequences fail closed. One deterministic primary consequence is then selected
per genomic variant.

This prevents a fixed annotation snapshot from collapsing a large temporal
candidate pool to only the small subset already represented in dbNSFP. It does
not turn later UniProt, AlphaFold, or dbNSFP covariates into a fully historical
deployment replay, so that limitation must remain in the paper.

Temporal identity is based primarily on stable ClinVar VariationID, with a
genomic fallback only when a stable ID is absent. Ambiguous remaps fail closed.
Every retained external candidate must also have current nonconflicting,
label-consistent contributing SCV evidence and at least one matching SCV that is
new or version-increased after the frozen cutoff. Review states that claim
multiple submitters require at least two distinct matching submitters. Aggregate
labels that did not change may remain eligible only through this genuine new
submission-level evidence; model scores and predictor availability never affect
cohort membership.

Pinned public-data acquisition, transformation, checksums, licenses, and the
ClinVar/ProteinGym contracts are documented in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Quick local start

```powershell
git clone https://github.com/asifahamed11/VariFuse.git
Set-Location VariFuse
python -m venv .venv-publication
.\.venv-publication\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt

python run_pipeline.py --local-publication --run-id reliability_temporal_v1 `
  --confirmation-split-seeds 1701 2903 4159 6841 7919 --dry-run
python run_pipeline.py --local-publication --run-id reliability_temporal_v1 `
  --confirmation-split-seeds 1701 2903 4159 6841 7919
```

The one-command route runs the primary experiment in the required order:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 08b -> 09
   -> 10 -> 14 -> 11 -> 12 -> 13
```

For easier recovery, use the segmented commands in
[LOCAL_PUBLICATION_RUN.md](LOCAL_PUBLICATION_RUN.md). They preserve the same
prespecified fixed-configuration confirmation split seeds, isolate the three Stage 10
datasets for easier recovery, bind the exact 2024-06/2026-08 ClinVar snapshot
pair, and capture the resolved software/hardware environment.

## Local resource behavior

The local profile uses ESM2 `esm2_t33_650M_UR50D` in FP16, one protein per GPU
batch, a 1,024-token budget, all label-independent DMS rows, a project-local
model cache, deterministic cuBLAS settings, an auto-discovered project-local
MMseqs2 binary, and strict external/homology checks. The model and maximum 1,022-
residue inference window have been smoke-tested on the GTX 1660 6 GB GPU.

Stage 10 remains the main bottleneck and a full all-assay DMS run can take many
hours or longer. The runner requires at least 15 GiB free before local ESM
extraction; the complete sources, AlphaFold structures, caches, embeddings, and
artifacts require substantially more. Stage 10 context caches and Stage 14's
Optuna/repeat checkpoints allow the identical command to continue after an
interruption. Close other GPU-heavy applications and never mix artifacts from
different run IDs.

## Result map

| Evidence | Authoritative artifact | Correct interpretation |
|---|---|---|
| Primary internal comparison | `outputs\14_tuning\architecture_selection.json` | Nested outer-fold estimate after inner-only selection |
| Primary row predictions | `outputs\14_tuning\nested_tuning_oof.npz` | Aligned internal OOF; never replaced by confirmation runs |
| Split-seed confirmation | `outputs\14_tuning\confirmatory_repeated_cv_results.json` | Fixed-configuration training-procedure stability, not new HPO |
| Deployment bundles | `outputs\11_train_and_evaluate\models\` | Locked models used for external scoring |
| External evidence | `outputs\12_external_validation\external_validation.json` | Unique-variant ClinVar, assay-macro DMS, uncertainty, baselines, robustness |
| Figure audit | `outputs\13_figures\figure_manifest.json` | Required figure and publication-contract completion status |
| Post-result research extensions | `outputs\15_research_extensions\` | Exploratory subgroup, aggregation, controls and functional sensitivity work |

Stage 12 suppresses inferential clinical comparisons when the de-overlapped
ClinVar cohort is underpowered. Missing-modality stress tests are descriptive.
`publication_complete=true` means the required software contract passed; it is
not a claim that the proposed model won or that the study is publishable.

If reliability-residual fusion does not beat a prespecified simple baseline,
report that result. Do not choose a favorable split seed, assay, mask level, or
subgroup after seeing performance.

## Verification

```powershell
pytest -q
ruff check src tests run_pipeline.py tools
python -m compileall -q src tests run_pipeline.py tools
```

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full run audit and manuscript
reporting checklist.

The Git repository provides code, pinned acquisition recipes and compact metrics.
It does not contain the large embeddings, fitted weights, private manuscript
files or all upstream datasets. Generating figures from compact metrics and
reproducing model fits are separate workflows. Code uses the existing [MIT
license](LICENSE); upstream resources retain their own terms.
