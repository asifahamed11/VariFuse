# ছোট validation ও full run

## ছোট run আবার চালানো

PowerShell-এ:

```powershell
Set-Location 'I:\VariFuse_2'
$Python = '.\.venv-publication\Scripts\python.exe'
& $Python -m pip check
& $Python -m pytest -q
& $Python -m ruff check src tests tools kaggle run_pipeline.py
& $Python tools/run_small_validation.py --output validation_runs/small_repeat_v1
```

শেষ command বাস্তব প্রস্তুত করা dataset থেকে ছোট নমুনা বানায়, local ESM-2
650M weights দিয়ে নতুন features ও masked/WT scores বের করে, পাঁচ fold-এ train
করে এবং evaluation ও ablation সংরক্ষণ করে। নতুন run-এর জন্য নতুন output নাম দিন।
Environment বা model weights পুনরায় download করার দরকার নেই।

কোনো ধাপে বন্ধ হলে একই output দিয়ে সেই ধাপ চালান:

```powershell
& $Python tools/run_small_validation.py --output validation_runs/small_repeat_v1 --phase extract
& $Python tools/run_small_validation.py --output validation_runs/small_repeat_v1 --phase train
```

Extraction content cache ব্যবহার করে। Training পাঁচ fold আবার চালায়।
ডেটা বা extraction code বদলে গেলে hash check training বন্ধ করবে; তখন নতুন
নমুনা অথবা নতুন extraction প্রয়োজন।

## কী যাচাই হয়

- ২০৪টি internal variant, ৬৪টি পুরোনো homology group; ৩২টি external ClinVar
  variant; ২টি ProteinGym assay থেকে ৩২টি variant।
- পাঁচটি neural architecture: concatenation, gated fusion, cross attention,
  reliability residual, evidential residual। সঙ্গে ESM logistic ও LightGBM।
- Evidential model-এর দুটি training ablation: RCDI auxiliary losses বন্ধ এবং
  quality attenuation ছাড়া পুরোনো gate। প্রতিটি model সর্বোচ্চ ৫ epoch,
  প্রতি fold-এ ১ ensemble member; architecture budget ছোট ও স্থির।
- Fit, early stopping, calibration, threshold selection ও test group আলাদা।
  Preprocessor শুধু fitting rows-এ fit হয়। External labels threshold বা model
  নির্বাচনে ব্যবহার হয় না।
- Neural checkpoint save/load prediction মিল এবং auxiliary evidence না থাকলে
  zero gate পরীক্ষা। ClinVar classification ও DMS assay-level functional
  Spearman আলাদা evaluation।

এটি engineering validation। ছোট protein ও class coverage দেখে নমুনা নেওয়া হয়েছে;
এটি natural-prevalence benchmark নয়। Raw dbNSFP থেকে Stage 01–09 এখানে আবার
চালানো হয় না: আগের Stage 08/09 tables নেওয়া হয়, বর্তমান external schema alignment
ও Stage 10 extraction functions চালানো হয়। পুরোনো homology/provenance inherited;
external dataset-এর নতুন homology clustering হয় না। Stage 14-এর পূর্ণ nested HPO,
confirmation repeats, Stage 11/12/13 production artifact chain এই ছোট runner
চালায় না। সেগুলোর জন্য নিচের full command ব্যবহার করুন।

ফল `validation_runs/small_20260909/`-এ:

- `dataset_manifest.json`: নমুনার সংখ্যা, selection ও source hashes।
- `environment-lock.json`, `pytest.log`, `extraction.log`, `training.log`:
  environment ও চালানোর প্রমাণ।
- `results.json`: ছোট run-এর metrics ও checks।
- `ablation_comparison.csv`, `ablation_comparison.png`: model তুলনা।
- `partitions.json`, `oof_predictions.csv`, `clinvar_predictions.csv`,
  `dms_predictions.csv`, `models/`: splits, predictions ও checkpoints।

## সম্পূর্ণ dataset দিয়ে চালানো

বর্তমান code-এর জন্য নতুন run ID ব্যবহার করুন। আগের `reliability_temporal_v1`
outputs-এর source hashes পুরোনো code-এর; সেগুলোর সঙ্গে নতুন training মেশাবেন না।

```powershell
Set-Location 'I:\VariFuse_2'
$Python = '.\.venv-publication\Scripts\python.exe'
& $Python run_pipeline.py --local-publication `
  --run-id full_20260909_v1 `
  --confirmation-split-seeds 1701 2903 4159 6841 7919
```

এটি `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 08b → 09 → 10 → 14 → 11 → 12 → 13`
চালাবে। নতুন পূর্ণ run হবে `publication_runs/full_20260909_v1/`-এ। Full training
ছোট runner-এর ৫ epoch বা ছোট architecture ব্যবহার করে না; production tuning ও
training budget প্রযোজ্য। Full run এই validation কাজের অংশ হিসেবে শুরু করা হয়নি।

একসঙ্গে সব চালানোর বদলে stage ধরে চালানো ও interruption recovery-এর বিস্তারিত
[LOCAL_PUBLICATION_RUN.md](LOCAL_PUBLICATION_RUN.md)-এ আছে। বড় dbNSFP এবং সব
ProteinGym assay নিয়ে full ESM extraction দীর্ঘ সময় নেবে; ছোট protein-এর এই
run থেকে পুরো সময় নির্ভরযোগ্যভাবে অনুমান করা যাবে না।

শেষে `outputs/14_tuning/architecture_selection.json`-এ primary nested OOF,
`outputs/12_external_validation/external_validation.json`-এ external results এবং
`figures/figure_manifest.json`-এ `completed` ও `publication_complete` দেখুন।
শুধু ছোট validation পাস করা পূর্ণ গবেষণা বা proposed model-এর শ্রেষ্ঠত্ব প্রমাণ করে না।
