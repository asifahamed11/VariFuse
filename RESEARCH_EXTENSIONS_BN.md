# ছয়টি Q1-readiness extension: অবস্থা ও চালানোর নিয়ম

এই extension-গুলো আগের audited `outputs` বদলায় না। নতুন সব artifact
`outputs/15_research_extensions`-এ যায়। এগুলো original result দেখার পরে করা হয়েছে,
তাই ফলগুলো exploratory; নতুন untouched confirmation হিসেবে লেখা যাবে না।

## বর্তমান অবস্থা

| কাজ | এখন কী হয়েছে | দাবি করার সীমা |
|---|---|---|
| 1. subgroup, gene-level ও paired analysis | Internal nested OOF এবং temporal ClinVar-এর analysis সম্পন্ন | Frozen prediction-ভিত্তিক exploratory interval; training uncertainty নয় |
| 2. ProteinGym aggregation/public comparison | Official protein → category → equal-category aggregation সম্পন্ন; public per-variant score importer প্রস্তুত | বর্তমান filtered subset-কে full leaderboard বলা যাবে না; public score file না দেওয়া পর্যন্ত direct public comparison অসম্পূর্ণ |
| 3. same-input baseline | Logistic, LightGBM ও MLP-এর পাঁচ fold সম্পন্ন; identical clinical cohort-এ frozen scoring সম্পন্ন | Post-result exploratory; neural MLP আলাদাভাবে retune করা হয়নি |
| 4. gate/component ablation | পাঁচ component model-এর সব 25 fold সম্পন্ন; মোট 8 model-এর 40 fold সম্পন্ন | Inherited training recipe; component differences optimal retuned performance নয় |
| 5. independent functional evidence | Label-blind structure-equipped sensitivity cohort তৈরি ও score করা হয়েছে | মাত্র ১৯ variant/১ assay; independent validation-এর জন্য অপর্যাপ্ত; নতুন untouched cohort এখনও দরকার |
| 6. reproducible release | Input/code hash, locked protocol, test এবং release capture যোগ হয়েছে | Project license owner নির্ধারণ করেননি; প্রকাশের আগে LICENSE নির্বাচন করতে হবে |

## দ্রুত analysis আবার চালানো

```powershell
Set-Location 'I:\VariFuse_2'
.\.venv-publication\Scripts\Activate.ps1
python -B tools\analyze_research_evidence.py --phase internal --bootstrap 1000
python -B tools\analyze_research_evidence.py --phase clinical --bootstrap 1000
python -B tools\analyze_research_evidence.py --phase dms
python -B tools\prepare_functional_validation.py
python -B tools\score_functional_validation.py
```

## Training একবারে একটি job

প্রথমে protocol check:

```powershell
python -B tools\run_research_controls.py --phase prepare
```

একটি model-এর একটি fold চালানোর command:

```powershell
python -B tools\run_research_controls.py --phase train --model hard_switch --fold 1
```

`--model`-এ নিচের আটটির একটি এবং `--fold`-এ 1 থেকে 5 দিন। অর্থাৎ মোট 40টি
স্বতন্ত্র, resumable job। একই completed command আবার দিলে checksum যাচাই করে skip করবে।

```text
reliability_control
fixed_quality_gate
hard_switch
no_modality_dropout
unbounded_residual
same_input_mlp
same_input_logistic
same_input_lightgbm
```

প্রতিটি job শেষ হওয়ার পর পরের fold দিন। সব 40টি শেষ হলে:

```powershell
python -B tools\summarize_research_controls.py
python -B tools\score_functional_validation.py
python -B tools\capture_research_release.py
```

`same_input_lightgbm` inner-fold grid selection করে এবং neural model-গুলো original final
training recipe/seed ব্যবহার করে। এগুলো সময়সাপেক্ষ; এই repair কাজের সময় full training
চালানো হয়নি। `--phase smoke` শুধু code-path check, scientific result নয়।

২০২৬-১০-০৪ update: আটটি model-এর ৪০টি full fold checkpoint এখন আছে। পুরোনো
`--phase summarize` pandas-এর object-string `row_ids` পড়তে গিয়ে NumPy error দেয়।
উপরের নতুন summary command checksum ও frozen row/label identity যাচাই করে সেগুলো
পড়বে এবং Unicode row IDs দিয়ে OOF export করবে। Training runner, locked protocol ও
original fold files বদলানো হয় না; training আবার চালানোর প্রয়োজন নেই।

## Public ProteinGym score যোগ করা

CSV-তে ঠিক `DMS_id,mutant,score` column লাগবে। `mutant` হবে `A123V` format। একই
evaluated variant subset-এ join না হলে script comparison প্রত্যাখ্যান করবে। উদাহরণ:

```powershell
python -B tools\analyze_research_evidence.py --phase dms `
  --public-score 'ESM1v|I:\scores\esm1v.csv|higher_is_functional'
```

## নতুন untouched cohort-এর শর্ত

নতুন dataset-এর source/release date, variant identity, reference allele/sequence check,
gene ও homology overlap, annotation coverage এবং outcome দেখার আগে locked protocol রাখতে
হবে। বর্তমান ProteinGym/ClinVar আবার subset করে তাকে untouched বলা যাবে না।

## যাচাই

```powershell
python -B -m pytest -q tests\test_research_extensions.py
python -B -m ruff check src tools tests run_pipeline.py
python -B tools\capture_research_release.py
```

মূল evidence files:

- `outputs/15_research_extensions/saved_prediction_analysis/internal_oof_metrics.csv`
- `outputs/15_research_extensions/saved_prediction_analysis/internal_paired_intervals.json`
- `outputs/15_research_extensions/saved_prediction_analysis/clinical_gene_macro.csv`
- `outputs/15_research_extensions/saved_prediction_analysis/clinical_paired_intervals.json`
- `outputs/15_research_extensions/saved_prediction_analysis/dms_corrected_aggregation.json`
- `outputs/15_research_extensions/functional_validation/functional_validation_results.json`
- `outputs/15_research_extensions/protocol.json`
