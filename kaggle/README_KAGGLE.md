# Kaggle-এ সম্পূর্ণ Stage 10, 14, 11, 12, 13

## কোন ZIP-এ কী আছে

- `VariFuse_Kaggle_Data.zip`: `full_stepwise_v1`-এর সম্পূর্ণ প্রস্তুত Stage 07–09
  datasets, original provenance/audits, ESM-2 650M pretrained weights, এবং export-এর
  সময় পর্যন্ত সম্পূর্ণ লেখা ও যাচাইকৃত internal ESM cache। এটি small dataset নয়।
- `VariFuse_Kaggle_Code.zip`: source code, tests, dependency list, setup utility,
  Kaggle runner ও `VariFuse_Kaggle.ipynb` notebook। `.venv` অন্তর্ভুক্ত নয়।

এই দুটি ZIP একই export-এর pair। অন্য version-এর code/data মেশালে setup বন্ধ হবে।
Raw dbNSFP, AlphaFold structure archive বা পুরোনো environment upload লাগবে না;
Stage 01–09 এখানে আবার চালানো হবে না।

## প্রথমবার setup

1. Kaggle-এ দুটি **Private Dataset** তৈরি করে একটিতে Data ZIP, অন্যটিতে Code ZIP
   upload করুন। ZIP expanded অবস্থায় থাকলেও setup তা খুঁজে পাবে।
2. Code ZIP খুলে `VariFuse_2/VariFuse_Kaggle.ipynb` notebook Kaggle-এ import করুন।
3. Notebook-এর **Add Input** থেকে ওই দুটি dataset যোগ করুন।
4. Accelerator হিসেবে **GPU T4 x2** এবং dependency installation-এর জন্য Internet
   চালু করুন। Runner দুইটি T4 না পেলে পরিষ্কার error দিয়ে থামবে।
5. Notebook-এর setup cells ক্রমানুসারে চালান। প্রথম cell দুই dataset মিলিয়ে
   `/kaggle/working/VariFuse_2` তৈরি করবে এবং প্রতিটি file-এর SHA-256 যাচাই করবে।
6. Dependency installation cell চালিয়ে package পরিবর্তন হলে notebook session
   restart করুন, তারপর setup ও preflight cell আবার চালান। Verified files পুনরায়
   copy হয় না। Kaggle-এর CUDA PyTorch dependency command বদলায় না।
7. Preflight পাস হলে stage cells একটি করে চালান, অথবা **Run All** ব্যবহার করুন।

Kaggle-এর runtime packages সময়ের সঙ্গে বদলায়; preflight actual imports, CUDA,
GPU count এবং data/source contracts পরীক্ষা করে। Platform reference:
https://github.com/Kaggle/docker-python এবং https://www.kaggle.com/docs/notebooks ।

## সঠিক ক্রম

```text
10 internal → 10 ClinVar → 10 DMS → 14 tuning → 11 training → 12 evaluation → 13 figures
```

Notebook-এ প্রতিটি stage-এর আলাদা cell আছে। একই session-এ terminal থেকে একসঙ্গে
চালানোর সমতুল্য command:

```bash
cd /kaggle/working/VariFuse_2
python kaggle/kaggle_runner.py all
```

শুধু একটি dataset বা stage:

```bash
python kaggle/kaggle_runner.py stage10 --dataset internal
python kaggle/kaggle_runner.py stage10 --dataset clinvar
python kaggle/kaggle_runner.py stage10 --dataset dms
python kaggle/kaggle_runner.py stage14
python kaggle/kaggle_runner.py stage11
python kaggle/kaggle_runner.py stage12
python kaggle/kaggle_runner.py stage13
```

Stage 14 production search/training budgets ও দুই reliability architecture-সহ
পূর্ণ comparator panel ব্যবহার করে। Fixed confirmation seeds:
`1701 2903 4159 6841 7919`। Small-validation epochs বা sample এখানে প্রযোজ্য নয়।
DMS sampling policy `all`: `dms_max_rows_per_assay=500` লেখা থাকলেও `all` policy-তে
৫০০-row cap ব্যবহার হয় না।

## সময়সীমা ও resume

সব stage একটি Kaggle session-এ শেষ হবে এমন নিশ্চয়তা নেই। Session বন্ধ হওয়ার আগে
Kaggle-এর persistence/output-saving ব্যবস্থা ব্যবহার করে পুরো working project
সংরক্ষণ করুন; mounted input datasets নতুন ফল নিজে থেকে সংরক্ষণ করে না। দীর্ঘ run
শুরুর আগে notebook-এ নিজের quota/session limits দেখে নিন।

নতুন session-এ সংরক্ষিত working project ফিরিয়ে একই location-এ রাখুন, preflight
চালিয়ে অসম্পূর্ণ stage আবার চালান। Stage 10-এর completed `.npz` cache বাঁচিয়ে
রাখলে successful contexts আবার গণনা করতে হয় না; `.tmp` final output নয়। Stage 14
resume করতে tuning directory-এর SQLite database, split plans ও checkpoints-সহ
সম্পূর্ণ directory রাখতে হবে। Source/configuration বদলাবেন না।

উদাহরণ: internal শেষ, ClinVar এখনো শেষ হয়নি:

```bash
python kaggle/kaggle_runner.py all --start-at 10-clinvar
```

প্রস্তুত input, model weights ও cache working directory-তে copy হয়; পরে নতুন
embeddings/models-এর জন্যও disk space লাগবে। Notebook-এর setup cell free space
দেখায়। চলমান local run এই export তৈরির সময় বন্ধ বা পরিবর্তন করা হয়নি।

## ফল কোথায় পাবেন

- `outputs/14_tuning/architecture_selection.json`: primary nested OOF results।
- `outputs/11_train_and_evaluate/results.json`: deployment training results।
- `outputs/12_external_validation/external_validation.json`: ClinVar ও DMS evaluation।
- `figures/figure_manifest.json`: `completed` এবং `publication_complete` flags।
- `kaggle_environment.json`: Kaggle environment ও source hashes।

এখানে তৈরি source ZIP-এর Stage 10 CLI-তে একটি provenance fix আছে: extraction-এর
সময় upstream validation পুনরায় চালানো হয়, ফলে validated production extraction
ভুলভাবে `explicitly_skipped_nonpublication` লেখা হয় না। Local চলমান source ফাইল
বদলানো হয়নি; export manifest-এ ওই এক লাইনের পরিবর্তন নথিভুক্ত আছে। Numerical
ESM extraction এবং cache format একই আছে। GPU batch-এর কারণে সামান্য numerical
rounding পার্থক্য হতে পারে; bitwise cross-hardware equality দাবি করা হচ্ছে না।
