# VariFuse: বর্তমান Q1-readiness সিদ্ধান্ত

হালনাগাদ: 2026-10-04; নিচের original comparison-এর assessment: 2026-09-20।

## সর্বশেষ completion update

সব আটটি control-এর 40টি fold এখন সম্পন্ন। Frozen clinical scoring-ও সম্পন্ন;
নতুন ফলগুলো post-result exploratory, untouched external confirmation নয়। Shared-input
LightGBM-এর development AP 0.92193 এবং clinical AP 0.96395; reliability control-এর
AP যথাক্রমে 0.90103 এবং 0.95675। Development paired AP difference +0.02090,
connected-group exploratory 95% interval [0.01664, 0.02570]। Original gated fusion-এর
সঙ্গে clinical difference +0.00150-এর interval [-0.00025, 0.00326]; স্পষ্ট superiority
প্রতিষ্ঠিত নয়। Owner-এর সম্মতিতে বিদ্যমান MIT license রাখা হয়েছে।

বর্তমান manuscript একটি comparative evaluation paper হিসেবে লেখা হয়েছে। নতুন untouched
functional evidence এখনও নেই; 19-variant sensitivity set independent validation নয়।
পুরোনো scalar-only LightGBM comparison ও নতুন shared-input comparison এক experiment নয়।
পূর্ণ-precision নতুন records: [publication/evidence](publication/evidence)।

## সিদ্ধান্ত

ছয়টি engineering/research solution-এর code path এখন আছে এবং lightweight অংশগুলো
চালানো হয়েছে। তবু বর্তমান evidence দিয়ে reliability-residual-কে শ্রেষ্ঠ architecture,
নতুন state of the art, বা clinically useful model বলা যাবে না। একটি সতর্ক method/evaluation
paper লেখা সম্ভব। Matched full training controls এখন সম্পন্ন; নতুন untouched external
evidence paper আরও শক্ত করতে পারে। Acceptance কোনো code change নিশ্চিত করতে পারে না।

## সবচেয়ে গুরুত্বপূর্ণ ফল

| Evaluation | Reliability AUPRC | তুলনা | AUPRC difference (reliability − comparator), 95% exploratory CI |
|---|---:|---|---:|
| Internal nested OOF | 0.9028 | raw ESM | +0.0557 [0.0458, 0.0666] |
| Internal nested OOF | 0.9028 | ESM + conservation | +0.0275 [0.0211, 0.0348] |
| Internal nested OOF | 0.9028 | LightGBM | +0.0223 [0.0144, 0.0313] |
| Internal nested OOF | 0.9028 | concatenation | −0.0068 [−0.0113, −0.0028] |
| Internal nested OOF | 0.9028 | gated fusion | −0.0136 [−0.0187, −0.0094] |
| Temporal ClinVar | 0.9574 | ESM + conservation | +0.0263 [0.0206, 0.0326] |
| Temporal ClinVar | 0.9574 | concatenation | −0.0022 [−0.0043, −0.0003] |
| Temporal ClinVar | 0.9574 | gated fusion | −0.0051 [−0.0071, −0.0032] |

Intervals gene-clustered এবং frozen predictions-এর উপর conditional; training/model-selection
uncertainty অন্তর্ভুক্ত নয় এবং multiple-comparison adjustment করা হয়নি।

Temporal ClinVar-এর same-coverage comparisons-এ reliability AlphaMissense-এর চেয়ে সামান্য
ভালো (+0.0059 AUPRC), কিন্তু REVEL-এর চেয়ে −0.0100 এবং MetaRNN-এর চেয়ে −0.0298। Predictor
গুলোর training-overlap independence এখানে প্রতিষ্ঠিত হয়নি, তাই এগুলো contextual comparison।

ProteinGym-এর corrected official-style hierarchy-তে reliability Spearman 0.41345। এটি raw
ESM/ESM+conservation-এর প্রায় সমান এবং gated fusion 0.38601-এর চেয়ে বেশি। Evaluated cohort
filtered single substitutions; full ProteinGym leaderboard-এর সঙ্গে সরাসরি সমতুল্য নয়।

Structure-equipped gene-disjoint sensitivity subset-এ মাত্র ১৯টি variant, HMGCR-এর একটি assay।
Reliability Spearman −0.3719 এবং AUROC 0.45। নমুনা খুব ছোট এবং আগে দেখা dataset থেকে নেওয়া,
তাই এটি negative/underpowered sensitivity result; independent validation নয়।

## ছয়টি কাজের completion gate

1. **Subgroup/gene/paired analysis:** সম্পন্ন। Internal ও ClinVar দুটির 1,000-replicate
   gene-clustered intervals সংরক্ষিত।
2. **ProteinGym/public comparator:** aggregation এবং safe same-row importer সম্পন্ন। Public
   per-variant score file অনুপস্থিত, তাই direct DMS public-model comparison pending।
3. **Same-input baseline:** code, locked partitions, resume এবং save/reload check সম্পন্ন। Full
   5-fold results এবং frozen clinical scoring সম্পন্ন।
4. **Gate/component ablation:** পাঁচটি matched neural control তৈরি; engineering smoke pass। Full
   5-fold results এবং frozen clinical scoring সম্পন্ন।
5. **Independent evidence:** available data দিয়ে sensitivity analysis সম্পন্ন; adequacy fail। নতুন
   untouched cohort acquisition pending।
6. **Versioned release:** compact source archive, package snapshot ও SHA256 manifest তৈরি করা যায়।
   Owner-এর সম্মতিতে বিদ্যমান MIT license রাখা হয়েছে।

Untouched cohort ছাড়া submission করলে paper-এর কেন্দ্র হওয়া উচিত rigorous
leakage-aware evaluation, temporal validation, modality failure এবং negative architectural result।
Reliability-residual superiorityকে কেন্দ্র করলে reviewer objection শক্ত হবে।

চালানোর command এবং file map: [RESEARCH_EXTENSIONS_BN.md](RESEARCH_EXTENSIONS_BN.md)।
