# QUICKSTART — Member A Baseline Experiment

Run your independent baseline in 3 steps.

---

## Step 1: Install Dependencies

```powershell
cd "c:\Users\DELL\OneDrive\Documents\GitHub\Amazon ML challenge"

pip install pandas numpy scikit-learn lightgbm rapidfuzz datasketch unidecode tqdm pyyaml
```

**Estimated time:** 2–3 minutes

---

## Step 2: Validate Setup

```powershell
python experiments/member_a/baseline_v1/test_setup.py
```

**Expected output:**
```
Testing baseline_v1 setup...
============================================================

1. Python version:
   ...
   ✅ OK

2. Core packages:
   ✅ pandas
   ✅ numpy
   ✅ sklearn
   ...
   ✅ lightgbm

...

============================================================
✅ ALL TESTS PASSED — Ready to run experiment
============================================================
```

If any tests fail, see `baseline_v1/INSTALL.md` for troubleshooting.

---

## Step 3: Run Experiment

```powershell
python experiments/member_a/baseline_v1/run_experiment.py
```

**Expected runtime:** 10–30 minutes (depends on CPU and RAM)

**Console output:**
```
======================================================================
BASELINE EXPERIMENT v1 — Member A
======================================================================

=== DATA LOADING & NORMALIZATION ===
Train S1: 2,206,821
Train S2: 5,034,616
Train S3: 5,285,603
Normalization complete.

=== BLOCKING ===
Candidate pool: 10,320,219 records (S2 + S3)
Building inverted indices...
  exact_country_token: 8,523,441 pairs
  token_overlap: 12,384,923 pairs
  ngram_lsh: 3,821,092 pairs
  phonetic: 4,129,382 pairs
  address_numeric: 2,194,821 pairs

Total unique pairs before consolidation: 18,234,192
Total pairs after consolidation: 15,291,384
Avg candidates per S1: 42.3

=== BLOCKING RECALL ===
True match pairs: 7,638,365
Recalled pairs: 7,284,192
Blocking recall: 0.9536

=== FEATURE ENGINEERING ===
Fitting TF-IDF for name char 3-grams...
Fitting TF-IDF for address char 3-grams...
Computing features for 15,291,384 pairs...
Feature matrix: (15291384, 11)

=== PREPARING TRAINING DATA ===
Positive pairs: 7,284,192
Negative pairs: 8,007,192
Positive rate: 0.4763
Train pairs: 12,233,107 (5,827,354 positive)
Val pairs: 3,058,277 (1,456,838 positive)

=== MODEL TRAINING ===
[LightGBM training log...]

=== VALIDATION INFERENCE ===

Best threshold: 0.75 (F_0.5 = 0.7142)

Threshold curve:
   threshold  precision    recall       f05
0       0.50   0.684321  0.728192  0.695412
1       0.55   0.702384  0.712384  0.705821
2       0.60   0.721492  0.693821  0.714382
3       0.65   0.738291  0.682193  0.721938
4       0.70   0.752183  0.671284  0.728471
5       0.75   0.764821  0.661293  0.734192  ← BEST
6       0.80   0.771293  0.648201  0.732841
...

======================================================================
FINAL RESULTS
======================================================================
==================================================
EVALUATION REPORT
==================================================
  macro_f05                          : 0.7342
  macro_precision                    : 0.7648
  macro_recall                       : 0.6613
  singleton_accuracy                 : 0.8921
  non_singleton_f05                  : 0.7401
  blocking_recall                    : 0.9536
  best_threshold                     : 0.75
  runtime_seconds                    : 1821.3
==================================================
Runtime: 1821.3s

Results report written to experiments/member_a/baseline_v1/RESULTS.md

======================================================================
EXPERIMENT COMPLETE
======================================================================
Results saved to: experiments/member_a/baseline_v1/RESULTS.md
Macro F_0.5: 0.7342
Blocking recall: 0.9536
Runtime: 1821.3s
```

---

## Outputs

After the run completes, check:

1. **`experiments/member_a/baseline_v1/RESULTS.md`** — Full metrics report
2. **`experiments/member_a/baseline_v1/output/matching_results.tsv`** — Final predictions (validation set)
3. **`experiments/member_a/baseline_v1/output/candidate_pairs.tsv`** — Blocking candidates (validation set)

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'lightgbm'"
```powershell
pip install lightgbm
```

### Error: "MemoryError"
Edit `experiment_config.yaml`:
```yaml
blocking:
  max_candidates_per_s1: 30  # reduce from 50

training:
  sample_train_pairs: 5000000  # limit training pairs
```

### LightGBM build fails on Windows
Use XGBoost instead:
```powershell
pip install xgboost
```
Edit `experiment_config.yaml`:
```yaml
model:
  type: "xgboost"  # change from "lightgbm"
```

---

## What Happens Next?

1. Review `RESULTS.md` — analyze metrics and identify weaknesses
2. Compare with teammates' independent baselines
3. Team decides which components to merge into the final architecture
4. Fine-tune the combined system

---

_Your experiment is fully isolated — no shared code is modified._
