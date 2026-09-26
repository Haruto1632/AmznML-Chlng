# Baseline Experiment v1 — Member A

**Status:** ✅ Implemented, ready to run

**Goal:** Establish a reproducible classical ML baseline using conservative normalization, multi-strategy blocking, and string similarity features with LightGBM.

---

## Architecture

```
student_resource/dataset/train/
  ├── train_source1.tsv (2.2M entities)
  ├── train_source2.tsv (5.0M entities)
  ├── train_source3.tsv (5.3M entities)
  └── train_ground_truth.tsv (7.6M match pairs)
         │
         ▼
    ┌─────────────────────┐
    │  1. Normalization    │
    │  - Lowercase         │
    │  - Legal suffixes    │
    │  - Transliteration   │
    │  - Token extraction  │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────┐
    │  2. Multi-Strategy   │
    │     Blocking         │
    │  - Exact token       │
    │  - Token overlap     │
    │  - N-gram LSH        │
    │  - Phonetic          │
    │  - Address numeric   │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────┐
    │  3. Feature Eng.     │
    │  - Name Jaccard      │
    │  - TF-IDF cosine     │
    │  - Levenshtein       │
    │  - Jaro-Winkler      │
    │  - Address overlap   │
    │  - Country exact     │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────┐
    │  4. LightGBM Model   │
    │  - Binary classifier │
    │  - Early stopping    │
    │  - Train/val split   │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────┐
    │  5. Calibration      │
    │  - Threshold sweep   │
    │  - Maximize F_0.5    │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────┐
    │  6. Output           │
    │  - matching_results  │
    │  - candidate_pairs   │
    │  - Metrics report    │
    └─────────────────────┘
```

---

## Files in This Experiment

| File | Purpose |
|---|---|
| `experiment_config.yaml` | All hyperparameters and config |
| `normalization.py` | Text normalization (conservative) |
| `blocking.py` | Multi-strategy blocking with inverted indices |
| `features.py` | Classical string similarity features |
| `model.py` | LightGBM/XGBoost wrapper |
| `pipeline.py` | End-to-end orchestrator |
| `run_experiment.py` | **Main entry point** |
| `test_setup.py` | Validate dependencies before running |
| `INSTALL.md` | Installation instructions |
| `README.md` | This file |

**Generated outputs:**
- `output/matching_results.tsv` — final predictions (validation set)
- `output/candidate_pairs.tsv` — blocking candidates (validation set)
- `RESULTS.md` — experiment report with metrics

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm rapidfuzz datasketch unidecode tqdm pyyaml
```

See `INSTALL.md` for detailed instructions and troubleshooting.

### 2. Test Setup

```bash
cd "c:\Users\DELL\OneDrive\Documents\GitHub\Amazon ML challenge"
python experiments/member_a/baseline_v1/test_setup.py
```

Expected output: `✅ ALL TESTS PASSED`

### 3. Run Experiment

```bash
python experiments/member_a/baseline_v1/run_experiment.py
```

**Expected runtime:** 10–30 minutes (depends on hardware)

**Expected memory:** ~8–12 GB RAM

---

## Configuration

Edit `experiment_config.yaml` to tune hyperparameters:

### Key Parameters

```yaml
blocking:
  max_candidates_per_s1: 50              # hard cap on candidates
  token_overlap_threshold: 0.2           # min Jaccard for token overlap
  ngram_lsh_threshold: 0.3               # min Jaccard for LSH

features:
  tfidf_max_features: 5000               # char n-gram vocabulary size

model:
  type: "lightgbm"                       # or "xgboost" or "histgbm"
  n_estimators: 300
  learning_rate: 0.05
  early_stopping_rounds: 30

training:
  val_split: 0.2                         # 80/20 train/val
  sample_train_pairs: null               # null = use all pairs
```

---

## Expected Metrics

Based on the challenge constraints and dataset characteristics, baseline targets:

| Metric | Target Range |
|---|---|
| Blocking recall | 92–96% |
| Macro F_0.5 | 0.65–0.75 (baseline) |
| Macro Precision | 0.70–0.80 |
| Macro Recall | 0.60–0.70 |
| Singleton accuracy | 0.80–0.90 |
| Avg candidates per S1 | 20–50 |

Actual results will be written to `RESULTS.md` after the run.

---

## Isolation from Shared Codebase

This experiment is fully isolated:

✅ **No modifications to `code/`** — only imports from shared modules
✅ **Self-contained** — all experiment code in `experiments/member_a/baseline_v1/`
✅ **Independent data flow** — reads from `student_resource/dataset/`, writes to `experiments/member_a/baseline_v1/output/`
✅ **No git conflicts** — teammates work in their own `experiments/member_*/` directories

### Reused Shared Modules (read-only)
- `code/business_entity_resolution/src/shared/data_loader.py` — TSV loaders
- `code/business_entity_resolution/src/shared/schemas.py` — column constants
- `code/business_entity_resolution/src/evaluation/evaluator.py` — F_0.5 computation
- `code/business_entity_resolution/src/calibration/calibrator.py` — threshold sweep
- `code/business_entity_resolution/src/normalization/country_mapper.py` — country normalization

---

## Next Steps After Baseline

1. **Error analysis:** Inspect false positives and false negatives
2. **Blocking improvements:** Add embedding-based blocking, tune thresholds
3. **Feature engineering:** Add more sophisticated features (edit distance bands, phonetic variants)
4. **Model ensemble:** Combine LightGBM with a small cross-encoder
5. **Transitive closure:** Graph-based refinement for consistent predictions

---

## Comparison with Team

After all 3 members run their independent baselines:
1. Compare F_0.5, precision, recall, blocking recall
2. Identify best-performing blocking strategies
3. Identify best-performing features
4. Merge successful components into unified architecture
5. Fine-tune the combined system

---

_This experiment is independent and can be deleted without breaking the shared codebase._
