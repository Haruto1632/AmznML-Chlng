# Member A — Independent Experiments

This directory contains my independent entity resolution experiments for the Amazon ML Challenge.

Each experiment is self-contained and can be run independently without modifying the shared codebase in `code/`.

---

## Experiments

### `baseline_v1/` — Classical ML Baseline

**Goal:** Establish a reproducible baseline using conservative normalization, multi-strategy blocking, and classical string similarity features with LightGBM.

**Architecture:**
```
Data → Normalization → Multi-strategy Blocking → Feature Engineering → LightGBM → Calibration → Output
```

**Key features:**
- Conservative text normalization (lowercase, legal suffix expansion, transliteration)
- 5 blocking strategies (exact token, token overlap, n-gram LSH, phonetic, address numeric)
- Classical features (Jaccard, Levenshtein, Jaro-Winkler, TF-IDF cosine)
- LightGBM binary classifier
- Threshold calibration on validation set
- Evaluation: macro F_0.5, precision, recall, singleton accuracy, blocking recall

**Run:**
```bash
cd "c:\Users\DELL\OneDrive\Documents\GitHub\Amazon ML challenge"
python experiments/member_a/baseline_v1/run_experiment.py
```

**Status:** ✅ Implemented, ready to run

---

## Future Experiments

### `baseline_v2/` — Improved Blocking (planned)
- Add embedding-based blocking (sentence-transformers)
- Tune blocking thresholds per strategy
- Target: >98% blocking recall

### `ensemble_v1/` — Model Ensemble (planned)
- Ensemble LightGBM + small cross-encoder
- Graph-based transitive closure
- Bipartite matching constraints

### `analysis_v1/` — Error Analysis (planned)
- Analyze false positives vs false negatives
- Break down by country, name length, address completeness
- Identify systematic errors

---

## Shared Resources (Reused from `code/`)

My experiments import and reuse these fully-implemented modules:
- `code/business_entity_resolution/src/shared/data_loader.py`
- `code/business_entity_resolution/src/shared/schemas.py`
- `code/business_entity_resolution/src/evaluation/evaluator.py`
- `code/business_entity_resolution/src/calibration/calibrator.py`
- `code/business_entity_resolution/src/normalization/country_mapper.py`

No modifications to `code/` are made — experiments are fully isolated.

---

## Collaboration

Once baseline results are measured and analyzed, the team will:
1. Compare baseline approaches from all 3 members
2. Identify best-performing components
3. Merge them into a unified team architecture
4. Fine-tune the combined system

My experiments remain isolated until that decision point.
