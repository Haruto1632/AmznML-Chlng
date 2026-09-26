# Architecture: Amazon ML Challenge — Business Entity Resolution

## Overview

Three independent business-record sources:

- **Source 1 (S1)** — deduplicated reference entities
- **Source 2 (S2)** — noisy business records
- **Source 3 (S3)** — noisy business records

**Task:** For every S1 entity, predict zero, one, or many matching entities from S2 and S3.

**Metric:** F_0.5 (precision-weighted, macro-averaged over all S1 entities).

---

## End-to-End Data Flow

```
dataset/train/                dataset/test/
train_source1.tsv             test_source1.tsv
train_source2.tsv             test_source2.tsv
train_source3.tsv             test_source3.tsv
train_ground_truth.tsv
        │
        ▼
┌─────────────────────────────────────────────────┐
│  1. DATA LOADING                                 │
│  src/shared/data_loader.py                       │
│  • Read all TSVs with sep="\t"                   │
│  • Validate schema (entity_id, business_name,    │
│    business_address, country)                    │
│  • Return typed DataFrames                       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  2. NORMALIZATION                [MEMBER A]      │
│  src/normalization/normalizer.py                 │
│  • Lowercase, strip whitespace                   │
│  • Normalize punctuation / legal suffixes        │
│  • "&" → "and", abbreviation expansion          │
│  • Token-order normalization                     │
│  • Address component cleanup                     │
│  • Transliteration (local, no API)               │
│  • Missing value handling                        │
│  OUTPUT: normalized DataFrame with extra cols:   │
│    business_name_normalized                      │
│    business_address_normalized                   │
│    (originals preserved)                         │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  3. MULTI-STRATEGY BLOCKING      [MEMBER A]      │
│  src/blocking/blocker.py                         │
│  Strategies (run in parallel, results unioned):  │
│  • exact_country_name_token                      │
│  • name_token_overlap                            │
│  • char_ngram_lsh                                │
│  • phonetic (Soundex/Metaphone)                  │
│  • address_token_overlap                         │
│  • country_aware                                 │
│  • embedding_ann (optional; pretrained encoder)  │
│                                                  │
│  OUTPUT per S1 entity:                           │
│    {source1_entity_id, candidate_entity_id,      │
│     blocking_reasons: [list of strategy names]}  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  4. CANDIDATE CONSOLIDATION      [MEMBER A]      │
│  src/candidate_generation/candidate_store.py     │
│  • Union candidates from all strategies          │
│  • Deduplicate pairs                             │
│  • Track blocking_reasons per pair               │
│  • Enforce candidate-count limits per S1 entity  │
│  • Measure: recall@candidates, reduction_ratio,  │
│    avg/median/p95 candidates per S1              │
│                                                  │
│  OUTPUT: candidate_pairs DataFrame               │
│    → also saved as output/candidate_pairs.tsv    │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  5. FEATURE ENGINEERING          [MEMBER B]      │
│  src/features/feature_builder.py                 │
│  Per candidate pair compute:                     │
│  NAME features:                                  │
│    exact_normalized_equality                     │
│    levenshtein_similarity                        │
│    jaro_winkler                                  │
│    token_jaccard                                 │
│    token_overlap                                 │
│    char_ngram_tfidf_cosine                       │
│    token_sort_ratio                              │
│    token_set_ratio                               │
│    length_diff                                   │
│    abbreviation_similarity                       │
│  ADDRESS features:                               │
│    normalized_exact_match                        │
│    token_jaccard                                 │
│    char_ngram_cosine                             │
│    edit_similarity                               │
│    numeric_token_overlap                         │
│    address_length_diff                           │
│  COUNTRY features:                               │
│    exact_match                                   │
│    normalized_country_similarity                 │
│  STRUCTURAL:                                     │
│    num_blocking_strategies                       │
│    strategy_flags (one per strategy)             │
│                                                  │
│  OUTPUT: feature DataFrame                       │
│    {source1_entity_id, candidate_entity_id,      │
│     feature_1 … feature_N}                       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  6. PAIRWISE MATCHING MODEL      [MEMBER B]      │
│  src/models/matcher.py                           │
│  Primary: GBM (LightGBM / XGBoost /             │
│           HistGradientBoosting)                  │
│  Optional ensemble:                              │
│    + small cross-encoder reranker               │
│  Training:                                       │
│    • Positives from train_ground_truth.tsv       │
│    • Hard negatives: same-name diff-address,     │
│      same-address diff-name, blocking candidates │
│      that are NOT true matches                   │
│    • Entity-level train/val split                │
│                                                  │
│  OUTPUT: predictions DataFrame                   │
│    {source1_entity_id, candidate_entity_id,      │
│     match_probability}                           │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  7. CALIBRATION & THRESHOLDING   [MEMBER C]      │
│  src/calibration/calibrator.py                   │
│  • Sweep thresholds 0.50 → 0.95 on val set      │
│  • Select threshold that maximises F_0.5         │
│  • Investigate per-evidence-type thresholds      │
│  • Singleton detection (no candidate above τ     │
│    → empty prediction → 1.0 for true singletons) │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  8. OUTPUT GENERATION            [MEMBER C]      │
│  src/pipeline/output_writer.py                   │
│  • matching_results.tsv                          │
│    source1_entity_id \t matched_entity_ids       │
│  • candidate_pairs.tsv                           │
│    source1_entity_id \t candidate_entity_ids     │
│  Guarantees:                                     │
│    ✓ Every S1 test entity has exactly one row    │
│    ✓ Empty lists for singletons                  │
│    ✓ No duplicates within any ID list            │
│    ✓ Only S2/S3 IDs that exist in test set       │
│    ✓ Every matched ID ∈ candidate set            │
│    ✓ No S1→S1 matches                            │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  9. VALIDATION                   [MEMBER C]      │
│  utils/validate_submission.py (provided)         │
│  src/evaluation/evaluator.py (internal)          │
│  Reports:                                        │
│    Precision, Recall, F_0.5                      │
│    Blocking recall, Reduction ratio              │
│    Avg/median/p95 candidates per S1              │
│    Singleton accuracy                            │
└─────────────────────────────────────────────────┘
```

---

## Module Ownership Map

| Module path | Owner | Imports from |
|---|---|---|
| `src/shared/data_loader.py` | shared (C writes first) | — |
| `src/shared/schemas.py` | shared (C writes first) | — |
| `src/normalization/` | **Member A** | `shared` |
| `src/blocking/` | **Member A** | `shared`, `normalization` |
| `src/candidate_generation/` | **Member A** | `shared`, `blocking` |
| `src/features/` | **Member B** | `shared`, `normalization` |
| `src/models/` | **Member B** | `shared`, `features` |
| `src/scoring/` | **Member B** | `shared`, `models` |
| `src/evaluation/` | **Member C** | `shared` |
| `src/calibration/` | **Member C** | `shared`, `scoring` |
| `src/pipeline/` | **Member C** | everything |

---

## Stable Interfaces (do not change without team agreement)

### `blocking.generate_candidates(source1, source2, source3, config) -> DataFrame`

```
Returns DataFrame with columns:
  source1_entity_id  (str)
  candidate_entity_id (str)
  blocking_reasons   (list[str])
```

### `features.build_pair_features(candidate_pairs, records_lookup, config) -> DataFrame`

```
Returns DataFrame with columns:
  source1_entity_id  (str)
  candidate_entity_id (str)
  <feature columns>  (float)
```

### `model.predict_proba(feature_df, config) -> DataFrame`

```
Returns DataFrame with columns:
  source1_entity_id  (str)
  candidate_entity_id (str)
  match_probability  (float in [0,1])
```

### `pipeline.run_pipeline(config) -> (matching_results_df, candidate_pairs_df)`

```
End-to-end: loads data, runs all stages, writes output files.
```

---

## Key Design Decisions

1. **candidate_pairs.tsv is FINAL** — it represents the exact candidate set passed to the model, not an intermediate blocking dump.
2. **Normalization creates new columns** — originals are never overwritten.
3. **Country is an open-set string** — never one-hot encoded, never hard-coded to {US, India}. France must work.
4. **Threshold is val-set derived** — never set to 0.5 by default without empirical justification.
5. **Singletons are first-class** — correctly predicting empty = full score for that entity.
6. **Hard negatives in training** — random negatives alone underfit the difficulty of the real candidate set.
7. **Reproducibility** — all random seeds fixed; pipeline is deterministic given fixed inputs.
