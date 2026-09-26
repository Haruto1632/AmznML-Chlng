# Team Task Assignments — Amazon ML Challenge 2026

> Last updated: project kickoff
> Metric: F_0.5 (precision-heavy, macro-averaged per S1 entity)
> Branch strategy: see GIT_WORKFLOW.md

---

## Quick Reference

| Member | Branch | Owns | Depends on |
|---|---|---|---|
| **A** | `feature/blocking` | normalization, blocking, candidate generation | shared contracts (C delivers first) |
| **B** | `feature/matching-model` | features, models, scoring | A's candidate output + shared contracts |
| **C** | `feature/evaluation-pipeline` | evaluation, calibration, pipeline, output, validation | A + B outputs |

---

## MEMBER A — Blocking / Candidate Generation

**Branch:** `feature/blocking`

### Files Owned

```
code/business_entity_resolution/src/normalization/
    __init__.py
    normalizer.py          ← main entry point
    name_cleaner.py        ← name-specific rules
    address_cleaner.py     ← address-specific rules
    country_mapper.py      ← country normalisation, open-set safe

code/business_entity_resolution/src/blocking/
    __init__.py
    blocker.py             ← orchestrates all strategies
    strategies/
        __init__.py
        exact_token.py     ← exact country+name token blocking
        token_overlap.py   ← name/address token overlap
        ngram_lsh.py       ← char n-gram + MinHash LSH
        phonetic.py        ← Soundex / Metaphone
        address_token.py   ← address component matching
        embedding_ann.py   ← ANN on pretrained embeddings (optional)

code/business_entity_resolution/src/candidate_generation/
    __init__.py
    candidate_store.py     ← union, dedup, limit, track reasons
    blocking_eval.py       ← recall@k, reduction_ratio utilities
```

**Do NOT touch:** `src/features/`, `src/models/`, `src/pipeline/`, `src/evaluation/`

---

### Task A-1 — Understand the Data (Day 1)

- [ ] Read all six train TSVs. Print shape, dtypes, null counts per column.
- [ ] Print 20 random rows from each source.
- [ ] Note country distribution in train vs expected test (France).
- [ ] Record findings in `experiments/data_exploration.md`.

**Expected output:** A markdown note with shapes, null rates, name/address noise examples.

---

### Task A-2 — Normalization (`src/normalization/`)

Implement `normalizer.py` with entry point:

```python
def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [entity_id, business_name,
                                     business_address, country]
    Output: same DataFrame PLUS:
              business_name_normalized   (str)
              business_address_normalized (str)
              country_normalized         (str)
    Originals are NEVER modified.
    """
```

Rules for **name normalization** (`name_cleaner.py`):
- Lowercase
- Strip leading/trailing whitespace; collapse internal whitespace
- Remove/replace punctuation (keep alphanumeric + space)
- `&` → `and`
- Legal suffix expansion table: `corp` ↔ `corporation`, `pvt` ↔ `private`, `ltd` ↔ `limited`, `llc`, `inc`, `co`, `plc`, `gmbh`, `sarl`, `sas`, `bv`, `nv`, etc.
- Sort tokens alphabetically for token-order normalization (create a separate `name_token_sorted` column)
- Strip stop-words for a separate `name_tokens` set column (used in blocking)
- Do NOT destroy information. Keep a "lightly cleaned" and a "heavily normalized" variant.

Rules for **address normalization** (`address_cleaner.py`):
- Lowercase, strip whitespace
- Abbreviation expansion: `st` → `street`, `rd` → `road`, `ave` → `avenue`, `blvd` → `boulevard`, `dr` → `drive`, `ln` → `lane`
- French address terms: `rue`, `avenue`, `boulevard`, `cedex`, `arrondissement` — keep as-is (no mapping to English)
- Numeric token extraction: pull all digit sequences into a separate `address_numbers` column
- Remove punctuation except hyphens in numbers

Rules for **country normalization** (`country_mapper.py`):
- Lowercase
- Common aliases: `usa` → `united states`, `u.s.a.` → `united states`, `u.s.` → `united states`, `uk` → `united kingdom`, `uae` → `united arab emirates`
- Return original if no alias found (open-set safe — do NOT drop unknowns)
- France variants: `france`, `fr`, `french republic` → `france`

**Tests:** `tests/test_normalizer.py`
- Test each rule with known input/output pairs
- Test that originals are preserved
- Test France strings are handled without crash

---

### Task A-3 — Multi-Strategy Blocking (`src/blocking/`)

Implement `blocker.py` with entry point:

```python
def generate_candidates(
    source1: pd.DataFrame,       # normalized S1 records
    source2: pd.DataFrame,       # normalized S2 records
    source3: pd.DataFrame,       # normalized S3 records
    config: dict
) -> pd.DataFrame:
    """
    Returns DataFrame:
      source1_entity_id    (str)
      candidate_entity_id  (str)
      blocking_reasons     (list[str])   — which strategies produced this pair
    """
```

Strategies to implement (each in its own file under `strategies/`):

| Strategy | File | Description |
|---|---|---|
| `exact_country_name_token` | `exact_token.py` | Same country + ≥1 shared normalised name token |
| `name_token_overlap` | `token_overlap.py` | Jaccard(name_tokens_A, name_tokens_B) ≥ threshold |
| `char_ngram_lsh` | `ngram_lsh.py` | MinHash LSH on character 3-grams of normalised name |
| `phonetic` | `phonetic.py` | Soundex/Metaphone of first name token matches |
| `address_token_overlap` | `address_token.py` | ≥1 shared address number + ≥1 shared street token |
| `embedding_ann` | `embedding_ann.py` | ANN search on sentence-transformer embeddings (optional, gated by config flag) |

Each strategy returns a DataFrame with `(source1_entity_id, candidate_entity_id)` — `blocker.py` unions them and adds `blocking_reasons`.

---

### Task A-4 — Candidate Consolidation (`src/candidate_generation/`)

Implement `candidate_store.py`:

```python
def consolidate_candidates(
    raw_pairs: pd.DataFrame,   # output of blocker.generate_candidates
    source1_ids: list,         # all S1 entity_ids (to guarantee full coverage)
    config: dict
) -> pd.DataFrame:
    """
    Returns final candidate_pairs DataFrame:
      source1_entity_id    (str)
      candidate_entity_id  (str)
      blocking_reasons     (list[str])
    Every S1 entity has a row (empty candidate list = singleton candidate).
    """
```

- Deduplicate `(source1_entity_id, candidate_entity_id)` pairs.
- Union `blocking_reasons` lists for pairs produced by multiple strategies.
- Apply optional per-S1 candidate count limit (config: `max_candidates_per_s1`, default 200).
- When trimming, prefer pairs with more blocking strategies (higher evidence).
- Guarantee every S1 entity has at least one row (empty candidate list is valid).

---

### Task A-5 — Blocking Evaluation (`src/candidate_generation/blocking_eval.py`)

```python
def evaluate_blocking(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame
) -> dict:
    """
    Returns dict with:
      blocking_recall         # fraction of true matches that survived blocking
      total_candidates        # total candidate pairs generated
      avg_candidates_per_s1
      median_candidates_per_s1
      p95_candidates_per_s1
      reduction_ratio         # 1 - (candidates / (|S1| * (|S2|+|S3|)))
      per_strategy_recall     # dict: strategy_name -> recall contribution
    """
```

**Log these numbers in `experiments/blocking_v1.md`** after each blocking iteration.

---

### Expected Outputs (Member A)

| File | Description |
|---|---|
| `experiments/data_exploration.md` | Data shapes, null rates, noise examples |
| `experiments/blocking_v1.md` | First blocking evaluation results |
| `output/candidate_pairs.tsv` | Final candidate pairs (written by pipeline, but A validates format) |
| `tests/test_normalizer.py` | Normalization unit tests |
| `tests/test_blocking.py` | Blocking recall tests on train data |

---

### Dependencies (Member A needs these first)

- `src/shared/schemas.py` — data contracts (C delivers Day 1)
- `src/shared/data_loader.py` — TSV loader (C delivers Day 1)
- Train/test TSV files in `dataset/`

---

## MEMBER B — Features + Matching Model

**Branch:** `feature/matching-model`

### Files Owned

```
code/business_entity_resolution/src/features/
    __init__.py
    feature_builder.py     ← main entry point
    name_features.py       ← all name similarity features
    address_features.py    ← all address similarity features
    country_features.py    ← country comparison features
    structural_features.py ← blocking evidence features

code/business_entity_resolution/src/models/
    __init__.py
    matcher.py             ← main model class
    gbm_model.py           ← LightGBM / XGBoost / HistGBM wrapper
    cross_encoder.py       ← optional small transformer reranker
    training_data.py       ← positive/negative pair construction
    hard_negatives.py      ← hard negative mining logic

code/business_entity_resolution/src/scoring/
    __init__.py
    scorer.py              ← wraps model, returns match_probability DataFrame
```

**Do NOT touch:** `src/normalization/`, `src/blocking/`, `src/pipeline/`, `src/evaluation/`

---

### Task B-1 — Feature Engineering (`src/features/`)

Implement `feature_builder.py` with entry point:

```python
def build_pair_features(
    candidate_pairs: pd.DataFrame,   # from candidate_store (has blocking_reasons)
    records: dict,                   # {entity_id: NormalizedRecord} lookup
    config: dict
) -> pd.DataFrame:
    """
    Returns DataFrame:
      source1_entity_id    (str)
      candidate_entity_id  (str)
      <all feature columns> (float)
    """
```

**Name features** (`name_features.py`):

| Feature | Notes |
|---|---|
| `name_exact_normalized` | 1.0 if normalized names equal |
| `name_levenshtein` | Normalized Levenshtein distance → similarity |
| `name_jaro_winkler` | jellyfish or rapidfuzz |
| `name_token_jaccard` | Jaccard on token sets |
| `name_token_overlap_ratio` | |overlap| / max(|A|, |B|) |
| `name_char3gram_cosine` | TF-IDF cosine on char 3-grams |
| `name_token_sort_ratio` | Sort both token lists, then edit distance |
| `name_token_set_ratio` | fuzzywuzzy-style token set ratio |
| `name_length_diff` | abs(len(A) - len(B)) / max(len(A), len(B)) |
| `name_abbrev_score` | Custom: check if one is abbreviation of other |
| `name_sorted_exact` | 1.0 if token-sorted normalized names equal |

**Address features** (`address_features.py`):

| Feature | Notes |
|---|---|
| `addr_exact_normalized` | 1.0 if normalized addresses equal |
| `addr_token_jaccard` | |
| `addr_char3gram_cosine` | |
| `addr_edit_similarity` | Normalized edit distance → similarity |
| `addr_numeric_overlap` | Jaccard on extracted numeric tokens |
| `addr_length_diff` | |
| `addr_both_empty` | 1.0 if both addresses are null/empty |
| `addr_one_empty` | 1.0 if exactly one is null/empty |

**Country features** (`country_features.py`):

| Feature | Notes |
|---|---|
| `country_exact_normalized` | 1.0 if normalized countries equal |
| `country_both_missing` | 1.0 if both null |

**Structural features** (`structural_features.py`):

| Feature | Notes |
|---|---|
| `num_blocking_strategies` | How many strategies produced this pair |
| `strategy_exact_token` | 1.0 if pair came from exact_country_name_token |
| `strategy_token_overlap` | 1.0 if pair came from name_token_overlap |
| `strategy_ngram_lsh` | 1.0 if pair came from char_ngram_lsh |
| `strategy_phonetic` | 1.0 if pair came from phonetic |
| `strategy_address_token` | 1.0 if pair came from address_token_overlap |
| `strategy_embedding` | 1.0 if pair came from embedding_ann |

---

### Task B-2 — Training Data Construction (`src/models/training_data.py`)

```python
def build_training_pairs(
    normalized_records: dict,
    ground_truth: pd.DataFrame,
    candidate_pairs: pd.DataFrame,   # blocking output on train data
    config: dict
) -> pd.DataFrame:
    """
    Returns labeled feature DataFrame:
      source1_entity_id, candidate_entity_id, <features>, label (0/1)
    """
```

**Positive pairs:** all true matches from `train_ground_truth.tsv`.

**Negative pairs — sampling strategy:**
1. Random negatives from candidate pairs not in ground truth (basic)
2. Hard negatives — from `hard_negatives.py`:
   - Same country, similar name (high name similarity, not a true match)
   - Same address prefix, different business name
   - Abbreviation confusers (e.g., "ABC Corp" vs. "ABC Co.")
   - Blocking candidates that are NOT true matches (these are the most realistic hard negatives)
3. Balance ratio: target ~1:5 positive:negative, use more hard negatives than random.

**Entity-level split:** When creating train/val split, split by S1 entity ID (not by pair) to avoid leakage. Recommend 80/20 stratified by whether S1 entity is a singleton.

---

### Task B-3 — GBM Model (`src/models/gbm_model.py`)

```python
class GBMMatcherModel:
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None): ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str): ...
    def load(self, path: str): ...
    def feature_importance(self) -> pd.DataFrame: ...
```

- Use LightGBM if available, else XGBoost, else sklearn HistGradientBoosting.
- Fix random seed.
- Log feature importances to `experiments/model_v1.md`.
- Save trained model to `code/business_entity_resolution/models/gbm_model.pkl`.

---

### Task B-4 — Scorer (`src/scoring/scorer.py`)

```python
def predict_match_probabilities(
    feature_df: pd.DataFrame,
    model,
    config: dict
) -> pd.DataFrame:
    """
    Returns DataFrame:
      source1_entity_id    (str)
      candidate_entity_id  (str)
      match_probability    (float in [0, 1])
    """
```

---

### Task B-5 — Baseline First

Before the full GBM, implement a **rule-based baseline** in `src/models/matcher.py`:

```python
def baseline_score(feature_row) -> float:
    """
    Simple weighted sum:
      0.5 * name_token_jaccard
    + 0.3 * addr_token_jaccard
    + 0.2 * country_exact_normalized
    """
```

Run it on validation. Record F_0.5 in `experiments/baseline.md`. Then improve with GBM.

---

### Expected Outputs (Member B)

| File | Description |
|---|---|
| `experiments/baseline.md` | Baseline rule-based score results |
| `experiments/model_v1.md` | GBM v1 results + feature importances |
| `models/gbm_model.pkl` | Serialized trained model |
| `tests/test_features.py` | Feature computation unit tests |
| `tests/test_model.py` | Model train/predict smoke tests |

---

### Dependencies (Member B needs these first)

- `src/shared/schemas.py` — data contracts
- `src/shared/data_loader.py` — TSV loader
- Member A's `candidate_store.py` (or a mock output for development)
- `dataset/train/` files

---

## MEMBER C — Evaluation / Calibration / Integration

**Branch:** `feature/evaluation-pipeline`

### Files Owned

```
code/business_entity_resolution/src/shared/
    __init__.py
    schemas.py             ← data contracts (deliver Day 1!)
    data_loader.py         ← TSV loader (deliver Day 1!)

code/business_entity_resolution/src/evaluation/
    __init__.py
    evaluator.py           ← F0.5, precision, recall, singleton accuracy
    metrics.py             ← formula implementations

code/business_entity_resolution/src/calibration/
    __init__.py
    calibrator.py          ← threshold sweep + selection

code/business_entity_resolution/src/pipeline/
    __init__.py
    pipeline.py            ← end-to-end runner
    output_writer.py       ← TSV output generation + integrity checks
    config.py              ← central config object

code/business_entity_resolution/
    README.md
    requirements.txt

output/
    matching_results.tsv   ← generated by pipeline
    candidate_pairs.tsv    ← generated by pipeline
```

**Do NOT touch:** `src/normalization/`, `src/blocking/`, `src/features/`, `src/models/`

---

### Task C-1 — Shared Contracts (DELIVER FIRST — Day 1)

Write `src/shared/schemas.py`:

```python
# NormalizedRecord, CandidatePair, FeatureRow, Prediction dataclasses + constants
# See ARCHITECTURE.md "Stable Interfaces" section
# See also: code/business_entity_resolution/src/shared/schemas.py
```

Write `src/shared/data_loader.py`:

```python
def load_source(path: str) -> pd.DataFrame:
    """Read a source TSV. Validates columns. Returns DataFrame."""

def load_ground_truth(path: str) -> pd.DataFrame:
    """Read train_ground_truth.tsv. Returns two-column DataFrame."""

def load_all_train(data_dir: str) -> tuple:
    """Returns (s1, s2, s3, ground_truth) DataFrames."""

def load_all_test(data_dir: str) -> tuple:
    """Returns (s1, s2, s3) DataFrames."""
```

**CRITICAL:** This is what A and B both import. Deliver it before they start.

---

### Task C-2 — Evaluation Framework (`src/evaluation/`)

Implement `evaluator.py`:

```python
def compute_f05_per_entity(
    predictions: dict,     # {s1_id: [matched_ids]}
    ground_truth: dict     # {s1_id: [true_ids]}
) -> pd.DataFrame:
    """
    Returns per-entity DataFrame:
      source1_entity_id, precision, recall, f05, is_singleton
    """

def compute_macro_f05(per_entity_df: pd.DataFrame) -> float:
    """Macro-average F_0.5 across all S1 entities."""

def full_evaluation_report(
    predictions: dict,
    ground_truth: dict,
    candidate_pairs: pd.DataFrame
) -> dict:
    """
    Returns dict with:
      macro_f05
      macro_precision
      macro_recall
      singleton_accuracy   (frac of true singletons predicted empty)
      false_positive_rate
      false_negative_rate
      blocking_recall
      avg_candidates_per_s1
      reduction_ratio
    """
```

**F_0.5 formula:**
```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
Singleton true positive: predict empty when truth is empty → F_0.5 = 1.0
Singleton false positive: predict non-empty when truth is empty → F_0.5 = 0.0
```

---

### Task C-3 — Threshold Calibration (`src/calibration/calibrator.py`)

```python
def calibrate_threshold(
    predictions_df: pd.DataFrame,   # {s1_id, candidate_id, match_probability}
    ground_truth: pd.DataFrame,
    thresholds: list = None          # default: [0.50, 0.55, ..., 0.95]
) -> dict:
    """
    Returns:
      best_threshold   (float)
      threshold_curve  (DataFrame: threshold, precision, recall, f05)
    """
```

Save threshold curve to `experiments/calibration_v1.md`.

Additional investigations:
- Does splitting threshold by `num_blocking_strategies` help? (high evidence vs. weak evidence)
- Does a higher threshold for name-only matches help precision?
- What is the optimal threshold for singleton prediction?

---

### Task C-4 — Output Writer (`src/pipeline/output_writer.py`)

```python
def write_matching_results(
    predictions: dict,          # {s1_id: [matched_ids]}
    all_s1_ids: list,           # ensure every S1 entity has a row
    path: str
) -> None: ...

def write_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    all_s1_ids: list,
    path: str
) -> None: ...

def validate_output_integrity(
    matching_path: str,
    candidate_path: str,
    s2_ids: set,
    s3_ids: set,
    s1_ids: set
) -> list:
    """
    Returns list of violation strings. Empty list = clean.
    Checks:
      - Every S1 entity present
      - No S1→S1 matches
      - All IDs exist in test set
      - No duplicate IDs per row
      - Every matched ID ∈ candidate set
    """
```

---

### Task C-5 — End-to-End Pipeline (`src/pipeline/pipeline.py`)

```python
def run_pipeline(config: dict) -> None:
    """
    1.  Load data (data_loader)
    2.  Normalize (normalizer)
    3.  Generate candidates (blocker + candidate_store)
    4.  Write candidate_pairs.tsv (output_writer)
    5.  Build features (feature_builder)
    6.  Load/apply model (scorer)
    7.  Calibrate threshold (calibrator)
    8.  Apply threshold → final predictions
    9.  Write matching_results.tsv (output_writer)
    10. Run internal validation (validate_output_integrity)
    11. Print evaluation report
    """
```

Entry point: `python -m business_entity_resolution.pipeline --config config.yaml`

---

### Task C-6 — Ablation Framework

Create `src/evaluation/ablation.py` that runs the pipeline with different config flags and collects results into `experiments/ablation_results.md`.

Track per ablation:
- Blocking strategy flags (which strategies enabled)
- Feature set
- Model type
- Threshold
- F_0.5, Precision, Recall
- Blocking recall
- Avg candidates per S1

---

### Expected Outputs (Member C)

| File | Description |
|---|---|
| `src/shared/schemas.py` | Shared data contracts (Day 1) |
| `src/shared/data_loader.py` | TSV loader (Day 1) |
| `experiments/calibration_v1.md` | Threshold sweep results |
| `experiments/ablation_results.md` | Ablation table |
| `output/matching_results.tsv` | Final predictions |
| `output/candidate_pairs.tsv` | Final candidates |
| `code/business_entity_resolution/README.md` | Reproduction guide |
| `code/business_entity_resolution/requirements.txt` | Pinned deps |
| `tests/test_evaluator.py` | F_0.5 formula unit tests |
| `tests/test_output_writer.py` | Output format tests |

---

### Dependencies (Member C)

- A's candidate output (mock with empty DataFrame during development)
- B's scorer output (mock with random probabilities during development)
- Both A and B must implement the stable interfaces from `ARCHITECTURE.md`

---

## Interface Summary

```
A produces  →  candidate_pairs.tsv  →  B consumes (features input)
B produces  →  match_probability DF →  C consumes (calibration input)
C produces  →  matching_results.tsv +  candidate_pairs.tsv (final outputs)
C delivers  →  schemas.py + data_loader.py  →  A and B both consume
```

## First Milestone (End of Day 1)

- [ ] C: `schemas.py` and `data_loader.py` pushed to `main` (shared immediately)
- [ ] A: Data exploration note written; normalization rules drafted
- [ ] B: Feature list reviewed and agreed upon; baseline plan outlined
- [ ] All: Git branches created; no merge conflicts on `main`
