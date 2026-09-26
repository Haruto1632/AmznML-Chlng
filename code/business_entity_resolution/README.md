# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline for resolving business entities across three noisy data sources.

**Metric:** F_0.5 (precision-heavy, macro-averaged per Source 1 entity)

---

## Repository Structure

```
code/business_entity_resolution/
├── src/
│   ├── shared/            # data contracts, loader — used by all members
│   │   ├── schemas.py
│   │   └── data_loader.py
│   ├── normalization/     # Member A: name + address + country normalisation
│   ├── blocking/          # Member A: multi-strategy candidate generation
│   ├── candidate_generation/  # Member A: consolidation + evaluation
│   ├── features/          # Member B: pairwise feature engineering
│   ├── models/            # Member B: GBM + optional cross-encoder
│   ├── scoring/           # Member B: probability output wrapper
│   ├── evaluation/        # Member C: F_0.5, precision, recall, ablation
│   ├── calibration/       # Member C: threshold sweep
│   └── pipeline/          # Member C: end-to-end runner + output writer
├── tests/
├── models/                # serialized model artifacts (not in git)
├── README.md
└── requirements.txt
```

---

## Environment Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Data Layout

Place the challenge data as follows (not included in the code submission):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## Running the Pipeline

All commands are run from the **workspace root** (the directory containing `dataset/`, `output/`, and `code/`).

### 1. Train the model

```bash
python -m business_entity_resolution.pipeline --mode train --data-dir dataset/train
```

This will:
- Load and normalize all training data
- Generate candidate pairs (blocking)
- Build pairwise features
- Train the GBM model
- Calibrate the decision threshold on a held-out validation set
- Save model artifacts to `code/business_entity_resolution/models/`

### 2. Run on test data

```bash
python -m business_entity_resolution.pipeline --mode test --data-dir dataset/test
```

This will:
- Load and normalize test data
- Generate candidate pairs → writes `output/candidate_pairs.tsv`
- Build pairwise features
- Apply trained model and calibrated threshold
- Write `output/matching_results.tsv`
- Run internal integrity checks

### 3. Validate output format

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Must print `PASS` before submitting.

### 4. (Optional) Run on training data with evaluation

```bash
python -m business_entity_resolution.pipeline --mode validate --data-dir dataset/train
```

Prints precision, recall, F_0.5, singleton accuracy, and blocking recall on the training validation split.

---

## Output Files

| File | Description |
|---|---|
| `output/matching_results.tsv` | Final predictions: `source1_entity_id \t matched_entity_ids` |
| `output/candidate_pairs.tsv` | Blocking candidates: `source1_entity_id \t candidate_entity_ids` |

Both use tab (`\t`) as separator. `matched_entity_ids` / `candidate_entity_ids` are comma-separated lists. Empty string for singletons.

---

## Pipeline Architecture

```
Data loading → Normalization → Multi-strategy blocking
→ Candidate consolidation → Feature engineering
→ GBM matching model → Threshold calibration
→ Output generation → Validation
```

See `ARCHITECTURE.md` in the workspace root for the detailed flow diagram.

---

## Key Design Decisions

- **candidate_pairs.tsv is the final candidate set** actually passed to the model, not a raw blocking dump.
- **Country is open-set** — the pipeline handles France and any unseen country without hard-coding.
- **Normalization creates new columns** — originals are always preserved.
- **Threshold is calibrated on validation data** — not defaulted to 0.5.
- **Hard negatives in training** — blocking candidates that are NOT true matches are used as hard negatives.
- **Singletons are explicitly handled** — no match above threshold → empty prediction.

---

## Reproducibility

- Random seed: `42` (set in `src/pipeline/config.py`)
- All library versions pinned in `requirements.txt`
- Pipeline is deterministic given fixed inputs and seed

---

## License Notes

All model weights and libraries used have MIT or Apache 2.0 licenses, complying with challenge rules.
Model parameters: all models used are ≤ 8 billion parameters.
No external APIs, geocoding services, or business databases are used.
