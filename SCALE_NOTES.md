# Scale Notes — Critical Design Constraints

This document captures key findings from the initial data audit that affect ALL
architectural decisions. Read before implementing anything.

---

## Dataset Sizes

| File | Records |
|---|---|
| train_source1.tsv | **2,206,821** |
| train_source2.tsv | **5,034,616** |
| train_source3.tsv | **5,285,603** |
| train_ground_truth.tsv | 2,206,821 (one row per S1 entity) |
| test_source1.tsv | **1,732,544** |
| test_source2.tsv | **4,887,273** |
| test_source3.tsv | **5,082,316** |

**Total train candidate universe:** 2.2M × (5M + 5.3M) = ~22.7 BILLION brute-force pairs.
Brute-force is absolutely impossible. Blocking is not optional — it is the entire solution.

---

## Ground Truth Statistics (Train)

| Metric | Value |
|---|---|
| Total S1 train entities | 2,206,821 |
| Singletons (no match) | 123,247 (5.6%) |
| Non-singletons | 2,083,574 (94.4%) |
| Total true match pairs | 7,638,365 |
| Max matches for one S1 entity | 11 |
| Avg matches per non-singleton S1 | 3.67 |

### Match count distribution:
```
0 matches: 123,247 entities  (5.6%)
1 matches: 119,157 entities
2 matches: 375,212 entities
3 matches: 530,841 entities
4 matches: 484,115 entities
5 matches: 321,957 entities
6 matches: 164,868 entities
7 matches:  63,968 entities
8 matches:  18,680 entities
9 matches:   4,205 entities
10 matches:    534 entities
11 matches:     37 entities
```

**Key insight:** Most S1 entities have 3–5 matches. The pipeline must scale to handle
7.6M positive training pairs.

---

## ID Format (Observed)

IDs are NOT zero-padded short numbers. They are large integers:
- S1: `S1-965667`, `S1-925783039` (variable length numbers)
- S2: `S2-681193310`, `S2-166376419`
- S3: `S3-775321672`, `S3-202863386`

No zero-padding; numbers go up to 9 digits.

---

## Data Language / Character Observations

From first rows:
- **US records:** ASCII business names, standard US addresses (`NC`, `OK`, `AZ`, etc.)
- **India records:** Mix of ASCII and **Devanagari script** (Hindi), e.g.:
  - `राम मार्केटिंग प्राइवेट लिमिटेड`
  - `आदित्य प्रॉपर्टीज एलएलपी`
  - Kannada script in addresses: `ಕರ್ನಾಟಕ`
- **Test France records:** Will have French address components

**Critical for normalization:**
- Unicode-aware processing is REQUIRED
- Devanagari text cannot be treated as ASCII
- Phonetic blocking on raw text will fail for non-Latin scripts
- For non-Latin scripts: consider transliteration (unidecode / anyascii) before string matching

---

## Actual Data Directory

All data lives under `student_resource/` (not `dataset/` at workspace root):
```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/          ← write outputs here for validation
├── utils/
│   └── validate_submission.py
├── README.md
└── Documentation_template.md
```

Validator must be run from `student_resource/`:
```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Scale-Driven Architectural Requirements

### 1. Inverted Index Blocking (MANDATORY)

Never iterate O(|S1| × |S2+S3|). Every blocking strategy MUST use inverted indices:
- Token → [entity_ids]: build once for S2+S3, query for each S1 entity
- Phonetic code → [entity_ids]: same pattern
- MinHash LSH: bulk insert S2+S3, query S1

Target: blocking must run in **under 30 minutes** on a single machine with 16GB RAM.

### 2. Chunked / Streaming Processing

Do NOT load all 7.6M training pairs into RAM simultaneously for feature computation.
Process in chunks of ~100,000 pairs at a time.

### 3. Candidate Limit Per S1 Entity

With 2.2M S1 entities, even 50 candidates per entity = 110M candidate pairs.
Feature computation on 110M pairs is expensive.

Recommended limits:
- Development/testing: max 20–30 candidates per S1 entity
- Final submission: tune based on recall vs. runtime tradeoff
- Target: avg 10–20 candidates per S1 after consolidation

### 4. Training Data Subsampling

With 7.6M positive pairs and potentially 30M+ negatives, training data must be sampled:
- Sample a representative subset for model training (e.g., 1M–2M total pairs)
- Ensure representative coverage of all match counts (1-match, 2-match, ..., 11-match)
- Maintain hard negative fraction from the subset

### 5. Sparse Feature Matrices

For TF-IDF and n-gram features, use scipy sparse matrices throughout.
Never densify a 110M × 50,000 feature matrix.

### 6. LightGBM is Required

HistGradientBoosting (sklearn) is too slow for 1M+ training rows.
LightGBM is mandatory for this dataset size.

---

## Memory Budget Estimate (Test Inference)

| Component | Approx. Memory |
|---|---|
| Load S1 (1.7M records) | ~500 MB |
| Load S2+S3 (10M records) | ~3 GB |
| Inverted index (name tokens) | ~2 GB |
| Candidate pairs (20 avg × 1.7M) = 34M pairs | ~2 GB |
| Feature matrix (34M × 30 features) | ~8 GB sparse |
| LightGBM inference | ~500 MB |
| **Total estimate** | **~16 GB** |

This pipeline requires a machine with at least 16GB RAM for test inference.
If memory is tight, reduce candidate limit or process in chunks.

---

## Noise Diversity Observed

From first few rows:
- S2 contains website URLs in names: `SHIVSHAKTI VIDYALAYA ... | www.shivshakti.com`
- Devanagari Hindi text: `राम मार्केटिंग प्राइवेट लिमिटेड`
- Inconsistent spacing, mixed case, duplicate words: `VIDYALAYA VIDYALAYA`
- S3: URL-style names: `wilfordhancock.com`
- Transliterations of Indian names in S3: `Pvt. EFS Print Ventures Ltd.`
- Empty addresses in S3 (e.g., `S3-859268022` has no address)
- Unicode accents: `Léarning Center` in S3

These all need to be handled in normalization.
