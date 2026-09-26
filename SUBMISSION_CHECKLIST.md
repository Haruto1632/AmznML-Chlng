# Submission Checklist — Amazon ML Challenge 2026

Run this checklist before every submission attempt.

---

## 1. Pipeline Correctness

- [ ] Train pipeline runs end-to-end without errors
  ```bash
  python -m business_entity_resolution.pipeline --mode train --config config.yaml
  ```
- [ ] Validation pipeline runs without errors
  ```bash
  python -m business_entity_resolution.pipeline --mode validate --config config.yaml
  ```
- [ ] Test pipeline runs end-to-end without errors
  ```bash
  python -m business_entity_resolution.pipeline --mode test --config config.yaml
  ```

---

## 2. Output File Format

- [ ] `output/matching_results.tsv` exists
- [ ] `output/candidate_pairs.tsv` exists
- [ ] Both files use tab (`\t`) as separator — NOT comma, NOT space
- [ ] Column names are exactly `source1_entity_id` and `matched_entity_ids` / `candidate_entity_ids`
- [ ] `matched_entity_ids` values are comma-separated (no spaces around commas)
- [ ] Singleton rows have empty second column (not "None", not "[]", not "nan")

---

## 3. Coverage and ID Integrity

- [ ] Every Source 1 test entity (`test_source1.tsv`) appears exactly once in `matching_results.tsv`
- [ ] Every Source 1 test entity appears exactly once in `candidate_pairs.tsv`
- [ ] No duplicate `source1_entity_id` rows in either file
- [ ] No duplicate entity IDs within any single `matched_entity_ids` or `candidate_entity_ids` cell
- [ ] No Source 1 IDs (`S1-*`) appear in the matched/candidate columns
- [ ] No self-matches (Source 1 entity matched to itself — impossible since S1 can't appear in S2/S3 columns, but verify)
- [ ] All predicted IDs exist in `test_source2.tsv` or `test_source3.tsv`
- [ ] No predicted IDs from `train_source2.tsv` or `train_source3.tsv` appear in test output

---

## 4. Candidate Integrity

- [ ] Every entity ID in `matching_results.tsv` also appears in the corresponding `candidate_pairs.tsv` row
  - No predicted match was skipped by the blocking step
- [ ] `candidate_pairs.tsv` represents the FINAL candidate set passed into the model
  - Not a raw/early blocking dump with unfiltered candidates
- [ ] For each S1 entity: `matched_ids ⊆ candidate_ids`

---

## 5. Model and Data Constraints

- [ ] No external business databases used
- [ ] No external lookup APIs called anywhere in the pipeline
- [ ] No geocoding APIs used
- [ ] No external internet data used to resolve businesses
- [ ] Countries NOT hard-coded to only {US, India} — France and other unseen countries handled
- [ ] All model weights are from pretrained models with MIT or Apache 2.0 licenses
- [ ] All models are ≤ 8 Billion parameters
- [ ] All model licenses documented in `requirements.txt` or `README.md`

---

## 6. Reproducibility

- [ ] `requirements.txt` is present and all versions are pinned
- [ ] Random seeds are fixed in all model training and sampling code
- [ ] Pipeline is deterministic given the same inputs
- [ ] No hardcoded absolute paths in source code
- [ ] Pipeline can be run from a clean environment using only `requirements.txt`

---

## 7. Official Validation Script

Run and verify PASS:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

- [ ] Script prints `PASS` (exit code 0)
- [ ] Zero violations listed
- [ ] If any violations appear — fix them ALL before submitting

---

## 8. Code Structure

- [ ] `code/business_entity_resolution/src/` contains all source code
- [ ] `code/business_entity_resolution/README.md` exists with end-to-end reproduction steps
- [ ] `code/business_entity_resolution/requirements.txt` exists with pinned versions
- [ ] README describes: data → blocking → matching → output, exactly what commands to run
- [ ] Pipeline can be reproduced by a reviewer with no prior context

---

## 9. Documentation

- [ ] `Documentation_template.md` is filled in (not blank template)
- [ ] Methodology section covers: blocking strategy, feature engineering, model choice, threshold calibration
- [ ] Candidate generation approach explained clearly (reviewers evaluate this separately)
- [ ] No placeholder text remaining in documentation

---

## 10. ZIP Structure

Verify the final ZIP contains:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md   (or .pdf)
```

- [ ] ZIP structure matches exactly
- [ ] No extra nested directories (e.g., not `output/output/matching_results.tsv`)
- [ ] ZIP is not password-protected
- [ ] ZIP does not contain `dataset/` (not required in submission)

---

## 11. Final Score Estimate

Before submitting, run internal evaluation on held-out validation set:

```bash
python -m business_entity_resolution.pipeline --mode validate --config config.yaml
```

Record expected performance:

| Metric | Value |
|---|---|
| Macro F_0.5 | |
| Macro Precision | |
| Macro Recall | |
| Singleton Accuracy | |
| Blocking Recall | |
| Avg Candidates/S1 | |

- [ ] F_0.5 is better than the rule-based baseline from `experiments/baseline.md`
- [ ] Singleton accuracy > 0.8 (false positives on singletons tank precision)
- [ ] Blocking recall > 0.95 (the recall ceiling must be very high)

---

## Sign-off

All team members review this checklist before submission:

- [ ] Member A sign-off (blocking/normalization)
- [ ] Member B sign-off (features/model)
- [ ] Member C sign-off (pipeline/output/validation)
