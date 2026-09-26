# Experiment: Baseline (Rule-Based)

**Date:** _fill in_
**Author:** Member B
**Branch:** `feature/matching-model`

---

## Configuration

| Parameter | Value |
|---|---|
| Normalization | basic lowercase + whitespace |
| Blocking strategy | exact_token + token_overlap |
| Model | rule-based weighted sum |
| Score formula | `0.5 * name_token_jaccard + 0.3 * addr_token_jaccard + 0.2 * country_exact` |
| Threshold | _fill in_ |

---

## Blocking Results

| Metric | Value |
|---|---|
| Blocking recall | _fill in_ |
| Total candidates | _fill in_ |
| Avg candidates per S1 | _fill in_ |
| Median candidates per S1 | _fill in_ |
| P95 candidates per S1 | _fill in_ |
| Reduction ratio | _fill in_ |

---

## Validation Results

| Metric | Value |
|---|---|
| Macro F_0.5 | _fill in_ |
| Macro Precision | _fill in_ |
| Macro Recall | _fill in_ |
| Singleton Accuracy | _fill in_ |
| False Positive Rate | _fill in_ |
| False Negative Rate | _fill in_ |

---

## Notes

_Observations about what the baseline gets right or wrong. What are the main failure modes?_

---

## Next Steps

_What to improve after this baseline? (blocking recall? precision? singleton handling?)_
