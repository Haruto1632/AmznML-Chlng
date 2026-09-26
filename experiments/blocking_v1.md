# Experiment: Blocking v1

**Date:** _fill in_
**Author:** Member A
**Branch:** `feature/blocking`

---

## Strategies Enabled

- [ ] exact_country_name_token
- [ ] name_token_overlap  (threshold: ___)
- [ ] char_ngram_lsh      (n=___, num_perm=___, threshold=___)
- [ ] phonetic            (algorithm: Soundex / Metaphone)
- [ ] address_token_overlap
- [ ] embedding_ann       (model: ___)

---

## Results per Strategy

| Strategy | Recall Contribution | Unique Pairs Added |
|---|---|---|
| exact_country_name_token | | |
| name_token_overlap | | |
| char_ngram_lsh | | |
| phonetic | | |
| address_token_overlap | | |
| embedding_ann | | |
| **UNION (all)** | | |

---

## Aggregate Blocking Metrics

| Metric | Value |
|---|---|
| Blocking recall (true matches surviving) | _fill in_ |
| Total candidate pairs | _fill in_ |
| Avg candidates per S1 | _fill in_ |
| Median candidates per S1 | _fill in_ |
| P95 candidates per S1 | _fill in_ |
| Reduction ratio | _fill in_ |

---

## Missed True Matches Analysis

_How many true matches did NOT survive blocking? What do they look like? 
Are they truly hard cases (completely different name/address) or fixable?_

Sample missed pairs:

| S1 entity | True match | Why missed |
|---|---|---|
| | | |

---

## Notes

_What to improve in blocking_v2?_
