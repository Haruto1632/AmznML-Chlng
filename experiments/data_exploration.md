# Data Exploration Notes

**Date:** _fill in_
**Author:** Member A
**Task:** A-1

---

## Dataset Sizes

| File | Rows | Columns |
|---|---|---|
| train_source1.tsv | _fill in_ | 4 |
| train_source2.tsv | _fill in_ | 4 |
| train_source3.tsv | _fill in_ | 4 |
| train_ground_truth.tsv | _fill in_ | 2 |
| test_source1.tsv | _fill in_ | 4 |
| test_source2.tsv | _fill in_ | 4 |
| test_source3.tsv | _fill in_ | 4 |

---

## Null / Empty Value Rates (Train)

| File | entity_id empty | business_name empty | business_address empty | country empty |
|---|---|---|---|---|
| Source 1 | | | | |
| Source 2 | | | | |
| Source 3 | | | | |

---

## Country Distribution

### Train
| Country | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| | | | |

### Test (fill in after inspecting test files)
| Country | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| France | | | |
| | | | |

---

## Ground Truth Statistics

| Metric | Value |
|---|---|
| Total S1 entities in train | |
| Singletons (no match) | |
| Singleton rate (%) | |
| Total true match pairs | |
| Max matches for one S1 entity | |
| Avg matches per non-singleton S1 | |

---

## Noise Examples

### Name Variations (examples from data)

| Source 1 name | Source 2/3 match | Type of variation |
|---|---|---|
| | | abbreviation |
| | | legal suffix |
| | | word order |
| | | punctuation |
| | | typo |

### Address Variations (examples from data)

| Source 1 address | Source 2/3 match | Type of variation |
|---|---|---|
| | | abbreviation |
| | | missing component |
| | | reordering |
| | | landmark-based |

---

## ID Format Observations

- S1 IDs: `S1-XXXXX` (___-digit zero-padded)
- S2 IDs: `S2-XXXXX`
- S3 IDs: `S3-XXXXX`

---

## Notes for Normalization Design

_What did you observe that should inform normalization rules?_

_What address formats are present for France vs US vs India?_

_Are there any unexpected characters (unicode, special chars)?_
