# Experiment: Model v1 (GBM)

**Date:** _fill in_
**Author:** Member B
**Branch:** `feature/matching-model`

---

## Configuration

| Parameter | Value |
|---|---|
| Model type | LightGBM / XGBoost / HistGBM |
| n_estimators | |
| learning_rate | |
| num_leaves | |
| random_seed | 42 |
| Negative sample ratio | |
| Hard negative fraction | |
| Val fraction | |
| Feature set | all (name + address + country + structural) |
| Threshold | _from calibration_ |

---

## Training Data

| Split | Positive pairs | Negative pairs | Hard negatives |
|---|---|---|---|
| Train | | | |
| Validation | | | |

---

## Feature Importances (Top 15)

| Feature | Importance |
|---|---|
| | |

---

## Validation Results

| Metric | Value |
|---|---|
| Macro F_0.5 | _fill in_ |
| Macro Precision | _fill in_ |
| Macro Recall | _fill in_ |
| Singleton Accuracy | _fill in_ |
| AUC-ROC | _fill in_ |
| AUC-PR | _fill in_ |

---

## Threshold Curve

| Threshold | Precision | Recall | F_0.5 |
|---|---|---|---|
| 0.50 | | | |
| 0.55 | | | |
| 0.60 | | | |
| 0.65 | | | |
| 0.70 | | | |
| 0.75 | | | |
| 0.80 | | | |
| 0.85 | | | |
| 0.90 | | | |
| 0.95 | | | |
| **Best** | | | |

---

## Improvement over Baseline

| Metric | Baseline | Model v1 | Delta |
|---|---|---|---|
| Macro F_0.5 | | | |
| Precision | | | |
| Recall | | | |
| Singleton Accuracy | | | |

---

## Notes

_What helped most? What hurt? What to try in model_v2?_
