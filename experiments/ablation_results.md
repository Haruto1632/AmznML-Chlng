# Ablation Results

**Owner:** Member C  |  Updated incrementally — do NOT overwrite rows, only add new ones.

---

## Ablation Table

| # | Description | Blocking Strategies | Features | Model | Threshold | F_0.5 | Precision | Recall | Blocking Recall | Avg Cands/S1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline rule-based | exact+token | name+addr+country | weighted sum | 0.60 | | | | | |
| 1 | +better normalization | exact+token | name+addr+country | weighted sum | 0.60 | | | | | |
| 2 | +ngram LSH blocking | exact+token+ngram | name+addr+country | weighted sum | 0.60 | | | | | |
| 3 | +phonetic blocking | all string | name+addr+country | weighted sum | 0.60 | | | | | |
| 4 | +all string features | all string | full | weighted sum | 0.60 | | | | | |
| 5 | +GBM model | all string | full | GBM | calibrated | | | | | |
| 6 | +hard negatives | all string | full | GBM+HN | calibrated | | | | | |
| 7 | +structural features | all string | full+structural | GBM+HN | calibrated | | | | | |
| 8 | +calibrated threshold | all string | full+structural | GBM+HN | best | | | | | |
| 9 | +embedding blocking | all+embedding | full+structural | GBM+HN | best | | | | | |
| 10 | +ensemble (GBM+CE) | all+embedding | full+structural | ensemble | best | | | | | |

---

## Notes

_Record which ablations were most impactful. This guides where to spend remaining effort._
