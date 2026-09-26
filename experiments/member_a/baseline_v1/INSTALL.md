# Installation Instructions — Baseline v1

## Prerequisites

- Python 3.8+
- pip

## Install Dependencies

From the repository root:

```bash
cd "c:\Users\DELL\OneDrive\Documents\GitHub\Amazon ML challenge"

# Install all required packages
pip install pandas numpy scikit-learn lightgbm rapidfuzz datasketch unidecode tqdm pyyaml
```

### Package Details

| Package | Purpose | License |
|---|---|---|
| pandas | Data manipulation | BSD |
| numpy | Numerical operations | BSD |
| scikit-learn | TF-IDF, metrics, train/test split | BSD |
| lightgbm | Gradient boosting model | MIT |
| rapidfuzz | Fast string similarity (Levenshtein, Jaro-Winkler) | MIT |
| datasketch | MinHash LSH for blocking | MIT |
| unidecode | Transliterate non-Latin scripts (Devanagari→ASCII) | GPL-2.0 |
| tqdm | Progress bars | MIT/MPL |
| pyyaml | Config file parsing | MIT |

### Optional: jellyfish (for phonetic blocking)

```bash
pip install jellyfish
```

If jellyfish is not installed, the phonetic blocking strategy will be skipped.

### Alternative: XGBoost or sklearn-only

If LightGBM installation fails:

**Option 1: Use XGBoost**
```bash
pip install xgboost
```
Then edit `experiment_config.yaml`:
```yaml
model:
  type: "xgboost"  # change from "lightgbm"
```

**Option 2: Use sklearn HistGradientBoosting (slower)**
Edit `experiment_config.yaml`:
```yaml
model:
  type: "histgbm"
```

## Verify Installation

```bash
python -c "import pandas, numpy, sklearn, lightgbm, rapidfuzz, datasketch, unidecode, tqdm, yaml; print('All packages installed successfully')"
```

## Run Experiment

```bash
python experiments/member_a/baseline_v1/run_experiment.py
```

Expected runtime: 10–30 minutes depending on hardware (2.2M S1 train entities).

## Troubleshooting

### ImportError: No module named 'unidecode'
```bash
pip install unidecode
```

### LightGBM build error on Windows
Download pre-built wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#lightgbm
Or use XGBoost instead (see above).

### Memory error during blocking
Reduce `max_candidates_per_s1` in `experiment_config.yaml`:
```yaml
blocking:
  max_candidates_per_s1: 30  # reduce from 50
```

### Slow TF-IDF fitting
Reduce `tfidf_max_features` in `experiment_config.yaml`:
```yaml
features:
  tfidf_max_features: 3000  # reduce from 5000
```
