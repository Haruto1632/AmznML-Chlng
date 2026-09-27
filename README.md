<p align="center">
  <h1 align="center">AMZNML-CHLNG</h1>
</p>

<p align="center">
  <strong>Amazon ML Challenge 2026 — Business Entity Resolution</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/Haruto1632/AmznML-Chlng?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
  <img src="https://img.shields.io/github/last-commit/Haruto1632/AmznML-Chlng?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
  <img src="https://img.shields.io/github/languages/top/Haruto1632/AmznML-Chlng?style=default&color=0080ff" alt="repo-top-language">
  <img src="https://img.shields.io/github/languages/count/Haruto1632/AmznML-Chlng?style=default&color=0080ff" alt="repo-language-count">
</p>

---

## Overview

This repository contains our solution for the **Amazon ML Challenge 2026 — Business Entity Resolution** task.

The objective is to identify records in **Source 2** and **Source 3** that correspond to each entity in **Source 1**. The pipeline is designed for large datasets where exhaustive all-pairs comparison is not practical.

For each Source 1 entity, the system generates a bounded candidate set, computes matching features, scores candidate pairs with a machine-learning model, calibrates the decision threshold, and writes the required challenge output files.

## Features

- Data loading for the challenge TSV datasets
- Name, address, and country normalization
- Bounded candidate generation / blocking
- Multiple blocking keys for names and addresses
- String and address similarity features
- Country-aware matching features
- LightGBM-based pair classification
- Threshold calibration for the challenge F0.5 objective
- Batched inference over the test Source 1 dataset
- Candidate-pair and matching-result generation
- Submission validation with the official challenge validator
- No external business-data lookup, geocoding, or API enrichment

## Project Structure

```text
AmznML-Chlng/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── tests/
│       ├── run.py
│       ├── requirements.txt
│       └── README.md
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── student_resource/
│   ├── dataset/
│   └── utils/
├── ARCHITECTURE.md
├── FULL_RUN_REPORT.md
├── GIT_WORKFLOW.md
├── LICENSE
├── PROJECT_HANDOFF.md
├── SCALE_NOTES.md
├── SUBMISSION_CHECKLIST.md
└── README.md
```

The detailed runnable documentation for the submitted pipeline is located at:

```text
code/business_entity_resolution/README.md
```

## Pipeline

The solution follows a multi-stage entity-resolution pipeline:

```text
Challenge TSV data
        │
        ▼
Normalization
        │
        ▼
Candidate Generation / Blocking
        │
        ▼
Pairwise Feature Extraction
        │
        ▼
LightGBM Matching Model
        │
        ▼
Threshold Calibration
        │
        ▼
Test Inference
        │
        ├── matching_results.tsv
        └── candidate_pairs.tsv
```

Candidate generation is intentionally bounded so that the system does not perform an all-pairs comparison between the source tables.

## Getting Started

### Prerequisites

- Python 3.12
- The challenge dataset in the expected directory structure
- Dependencies listed in `code/business_entity_resolution/requirements.txt`

### Installation

Create a virtual environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
```

### Training

A reproducible training run can be started with:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code/business_entity_resolution/run.py train --train-entities 30000 --threads 4
```

The training pipeline uses a sampled set of Source 1 entities and produces the model and calibration information used for inference.

### Inference

Run test inference with:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code/business_entity_resolution/run.py infer
```

### Testing

Run the project tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests -q
```

## Output Files

The inference pipeline produces the two challenge output files:

### `output/matching_results.tsv`

Contains the predicted matching Source 2 and Source 3 entity IDs for each Source 1 entity.

Each Source 1 entity appears exactly once. Entities for which no match is predicted have an empty match field.

### `output/candidate_pairs.tsv`

Contains the final bounded candidate set considered by the matching pipeline.

Candidate generation is an important part of the solution because a true match cannot be recovered by the ranking model if it is absent from the candidate set.

## Validation

The official submission validator can be run with:

```powershell
.\.venv\Scripts\python.exe -X utf8 student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test `
  --check-ids
```

The current generated submission files have been checked with the validator and pass the structural/ID validation.

## Project Roadmap

### Completed

- [x] Set up the Business Entity Resolution project structure
- [x] Implement data loading and TSV handling
- [x] Implement name, address, and country normalization
- [x] Implement bounded candidate generation / blocking
- [x] Implement pairwise matching features
- [x] Integrate the LightGBM matching model
- [x] Implement threshold calibration
- [x] Implement full test-set inference
- [x] Generate `matching_results.tsv`
- [x] Generate `candidate_pairs.tsv`
- [x] Run the official submission validator

### Current / Future Work

- [ ] Improve blocking recall without causing uncontrolled candidate growth
- [ ] Analyze true matches missed during candidate generation
- [ ] Benchmark alternative blocking strategies on controlled validation samples
- [ ] Further tune matching thresholds and model parameters based on validation results
- [ ] Keep the final submission package reproducible and within the challenge constraints

## Challenge Constraints

The solution is designed around the challenge requirements:

- Challenge data is processed from the provided TSV files.
- No external business-data lookup, geocoding, or API enrichment is used.
- Candidate generation is bounded rather than an exhaustive all-pairs search.
- Country handling is not restricted to a fixed US/India-only list.
- The final submission uses the required TSV output format.
- The submitted model is within the challenge model-size and licensing requirements.

## Documentation

Additional project documentation is available in:

- `ARCHITECTURE.md` — system architecture and pipeline design
- `FULL_RUN_REPORT.md` — full-run experiment and inference information
- `SCALE_NOTES.md` — dataset and scalability notes
- `PROJECT_HANDOFF.md` — development handoff information
- `SUBMISSION_CHECKLIST.md` — submission preparation checklist
- `code/business_entity_resolution/README.md` — runnable pipeline documentation

## Contributing

Development work is organized through Git branches. Before making changes:

```powershell
git checkout -b feature/<name>
```

Run the relevant tests before committing changes.

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for the complete license text.

## Acknowledgments

This project was developed as part of the **Amazon ML Challenge 2026**.
