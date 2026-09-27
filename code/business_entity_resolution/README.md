# Amazon ML Challenge 2026 — Business Entity Resolution

This project implements a scalable machine-learning pipeline for business entity
resolution across three independent data sources.

For every Source 1 entity, the pipeline identifies zero, one, or multiple
matching records from Source 2 and Source 3.

## Pipeline Overview

The pipeline consists of the following stages:

1. Data loading and normalization
2. Bounded candidate generation / blocking
3. String and address similarity feature extraction
4. LightGBM binary classification
5. Threshold calibration
6. Batched test inference
7. Generation of the required TSV outputs

The system uses only the data provided by the challenge and does not perform
external business lookups, geocoding, or external data augmentation.

## Directory Structure

```text
business_entity_resolution/
├── src/                    # Pipeline source code
├── tests/                  # Pipeline tests
├── run.py                  # Training and inference entry point
├── requirements.txt        # Python dependencies
└── README.md
