# Amazon ML Challenge 2026 — Business Entity Resolution

The shared pipeline runs normalization, bounded indexed candidate retrieval,
classical string features, LightGBM, calibration and batched TSV inference.
See PROJECT_HANDOFF.md for history and FULL_RUN_REPORT.md for the current run status.

## Reproduce

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
.\.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests -q
.\.venv\Scripts\python.exe -u -X utf8 code/business_entity_resolution/run.py train --train-entities 30000 --threads 4
.\.venv\Scripts\python.exe -u -X utf8 code/business_entity_resolution/run.py infer
.\.venv\Scripts\python.exe -X utf8 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Supply the challenge TSVs under `student_resource/dataset/train` and `test`.
No external entity data is used. `--data-dir`, `--work-dir`, `--model-dir` and
`--output-dir` override paths. Paths are relative to the working directory.
From a standalone code package, run `python run.py train --data-dir <dataset>`.
Do not use the old `python -m business_entity_resolution.pipeline` command.

## Training and evaluation

All training S2/S3 records form the candidate universe. A reproducible sample of
30,000 S1 records is selected without using labels (seed 42). The sample is split
70% model fitting, 10% early stopping, 10% threshold calibration and 10% untouched
evaluation. Zero-candidate entities remain in the metrics. The sample size is
explicit, adjustable with `--train-entities`, and saved with the model.

TF-IDF vocabularies are fitted only on the fitting cohort and its retrieved
training targets. Test data never trains the model, vocabulary or threshold.
`output/model/` contains the model/feature bundle, LightGBM text model, held-out
metrics, threshold curve and evaluation pair scores. Artifacts are local and ignored
by Git. The experiment model import now uses the shared LightGBM wrapper.

## Scale and candidates

Shared normalization runs in 100,000-row chunks into memory-mapped Arrow files.
The existing token, phonetic and address blocking ideas use compact sorted inverted
postings instead of huge Python dictionaries. Keys include name tokens, sorted
normalized name, country-aware Soundex, address numbers and number/name combinations.
At most six name tokens and two address numbers contribute keys per record.

Buckets larger than 128 records are discarded before querying; each S1 queries
its four rarest usable keys, then keeps at most 24 candidates ranked by distinct
strategy evidence, inverse bucket size and stable row order. This bounds pair
materialization before feature generation. MinHash/LSH and embeddings are not
used in the first full-scale baseline. The tradeoff is missed matches, measured
on a holdout against the full training candidate universe.

Features retain the baseline's name/address token and numeric overlap,
Levenshtein, Jaro-Winkler, character-trigram TF-IDF cosine, country agreement,
and blocking evidence; missing-name/address indicators are included. Sparse
TF-IDF is transformed in batches, with rowwise sparse products, never a dense
all-pairs similarity matrix. LightGBM is the original baseline model family.

## Inference and outputs

Inference always processes ALL test S1 records, with 1,000 S1 records per batch.
The saved threshold and feature transforms are reused. Every scored candidate
appears in `output/candidate_pairs.tsv`; matches are a subset of those exact
scored pairs. Empty lists are empty TSV cells. No cross-entity uniqueness constraint
is imposed, since each S1 may have zero, one or multiple matches.

Checkpoint files in `output/work/` permit restarting the same inference command.
Final filenames are assembled only after all batches finish. Source file metadata,
model hash and batch size identify caches/checkpoints; rebuild if normalization
or blocking code changes. `output/inference_metrics.json` records counts, candidate
quantiles, probability histogram, runtime and approximate process peak memory.
Run the unmodified official validator with `--check-ids`; inspect warnings as well
as the exit code. A code/test pass alone does not establish submission readiness.

## Constraints and limitations

Country normalization is open-set, including France. LightGBM is MIT licensed
and far below 8B parameters. Other dependency licenses are documented separately
in requirements (including GPL Unidecode and BSD utilities); they are not model
weights. No business lookup, geocoding, external augmentation, transformer or
cross-encoder is used. The sampling and bucket/candidate caps limit quality.
The previous small enriched smoke score is not a generalization estimate.

Datasets, environments, normalized caches, checkpoints and large outputs must
remain untracked. Preserve the official validator under `student_resource/utils/`.
