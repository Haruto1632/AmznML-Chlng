# Full pipeline run report — 2026-09-27

Status: IN PROGRESS. Full outputs and official validation are not yet complete.

Base revision: 214c22b on feature/blocking. Only pre-existing untracked exp_log.txt
was present at the start of this session. No branch changes, commits or pushes.

Verified before full launch: 42 tests pass; integrated train/save/load/infer on
real-data subset passes the unchanged official validator with --check-ids.

Full training command:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code/business_entity_resolution/run.py train --train-entities 30000 --threads 4
```

Full inference command (after training):

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code/business_entity_resolution/run.py infer
```

Official validator command (after inference):

```powershell
.\.venv\Scripts\python.exe -X utf8 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

Training normalization/index generation uses the full provided training data;
model fitting uses a fixed random 30,000-S1 sample against full S2/S3. Cohorts:
21,000 fit, 3,000 early stopping, 3,000 calibration, 3,000 untouched evaluation.
Inference has no test sampling. Resource budget: 24 candidates/S1, bucket cap128,
four rarest keys/S1, 1,000-S1 inference batches, four LightGBM threads.

Logs: output/full_train.log and output/full_prepare_test.log. Raw data, caches,
model files, checkpoints and output files are ignored by Git.
