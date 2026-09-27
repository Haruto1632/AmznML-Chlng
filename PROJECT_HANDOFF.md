# Project handoff — 2026-09-27

## Repository and history

Audited local branch `feature/blocking`, HEAD `a488aa0`, tracking
`origin/feature/blocking`; local `main` and cached `origin/main` are `09aa720`.
Remote freshness is UNKNOWN / NEEDS VERIFICATION (no fetch performed).
No commits reset, rewritten, or pushed during this session.

Pre-existing changes preserved: `src/blocking/blocker.py` changes token bucket
defaults; `src/blocking/strategies/token_blocking.py` removes an intermediate
country/token DataFrame construction. Untracked `exp_log.txt`, blocking `cache/`
and `results/` existed before takeover. Do not stage them indiscriminately.
No AGENTS.md, pyproject.toml, setup.py or setup.cfg found in the repository.
The named `Business Entity Resolution — Task PRD.md` was not found: UNKNOWN /
NEEDS VERIFICATION. Root README contains only a title. The actual methodology
template is `student_resource/Documentation_template.md` and is still blank.

Commit evidence:

| Commit | Verified contents |
|---|---|
| ae68b68 | Initial repository |
| 9ad6e0e | Shared contracts, planning documents and module scaffolds |
| db70d4d | Separate member_a baseline normalization/blocking/features/GBM experiment |
| 09aa720 | Baseline path bootstrap and data path fixes |
| bdbddf7 | Shared normalization, indexed blocking, consolidation and experiment suite |
| f66bf92 | Legal suffix exclusion, token bucket caps, index optimization |
| a488aa0 | Preprocessing cache script and experiment configuration updates |

Kiro versus Claude/Antigravity authorship cannot be established from Git: commits
are attributed to the user's Git identities, not agents. Agent-specific completion
is UNKNOWN / NEEDS VERIFICATION. Use the commit contents above, not guessed attribution.

## Architecture and responsibilities

Documented flow: load -> normalize -> multiple blockers -> consolidate -> pair
features -> GBM -> macro F0.5 threshold calibration -> two TSVs -> official validator.
Stable candidate, feature and prediction DataFrame contracts live in
`code/business_entity_resolution/src/shared/schemas.py`.

Repository ownership differs from the new proposed assignments: TEAM_TASKS.md,
ARCHITECTURE.md and GIT_WORKFLOW.md assign A normalization/blocking/candidates,
B features/model/scoring, C loader/evaluation/calibration/pipeline/output.
Actual human reassignment is UNKNOWN / NEEDS VERIFICATION. This takeover fixes
evaluation/integration without overwriting existing blocking edits.

## Implemented versus incomplete

- Shared normalization: name/address/country cleaning and added columns implemented.
- Active blocking: token_blocking.py, phonetic_blocking.py, ngram_blocking.py,
  address_blocking.py and blocker.py implemented; consolidation implemented.
  Older exact_token.py/token_overlap.py/phonetic.py/ngram_lsh.py/address_token.py
  are scaffold alternatives, not the active orchestrator imports.
- Embedding ANN is optional and not established as usable; no cross-encoder exists.
- Shared country and structural features implemented; name/address features and
  feature_builder remain NotImplementedError stubs.
- Shared model, training-pair construction, hard negatives and scoring remain stubs.
- Shared evaluator, calibration, loader and output writer have implementations,
  with evaluation/integrity gaps found during takeover.
- Shared pipeline is a scaffold that catches missing implementation/model errors
  and can write all-singleton output. It is NOT a trained submission pipeline.
- Separate baseline_v1 has real feature and LightGBM implementations, but eagerly
  loads all data and computes features row by row. It is not production scale.
  Test inference/artifact integration is unfinished; run_on_test does not implement it.
- Many original test classes contain only pass/comments, so passing those tests
  does not establish model/blocker readiness.

## Exact stop point and existing measurements

`exp_log.txt` ends at B3 phonetic generation: **93,021,748 pairs for 5,000 S1**,
after building indices over 10,320,219 S2/S3 records. No B3 result JSON exists.
This establishes the last recorded operation; process termination cause is UNKNOWN.
B0/B1/B2 JSONs exist. Historical B1 recall 0.2954035552 (5,135/17,383),
547,598 pairs, 62.29 s. B2 recall 0.5976528792 (10,389/17,383), 652,791 pairs,
136.59 s. These are inherited artifacts, not reproduced measurements.
Their mean/median excluded zero-candidate S1 records: corrected all-5,000 means
would be 109.5196 and 130.5582 respectively. Recall remains a historical observation.
No representative trained-model validation score or full submission exists.
The takeover's bounded integration score is recorded below separately.

## Dataset

All seven real TSVs exist under `student_resource/dataset/`:

| Relative file | Bytes |
|---|---:|
| train/train_source1.tsv | 210069713 |
| train/train_source2.tsv | 489301488 |
| train/train_source3.tsv | 503705637 |
| train/train_ground_truth.tsv | 127015583 |
| test/test_source1.tsv | 175022086 |
| test/test_source2.tsv | 509456422 |
| test/test_source3.tsv | 506002772 |

`__MACOSX` contains metadata sidecars, not usable data. Root dataset/ is not the
actual dataset. SCALE_NOTES.md records full row counts; these have not been
independently recounted. Its 2.2M x 10.3M product is approximately 22.8 TRILLION,
not billion. No external entity data has been accessed.
All seven TSV headers were checked and match the source/ground-truth contracts.
Six existing normalized Parquet caches (train/test S1/S2/S3) are present under
`experiments/member_a/blocking/cache/`; cache contents/freshness have not been
revalidated. Preprocessing's chunk_size argument does not prevent full-file loading.

## Takeover work and priority

The immediate bottleneck chosen was trustworthy validation, ahead of blocking
optimization or advanced models. Original import issue is already fixed by
_project_paths.py importing src.*; avoid reintroducing stdlib `code` collisions.
The documented `python -m business_entity_resolution.pipeline` command does not
match the source layout and no installable package configuration exists.

Changes this session: retain literal TSV strings; reject duplicate ground truth;
remove iterrows from ground-truth conversion and threshold application; validate
calibration cohorts/probabilities; deduplicate threshold matches; conservatively
break threshold ties; include zero-candidate entities in blocking statistics;
complete evaluation report blocking integration; split baseline from all S1 IDs
and sample only training pairs; export validation-only scored candidates/matches;
retain validation blocking metrics in baseline reports; add parent package marker
for test relative imports. Also repaired candidate output duplicate/extra-row/schema
checks, added deterministic candidate-cap tie breaking to the separate baseline,
and an optional LightGBM thread setting (smoke uses two). Added a bounded real-data
smoke runner and regression tests; original blocking edits remain intact.

## Next implementation steps and risks

1. Use the regression/smoke commands below before running large experiments.
2. Bound phonetic/address bucket expansion BEFORE materializing raw pairs, then
   rerun B1/B2/B3 with corrected cohort metrics and peak-memory measurements.
3. Reuse baseline feature/model implementations to fill the shared contracts;
   batch sparse transforms/scoring, remove full-data Python dictionaries/iterrows.
4. Persist model, feature transforms, split IDs and threshold; separate calibration
   from untouched evaluation data. Current baseline score is a tuning-set score.
5. Replace shared pipeline's silent all-singleton fallbacks with explicit failures
   while wiring real training/inference. Validate final candidate/scored-pair equality.
6. Run full test inference, official validation, fill methodology, then package.

Full-scale memory/runtime and model quality are unverified. A final consolidation
cap does not prevent raw-pair explosion. Baseline TF-IDF is fitted on the full
development corpus (transductive), and early stopping/calibration share validation.
README's claim that every dependency is MIT/Apache is false: requirements includes
GPL Unidecode and BSD libraries; final model license is distinct and must be documented.
No model weights selected beyond the existing LightGBM baseline.

## Environment, tests and continuation commands

Default Python 3.13 lacks dependencies. Existing Python 3.11 also lacks the ML
stack. Workspace `.venv` uses Python 3.12.14, created from the available bundled
runtime with system-site-packages enabled. All repository-pinned requirements
installed successfully into the venv; a fresh machine can use an ordinary Python
3.12 venv with the same requirements. There is no editable/installable project
distribution; scripts use the existing src-package bootstrap.

Executed:

- `python -m pytest ...` initially failed on default Python (pytest missing).
- After environment setup, `.venv` suite: **39 passed in 2.26 s**. This includes
  14 new regression/integration cases, alongside existing tests; empty scaffold
  test classes are not evidence of shared model coverage.
- Baseline `test_setup.py`: **31 checks passed**, including imports and real TSV
  reads. This confirms the old `code`/stdlib import issue is already resolved.
- `compileall` on shared code and baseline: passed.
- `git diff --check`: passed (only Git CRLF conversion notices).
- Real-data `run_smoke.py --entities 100`: passed; 100 S1, 347 S2, 375 S3;
  80/20 S1 split, 3,129 train pairs and 671 scored validation pairs. Sample selection
  streams all train GT/S2/S3 chunks to retain actual positives and 200 distractors
  per source. No invented business rows. This is not a full-data experiment.
- Official `student_resource/utils/validate_submission.py --check-ids`: **PASS**
  on the smoke validation fixture, 20 rows in each output, 722 valid S2/S3 IDs.
  Internal integrity check also passed. Full test submission validation NOT run.

Measured smoke calibration results (20 entities, 2 true singletons):

| Metric | Value |
|---|---:|
| Macro F0.5 | 0.9788690476 |
| Macro precision | 1.0 |
| Macro recall | 0.9525 |
| Singleton accuracy | 1.0 (2/2) |
| Validation blocking recall | 0.9545454545 (63/66) |
| Average / median candidates | 33.55 / 43 |
| Candidate reduction ratio | 0.9535318560 |
| Selected threshold | 0.70 |
| Pipeline runtime | 7.98 s |
| Selection + validator overhead | 20.65 s |

These are integration/calibration measurements on a true-match-enriched small
pool, NOT representative generalization or leaderboard estimates. Peak memory
was not measured. Metrics/config/sample/validation fixture and both TSVs are
under ignored `output/smoke/`. No full submission or final trained model is packaged.

Run from repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests -q
.\.venv\Scripts\python.exe -X utf8 experiments/member_a/baseline_v1/test_setup.py
.\.venv\Scripts\python.exe -X utf8 experiments/member_a/baseline_v1/run_smoke.py --entities 100
.\.venv\Scripts\python.exe -X utf8 student_resource/utils/validate_submission.py --matching output/smoke/matching_results.tsv --candidate output/smoke/candidate_pairs.tsv --test-dir output/smoke/validation_sources --check-ids
```

After bounding raw blocking expansion, a next blocking benchmark command is:

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/member_a/blocking/run_experiments.py --sample 5000 --experiment B3
```

That command still loads the FULL S2/S3 pool; do not mistake --sample for a memory
limit. Parquet experiment reads also require a parquet engine (not pinned in
requirements; present in this machine's inherited runtime, clean-environment
support NEEDS VERIFICATION). Final official validation, once full outputs exist:

```powershell
.\.venv\Scripts\python.exe -X utf8 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids
```

## Final working tree and changed files

Still on `feature/blocking` at a488aa0; no commits or pushes made. Nine previously
tracked files changed by takeover, four new source/report/test files, plus the
two original modified blocking files and original untracked log/cache/results.

Takeover files:

- README.md; PROJECT_HANDOFF.md
- code/business_entity_resolution/__init__.py
- code/business_entity_resolution/src/shared/data_loader.py
- code/business_entity_resolution/src/evaluation/evaluator.py
- code/business_entity_resolution/src/calibration/calibrator.py
- code/business_entity_resolution/src/candidate_generation/blocking_eval.py
- code/business_entity_resolution/src/pipeline/output_writer.py
- code/business_entity_resolution/tests/test_validation_regressions.py
- experiments/member_a/baseline_v1/pipeline.py
- experiments/member_a/baseline_v1/blocking.py
- experiments/member_a/baseline_v1/model.py
- experiments/member_a/baseline_v1/run_smoke.py

Next team action: A (per repository ownership) bounds raw phonetic/address
generation and benchmarks recall/memory; B integrates baseline features/LightGBM
behind shared contracts; C uses the corrected complete-cohort evaluation and
finishes fail-fast training/inference and final validation integration.
