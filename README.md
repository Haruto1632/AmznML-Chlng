# AmznML-Chlng

Amazon ML Challenge 2026: business entity resolution. Start with
[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) for the verified implementation status,
existing experiment results, ownership and next tasks.

The shared training/inference pipeline is still scaffolded. The existing
`experiments/member_a/baseline_v1/` implements a development baseline.
All seven challenge TSVs belong under `student_resource/dataset/`; `__MACOSX`
contains metadata only.

From the repository root, use Python 3.12 with the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/business_entity_resolution/requirements.txt
.\.venv\Scripts\python.exe -m pytest code/business_entity_resolution/tests -q
.\.venv\Scripts\python.exe -X utf8 experiments/member_a/baseline_v1/test_setup.py
.\.venv\Scripts\python.exe -X utf8 experiments/member_a/baseline_v1/run_smoke.py --entities 100
```

The smoke runner streams training files to select real rows, limits the working
sample, runs LightGBM and validates both TSVs against a validation-only source
fixture. Its true-match-enriched pool is for integration testing, not estimating
leaderboard performance. Outputs and metrics go to `output/smoke/`. It does not
produce a full test submission. The full baseline and blocking runners load
millions of records; review the handoff's scalability risks before using them.
