# Git Workflow — Amazon ML Challenge 2026

---

## Branch Structure

```
main
├── feature/blocking             (Member A)
├── feature/matching-model       (Member B)
└── feature/evaluation-pipeline  (Member C)
```

**Rule:** Never commit directly to `main`. All work goes in feature branches and is merged via pull request.

---

## Initial Setup (run once per teammate)

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Create your branch from main
git checkout main
git pull origin main

# Member A:
git checkout -b feature/blocking

# Member B:
git checkout -b feature/matching-model

# Member C:
git checkout -b feature/evaluation-pipeline
```

---

## Daily Workflow

```bash
# Start of day: sync with main
git fetch origin
git rebase origin/main   # or git merge origin/main

# Work on your files...

# Commit often (small, focused commits)
git add src/normalization/normalizer.py
git commit -m "feat(normalization): add legal suffix expansion table"

# Push your branch
git push origin feature/blocking
```

---

## Merge Process

1. Push your branch to origin
2. Open a pull request to `main`
3. Another teammate reviews (quick review, not blocking)
4. Merge once CI passes (or manually if no CI)
5. All teammates pull main and rebase after a merge

---

## File Ownership — Conflict Prevention

The key rule: **each file is owned by exactly one member**.

| Path | Owner | Others should NOT edit |
|---|---|---|
| `src/shared/schemas.py` | C | A, B read only — request changes via PR comment |
| `src/shared/data_loader.py` | C | A, B read only |
| `src/normalization/` | **A** | B, C do not touch |
| `src/blocking/` | **A** | B, C do not touch |
| `src/candidate_generation/` | **A** | B, C do not touch |
| `src/features/` | **B** | A, C do not touch |
| `src/models/` | **B** | A, C do not touch |
| `src/scoring/` | **B** | A, C do not touch |
| `src/evaluation/` | **C** | A, B do not touch |
| `src/calibration/` | **C** | A, B do not touch |
| `src/pipeline/` | **C** | A, B do not touch |
| `experiments/data_exploration.md` | A | |
| `experiments/blocking_v*.md` | A | |
| `experiments/baseline.md` | B | |
| `experiments/model_v*.md` | B | |
| `experiments/calibration_v*.md` | C | |
| `experiments/ablation_results.md` | C | A, B add rows only (no rewrites) |
| `ARCHITECTURE.md` | shared — discuss before changing | |
| `TEAM_TASKS.md` | shared — discuss before changing | |
| `output/` | C generates final outputs | |

---

## Interface Change Protocol

If any member needs to change a **shared interface** (function signatures in schemas, data_loader, or the three stable interfaces in ARCHITECTURE.md):

1. Open a GitHub issue or team chat thread describing the change.
2. All three members must agree.
3. The member who owns the file makes the change and notifies others.
4. Other members update their usage in their own branch immediately.

---

## Priority Merges (do these first)

Because A and B both depend on `src/shared/`, Member C must merge the shared contracts to `main` **before** A and B start coding.

**Day 1 sequence:**
1. C: push `src/shared/schemas.py` + `src/shared/data_loader.py` → PR → merge to main
2. A: pull main, start `feature/blocking`
3. B: pull main, start `feature/matching-model`

---

## Commit Message Convention

```
<type>(<scope>): <short description>

Types: feat, fix, refactor, test, docs, experiment
Scopes: normalization, blocking, candidates, features, model, scoring, evaluation, calibration, pipeline, shared

Examples:
feat(blocking): implement MinHash LSH strategy
fix(normalization): handle missing country values without crash
experiment(blocking): log v1 recall results
test(features): add name Levenshtein feature tests
docs(readme): add test pipeline run instructions
```

---

## Commands Each Teammate Should Run Daily

### Member A
```bash
git checkout feature/blocking
git rebase origin/main
# work on src/normalization/, src/blocking/, src/candidate_generation/
python -m pytest tests/test_normalizer.py tests/test_blocking.py -v
git add src/normalization/ src/blocking/ src/candidate_generation/
git commit -m "feat(blocking): ..."
git push origin feature/blocking
```

### Member B
```bash
git checkout feature/matching-model
git rebase origin/main
# work on src/features/, src/models/, src/scoring/
python -m pytest tests/test_features.py tests/test_model.py -v
git add src/features/ src/models/ src/scoring/
git commit -m "feat(model): ..."
git push origin feature/matching-model
```

### Member C
```bash
git checkout feature/evaluation-pipeline
git rebase origin/main
# work on src/shared/, src/evaluation/, src/calibration/, src/pipeline/
python -m pytest tests/test_evaluator.py tests/test_output_writer.py -v
git add src/shared/ src/evaluation/ src/calibration/ src/pipeline/
git commit -m "feat(pipeline): ..."
git push origin feature/evaluation-pipeline
```

---

## Integration Testing

Once A and B have their modules working, C will:

1. Pull both branches into a local integration branch
2. Run the full pipeline end-to-end on train data
3. Check that `output/matching_results.tsv` and `output/candidate_pairs.tsv` are generated
4. Run `utils/validate_submission.py` — must pass
5. Report F_0.5 to the team

```bash
git checkout -b integration/test-run
git merge feature/blocking
git merge feature/matching-model
git merge feature/evaluation-pipeline
python -m business_entity_resolution.pipeline --mode train
python -m business_entity_resolution.pipeline --mode test
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
