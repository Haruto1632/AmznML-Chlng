"""
test_setup.py — Quick pre-flight check for baseline_v1.

Verifies dependencies, project structure, and data loading
BEFORE running the full (hours-long) experiment.

Run from the repository root:
    python experiments/member_a/baseline_v1/test_setup.py
"""

from __future__ import annotations

# ── Path bootstrap ────────────────────────────────────────────────────────
import _project_paths  # noqa: F401  — configures sys.path

import sys
from pathlib import Path

REPO_ROOT    = _project_paths.REPO_ROOT
EXPERIMENT   = Path(__file__).parent
OK   = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"

pass_count = fail_count = 0

def ok(msg):
    global pass_count
    pass_count += 1
    print(f"{OK} {msg}")

def warn(msg):
    print(f"{WARN} {msg}")

def fail(msg):
    global fail_count
    fail_count += 1
    print(f"{FAIL} {msg}")


print("=" * 60)
print("baseline_v1 — pre-flight check")
print(f"Repo root : {REPO_ROOT}")
print("=" * 60)

# ── 1. Python version ─────────────────────────────────────────────────────
print("\n[1] Python version")
if sys.version_info >= (3, 8):
    ok(f"Python {sys.version.split()[0]}")
else:
    fail(f"Python {sys.version.split()[0]} — 3.8+ required")

# ── 2. Required packages ──────────────────────────────────────────────────
print("\n[2] Required packages")
required = {
    "pandas":     "pandas",
    "numpy":      "numpy",
    "sklearn":    "scikit-learn",
    "yaml":       "pyyaml",
    "tqdm":       "tqdm",
    "lightgbm":   "lightgbm",
    "rapidfuzz":  "rapidfuzz",
    "datasketch": "datasketch",
    "unidecode":  "unidecode",
}
for import_name, pip_name in required.items():
    try:
        __import__(import_name)
        ok(import_name)
    except ImportError:
        fail(f"{import_name}  →  pip install {pip_name}")

print("\n[3] Optional packages")
try:
    import jellyfish
    ok("jellyfish  (phonetic blocking enabled)")
except ImportError:
    warn("jellyfish not found — phonetic blocking will be skipped  "
         "(pip install jellyfish)")

# ── 4. Shared project modules ─────────────────────────────────────────────
print("\n[4] Shared project modules")
shared_imports = [
    ("src.shared.data_loader",               "data_loader"),
    ("src.shared.schemas",                   "schemas"),
    ("src.evaluation.evaluator",             "evaluator"),
    ("src.calibration.calibrator",           "calibrator"),
    ("src.normalization.country_mapper",     "normalize_country"),
]
for dotted, symbol in shared_imports:
    try:
        import importlib
        # e.g. "src.shared.data_loader" → import src.shared.data_loader, get attr "data_loader"
        mod = importlib.import_module(dotted)
        # confirm the symbol is accessible via the parent module too
        ok(dotted)
    except Exception as e:
        fail(f"{dotted}  →  {e}")

# ── 5. Experiment-local modules ───────────────────────────────────────────
print("\n[5] Experiment-local modules")
local_imports = [
    ("normalization", ["clean_name", "clean_address"]),
    ("blocking",      ["MultiStrategyBlocker"]),
    ("features",      ["FeatureBuilder"]),
    ("model",         ["MatchingModel"]),
    ("pipeline",      ["BaselinePipeline"]),
]
for mod_name, attrs in local_imports:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        for attr in attrs:
            getattr(mod, attr)
        ok(f"{mod_name}  ({', '.join(attrs)})")
    except Exception as e:
        fail(f"{mod_name}  →  {e}")

# ── 6. Data files ─────────────────────────────────────────────────────────
print("\n[6] Data files")
data_files = [
    ("student_resource/dataset/train/train_source1.tsv",      "S1 train"),
    ("student_resource/dataset/train/train_source2.tsv",      "S2 train"),
    ("student_resource/dataset/train/train_source3.tsv",      "S3 train"),
    ("student_resource/dataset/train/train_ground_truth.tsv", "Ground truth"),
    ("student_resource/dataset/test/test_source1.tsv",        "S1 test"),
    ("student_resource/dataset/test/test_source2.tsv",        "S2 test"),
    ("student_resource/dataset/test/test_source3.tsv",        "S3 test"),
]
for rel_path, label in data_files:
    abs_path = REPO_ROOT / rel_path
    if abs_path.exists():
        size_mb = abs_path.stat().st_size / 1_048_576
        ok(f"{label:20s}  {size_mb:7.1f} MB  ({rel_path})")
    else:
        fail(f"{label:20s}  NOT FOUND  ({abs_path})")

# ── 7. Config file ────────────────────────────────────────────────────────
print("\n[7] Config file")
config_path = EXPERIMENT / "experiment_config.yaml"
if config_path.exists():
    import yaml
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    ok(f"experiment_config.yaml loaded")
    print(f"     model type          : {cfg['model']['type']}")
    print(f"     max candidates/S1   : {cfg['blocking']['max_candidates_per_s1']}")
    print(f"     train_dir (config)  : {cfg['data']['train_dir']}")
else:
    fail(f"experiment_config.yaml not found at {config_path}")

# ── 8. Smoke-test data loading ────────────────────────────────────────────
print("\n[8] Data loading smoke test (first 100 rows)")
try:
    import pandas as pd
    train_dir = REPO_ROOT / "student_resource" / "dataset" / "train"
    s1_head = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", nrows=100, dtype=str)
    gt_head = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", nrows=100, dtype=str)
    assert list(s1_head.columns) == ["entity_id", "business_name", "business_address", "country"], \
        f"Unexpected S1 columns: {list(s1_head.columns)}"
    assert list(gt_head.columns) == ["source1_entity_id", "matched_entity_ids"], \
        f"Unexpected GT columns: {list(gt_head.columns)}"
    ok(f"S1 head: {len(s1_head)} rows, columns={list(s1_head.columns)}")
    ok(f"GT head: {len(gt_head)} rows, columns={list(gt_head.columns)}")
    # Show one sample row
    sample = s1_head.iloc[0]
    print(f"     Sample S1 row: id={sample['entity_id']}  "
          f"name='{sample['business_name'][:40]}'  "
          f"country='{sample['country']}'")
except Exception as e:
    fail(f"Data loading failed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if fail_count == 0:
    print(f"✅  ALL CHECKS PASSED ({pass_count} passed)")
    print("\nRun the experiment:")
    print(f'  python experiments/member_a/baseline_v1/run_experiment.py \\')
    print(f'      --data-dir "student_resource/dataset"')
else:
    print(f"❌  {fail_count} CHECK(S) FAILED  ({pass_count} passed)")
    print("\nFix the issues above, then re-run this check.")
    sys.exit(1)
print("=" * 60)
