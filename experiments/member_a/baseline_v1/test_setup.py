"""
test_setup.py — Quick validation that all dependencies and data are accessible.

Run this before running the full experiment.
"""

import sys
from pathlib import Path

print("Testing baseline_v1 setup...")
print("=" * 60)

# Test 1: Python version
print("\n1. Python version:")
print(f"   {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required")
    sys.exit(1)
print("   ✅ OK")

# Test 2: Core packages
print("\n2. Core packages:")
packages = [
    'pandas', 'numpy', 'sklearn', 'yaml', 'tqdm',
    'lightgbm', 'rapidfuzz', 'datasketch', 'unidecode'
]

missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg} (missing)")
        missing.append(pkg)

if missing:
    print(f"\n   Install missing packages:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)

# Test 3: Optional packages
print("\n3. Optional packages:")
try:
    import jellyfish
    print("   ✅ jellyfish (phonetic blocking enabled)")
except ImportError:
    print("   ⚠️  jellyfish (phonetic blocking will be skipped)")

# Test 4: Project structure
print("\n4. Project structure:")
project_root = Path(__file__).parents[3]
required_paths = [
    project_root / "student_resource/dataset/train/train_source1.tsv",
    project_root / "student_resource/dataset/train/train_source2.tsv",
    project_root / "student_resource/dataset/train/train_source3.tsv",
    project_root / "student_resource/dataset/train/train_ground_truth.tsv",
    project_root / "code/business_entity_resolution/src/shared/data_loader.py",
    project_root / "code/business_entity_resolution/src/evaluation/evaluator.py",
]

all_exist = True
for p in required_paths:
    if p.exists():
        print(f"   ✅ {p.relative_to(project_root)}")
    else:
        print(f"   ❌ {p.relative_to(project_root)} (missing)")
        all_exist = False

if not all_exist:
    print("\n   Missing required files. Check dataset placement.")
    sys.exit(1)

# Test 5: Import shared modules
print("\n5. Shared modules:")
sys.path.insert(0, str(project_root))

try:
    from code.business_entity_resolution.src.shared import data_loader, schemas
    print("   ✅ data_loader")
    print("   ✅ schemas")
except ImportError as e:
    print(f"   ❌ Failed to import shared modules: {e}")
    sys.exit(1)

try:
    from code.business_entity_resolution.src.evaluation import evaluator
    print("   ✅ evaluator")
except ImportError as e:
    print(f"   ❌ Failed to import evaluator: {e}")
    sys.exit(1)

try:
    from code.business_entity_resolution.src.calibration import calibrator
    print("   ✅ calibrator")
except ImportError as e:
    print(f"   ❌ Failed to import calibrator: {e}")
    sys.exit(1)

# Test 6: Load a small sample
print("\n6. Data loading test:")
try:
    s1, s2, s3, gt = data_loader.load_all_train(
        str(project_root / "student_resource/dataset/train")
    )
    print(f"   ✅ Loaded train S1: {len(s1):,} records")
    print(f"   ✅ Loaded train S2: {len(s2):,} records")
    print(f"   ✅ Loaded train S3: {len(s3):,} records")
    print(f"   ✅ Loaded ground truth: {len(gt):,} rows")
except Exception as e:
    print(f"   ❌ Failed to load data: {e}")
    sys.exit(1)

# Test 7: Config file
print("\n7. Configuration:")
config_path = Path(__file__).parent / "experiment_config.yaml"
if config_path.exists():
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"   ✅ Config loaded from {config_path.name}")
    print(f"   Model type: {config['model']['type']}")
    print(f"   Max candidates per S1: {config['blocking']['max_candidates_per_s1']}")
else:
    print(f"   ❌ Config file missing: {config_path}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED — Ready to run experiment")
print("=" * 60)
print("\nRun the experiment:")
print(f"  python {Path(__file__).parent / 'run_experiment.py'}")
