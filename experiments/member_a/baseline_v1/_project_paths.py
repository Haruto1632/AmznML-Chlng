"""
_project_paths.py — sys.path bootstrap for the member_a baseline experiment.

Import this module FIRST in any script that needs shared project modules.
It is idempotent: importing it multiple times is safe.

Root cause of the original ModuleNotFoundError
-----------------------------------------------
The shared modules live under:

    code/business_entity_resolution/src/
        shared/        (data_loader, schemas)
        evaluation/    (evaluator)
        calibration/   (calibrator)
        normalization/ (country_mapper)

They use intra-package relative imports such as:

    from ..shared.schemas import COL_SOURCE1_ID          # in evaluator.py
    from ..evaluation.evaluator import compute_macro_f05 # in calibrator.py

Relative imports ONLY work when a module is loaded as part of a proper package.
'src' is that package — it has an __init__.py.

Attempt 1 (broken): add src/ to sys.path → import as `shared`, `evaluation`, …
  • Python finds and loads them as TOP-LEVEL modules, not as `src.shared`, `src.evaluation`.
  • When evaluator.py executes `from ..shared.schemas import …` it fails with
    "attempted relative import beyond top-level package" because the parent
    package (src) was never set up.

Fix: add code/business_entity_resolution/ to sys.path.
  • Python can now do `import src` (which IS an __init__.py package).
  • Every submodule loads as src.shared, src.evaluation, etc.
  • All relative imports (`from ..shared import …`) resolve correctly.
  • There is no naming clash with stdlib.

Imports in experiment code
--------------------------
    from src.shared      import data_loader, schemas
    from src.evaluation  import evaluator
    from src.calibration import calibrator
    from src.normalization.country_mapper import normalize_country
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Absolute paths ──────────────────────────────────────────────────────────

# Repository root  (parents: baseline_v1 → member_a → experiments → repo_root)
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# Parent of the 'src' package: adding this lets Python do  import src.shared …
SHARED_PKG_PARENT: Path = REPO_ROOT / "code" / "business_entity_resolution"

# The experiment directory itself (for normalization.py, blocking.py, …)
EXPERIMENT_DIR: Path = Path(__file__).resolve().parent

# Default dataset locations
DEFAULT_TRAIN_DIR: Path = REPO_ROOT / "student_resource" / "dataset" / "train"
DEFAULT_TEST_DIR:  Path = REPO_ROOT / "student_resource" / "dataset" / "test"


# ── sys.path helper ─────────────────────────────────────────────────────────

def _insert_once(path: str) -> None:
    """Prepend *path* to sys.path only if not already present."""
    if path not in sys.path:
        sys.path.insert(0, path)


def setup() -> None:
    """Configure sys.path.  Safe to call multiple times."""
    # 1. Allows:  from src.shared import …   (relative imports inside src work)
    _insert_once(str(SHARED_PKG_PARENT))
    # 2. Allows:  from normalization import clean_name   (experiment-local files)
    _insert_once(str(EXPERIMENT_DIR))


# Run automatically on import.
setup()
