"""Compatibility import for the baseline model now shared with production."""
import _project_paths  # configures the existing src package
from src.models.baseline_model import MatchingModel

__all__ = ["MatchingModel"]
