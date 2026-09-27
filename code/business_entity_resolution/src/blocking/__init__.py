"""
blocking/ — Multi-strategy candidate blocking package.

Owner: Member A  |  Branch: feature/blocking

Public interface:
    from .blocker import generate_candidates
"""

from .blocker import generate_candidates

__all__ = ["generate_candidates"]
