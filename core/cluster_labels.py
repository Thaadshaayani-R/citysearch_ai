# core/cluster_labels.py (UPDATED)
"""
Backwards compatibility wrapper for cluster labels.
Uses cluster_definitions.py as single source of truth.
"""

from .cluster_definitions import CLUSTER_LABELS

# Re-export for backwards compatibility
__all__ = ["CLUSTER_LABELS"]
