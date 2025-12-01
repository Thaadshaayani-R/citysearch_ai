# core/cluster_explain.py (UPDATED)
"""
Backwards compatibility wrapper for cluster explanations.
Uses cluster_definitions.py as single source of truth.
"""

from .cluster_definitions import (
    CLUSTER_NAMES,
    CLUSTER_TAGLINES,
    CLUSTER_LIFESTYLE,
    INTERPRETATION_GUIDE,
    explain_cluster,
)

# Re-export for backwards compatibility
__all__ = [
    "CLUSTER_NAMES",
    "CLUSTER_TAGLINES", 
    "CLUSTER_LIFESTYLE",
    "INTERPRETATION_GUIDE",
    "explain_cluster",
]
