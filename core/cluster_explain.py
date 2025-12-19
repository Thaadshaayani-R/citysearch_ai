# core/cluster_explain.py (UPDATED)


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
