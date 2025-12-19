# core/cluster_definitions.py (NEW - UNIFIED)

# CONFIGURATION
 
CLUSTER_COUNT = 5  # Must match n_clusters in training script


 
# CLUSTER DEFINITIONS
 
CLUSTERS = {
    0: {
        "name": "Growing Suburban Cities",
        "tagline": "Mid-to-large population suburbs with balanced demographics.",
        "summary": """
Cities in this cluster tend to be mid-to-large population suburbs.
They usually have balanced age groups and medium-sized households.
These cities offer steady population growth and family-oriented living.
        """.strip(),
        "best_for": ["Families", "Professionals seeking stability"],
    },
    1: {
        "name": "Large Urban Hubs",
        "tagline": "Big metropolitan cities with diverse populations.",
        "summary": """
This cluster contains big metropolitan cities.
Populations are high, ages vary widely, and household sizes are smaller.
These cities attract diverse residents and have major economic activity.
        """.strip(),
        "best_for": ["Young professionals", "Career-focused individuals"],
    },
    2: {
        "name": "Affordable Mid-Size Cities",
        "tagline": "Moderately populated, often more affordable options.",
        "summary": """
These cities are moderately populated, often more affordable,
and tend to have a slightly younger working population demographic.
Household sizes vary but remain moderate.
        """.strip(),
        "best_for": ["First-time homebuyers", "Young families"],
    },
    3: {
        "name": "Retirement-Friendly Areas",
        "tagline": "Higher median ages, slower pace, calmer lifestyles.",
        "summary": """
Cities in this group often have older median ages,
slower growth, and small to medium population sizes.
They are ideal for retirees seeking calmer lifestyles.
        """.strip(),
        "best_for": ["Retirees", "Seniors", "Those seeking quiet living"],
    },
    4: {
        "name": "Young Professional Hotspots",
        "tagline": "Fast-growing cities attracting younger residents.",
        "summary": """
This cluster includes fast-growing cities with younger median age.
They attract students, young workers, and early-career professionals.
Household sizes tend to be smaller and median age is lower.
        """.strip(),
        "best_for": ["Young professionals", "Students", "Early-career workers"],
    },
}


 
# INTERPRETATION GUIDE
 
INTERPRETATION_GUIDE = """
Cities in this cluster tend to share similar traits:
- Population level  
- How age is distributed across residents  
- How many people typically live together in a home  

If you like one city in this cluster, you will probably enjoy the others in it.
""".strip()


 
# PUBLIC FUNCTIONS
 
def get_cluster_info(cluster_id: int) -> dict:
    """
    Get information about a specific cluster.
    
    Args:
        cluster_id: Cluster ID (0 to CLUSTER_COUNT-1)
    
    Returns:
        dict with keys: name, tagline, summary, best_for
    """
    if cluster_id not in CLUSTERS:
        return {
            "name": f"Unknown Cluster {cluster_id}",
            "tagline": "No information available.",
            "summary": "This cluster ID is not recognized.",
            "best_for": [],
        }
    return CLUSTERS[cluster_id]


def get_cluster_name(cluster_id: int) -> str:
    """Get just the cluster name."""
    return get_cluster_info(cluster_id)["name"]


def get_cluster_tagline(cluster_id: int) -> str:
    """Get just the cluster tagline."""
    return get_cluster_info(cluster_id)["tagline"]


def explain_cluster(cluster_id: int = 0, return_dict: bool = False):
    """
    Get full explanation for a cluster.
    
    Args:
        cluster_id: Cluster ID
        return_dict: If True, return dict; if False, return Markdown
    
    Returns:
        dict or str depending on return_dict
    """
    info = get_cluster_info(cluster_id)
    
    if return_dict:
        return {
            "cluster_id": cluster_id,
            "cluster_name": info["name"],
            "cluster_summary": info["tagline"],
            "detailed_summary": info["summary"],
            "best_for": info["best_for"],
            "how_to_read": INTERPRETATION_GUIDE,
        }
    
    # Markdown format
    best_for_str = ", ".join(info["best_for"]) if info["best_for"] else "General population"
    
    return f"""
### 🧠 Cluster {cluster_id} — {info["name"]}

**{info["tagline"]}**

{info["summary"]}

**Best for:** {best_for_str}

**How to interpret this cluster:**  
{INTERPRETATION_GUIDE}
"""


def get_all_cluster_names() -> dict:
    """Get mapping of cluster_id to name."""
    return {cid: info["name"] for cid, info in CLUSTERS.items()}


 
# BACKWARDS COMPATIBILITY
 
# These match the old cluster_labels.py format
CLUSTER_LABELS = {
    cid: {"name": info["name"], "summary": info["summary"]}
    for cid, info in CLUSTERS.items()
}

# These match the old cluster_explain.py format
CLUSTER_NAMES = {cid: info["name"] for cid, info in CLUSTERS.items()}
CLUSTER_TAGLINES = {cid: info["tagline"] for cid, info in CLUSTERS.items()}
CLUSTER_LIFESTYLE = {cid: info["summary"] for cid, info in CLUSTERS.items()}
