# core/cluster_explain.py
"""
Return human-friendly cluster names and explanations.
Supports both:
- Markdown explanation
- Dictionary explanation (for UI cards)
"""

# -------------------------------
# 1. Friendly Cluster Names
# -------------------------------
CLUSTER_NAMES = {
    0: "Balanced Mid-Size Cities",
    1: "Young Urban Growth Cities",
    2: "High-Age Suburban Towns",
    3: "Large Diverse Metros",
}

# -------------------------------
# 2. Short Taglines
# -------------------------------
CLUSTER_TAGLINES = {
    0: "Mid-to-large cities with balanced ages and medium-sized households.",
    1: "Fast-growing cities attracting younger adults with compact households.",
    2: "Stable suburban towns with higher median ages and smaller families.",
    3: "Large metro areas with diverse age groups and varied household structures.",
}

# -------------------------------
# 3. Lifestyle Summaries
# -------------------------------
CLUSTER_LIFESTYLE = {
    0: """
Medium to large cities with a balanced mix of ages and medium-sized households.
These places are stable, family-friendly, and offer predictable growth.
""",
    1: """
Cities in this cluster tend to attract younger residents.
They often have higher population growth and smaller household sizes.
Good for students, professionals, and first-time movers.
""",
    2: """
Older suburban-style cities with higher median ages.
Relaxed pace of life, quieter neighborhoods, and stable populations.
""",
    3: """
These are large, diverse, multi-cultural metro cities.
They have big populations, wide age distribution, and mixed household structures.
""",
}

# -------------------------------
# 4. Interpretation Guide
# -------------------------------
INTERPRETATION_GUIDE = """
Cities in this cluster tend to share similar traits:

- Population level  
- How age is distributed across residents  
- How many people typically live together in a home  

If you like one city in this cluster, you will probably enjoy the others in it.
"""

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def explain_cluster(df=None, cluster_id=0, return_dict=False):
    """
    Returns either:
    - Markdown explanation (default)
    - Dictionary for UI (return_dict=True)
    """

    name = CLUSTER_NAMES.get(cluster_id, "Unknown Cluster")
    tagline = CLUSTER_TAGLINES.get(cluster_id, "")
    lifestyle = CLUSTER_LIFESTYLE.get(cluster_id, "")
    howto = INTERPRETATION_GUIDE

    if return_dict:
        return {
            "cluster_id": cluster_id,
            "cluster_name": name,
            "cluster_summary": tagline,
            "detailed_summary": lifestyle,
            "how_to_read": howto,
        }

    # Markdown fallback version
    return f"""
### 🧠 Cluster {cluster_id} — {name}

**Lifestyle Summary**  
{lifestyle}

**How to interpret this cluster:**  
{howto}
"""
