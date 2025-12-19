# core/intent_classifier.py

import re

US_STATES = [
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware",
    "florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
    "louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi",
    "missouri","montana","nebraska","nevada","new hampshire","new jersey","new mexico",
    "new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania",
    "rhode island","south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming"
]


def _detect_state(text: str):
    text = text.lower()
    for s in US_STATES:
        if s in text:
            return s.title()
    return None


def classify_query_intent(query: str):
    """
    Classify the query into:
      - mode: "sql", "semantic", or "hybrid"
      - state_filter: optional state name (e.g. "Texas") if present

    Rules (simple but effective):
    - Numeric / structured → SQL
    - Subjective / lifestyle → Semantic
    - Subjective + State mention → Hybrid (semantic within that state)
    """
    q = query.lower()
    
    # Check for lifestyle-related queries by keywords
    lifestyle_keywords = [
        "life in", "living in", "what is life like in", "what is it like in", "tell me about", "living in", "life in"
    ]
    if any(keyword in q for keyword in lifestyle_keywords):
        # If there is a state mentioned with lifestyle keywords, classify as hybrid
        state = _detect_state(q)
        if state:
            return "hybrid", state  # Lifestyle with state -> hybrid
        return "semantic", None  # Lifestyle without state -> semantic

     
    # ML INTENTS (Family / Young Adults / Retirement)
     
    # Family keywords
    if any(word in q for word in ["family", "families", "kids", "children", "child", "kid", "schools", "raising"]):
        return "ml_family", None

    # Young professionals keywords
    if any(word in q for word in ["young", "young adult", "young professionals", "adults", "adult", "career", "jobs", "millennials", "working professionals"]):
        return "ml_young", None

    # Retirement keywords
    if any(word in q for word in ["retire", "retirement", "seniors", "senior", "elderly", "retirees", "older adults"]):
        return "ml_retirement", None

     
    # Compare two cities (ML + GPT explanation)
     
    if "compare" in q:
        return "ml_compare_cities", None

     
    # Predict score for single city
     
    if "score for" in q or "predict" in q:
        return "ml_single_city", None

    # numeric / structured signals
    numeric_keywords = [
        "population", "median age", "median_age", "avg household size",
        "average household size", "avg_household_size",
        "how many", "number of", "count of", "total cities", "total number",
        "percentage of", "what percentage", "percent of", "what percent",
        "group by", "by state", "per state", "average population by"
    ]
    comparison_tokens = [">", "<", ">=", "<=", "="]

    has_number = bool(re.search(r"\d", q))
    has_comparison = any(tok in q for tok in comparison_tokens)
    has_numeric_keywords = any(k in q for k in numeric_keywords)
    has_top_pattern = bool(re.search(r"\btop\s+\d+", q))

    numeric_flag = has_number or has_comparison or has_numeric_keywords or has_top_pattern

    # semantic / lifestyle signals
    semantic_keywords = [
        "similar to", "similar cities", "like ", "vibe", "lifestyle",
        "family-friendly", "family friendly", "good for families",
        "retirement", "retire", "senior", "young professionals",
        "nightlife", "beach", "coastal", "coast", "tourist",
        "college town", "students", "student", "startup", "tech hub",
        "culture", "vibrant", "affordable to live", "affordable city",
        "quality of life", "good to live", "good to raise a family",
        "safe city", "safe cities", "crime", "walkable", "walkability",
        "best cities", "best city", "best place", "best places"
    ]

    semantic_flag = any(k in q for k in semantic_keywords)

    # special: "best cities for X" without numbers is semantic
    if "best " in q and not has_number and not has_comparison:
        semantic_flag = True

    state = _detect_state(q)

    # classification logic
    # 1) numeric-only or numeric-dominant → SQL
    if numeric_flag and not semantic_flag:
        return "sql", state

    # for now, if both numeric + semantic → favor structured SQL (safe & accurate)
    if numeric_flag and semantic_flag:
        return "sql", state

    # 2) semantic + state → hybrid (semantic within that state)
    if semantic_flag and state and not numeric_flag:
        return "hybrid", state

    # 3) purely semantic → semantic
    if semantic_flag and not state:
        return "semantic", None

    # 4) mention only state like "cities in Texas" → SQL
    if state and not semantic_flag and not numeric_flag:
        return "sql", state

    # default fallback: treat as SQL
    return "sql", state
