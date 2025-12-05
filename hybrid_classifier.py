"""
CitySearch AI - Hybrid Query Classifier
========================================
Tiered classification system for optimal cost/accuracy balance.

Architecture:
    TIER 1: Cache Check (FREE, instant)
    TIER 2: High-Confidence Patterns (FREE, fast)
    TIER 3: LLM Classification ($0.000015, accurate)
    TIER 4: Rule-Based Fallback (FREE, basic)

Cost Analysis:
    - Pattern match: $0 (FREE)
    - LLM (GPT-4o-mini): ~$0.000015 per query
    - 10,000 queries with 70% pattern hit rate: ~$0.05

Usage:
    from hybrid_classifier import classify_query_hybrid
    
    classification = classify_query_hybrid("Population of Denver")
    # Returns: {
    #     "query_type": "single_city",
    #     "cities": ["Denver"],
    #     "metric": "population",
    #     "source": "pattern",  # or "cache", "llm", "rule_based"
    #     ...
    # }
"""

import streamlit as st
import json
import hashlib
import re

# =============================================================================
# CLASSIFICATION CACHE (In-memory, resets on app restart)
# =============================================================================
# In production, use Redis or database for persistence
_classification_cache = {}
def clear_classification_cache():
    """Clear the cache - call this when patterns are updated."""
    global _classification_cache
    _classification_cache = {}
    return "Cache cleared"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def classify_query_hybrid(query: str) -> dict:
    """
    Classify a query using tiered approach for optimal cost/accuracy.
    
    Tiers:
        1. Cache - instant, free
        2. Pattern - fast, free, high-confidence only
        3. LLM - accurate, costs ~$0.000015
        4. Rule-based - fallback
    
    Args:
        query: User's natural language query
        
    Returns:
        Classification dict with query_type, metric, cities, states, etc.
    """
    q_lower = query.lower().strip()
    
    # DEBUG: Print query to logs (remove in production)
    print(f"Classifying: {query}")
    print(f"Query lower: {q_lower}")
    
    
    # =========================================================================
    # TIER 1: Check Cache (FREE, instant)
    # =========================================================================
    cache_key = hashlib.md5(q_lower.encode()).hexdigest()
    if cache_key in _classification_cache:
        cached = _classification_cache[cache_key].copy()
        cached["source"] = "cache"
        return cached
    
    # =========================================================================
    # TIER 2: High-Confidence Patterns (FREE, fast)
    # =========================================================================
    pattern_result = _check_high_confidence_patterns(q_lower, query)
    if pattern_result:
        pattern_result["source"] = "pattern"
        _classification_cache[cache_key] = pattern_result
        return pattern_result
    
    # =========================================================================
    # TIER 3: LLM Classification (Accurate, ~$0.000015)
    # =========================================================================
    llm_result = _llm_classify(query)
    if llm_result and llm_result.get("success"):
        llm_result["source"] = "llm"
        _classification_cache[cache_key] = llm_result
        return llm_result
    
    # =========================================================================
    # TIER 4: Rule-Based Fallback (FREE, basic)
    # =========================================================================
    fallback = _rule_based_fallback(query)
    fallback["source"] = "rule_based"
    return fallback


# =============================================================================
# TIER 2: PATTERN MATCHING
# =============================================================================

def _check_high_confidence_patterns(q: str, original_query: str) -> dict:
    """
    Check for simple, unambiguous patterns.
    Only return if we're 100% confident.
    
    Patterns covered:
        - "how many cities in [STATE]"
        - "compare [CITY] and [CITY]"
        - "life in [CITY]"
        - "best city for families/retirement/young professionals"
        - "population of [CITY/STATE]"
    """
    
    # -----------------------------------------------------------------
    # Pattern: "how many cities in [STATE]"
    # -----------------------------------------------------------------
    match = re.match(r"how many cities (?:are )?(?:in |are in )(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        return _build_result("aggregate", states=[state.title()], metric="count")
    
    # -----------------------------------------------------------------
    # Pattern: "compare [CITY/STATE] and [CITY/STATE]"
    # -----------------------------------------------------------------
    match = re.match(r"compare (.+?) (?:and|vs|versus|or) (.+?)(?:\?|$)", q)
    if match:
        entity1 = match.group(1).strip().rstrip("?.,!")
        entity2 = match.group(2).strip().rstrip("?.,!")
        
        # Check if states
        if _is_state(entity1) and _is_state(entity2):
            return _build_result("comparison", 
                                 states=[entity1.title(), entity2.title()],
                                 comparison_type="state_vs_state")
        else:
            return _build_result("comparison",
                                 cities=[entity1.title(), entity2.title()],
                                 comparison_type="city_vs_city")
    
    # -----------------------------------------------------------------
    # Pattern: "life in [CITY]" / "living in [CITY]"
    # -----------------------------------------------------------------
    match = re.match(r"(?:life|living|lifestyle) in (.+?)(?:\?|$)", q)
    if match:
        city = match.group(1).strip().rstrip("?.,!")
        return _build_result("lifestyle", cities=[city.title()])
    
    # -----------------------------------------------------------------
    # Pattern: "population of [CITY/STATE]"
    # -----------------------------------------------------------------
    match = re.search(r"(?:what is the |what's the )?population (?:of |in )(.+?)(?:\?|$)", q)
    if match:
        name = match.group(1).strip().rstrip("?.,!")
        if _is_state(name):
            return _build_result("single_state", states=[name.title()], metric="population")
        else:
            return _build_result("single_city", cities=[name.title()], metric="population")
    
    # -----------------------------------------------------------------
    # Pattern: "median age in/of [CITY]"
    # -----------------------------------------------------------------
    match = re.search(r"(?:median )?age (?:of |in )(.+?)(?:\?|$)", q)
    if match:
        name = match.group(1).strip().rstrip("?.,!")
        return _build_result("single_city", cities=[name.title()], metric="median_age")
    
    # -----------------------------------------------------------------
    # Pattern: "best city for families/kids/children"
    # -----------------------------------------------------------------
    if "best" in q:
        if any(w in q for w in ["family", "families", "kids", "children", "kid", "child"]):
            return _build_result("ranking", intent="families")
        
        if any(w in q for w in ["retire", "retirement", "senior", "seniors", "elderly", "retiree"]):
            return _build_result("ranking", intent="retirement")
        
        if any(w in q for w in ["young professional", "young professionals", "career", "millennials"]):
            return _build_result("ranking", intent="young_professionals")
    
    # -----------------------------------------------------------------
    # Pattern: "top N [metric] cities"
    # -----------------------------------------------------------------
    match = re.search(r"top (\d+)\s+(?:most\s+)?(?:populated|largest|biggest)\s+cit", q)
    if match:
        limit = int(match.group(1))
        return _build_result("superlative", metric="population", direction="highest", limit=limit)
    
    match = re.search(r"top (\d+)\s+(?:smallest|least populated)\s+cit", q)
    if match:
        limit = int(match.group(1))
        return _build_result("superlative", metric="population", direction="lowest", limit=limit)
    
    # -----------------------------------------------------------------
    # Pattern: "which city has highest/lowest [metric]"
    # -----------------------------------------------------------------
    match = re.search(r"(?:which|what) city (?:has|have) (?:the )?(?:highest|largest|most|biggest) (.+?)(?:\?|$)", q)
    if match:
        metric = _extract_metric(match.group(1))
        return _build_result("superlative", metric=metric, direction="highest", limit=1)
    
    match = re.search(r"(?:which|what) city (?:has|have) (?:the )?(?:lowest|smallest|least) (.+?)(?:\?|$)", q)
    if match:
        metric = _extract_metric(match.group(1))
        return _build_result("superlative", metric=metric, direction="lowest", limit=1)
    
    # -----------------------------------------------------------------
    # Pattern: "cities in [STATE]"
    # -----------------------------------------------------------------
    match = re.search(r"(?:cities|city) in (.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("city_list", states=[state.title()])

    # -----------------------------------------------------------------
    # Pattern: "cities with population > N" (handles >, >=, greater than, more than, etc.)
    # -----------------------------------------------------------------
    # Pattern 1: "cities with population > 1000000" or "population > 1000000"
    match = re.search(r"population\s*(?:>|>=)\s*(\d[\d,]*)", q)
    if match:
        threshold = int(match.group(1).replace(",", ""))
        return _build_result("filter", metric="population", filter_op="gt", filter_value=threshold)
    
    # Pattern 2: "cities with population greater than 1000000"
    match = re.search(r"population\s+(?:greater than|more than|over|above)\s+(\d[\d,]*)", q)
    if match:
        threshold = int(match.group(1).replace(",", ""))
        return _build_result("filter", metric="population", filter_op="gt", filter_value=threshold)
    
    # Pattern 3: "cities with population < 100000"
    match = re.search(r"population\s*(?:<|<=)\s*(\d[\d,]*)", q)
    if match:
        threshold = int(match.group(1).replace(",", ""))
        return _build_result("filter", metric="population", filter_op="lt", filter_value=threshold)
    
    # Pattern 4: "cities with population less than 100000"
    match = re.search(r"population\s+(?:less than|under|below)\s+(\d[\d,]*)", q)
    if match:
        threshold = int(match.group(1).replace(",", ""))
        return _build_result("filter", metric="population", filter_op="lt", filter_value=threshold)
    # -----------------------------------------------------------------
    # Pattern: "population > N" without "cities with"
    # -----------------------------------------------------------------
    match = re.search(r"population\s*(?:>|greater than|more than|over|above)\s*(\d[\d,]*)", q)
    if match:
        threshold = int(match.group(1).replace(",", ""))
        return _build_result("filter", metric="population", filter_op="gt", filter_value=threshold)
    
    match = re.search(r"cities? with population\s*(?:<|less than|under|below)\s*(\d+)", q)
    if match:
        threshold = int(match.group(1))
        return _build_result("filter", metric="population", filter_op="lt", filter_value=threshold)
    
    # -----------------------------------------------------------------
    # Pattern: "cities similar to [CITY]" or "cities like [CITY]"
    # -----------------------------------------------------------------
    match = re.search(r"cities?\s+(?:similar to|like)\s+(.+?)(?:\?|$)", q)
    if match:
        city = match.group(1).strip().rstrip("?.,!")
        return _build_result("similar_cities", cities=[city.title()])
    # No high-confidence pattern matched
    return None


def _build_result(query_type: str, **kwargs) -> dict:
    """Build a standardized classification result."""
    result = {
        "success": True,
        "query_type": query_type,
        "original_mode": _map_query_type_to_mode(query_type, kwargs.get("intent")),
        "cities": kwargs.get("cities", []),
        "states": kwargs.get("states", []),
        "metric": kwargs.get("metric"),
        "direction": kwargs.get("direction"),
        "limit": kwargs.get("limit", 10),
        "intent": kwargs.get("intent", "general"),
        "comparison_type": kwargs.get("comparison_type"),
        "is_city_related": True,
        "response_type": query_type,
    }
    return result


def _extract_metric(text: str) -> str:
    """Extract metric name from text."""
    text = text.lower().strip()
    
    if any(w in text for w in ["population", "people", "resident"]):
        return "population"
    if any(w in text for w in ["age", "old", "young"]):
        return "median_age"
    if any(w in text for w in ["household", "family size"]):
        return "avg_household_size"
    
    return "population"


def _is_state(name: str) -> bool:
    """Check if a name is a US state."""
    us_states = {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming"
    }
    return name.lower().strip() in us_states


# =============================================================================
# TIER 3: LLM CLASSIFICATION
# =============================================================================

def _llm_classify(query: str) -> dict:
    """
    Use LLM to understand query intent accurately.
    
    Cost: ~$0.000015 per query (GPT-4o-mini)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a query classifier for a US cities database.

The database has columns: city, state, population, median_age, avg_household_size, state_code

Analyze the query and return JSON:
{
    "query_type": "single_city" | "single_state" | "city_list" | "superlative" | "comparison" | "ranking" | "aggregate" | "lifestyle",
    "cities": ["city names mentioned"],
    "states": ["state names mentioned"],
    "metric": "population" | "median_age" | "avg_household_size" | null,
    "direction": "highest" | "lowest" | null,
    "limit": number or null,
    "intent": "families" | "young_professionals" | "retirement" | "general",
    "comparison_type": "city_vs_city" | "state_vs_state" | null,
    "is_city_related": true | false
}

Classification rules:
- "Population of Denver" → single_city, metric: population
- "Population of Texas" → single_state, metric: population  
- "Which city has highest population" → superlative, metric: population, direction: highest
- "List top 5 most populated cities" → superlative, metric: population, direction: highest, limit: 5
- "Best cities for families" → ranking, intent: families
- "Best city for adults" → ranking, intent: young_professionals
- "Compare Miami and Austin" → comparison, comparison_type: city_vs_city
- "How many cities in Texas" → aggregate
- "Life in Dallas" → lifestyle
- "Cities with population > 1 million" → city_list"""
            }, {
                "role": "user",
                "content": query
            }],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        result["success"] = True
        result["original_mode"] = _map_query_type_to_mode(
            result.get("query_type", "city_list"),
            result.get("intent")
        )
        
        return result
        
    except Exception as e:
        print(f"LLM classification error: {e}")
        return None


# =============================================================================
# TIER 4: RULE-BASED FALLBACK
# =============================================================================

def _rule_based_fallback(query: str) -> dict:
    """
    Basic rule-based classification as final fallback.
    """
    try:
        from core.intent_classifier import classify_query_intent
        mode, state = classify_query_intent(query)
        
        return {
            "success": True,
            "query_type": mode if mode else "city_list",
            "original_mode": mode if mode else "sql",
            "states": [state] if state else [],
            "cities": [],
            "metric": None,
            "direction": None,
            "limit": 10,
            "intent": "general",
            "is_city_related": True,
            "response_type": "city_list"
        }
    except Exception:
        return {
            "success": True,
            "query_type": "city_list",
            "original_mode": "sql",
            "states": [],
            "cities": [],
            "metric": None,
            "direction": None,
            "limit": 10,
            "intent": "general",
            "is_city_related": True,
            "response_type": "city_list"
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _map_query_type_to_mode(query_type: str, intent: str = None) -> str:
    """Map query_type to internal processing mode."""
    if query_type == "ranking":
        intent_map = {
            "families": "ml_family",
            "young_professionals": "ml_young",
            "retirement": "ml_retirement"
        }
        return intent_map.get(intent, "sql")
    
    mapping = {
        "single_city": "single_city",
        "single_state": "single_state",
        "city_list": "sql",
        "superlative": "superlative",
        "comparison": "ml_compare_cities",
        "aggregate": "sql",
        "lifestyle": "lifestyle",
        "filter": "filter",
        "similar_cities": "similar_cities",
    }
    return mapping.get(query_type, "sql")


def is_city_related_query(query: str) -> bool:
    """
    Check if a query is related to US cities/states.
    Used to filter out completely off-topic queries.
    """
    city_keywords = [
        "city", "cities", "town", "population", "state", "states",
        "live", "living", "move", "moving", "best", "top", "largest",
        "smallest", "compare", "vs", "versus", "family", "families",
        "retire", "retirement", "young", "professional", "median age",
        "household", "denver", "austin", "miami", "seattle", "chicago",
        "texas", "california", "florida", "new york"
    ]
    
    q_lower = query.lower()
    return any(kw in q_lower for kw in city_keywords)


def clear_cache():
    """Clear the classification cache. Useful for testing."""
    global _classification_cache
    _classification_cache = {}


def get_cache_stats():
    """Get cache statistics for monitoring."""
    return {
        "size": len(_classification_cache),
        "keys": list(_classification_cache.keys())[:10]  # First 10 keys
    }
