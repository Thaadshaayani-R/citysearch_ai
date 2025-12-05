"""
CitySearch AI - Hybrid Query Classifier
========================================
Cost-efficient query classification:
1. Try rule-based classifier first (FREE)
2. If unsure, fallback to LLM (costs $)
3. Handle city-related vs out-of-scope queries

This approach saves 80-90% on API costs while maintaining accuracy.
"""

import streamlit as st

# Import your existing rule-based classifier
from core.intent_classifier import classify_query_intent

# Import LLM classifier for fallback
from llm_classifier import classify_query_with_llm


# =============================================================================
# CONFIDENCE KEYWORDS
# =============================================================================

# Keywords that indicate HIGH confidence for each mode
CONFIDENCE_PATTERNS = {
    "ml_family": ["family", "families", "kids", "children", "child", "kid", "raising kids", "schools"],
    "ml_young": ["young", "professional", "millennials", "adults", "adult", "young adult", "career", "jobs"],
    "ml_retirement": ["retire", "retirement", "senior", "seniors", "elderly", "retirees", "older"],
    "ml_compare_cities": ["compare", " vs ", "versus", "difference between", "or better"],
    "ml_single_city": ["score for", "predict for", "rating for"],
    "semantic": ["life in", "living in", "like in", "lifestyle", "what's it like", "culture in"],
    "sql": [
        "population", "how many", "count", "total", "average", "avg",
        "top", "largest", "smallest", "biggest", "highest", "lowest",
        "median age", "household size", "list", "show me", "cities in",
        "greater than", "less than", "more than", "under", "over",
        "between", "sort by", "order by"
    ],
}

# Modes that are considered "confident" when matched
CONFIDENT_MODES = [
    "ml_family", "ml_young", "ml_retirement", 
    "ml_compare_cities", "ml_single_city",
    "semantic"
]


# =============================================================================
# MAIN HYBRID CLASSIFIER
# =============================================================================

def classify_query_hybrid(query: str) -> dict:
    """
    Hybrid classification: LLM-first for accuracy, with rule-based fallback.
    
    LLM classification is cheap (~$0.0001 per query) and much more accurate.
    """
    
    # Always try LLM first for better accuracy
    llm_result = _llm_classify(query)
    
    if llm_result and llm_result.get("success"):
        return llm_result
    
    # Fallback to rule-based if LLM fails
    return _rule_based_fallback(query)


def _llm_classify(query: str) -> dict:
    """Use LLM to understand query intent accurately."""
    try:
        import streamlit as st
        from openai import OpenAI
        import json
        
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a query classifier for a US cities database. 
                
The database has these columns: city, state, population, median_age, avg_household_size, state_code

Analyze the user's query and return JSON:
{
    "query_type": "single_city" | "single_state" | "city_list" | "superlative" | "comparison" | "ranking" | "aggregate" | "lifestyle",
    "cities": ["city names if mentioned"],
    "states": ["state names if mentioned"],
    "metric": "population" | "median_age" | "avg_household_size" | null,
    "direction": "highest" | "lowest" | null,
    "limit": number or null,
    "intent": "families" | "young_professionals" | "retirement" | "general",
    "is_city_related": true | false,
    "response_type": "single_city" | "single_state" | "city_list" | "superlative" | "comparison" | "ranking" | "aggregate" | "lifestyle"
}

Examples:
- "List top 5 most populated cities" → query_type: "superlative", metric: "population", direction: "highest", limit: 5
- "What is the population of Denver?" → query_type: "single_city", cities: ["Denver"], metric: "population"
- "Which city has the highest median age?" → query_type: "superlative", metric: "median_age", direction: "highest", limit: 1
- "Best cities for families" → query_type: "ranking", intent: "families"
- "Compare Miami and Austin" → query_type: "comparison", cities: ["Miami", "Austin"]
- "How many cities are in Texas?" → query_type: "aggregate", states: ["Texas"]
- "Population of California" → query_type: "single_state", states: ["California"], metric: "population"
"""
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
        result["source"] = "llm"
        result["original_mode"] = _map_query_type_to_mode(result.get("query_type", "city_list"))
        
        return result
        
    except Exception as e:
        print(f"LLM classification error: {e}")
        return None


def _map_query_type_to_mode(query_type: str) -> str:
    """Map LLM query_type to internal mode."""
    mapping = {
        "single_city": "single_city",
        "single_state": "single_state",
        "city_list": "sql",
        "superlative": "superlative",
        "comparison": "ml_compare_cities",
        "ranking": "ranking",
        "aggregate": "sql",
        "lifestyle": "lifestyle",
    }
    return mapping.get(query_type, "sql")


def _rule_based_fallback(query: str) -> dict:
    """Fallback to rule-based classification if LLM fails."""
    try:
        from core.intent_classifier import classify_query_intent
        mode, state = classify_query_intent(query)
        
        return {
            "success": True,
            "source": "rule_based",
            "original_mode": mode,
            "query_type": mode,
            "states": [state] if state else [],
            "cities": [],
            "metric": None,
            "direction": None,
            "limit": 10,
            "intent": "general",
            "is_city_related": True,
            "response_type": "city_list"
        }
    except Exception as e:
        return {
            "success": True,
            "source": "fallback",
            "original_mode": "sql",
            "query_type": "city_list",
            "is_city_related": True,
            "response_type": "city_list"
        }


def is_city_related_query(query: str) -> bool:
    """Check if query is about US cities."""
    city_keywords = [
        "city", "cities", "town", "population", "state", "states",
        "live", "living", "move", "moving", "best", "top", "largest",
        "smallest", "compare", "vs", "versus", "family", "families",
        "retire", "retirement", "young", "professional"
    ]
    
    us_states = [
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
    ]
    
    q_lower = query.lower()
    
    if any(kw in q_lower for kw in city_keywords):
        return True
    if any(state in q_lower for state in us_states):
        return True
    
    return False

# =============================================================================
# CONFIDENCE CHECKER
# =============================================================================

def _check_confidence(query: str, mode: str) -> str:
    """
    Determine confidence level of rule-based classification.
    
    Returns: "high", "medium", or "low"
    """
    q = query.lower()
    
    # IMPORTANT: "best" queries should be ML ranking, not semantic/sql
    # If rule-based didn't detect ML mode for "best" query, use LLM
    if "best" in q and mode not in ["ml_family", "ml_young", "ml_retirement"]:
        return "low"  # Force LLM fallback
    
    # Check if query matches confidence patterns for detected mode
    if mode in CONFIDENCE_PATTERNS:
        patterns = CONFIDENCE_PATTERNS[mode]
        if any(p in q for p in patterns):
            return "high"

# =============================================================================
# BUILD RULE-BASED RESULT
# =============================================================================

def _build_rule_based_result(query: str, mode: str, state_filter: str, confidence: str) -> dict:
    """
    Build standardized result from rule-based classification.
    """
    q = query.lower()
    
    # Map your mode names to response_type
    mode_mapping = {
        "sql": "sql_query",
        "semantic": "semantic_search",
        "hybrid": "hybrid_search",
        "ml_family": "recommendation",
        "ml_young": "recommendation",
        "ml_retirement": "recommendation",
        "ml_compare_cities": "comparison",
        "ml_single_city": "single_city_score",
    }
    
    response_type = mode_mapping.get(mode, mode)
    
    # Detect metric from query
    metric = _extract_metric(q)
    
    # Detect specific intent for recommendations
    specific_intent = _extract_specific_intent(mode)
    
    return {
        "response_type": response_type,
        "original_mode": mode,  # Keep original mode for routing
        "state_filter": state_filter,
        "mentioned_cities": [],  # Rule-based doesn't extract cities well
        "mentioned_states": [state_filter] if state_filter else [],
        "metric": metric,
        "specific_intent": specific_intent,
        "is_city_related": True,  # Rule-based assumes city queries
        "can_answer_from_db": True,
        "use_gpt_knowledge": False,
        "source": "rule_based",
        "confidence": confidence,
    }


def _extract_metric(query: str) -> str:
    """Extract metric from query."""
    q = query.lower()
    
    if "population" in q:
        return "population"
    elif "median age" in q or "age" in q:
        return "median_age"
    elif "household" in q:
        return "avg_household_size"
    else:
        return "all"


def _extract_specific_intent(mode: str) -> str:
    """Extract specific intent from mode."""
    if mode == "ml_family":
        return "families"
    elif mode == "ml_young":
        return "young_professionals"
    elif mode == "ml_retirement":
        return "retirement"
    else:
        return "general"


# =============================================================================
# LLM FALLBACK
# =============================================================================

def _llm_fallback(query: str) -> dict:
    """
    Fallback to LLM classifier when rule-based is unsure.
    """
    # Get LLM classification
    llm_result = classify_query_with_llm(query)
    
    # Add source info
    llm_result["source"] = "llm"
    llm_result["confidence"] = "high"  # LLM is generally confident
    
    # Check if it's a city-related query that we can't answer from DB
    if llm_result.get("is_city_related", True) and not llm_result.get("can_answer_from_db", True):
        llm_result["use_gpt_knowledge"] = True
    else:
        llm_result["use_gpt_knowledge"] = False
    
    return llm_result


# =============================================================================
# HELPER: CHECK IF QUERY IS CITY-RELATED
# =============================================================================

def is_city_related_query(query: str) -> bool:
    """
    Check if query is related to cities/states/USA.
    Used to determine if we should use GPT knowledge or show out-of-scope.
    """
    q = query.lower()
    
    # City-related keywords
    city_keywords = [
        "city", "cities", "town", "towns", "metro", "metropolitan",
        "state", "states", "usa", "america", "american", "us",
        "population", "median age", "household", "living", "life in",
        "move to", "relocate", "best place", "where to live",
        "county", "region", "area"
    ]
    
    # State names (check if any US state is mentioned)
    us_states = [
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
    ]
    
    # Check for keywords
    if any(kw in q for kw in city_keywords):
        return True
    
    # Check for state names
    if any(state in q for state in us_states):
        return True
    
    return False


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "population of Dallas",  # Should be SQL, high confidence
        "best cities for families in Texas",  # Should be ml_family, high
        "life in Miami",  # Should be semantic, high
        "compare Austin and Denver",  # Should be compare, high
        "what's the weather like",  # Should fallback to LLM, low confidence
        "tell me about pizza",  # Not city-related, out of scope
    ]
    
    for q in test_queries:
        result = classify_query_hybrid(q)
        print(f"\nQuery: {q}")
        print(f"  Mode: {result.get('response_type')} | Source: {result.get('source')} | Confidence: {result.get('confidence')}")
