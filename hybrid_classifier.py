import streamlit as st
import json
import hashlib

# Simple in-memory cache (use Redis in production)
_classification_cache = {}

def classify_query_hybrid(query: str) -> dict:
    """
    Tiered classification:
    1. Cache check (FREE)
    2. High-confidence patterns (FREE)
    3. LLM fallback ($)
    """
    
    q_lower = query.lower().strip()
    
    # ========================================
    # TIER 1: Check Cache (FREE)
    # ========================================
    cache_key = hashlib.md5(q_lower.encode()).hexdigest()
    if cache_key in _classification_cache:
        cached = _classification_cache[cache_key].copy()
        cached["source"] = "cache"
        return cached
    
    # ========================================
    # TIER 2: High-Confidence Patterns (FREE)
    # ========================================
    pattern_result = _check_high_confidence_patterns(q_lower)
    if pattern_result:
        pattern_result["source"] = "pattern"
        _classification_cache[cache_key] = pattern_result  # Cache it
        return pattern_result
    
    # ========================================
    # TIER 3: LLM Classification ($)
    # ========================================
    llm_result = _llm_classify(query)
    if llm_result and llm_result.get("success"):
        llm_result["source"] = "llm"
        _classification_cache[cache_key] = llm_result  # Cache it
        return llm_result
    
    # ========================================
    # FALLBACK: Rule-based
    # ========================================
    return _rule_based_fallback(query)


def _check_high_confidence_patterns(q: str) -> dict:
    """
    Check for simple, unambiguous patterns.
    Only return if we're 100% confident.
    """
    import re
    
    # Pattern: "how many cities in [STATE]"
    match = re.match(r"how many cities (?:are )?(?:in |are in )(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        return {
            "success": True,
            "query_type": "aggregate",
            "original_mode": "sql",
            "states": [state.title()],
            "cities": [],
            "metric": "count",
            "is_city_related": True,
            "response_type": "aggregate"
        }
    
    # Pattern: "compare [CITY] and [CITY]"
    match = re.match(r"compare (.+?) (?:and|vs|versus) (.+?)(?:\?|$)", q)
    if match:
        city1 = match.group(1).strip().rstrip("?.,!")
        city2 = match.group(2).strip().rstrip("?.,!")
        return {
            "success": True,
            "query_type": "comparison",
            "original_mode": "ml_compare_cities",
            "cities": [city1.title(), city2.title()],
            "states": [],
            "is_city_related": True,
            "response_type": "comparison"
        }
    
    # Pattern: "life in [CITY]" or "living in [CITY]"
    match = re.match(r"(?:life|living) in (.+?)(?:\?|$)", q)
    if match:
        city = match.group(1).strip().rstrip("?.,!")
        return {
            "success": True,
            "query_type": "lifestyle",
            "original_mode": "lifestyle",
            "cities": [city.title()],
            "states": [],
            "is_city_related": True,
            "response_type": "lifestyle"
        }
    
    # Pattern: "best city for families" (exact ML intents)
    if "best" in q:
        if any(w in q for w in ["family", "families", "kids", "children"]):
            return {
                "success": True,
                "query_type": "ranking",
                "original_mode": "ml_family",
                "intent": "families",
                "cities": [],
                "states": [],
                "is_city_related": True,
                "response_type": "ranking"
            }
        if any(w in q for w in ["retire", "retirement", "senior", "seniors"]):
            return {
                "success": True,
                "query_type": "ranking",
                "original_mode": "ml_retirement",
                "intent": "retirement",
                "cities": [],
                "states": [],
                "is_city_related": True,
                "response_type": "ranking"
            }
        if any(w in q for w in ["young professional", "young professionals"]):
            return {
                "success": True,
                "query_type": "ranking",
                "original_mode": "ml_young",
                "intent": "young_professionals",
                "cities": [],
                "states": [],
                "is_city_related": True,
                "response_type": "ranking"
            }
    
    # No high-confidence pattern found
    return None


def _llm_classify(query: str) -> dict:
    """Use LLM to understand query intent accurately."""
    try:
        from openai import OpenAI
        
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
    "is_city_related": true | false
}

Examples:
- "List top 5 most populated cities" → query_type: "superlative", metric: "population", direction: "highest", limit: 5
- "What is the population of Denver?" → query_type: "single_city", cities: ["Denver"], metric: "population"
- "Which city has the highest median age?" → query_type: "superlative", metric: "median_age", direction: "highest", limit: 1
- "Best cities for families" → query_type: "ranking", intent: "families"
- "Compare Miami and Austin" → query_type: "comparison", cities: ["Miami", "Austin"]
- "How many cities are in Texas?" → query_type: "aggregate", states: ["Texas"]
- "Population of California" → query_type: "single_state", states: ["California"], metric: "population"
- "Best city for adults" → query_type: "ranking", intent: "young_professionals"
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
        result["original_mode"] = _map_query_type_to_mode(result.get("query_type", "city_list"), result.get("intent"))
        
        return result
        
    except Exception as e:
        print(f"LLM classification error: {e}")
        return None


def _map_query_type_to_mode(query_type: str, intent: str = None) -> str:
    """Map LLM query_type to internal mode."""
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
    }
    return mapping.get(query_type, "sql")


def _rule_based_fallback(query: str) -> dict:
    """Fallback to rule-based classification if everything else fails."""
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
    except Exception:
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
    
    q_lower = query.lower()
    return any(kw in q_lower for kw in city_keywords)
