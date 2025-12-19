"""
CitySearch AI - Hybrid Query Classifier v2
============================================
Tiered classification system with MINIMAL patterns + Smart LLM.

Strategy: "Less Pattern, More Intelligence"
- Only 8 high-confidence patterns (reduced from 30+)
- Detect unsupported data queries FIRST
- Let LLM handle most queries with schema context

Architecture:
    TIER 1: Cache Check (FREE, instant)
    TIER 2: Unsupported Data Detection (FREE, instant) ← NEW!
    TIER 3: Minimal High-Confidence Patterns (FREE, fast) ← REDUCED!
    TIER 4: LLM Classification with Schema RAG ($0.000015, accurate)
    TIER 5: Rule-Based Fallback (FREE, basic)

Cost Analysis:
    - Pattern match: $0 (FREE)
    - LLM (GPT-4o-mini): ~$0.000015 per query
    - 10,000 queries with 30% pattern hit rate: ~$0.10

Author: CitySearch AI
Version: 2.0
"""

import streamlit as st
import json
import hashlib
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data we DON'T have - detect these FIRST to avoid wrong results
UNSUPPORTED_DATA_KEYWORDS = {
    # Cost related
    "cheap": "cost of living",
    "cheapest": "cost of living",
    "expensive": "cost of living",
    "affordable": "cost of living",
    "cost of living": "cost of living",
    "housing cost": "cost of living",
    "rent": "cost of living",
    "housing price": "cost of living",
    
    # Safety related
    "safe": "crime/safety statistics",
    "safest": "crime/safety statistics",
    "dangerous": "crime/safety statistics",
    "crime": "crime/safety statistics",
    "violent": "crime/safety statistics",
    "unsafe": "crime/safety statistics",
    
    # Growth related
    "growing": "population growth rate",
    "fastest growing": "population growth rate",
    "growth rate": "population growth rate",
    "shrinking": "population growth rate",
    "declining": "population growth rate",
    
    # Employment related
    "unemployment": "unemployment rate",
    "unemployed": "unemployment rate",
    "job market": "employment statistics",
    "job opportunities": "employment statistics",
    
    # Income related
    "income": "income/salary data",
    "salary": "income/salary data",
    "wages": "income/salary data",
    "rich": "income/salary data",
    "poor": "income/salary data",
    "wealthy": "income/salary data",
    "median income": "income/salary data",
    
    # Weather related
    "weather": "weather/climate data",
    "climate": "weather/climate data",
    "temperature": "weather/climate data",
    "sunny": "weather/climate data",
    "rainy": "weather/climate data",
    "snow": "weather/climate data",
    "humid": "weather/climate data",
    
    # Education related
    "school": "education statistics",
    "schools": "education statistics",
    "education": "education statistics",
    "university": "education statistics",
    "college": "education statistics",
}

# US States for validation
US_STATES = {
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

# =============================================================================
# CLASSIFICATION CACHE (In-memory, resets on app restart)
# =============================================================================

_classification_cache = {}

def clear_classification_cache():
    """Clear the cache - call this when patterns are updated."""
    global _classification_cache
    _classification_cache = {}
    return "Cache cleared"

def clear_cache():
    """Alias for clear_classification_cache."""
    return clear_classification_cache()

def get_cache_stats():
    """Get cache statistics for monitoring."""
    return {
        "size": len(_classification_cache),
        "keys": list(_classification_cache.keys())[:10]
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def classify_query_hybrid(query: str) -> dict:
    """
    Classify a query using tiered approach for optimal cost/accuracy.
    
    Tiers:
        1. Cache - instant, free
        2. Unsupported Data - detect queries we can't answer
        3. Minimal Patterns - only 8 high-confidence patterns
        4. LLM - accurate, costs ~$0.000015
        5. Rule-based - fallback
    
    Args:
        query: User's natural language query
        
    Returns:
        Classification dict with query_type, metric, cities, states, etc.
    """
    q_lower = query.lower().strip()
    
    # DEBUG: Print query to logs
    print(f"[Classifier] Query: {query}")
    
    # =========================================================================
    # TIER 1: Check Cache (FREE, instant)
    # =========================================================================
    cache_key = hashlib.md5(q_lower.encode()).hexdigest()
    if cache_key in _classification_cache:
        cached = _classification_cache[cache_key].copy()
        cached["source"] = "cache"
        print(f"[Classifier] Cache hit: {cached.get('query_type')}")
        return cached
    
    # =========================================================================
    # TIER 2: Unsupported Data Detection (FREE, instant) ← NEW!
    # =========================================================================
    unsupported = _check_unsupported_data(q_lower)
    if unsupported:
        unsupported["source"] = "data_check"
        _classification_cache[cache_key] = unsupported
        print(f"[Classifier] Unsupported data: {unsupported.get('missing_data')}")
        return unsupported
    
    # =========================================================================
    # TIER 3: Minimal High-Confidence Patterns (FREE, fast) ← REDUCED!
    # =========================================================================
    pattern_result = _check_minimal_patterns(q_lower, query)
    if pattern_result:
        pattern_result["source"] = "pattern"
        _classification_cache[cache_key] = pattern_result
        print(f"[Classifier] Pattern match: {pattern_result.get('query_type')}")
        return pattern_result
    
    # =========================================================================
    # TIER 4: LLM Classification with Schema RAG (Accurate, ~$0.000015)
    # =========================================================================
    llm_result = _llm_classify(query)
    if llm_result and llm_result.get("success"):
        llm_result["source"] = "llm"
        _classification_cache[cache_key] = llm_result
        print(f"[Classifier] LLM: {llm_result.get('query_type')}")
        return llm_result
    
    # =========================================================================
    # TIER 5: Rule-Based Fallback (FREE, basic)
    # =========================================================================
    fallback = _rule_based_fallback(query)
    fallback["source"] = "rule_based"
    print(f"[Classifier] Fallback: {fallback.get('query_type')}")
    return fallback


# =============================================================================
# TIER 2: UNSUPPORTED DATA DETECTION (NEW!)
# =============================================================================

def _check_unsupported_data(q: str) -> dict:
    """
    Check if query asks for data we don't have.
    Returns classification with 'unsupported_data' type if matched.
    """
    for keyword, data_type in UNSUPPORTED_DATA_KEYWORDS.items():
        if keyword in q:
            # Extract state if present
            state = _extract_state(q)
            
            return {
                "success": True,
                "query_type": "unsupported_data",
                "original_mode": "unsupported_data",
                "missing_data": data_type,
                "keyword_matched": keyword,
                "states": [state] if state else [],
                "cities": [],
                "is_city_related": True,
                "response_type": "unsupported_data",
            }
    
    return None


def _extract_state(text: str) -> str:
    """Extract state name from text."""
    text_lower = text.lower()
    
    for state in US_STATES:
        if state in text_lower:
            return state.title()
    
    return None


# =============================================================================
# TIER 3: MINIMAL PATTERN MATCHING (REDUCED!)
# =============================================================================

def _check_minimal_patterns(q: str, original_query: str) -> dict:
    """
    MINIMAL pattern matching - only 100% reliable patterns.
    Let LLM handle everything else.
    
    Only 8 pattern categories kept:
    1. population of [city/state]
    2. how many cities in [state]
    3. compare [X] and [Y]
    4. life in [city] / tell me about [city]
    5. cities similar to [city]
    6. top N largest in [state]
    7. total population of [state]
    8. cities in [state] (basic list)
    """
    
    # -----------------------------------------------------------------
    # Pattern 1: "population of [CITY/STATE]" - Very specific
    # -----------------------------------------------------------------
    match = re.search(r"(?:what(?:'s| is) (?:the )?)?population (?:of |in )(.+?)(?:\?|$)", q)
    if match:
        name = match.group(1).strip().rstrip("?.,!")
        if _is_state(name):
            return _build_result("single_state", states=[name.title()], metric="population")
        else:
            return _build_result("single_city", cities=[name.title()], metric="population")
    
    # -----------------------------------------------------------------
    # Pattern 2: "how many cities in [STATE]" - Aggregate count
    # -----------------------------------------------------------------
    match = re.match(r"how many cities (?:are )?(?:there )?(?:in |does )(.+?)(?:\s+have)?(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("aggregate", states=[state.title()], metric="count")
    
    # -----------------------------------------------------------------
    # Pattern 3: "compare [X] and [Y]" - Explicit comparison
    # -----------------------------------------------------------------
    match = re.match(r"compare (.+?) (?:and|vs\.?|versus|with|or) (.+?)(?:\?|$)", q)
    if match:
        entity1 = match.group(1).strip().rstrip("?.,!")
        entity2 = match.group(2).strip().rstrip("?.,!")
        
        if _is_state(entity1) and _is_state(entity2):
            return _build_result("comparison", 
                                 states=[entity1.title(), entity2.title()],
                                 comparison_type="state_vs_state")
        else:
            return _build_result("comparison",
                                 cities=[entity1.title(), entity2.title()],
                                 comparison_type="city_vs_city")
    
    # -----------------------------------------------------------------
    # Pattern 4: "life in [CITY]" / "living in [CITY]" - Lifestyle (RAG)
    # -----------------------------------------------------------------
    match = re.match(r"(?:life|living|lifestyle) in (.+?)(?:\?|$)", q)
    if match:
        city = match.group(1).strip().rstrip("?.,!")
        if not _is_state(city):
            return _build_result("lifestyle", cities=[city.title()])
    
    # "tell me about [CITY]" - Also lifestyle
    match = re.match(r"tell me about (.+?)(?:\?|$)", q)
    if match:
        name = match.group(1).strip().rstrip("?.,!")
        if not _is_state(name):
            return _build_result("lifestyle", cities=[name.title()])
    
    # "city profile for [CITY]"
    match = re.match(r"city profile (?:for |of )?(.+?)(?:\?|$)", q)
    if match:
        city = match.group(1).strip().rstrip("?.,!")
        return _build_result("lifestyle", cities=[city.title()])
    
    # -----------------------------------------------------------------
    # Pattern 5: "cities similar to [CITY]" - Similar cities (RAG)
    # -----------------------------------------------------------------
    match = re.search(r"(?:cities?|places?) (?:similar to|like) (.+?)(?:\?|$)", q)
    if match:
        city = match.group(1).strip().rstrip("?.,!")
        return _build_result("similar_cities", cities=[city.title()])
    
    # -----------------------------------------------------------------
    # Pattern 6: "top N largest cities in [STATE]" - Superlative
    # -----------------------------------------------------------------
    match = re.search(r"top (\d+)\s+(?:most\s+)?(?:populated|largest|biggest)\s+cit(?:y|ies)?\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        limit = int(match.group(1))
        state = match.group(2).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="highest", 
                               limit=limit, states=[state.title()])
    
    # "largest city in [STATE]"
    match = re.search(r"(?:largest|biggest|most populated)\s+cit(?:y|ies)\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="highest", 
                               limit=1, states=[state.title()])
    
    # "top N cities in [STATE]" (default to population)
    match = re.search(r"top (\d+)\s+cit(?:y|ies)\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        limit = int(match.group(1))
        state = match.group(2).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="highest", 
                               limit=limit, states=[state.title()])
    
    # -----------------------------------------------------------------
    # Pattern 7: "total population of [STATE]" - Aggregate sum
    # -----------------------------------------------------------------
    match = re.search(r"(?:total|sum|combined)\s+population\s+(?:of\s+)?(?:all\s+)?(?:cities\s+)?(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("aggregate", states=[state.title()], metric="total_population")
    
    # -----------------------------------------------------------------
    # Pattern 8: Simple "cities in [STATE]" - Basic list
    # But ONLY if no other qualifiers that need LLM
    # -----------------------------------------------------------------
    # Skip if has qualifiers that need more intelligence
    skip_qualifiers = ["best", "top", "worst", "most", "least", "with", "where", "that", "which"]
    has_qualifier = any(qual in q for qual in skip_qualifiers)
    
    if not has_qualifier:
        match = re.search(r"(?:all )?(?:cities|city) in (.+?)(?:\?|$)", q)
        if match:
            state = match.group(1).strip().rstrip("?.,!")
            if _is_state(state):
                return _build_result("city_list", states=[state.title()])
    
    # -----------------------------------------------------------------
    # NO PATTERN MATCHED - Let LLM handle it
    # -----------------------------------------------------------------
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
        # Filter-specific fields
        "filter_op": kwargs.get("filter_op"),
        "filter_value": kwargs.get("filter_value"),
        "filter_min": kwargs.get("filter_min"),
        "filter_max": kwargs.get("filter_max"),
    }
    return result


def _is_state(name: str) -> bool:
    """Check if a name is a US state."""
    return name.lower().strip() in US_STATES


# =============================================================================
# TIER 4: LLM CLASSIFICATION WITH SCHEMA RAG
# =============================================================================

def _llm_classify(query: str) -> dict:
    """
    Use LLM to understand query intent accurately.
    Includes schema context for better SQL generation.
    
    Cost: ~$0.000015 per query (GPT-4o-mini)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a SQL query classifier for a US cities database.

DATABASE SCHEMA:
Table: dbo.cities
Columns:
  - city (VARCHAR): City name
  - state (VARCHAR): Full state name (e.g., "California", "Texas")
  - state_code (VARCHAR): 2-letter code (e.g., "CA", "TX")
  - population (INT): City population
  - median_age (FLOAT): Median age of residents
  - avg_household_size (FLOAT): Average household size

⚠️ IMPORTANT: We ONLY have these 3 metrics: population, median_age, avg_household_size
We do NOT have: cost of living, crime, safety, growth rate, unemployment, income, weather, schools

TASK: Analyze the user's query and return JSON.

For queries about data we DO have, generate SQL.
For queries needing "best cities" ranking, use population as a reasonable proxy.

Return JSON:
{
    "query_type": "filter" | "superlative" | "aggregate" | "single_best" | "group_by" | "comparison" | "pattern_match" | "single_city" | "single_state" | "ranking" | "city_list" | "general_knowledge" | "out_of_scope",
    "sql": "SELECT ... FROM dbo.cities WHERE ...",
    "cities": [],
    "states": [],
    "metric": null,
    "direction": null,
    "limit": 10,
    "intent": "general" | "families" | "retirement" | "young_professionals",
    "is_city_related": true,
    "explanation": "Brief explanation"
}

SQL RULES:
- Use TOP N instead of LIMIT (SQL Server syntax)
- Table name is dbo.cities
- Use LOWER() for case-insensitive string comparisons
- Always include ORDER BY for ranked queries
- For "best cities" queries, ORDER BY population DESC (as proxy for livability)

EXAMPLES:

Query: "best cities to live in Texas"
{
    "query_type": "ranking",
    "sql": "SELECT TOP 10 * FROM dbo.cities WHERE LOWER(state) = 'texas' ORDER BY population DESC",
    "states": ["Texas"],
    "intent": "general",
    "explanation": "Using population as proxy for livability - larger cities typically have more amenities"
}

Query: "cities with population > 500000"
{
    "query_type": "filter",
    "sql": "SELECT * FROM dbo.cities WHERE population > 500000 ORDER BY population DESC",
    "is_city_related": true
}

Query: "youngest cities in America"
{
    "query_type": "superlative",
    "sql": "SELECT TOP 10 * FROM dbo.cities ORDER BY median_age ASC",
    "metric": "median_age",
    "direction": "lowest"
}

Query: "cities starting with San"
{
    "query_type": "pattern_match",
    "sql": "SELECT * FROM dbo.cities WHERE city LIKE 'San%' ORDER BY population DESC",
    "is_city_related": true
}

Query: "best cities for families"
{
    "query_type": "ranking",
    "sql": "SELECT TOP 10 * FROM dbo.cities WHERE avg_household_size > 2.5 ORDER BY population DESC",
    "intent": "families",
    "explanation": "Using household size > 2.5 as family indicator, sorted by population"
}

Query: "which is best city for families"
{
    "query_type": "single_best",
    "sql": "SELECT TOP 1 * FROM dbo.cities WHERE avg_household_size > 2.5 ORDER BY population DESC",
    "intent": "families",
    "limit": 1,
    "needs_summary": true,
    "explanation": "Single best city for families based on household size and population"
}

Query: "best cities for retirement"
{
    "query_type": "ranking",
    "sql": "SELECT TOP 10 * FROM dbo.cities WHERE median_age > 38 ORDER BY population DESC",
    "intent": "retirement",
    "explanation": "Using higher median age as retirement indicator"
}

Query: "best cities for young professionals"
{
    "query_type": "ranking",
    "sql": "SELECT TOP 10 * FROM dbo.cities WHERE median_age < 35 ORDER BY population DESC",
    "intent": "young_professionals",
    "explanation": "Using lower median age as indicator for young professional cities"
}

Query: "what is the weather in Miami"
{
    "query_type": "out_of_scope",
    "sql": null,
    "is_city_related": false,
    "explanation": "Weather data not available in our database"
}

Query: "who is the president"
{
    "query_type": "out_of_scope",
    "sql": null,
    "is_city_related": false,
    "explanation": "Not a city-related query"
}"""
            }, {
                "role": "user",
                "content": query
            }],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        result["success"] = True
        result["original_mode"] = _map_query_type_to_mode(
            result.get("query_type", "city_list"),
            result.get("intent")
        )
        
        # Ensure required fields exist
        result.setdefault("cities", [])
        result.setdefault("states", [])
        result.setdefault("limit", 10)
        result.setdefault("is_city_related", True)
        
        # Validate SQL exists
        if "sql" in result and result["sql"]:
            result["has_sql"] = True
        else:
            result["has_sql"] = False
        
        return result
        
    except Exception as e:
        print(f"[Classifier] LLM error: {e}")
        return None


# =============================================================================
# TIER 5: RULE-BASED FALLBACK
# =============================================================================

def _rule_based_fallback(query: str) -> dict:
    """
    Basic rule-based classification as final fallback.
    """
    # Check if it's city-related at all
    if not is_city_related_query(query):
        return {
            "success": True,
            "query_type": "out_of_scope",
            "original_mode": "out_of_scope",
            "states": [],
            "cities": [],
            "metric": None,
            "is_city_related": False,
            "response_type": "out_of_scope"
        }
    
    # Try to use legacy classifier
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
        # Ultimate fallback - just return city list
        state = _extract_state(query.lower())
        return {
            "success": True,
            "query_type": "city_list",
            "original_mode": "sql",
            "states": [state] if state else [],
            "cities": [],
            "metric": None,
            "direction": None,
            "limit": 20,
            "intent": "general",
            "is_city_related": True,
            "response_type": "city_list",
            "sql": f"SELECT TOP 20 * FROM dbo.cities{' WHERE LOWER(state) = ' + repr(state.lower()) if state else ''} ORDER BY population DESC"
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
        return intent_map.get(intent, "llm_sql")
    
    mapping = {
        "single_city": "single_city",
        "single_state": "single_state",
        "city_list": "sql",
        "superlative": "superlative",
        "comparison": "ml_compare_cities",
        "aggregate": "aggregate",
        "lifestyle": "lifestyle",
        "filter": "llm_sql",
        "filter_range": "llm_sql",
        "similar_cities": "similar_cities",
        "general_knowledge": "general_knowledge",
        "group_by": "llm_sql",
        "pattern_match": "llm_sql",
        "out_of_scope": "out_of_scope",
        "unsupported_data": "unsupported_data",
        "single_best": "single_best",
    }
    return mapping.get(query_type, "sql")


def is_city_related_query(query: str) -> bool:
    """
    Check if a query is related to US cities/states.
    Returns TRUE for any geography, demographics, or location-related question.
    """
    city_keywords = [
        # Direct mentions
        "city", "cities", "town", "state", "states", "county",
        # Demographics
        "population", "people", "residents", "median age", "household",
        # Actions
        "live", "living", "move", "moving", "relocate", "visit",
        # Comparisons
        "best", "top", "largest", "smallest", "biggest", "compare", "vs", "versus",
        # Categories
        "family", "families", "retire", "retirement", "young", "professional",
        # Questions about places
        "tell me about",
        # Include unsupported keywords - they're still city-related!
        "cheap", "expensive", "affordable", "safe", "safest", "crime",
        "growing", "unemployment", "income", "weather", "schools",
    ]
    
    # Add state names to keywords
    city_keywords.extend(list(US_STATES))
    
    # Common city names
    common_cities = [
        "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
        "san antonio", "san diego", "dallas", "austin", "miami", "seattle", "denver",
        "boston", "atlanta", "detroit", "portland", "las vegas"
    ]
    city_keywords.extend(common_cities)
    
    q_lower = query.lower()
    return any(kw in q_lower for kw in city_keywords)


# =============================================================================
# TESTING (run with: python hybrid_classifier.py)
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        # Should detect unsupported data (TIER 2)
        ("Cheap cities in California", "unsupported_data"),
        ("Safest cities in Florida", "unsupported_data"),
        ("Fast growing cities in USA", "unsupported_data"),
        ("Cities with low unemployment", "unsupported_data"),
        
        # Should use pattern matching (TIER 3)
        ("Population of Denver", "single_city"),
        ("How many cities in Texas", "aggregate"),
        ("Compare Miami and Austin", "comparison"),
        ("Life in Austin", "lifestyle"),
        ("Cities similar to Chicago", "similar_cities"),
        ("Top 10 largest cities in California", "superlative"),
        ("Total population of Florida", "aggregate"),
        ("Cities in Texas", "city_list"),
        
        # Should use LLM (TIER 4)
        ("Best cities to live in Texas", "ranking"),
        ("Cities with population over 1 million", "filter"),
        ("Youngest cities in America", "superlative"),
        ("Cities starting with San", "pattern_match"),
        ("Best cities for families", "ranking"),
        ("Best cities for young professionals", "ranking"),
        
        # Should be out of scope (TIER 5)
        ("Who is the president", "out_of_scope"),
        ("What is machine learning", "out_of_scope"),
    ]
    
    print("=" * 70)
    print("HYBRID CLASSIFIER v2 TEST")
    print("=" * 70)
    
    for query, expected in test_queries:
        result = classify_query_hybrid(query)
        actual = result.get("query_type")
        source = result.get("source")
        status = "✅" if actual == expected else "❌"
        
        print(f"\n{status} Query: {query}")
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual} (via {source})")
        if result.get("missing_data"):
            print(f"   Missing:  {result.get('missing_data')}")
        if result.get("sql"):
            print(f"   SQL:      {result.get('sql')[:60]}...")
    
    print("\n" + "=" * 70)