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
    # Pattern: "top N [metric] cities in [STATE]"
    # -----------------------------------------------------------------
    match = re.search(r"top (\d+)\s+(?:most\s+)?(?:populated|largest|biggest)\s+cit(?:y|ies)?\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        limit = int(match.group(1))
        state = match.group(2).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="highest", limit=limit, states=[state.title()])
    
    # Pattern: "top N cities in [STATE]" (without metric - default to population)
    match = re.search(r"top (\d+)\s+cit(?:y|ies)\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        limit = int(match.group(1))
        state = match.group(2).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="highest", limit=limit, states=[state.title()])
    
    # Pattern: "largest city in [STATE]" or "most populated city in [STATE]"
    match = re.search(r"(?:largest|biggest|most populated)\s+cit(?:y|ies)\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="highest", limit=1, states=[state.title()])
    
    # Pattern: "smallest city in [STATE]" or "least populated city in [STATE]"
    match = re.search(r"(?:smallest|least populated)\s+cit(?:y|ies)\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="lowest", limit=1, states=[state.title()])
    
    # Pattern: "highest/lowest [metric] cities" or "[metric] cities"
    match = re.search(r"(?:highest|oldest)\s+(?:median\s+)?age\s+cit", q)
    if match:
        return _build_result("superlative", metric="median_age", direction="highest", limit=10)
    
    match = re.search(r"(?:lowest|youngest)\s+(?:median\s+)?age\s+cit", q)
    if match:
        return _build_result("superlative", metric="median_age", direction="lowest", limit=10)
    
    match = re.search(r"youngest\s+cit", q)
    if match:
        return _build_result("superlative", metric="median_age", direction="lowest", limit=10)
    
    match = re.search(r"oldest\s+cit", q)
    if match:
        return _build_result("superlative", metric="median_age", direction="highest", limit=10)
    
    # Pattern: "top N youngest/oldest cities"
    match = re.search(r"top (\d+)\s+(?:youngest|lowest age)\s+cit", q)
    if match:
        limit = int(match.group(1))
        return _build_result("superlative", metric="median_age", direction="lowest", limit=limit)
    
    match = re.search(r"top (\d+)\s+(?:oldest|highest age)\s+cit", q)
    if match:
        limit = int(match.group(1))
        return _build_result("superlative", metric="median_age", direction="highest", limit=limit)
    
    # Pattern: "largest households" or "biggest household size"
    match = re.search(r"(?:largest|biggest|highest)\s+(?:household|family)\s*(?:size)?", q)
    if match:
        return _build_result("superlative", metric="avg_household_size", direction="highest", limit=10)
    
    match = re.search(r"(?:smallest|lowest)\s+(?:household|family)\s*(?:size)?", q)
    if match:
        return _build_result("superlative", metric="avg_household_size", direction="lowest", limit=10)
    
    # Pattern: "lowest/highest population cities in [STATE]"
    match = re.search(r"(?:lowest|smallest)\s+population\s+cit(?:y|ies)\s+(?:in|of)\s+(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("superlative", metric="population", direction="lowest", limit=10, states=[state.title()])
    
    # Pattern: "top N [metric] cities" without state
    match = re.search(r"top (\d+)\s+(?:most\s+)?(?:populated|largest|biggest)\s+cit", q)
    if match:
        limit = int(match.group(1))
        return _build_result("superlative", metric="population", direction="highest", limit=limit)
    
    # -----------------------------------------------------------------
    # Pattern: "best cities for [INTENT] in [STATE]"
    # -----------------------------------------------------------------
    match = re.search(r"best\s+cit(?:y|ies)\s+for\s+(.+?)\s+in\s+(.+?)(?:\?|$)", q)
    if match:
        intent_text = match.group(1).strip()
        state = match.group(2).strip().rstrip("?.,!")
        intent = _extract_intent(intent_text)
        if _is_state(state) and intent:
            return _build_result("ranking", intent=intent, states=[state.title()])
    
    # Pattern: "top cities for retirement in [STATE]"
    match = re.search(r"top\s+cit(?:y|ies)\s+for\s+(.+?)\s+in\s+(.+?)(?:\?|$)", q)
    if match:
        intent_text = match.group(1).strip()
        state = match.group(2).strip().rstrip("?.,!")
        intent = _extract_intent(intent_text)
        if _is_state(state) and intent:
            return _build_result("ranking", intent=intent, states=[state.title()])
    
    # -----------------------------------------------------------------
    # Pattern: "total population of [STATE]"
    # -----------------------------------------------------------------
    match = re.search(r"total\s+population\s+(?:of|in)\s+(.+?)(?:\?|$)", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("single_state", states=[state.title()], metric="population")
    
    # -----------------------------------------------------------------
    # Pattern: "how many cities does [STATE] have"
    # -----------------------------------------------------------------
    match = re.search(r"how many cities (?:does|do)\s+(.+?)\s+have", q)
    if match:
        state = match.group(1).strip().rstrip("?.,!")
        if _is_state(state):
            return _build_result("aggregate", states=[state.title()], metric="count")
    
    # -----------------------------------------------------------------
    # Pattern: "cities with population between X and Y"
    # -----------------------------------------------------------------
    match = re.search(r"cities?\s+with\s+population\s+between\s+(\d[\d,k]*)\s+and\s+(\d[\d,k]*)", q)
    if match:
        min_val = _parse_number(match.group(1))
        max_val = _parse_number(match.group(2))
        return _build_result("filter_range", metric="population", filter_min=min_val, filter_max=max_val)
    
    # -----------------------------------------------------------------
    # Pattern: "cities in [STATE] with [METRIC] > N"
    # -----------------------------------------------------------------
    match = re.search(r"cities?\s+in\s+(.+?)\s+with\s+(?:median\s+)?age\s*(?:>|greater than|over|above)\s*(\d+)", q)
    if match:
        state = match.group(1).strip()
        threshold = int(match.group(2))
        if _is_state(state):
            return _build_result("filter", metric="median_age", filter_op="gt", filter_value=threshold, states=[state.title()])
    
    match = re.search(r"cities?\s+in\s+(.+?)\s+with\s+(?:household|family)\s*(?:size)?\s*(?:>|greater than|over|above)\s*(\d+\.?\d*)", q)
    if match:
        state = match.group(1).strip()
        threshold = float(match.group(2))
        if _is_state(state):
            return _build_result("filter", metric="avg_household_size", filter_op="gt", filter_value=threshold, states=[state.title()])
    
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
    # match = re.search(r"population\s*(?:>|>=)\s*(\d[\d,]*)", q)
    # if match:
    #     threshold = int(match.group(1).replace(",", ""))
    #     return _build_result("filter", metric="population", filter_op="gt", filter_value=threshold)
    
    # # Pattern 2: "cities with population greater than 1000000"
    # match = re.search(r"population\s+(?:greater than|more than|over|above)\s+(\d[\d,]*)", q)
    # if match:
    #     threshold = int(match.group(1).replace(",", ""))
    #     return _build_result("filter", metric="population", filter_op="gt", filter_value=threshold)
    
    # # Pattern 3: "cities with population < 100000"
    # match = re.search(r"population\s*(?:<|<=)\s*(\d[\d,]*)", q)
    # if match:
    #     threshold = int(match.group(1).replace(",", ""))
    #     return _build_result("filter", metric="population", filter_op="lt", filter_value=threshold)
    
    # # Pattern 4: "cities with population less than 100000"
    # match = re.search(r"population\s+(?:less than|under|below)\s+(\d[\d,]*)", q)
    # if match:
    #     threshold = int(match.group(1).replace(",", ""))
    #     return _build_result("filter", metric="population", filter_op="lt", filter_value=threshold)
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
        # Filter-specific fields
        "filter_op": kwargs.get("filter_op"),
        "filter_value": kwargs.get("filter_value"),
        "filter_min": kwargs.get("filter_min"),
        "filter_max": kwargs.get("filter_max"),
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

def _extract_intent(text: str) -> str:
    """Extract intent from text like 'families', 'retirement', etc."""
    text = text.lower()
    
    if any(w in text for w in ["family", "families", "kids", "children"]):
        return "families"
    if any(w in text for w in ["retire", "retirement", "senior", "seniors", "retiree"]):
        return "retirement"
    if any(w in text for w in ["young", "professional", "career", "millennial", "remote", "worker", "student"]):
        return "young_professionals"
    
    return None


def _parse_number(text: str) -> int:
    """Parse numbers like '100k', '1,000,000', '500000'."""
    text = text.lower().replace(",", "").strip()
    
    if text.endswith("k"):
        return int(float(text[:-1]) * 1000)
    if text.endswith("m"):
        return int(float(text[:-1]) * 1000000)
    
    return int(text)


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
                "content": """You are a SQL query generator for a US cities database.

DATABASE SCHEMA:
Table: dbo.cities
Columns:
  - city (VARCHAR): City name
  - state (VARCHAR): Full state name (e.g., "California", "Texas")
  - state_code (VARCHAR): 2-letter code (e.g., "CA", "TX")
  - population (INT): City population
  - median_age (FLOAT): Median age of residents
  - avg_household_size (FLOAT): Average household size

TASK: Analyze the user's query and return JSON with:
1. Classification info
2. A valid SQL Server query (use TOP instead of LIMIT)

Return JSON:
{
    "query_type": "filter" | "superlative" | "aggregate" | "group_by" | "comparison" | "pattern_match" | "single_city" | "single_state" | "ranking" | "general_knowledge",
    "sql": "SELECT ... FROM dbo.cities WHERE ...",
    "cities": [],
    "states": [],
    "metric": null,
    "direction": null,
    "limit": null,
    "intent": "general",
    "is_city_related": true,
    "needs_gpt_knowledge": false,
    "explanation": "Brief explanation of what the query does"
}

SQL RULES:
- Use TOP N instead of LIMIT (SQL Server syntax)
- Table name is dbo.cities
- Use LOWER() for case-insensitive string comparisons
- For "between", use: column BETWEEN value1 AND value2
- For "starting with", use: column LIKE 'prefix%'
- For "contains", use: column LIKE '%word%'
- Always include ORDER BY for ranked queries
- For aggregates, use COUNT(*), SUM(), AVG()
- For GROUP BY, include HAVING if filtering groups

EXAMPLES:
Query: "cities with population > 500000"
{
    "query_type": "filter",
    "sql": "SELECT * FROM dbo.cities WHERE population > 500000 ORDER BY population DESC",
    "is_city_related": true
}

Query: "top 10 largest cities in Florida"
{
    "query_type": "superlative",
    "sql": "SELECT TOP 10 * FROM dbo.cities WHERE LOWER(state) = 'florida' ORDER BY population DESC",
    "states": ["Florida"],
    "is_city_related": true
}

Query: "Show cities sorted by population per household size"
{
    "query_type": "filter",
    "sql": "SELECT *, (population / avg_household_size) as pop_per_household FROM dbo.cities ORDER BY pop_per_household DESC",
    "is_city_related": true,
    "explanation": "Calculated derived metric: population divided by household size"
}

Query: "cities in Texas with median age < 35"
{
    "query_type": "filter",
    "sql": "SELECT * FROM dbo.cities WHERE LOWER(state) = 'texas' AND median_age < 35 ORDER BY median_age ASC",
    "states": ["Texas"],
    "is_city_related": true
}

Query: "average median age in Ohio"
{
    "query_type": "aggregate",
    "sql": "SELECT AVG(median_age) as avg_median_age, COUNT(*) as city_count FROM dbo.cities WHERE LOWER(state) = 'ohio'",
    "states": ["Ohio"],
    "is_city_related": true
}

Query: "city count for each state"
{
    "query_type": "group_by",
    "sql": "SELECT state, COUNT(*) as city_count FROM dbo.cities GROUP BY state ORDER BY city_count DESC",
    "is_city_related": true
}

Query: "cities starting with San"
{
    "query_type": "pattern_match",
    "sql": "SELECT * FROM dbo.cities WHERE city LIKE 'San%' ORDER BY population DESC",
    "is_city_related": true
}

Query: "cities where median age > 40 and population < 100000"
{
    "query_type": "filter",
    "sql": "SELECT * FROM dbo.cities WHERE median_age > 40 AND population < 100000 ORDER BY population DESC",
    "is_city_related": true
}

Query: "which is bigger Miami or Tampa"
{
    "query_type": "comparison",
    "sql": "SELECT * FROM dbo.cities WHERE LOWER(city) IN ('miami', 'tampa') ORDER BY population DESC",
    "cities": ["Miami", "Tampa"],
    "is_city_related": true
}

Query: "states with more than 20 cities"
{
    "query_type": "group_by",
    "sql": "SELECT state, COUNT(*) as city_count FROM dbo.cities GROUP BY state HAVING COUNT(*) > 20 ORDER BY city_count DESC",
    "is_city_related": true
}"""
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
        
        # Validate SQL exists
        if "sql" in result and result["sql"]:
            result["has_sql"] = True
        else:
            result["has_sql"] = False
        
        return result
        
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
        "general_knowledge": "general_knowledge",
        "filter_range": "filter_range",
        "filter": "llm_sql",
        "group_by": "llm_sql",
        "pattern_match": "llm_sql",
        "aggregate": "llm_sql",
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
        # Descriptive
        "big", "small", "expensive", "cheap", "affordable", "safe", "dangerous",
        # Questions about places
        "why", "what makes", "tell me about", "how is", "history of",
        # Common city names
        "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
        "san antonio", "san diego", "dallas", "austin", "miami", "seattle", "denver",
        "boston", "atlanta", "detroit", "portland", "las vegas",
        # Common state names
        "texas", "california", "florida", "new york", "illinois", "pennsylvania",
        "ohio", "georgia", "michigan", "arizona", "washington", "colorado"
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
