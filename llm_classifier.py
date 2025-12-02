"""
CitySearch AI - LLM Query Classifier
=====================================
Uses GPT to understand user intent and classify queries.
Replaces rule-based pattern matching with intelligent understanding.
"""

import json
import streamlit as st
from config import LLM_MODEL, LLM_TEMPERATURE_CLASSIFICATION, LLM_MAX_TOKENS_CLASSIFICATION


def get_openai_client():
    """Get OpenAI client from secrets."""
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception:
        pass
    return None


def classify_query_with_llm(query: str) -> dict:
    """
    Use LLM to understand user intent and extract query parameters.
    
    Returns structured classification:
    {
        "response_type": str,       # Type of response needed
        "specific_intent": str,     # families, young_professionals, retirement, general
        "mentioned_cities": list,   # City names found in query
        "mentioned_states": list,   # State names found in query
        "metric": str,              # population, median_age, avg_household_size, all
        "sort_direction": str,      # highest, lowest, null
        "limit": int,               # Number of results (e.g., 10 for "top 10")
        "comparison_type": str,     # city_vs_city, state_vs_state, null
        "can_answer_from_database": bool,
        "is_city_related": bool,    # Is this about US cities/states?
        "needs_ai_summary": bool,
        "summary_style": str,       # none, brief, detailed, recommendation, comparison
        "sql_hint": str             # Hint for SQL generation
    }
    """
    
    client = get_openai_client()
    if not client:
        return get_fallback_classification(query)
    
    prompt = f"""
You are a query classifier for a US Cities database application.

DATABASE SCHEMA:
- Table: dbo.cities
- Columns: city, state, state_code, population, median_age, avg_household_size

Analyze this user query and return JSON classification.

User Query: "{query}"

Return ONLY valid JSON (no markdown, no explanation):
{{
    "response_type": "single_city" | "single_state" | "city_list" | "state_list" | "comparison" | "city_profile" | "state_profile" | "aggregate" | "recommendation" | "cluster" | "similar_cities" | "lifestyle" | "general_question",
    
    "specific_intent": "families" | "young_professionals" | "retirement" | "general",
    
    "mentioned_cities": ["list of city names found in query"],
    
    "mentioned_states": ["list of state names found in query"],
    
    "metric": "population" | "median_age" | "avg_household_size" | "all" | null,
    
    "sort_direction": "highest" | "lowest" | null,
    
    "limit": null or number (e.g., 5, 10),
    
    "comparison_type": "city_vs_city" | "state_vs_state" | null,
    
    "can_answer_from_database": true | false,
    
    "is_city_related": true | false,
    
    "needs_ai_summary": true | false,
    
    "summary_style": "none" | "brief" | "detailed" | "recommendation" | "comparison" | "highlights",
    
    "sql_hint": "brief description of what SQL might help, or null"
}}

CLASSIFICATION RULES:

1. RESPONSE TYPES:
   - "single_city": Questions about ONE specific city ("What is the population of Denver?", "Which city has the highest population?")
   - "single_state": Questions about ONE specific state ("What is the population of Texas?")
   - "city_list": Questions expecting multiple cities ("Top 10 cities in Florida", "List all cities in Texas")
   - "comparison": Comparing two entities ("Denver vs Dallas", "Compare Florida and Texas")
   - "city_profile": Detailed info about a city ("Tell me about Miami", "City profile for Denver", "Life in Austin")
   - "state_profile": Detailed info about a state ("Tell me about California")
   - "aggregate": Count/sum/average questions ("How many cities in Texas?", "Total population of Florida")
   - "recommendation": Best/worst questions ("Best city for families", "Best retirement city")
   - "cluster": Clustering questions ("Which cluster is Miami in?")
   - "similar_cities": Similarity questions ("Cities similar to Chicago")
   - "lifestyle": Lifestyle questions ("Life in Dallas", "What's it like living in Miami?")
   - "general_question": Non-city questions ("What is the capital of France?")

2. SPECIFIC INTENT:
   - "families": Questions about family-friendliness, schools, safety, kids
   - "young_professionals": Questions about careers, nightlife, tech jobs, young people
   - "retirement": Questions about seniors, healthcare, cost of living for retirees
   - "general": Everything else

3. NEEDS AI SUMMARY:
   - TRUE for: recommendations, comparisons, profiles, lifestyle, opinion questions
   - FALSE for: simple facts, counts, basic lists

4. SUMMARY STYLE:
   - "none": Simple facts (no AI narrative needed)
   - "brief": 1-2 bullet points of context
   - "detailed": 2-3 sentences describing the city/result
   - "recommendation": Pros/cons + clear recommendation
   - "comparison": Side-by-side analysis with winner
   - "highlights": Key features as bullet points

5. CAN ANSWER FROM DATABASE:
   - TRUE: Questions about US cities/states that our data can answer
   - FALSE: Questions about non-US locations, general knowledge, unrelated topics

6. IS CITY RELATED:
   - TRUE: Any question about US cities, states, population, demographics
   - FALSE: Questions about other countries, general knowledge, unrelated topics

EXAMPLES:

Query: "What is the population of Denver?"
{{"response_type": "single_city", "specific_intent": "general", "mentioned_cities": ["Denver"], "mentioned_states": [], "metric": "population", "sort_direction": null, "limit": null, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": false, "summary_style": "brief", "sql_hint": "SELECT * FROM cities WHERE city = 'Denver'"}}

Query: "Which city has the highest population?"
{{"response_type": "single_city", "specific_intent": "general", "mentioned_cities": [], "mentioned_states": [], "metric": "population", "sort_direction": "highest", "limit": 1, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": false, "summary_style": "brief", "sql_hint": "SELECT TOP 1 * FROM cities ORDER BY population DESC"}}

Query: "Best cities for families in Texas"
{{"response_type": "recommendation", "specific_intent": "families", "mentioned_cities": [], "mentioned_states": ["Texas"], "metric": "all", "sort_direction": "highest", "limit": 10, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "recommendation", "sql_hint": "SELECT * FROM cities WHERE state = 'Texas'"}}

Query: "Compare Miami and Dallas"
{{"response_type": "comparison", "specific_intent": "general", "mentioned_cities": ["Miami", "Dallas"], "mentioned_states": [], "metric": "all", "sort_direction": null, "limit": null, "comparison_type": "city_vs_city", "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "comparison", "sql_hint": "SELECT * FROM cities WHERE city IN ('Miami', 'Dallas')"}}

Query: "Tell me about Austin"
{{"response_type": "city_profile", "specific_intent": "general", "mentioned_cities": ["Austin"], "mentioned_states": [], "metric": "all", "sort_direction": null, "limit": null, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "detailed", "sql_hint": "SELECT * FROM cities WHERE city = 'Austin'"}}

Query: "How many cities are in California?"
{{"response_type": "aggregate", "specific_intent": "general", "mentioned_cities": [], "mentioned_states": ["California"], "metric": "population", "sort_direction": null, "limit": null, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": false, "summary_style": "none", "sql_hint": "SELECT COUNT(*) FROM cities WHERE state = 'California'"}}

Query: "Top 10 largest cities in Florida"
{{"response_type": "city_list", "specific_intent": "general", "mentioned_cities": [], "mentioned_states": ["Florida"], "metric": "population", "sort_direction": "highest", "limit": 10, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": false, "summary_style": "brief", "sql_hint": "SELECT TOP 10 * FROM cities WHERE state = 'Florida' ORDER BY population DESC"}}

Query: "Cities similar to Chicago"
{{"response_type": "similar_cities", "specific_intent": "general", "mentioned_cities": ["Chicago"], "mentioned_states": [], "metric": "all", "sort_direction": null, "limit": 10, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "brief", "sql_hint": null}}

Query: "Life in Dallas"
{{"response_type": "lifestyle", "specific_intent": "general", "mentioned_cities": ["Dallas"], "mentioned_states": [], "metric": "all", "sort_direction": null, "limit": null, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "detailed", "sql_hint": "SELECT * FROM cities WHERE city = 'Dallas'"}}

Query: "What is the capital of France?"
{{"response_type": "general_question", "specific_intent": "general", "mentioned_cities": [], "mentioned_states": [], "metric": null, "sort_direction": null, "limit": null, "comparison_type": null, "can_answer_from_database": false, "is_city_related": false, "needs_ai_summary": false, "summary_style": "none", "sql_hint": null}}

Query: "Best city for young professionals"
{{"response_type": "recommendation", "specific_intent": "young_professionals", "mentioned_cities": [], "mentioned_states": [], "metric": "all", "sort_direction": "highest", "limit": 10, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "recommendation", "sql_hint": null}}

Query: "Best retirement cities"
{{"response_type": "recommendation", "specific_intent": "retirement", "mentioned_cities": [], "mentioned_states": [], "metric": "all", "sort_direction": "highest", "limit": 10, "comparison_type": null, "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "recommendation", "sql_hint": null}}

Query: "Which is better Florida or Texas for families?"
{{"response_type": "comparison", "specific_intent": "families", "mentioned_cities": [], "mentioned_states": ["Florida", "Texas"], "metric": "all", "sort_direction": null, "limit": null, "comparison_type": "state_vs_state", "can_answer_from_database": true, "is_city_related": true, "needs_ai_summary": true, "summary_style": "comparison", "sql_hint": null}}

Now classify the query. Return ONLY the JSON object.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE_CLASSIFICATION,
            max_tokens=LLM_MAX_TOKENS_CLASSIFICATION
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up response (remove markdown if present)
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        classification = json.loads(result_text)
        
        # Ensure all required fields exist
        return ensure_classification_fields(classification)
        
    except Exception as e:
        # Fallback to basic classification
        return get_fallback_classification(query, error=str(e))


def ensure_classification_fields(classification: dict) -> dict:
    """Ensure all required fields exist in classification."""
    
    defaults = {
        "response_type": "city_list",
        "specific_intent": "general",
        "mentioned_cities": [],
        "mentioned_states": [],
        "metric": None,
        "sort_direction": None,
        "limit": None,
        "comparison_type": None,
        "can_answer_from_database": True,
        "is_city_related": True,
        "needs_ai_summary": False,
        "summary_style": "none",
        "sql_hint": None
    }
    
    for key, default_value in defaults.items():
        if key not in classification:
            classification[key] = default_value
    
    return classification


def get_fallback_classification(query: str, error: str = None) -> dict:
    """
    Fallback classification when LLM is unavailable.
    Uses simple pattern matching.
    """
    
    q_lower = query.lower()
    
    # Default classification
    classification = {
        "response_type": "city_list",
        "specific_intent": "general",
        "mentioned_cities": [],
        "mentioned_states": [],
        "metric": None,
        "sort_direction": None,
        "limit": None,
        "comparison_type": None,
        "can_answer_from_database": True,
        "is_city_related": True,
        "needs_ai_summary": False,
        "summary_style": "none",
        "sql_hint": None,
        "fallback": True,
        "error": error
    }
    
    # Simple pattern detection
    if "compare" in q_lower or " vs " in q_lower or " versus " in q_lower:
        classification["response_type"] = "comparison"
        classification["needs_ai_summary"] = True
        classification["summary_style"] = "comparison"
    
    elif "best" in q_lower or "recommend" in q_lower:
        classification["response_type"] = "recommendation"
        classification["needs_ai_summary"] = True
        classification["summary_style"] = "recommendation"
        
        if "famil" in q_lower:
            classification["specific_intent"] = "families"
        elif "young" in q_lower or "professional" in q_lower:
            classification["specific_intent"] = "young_professionals"
        elif "retire" in q_lower:
            classification["specific_intent"] = "retirement"
    
    elif "profile" in q_lower or "tell me about" in q_lower or "life in" in q_lower:
        classification["response_type"] = "city_profile"
        classification["needs_ai_summary"] = True
        classification["summary_style"] = "detailed"
    
    elif "how many" in q_lower or "count" in q_lower or "total" in q_lower:
        classification["response_type"] = "aggregate"
        classification["summary_style"] = "none"
    
    elif "top " in q_lower or "largest" in q_lower or "smallest" in q_lower:
        classification["response_type"] = "city_list"
        
        # Try to extract limit
        import re
        limit_match = re.search(r'top\s+(\d+)', q_lower)
        if limit_match:
            classification["limit"] = int(limit_match.group(1))
        
        if "largest" in q_lower or "highest" in q_lower or "most" in q_lower:
            classification["sort_direction"] = "highest"
        elif "smallest" in q_lower or "lowest" in q_lower or "least" in q_lower:
            classification["sort_direction"] = "lowest"
    
    elif "similar" in q_lower:
        classification["response_type"] = "similar_cities"
        classification["needs_ai_summary"] = True
    
    elif "cluster" in q_lower:
        classification["response_type"] = "cluster"
        classification["needs_ai_summary"] = True
    
    elif "population" in q_lower or "median age" in q_lower or "household" in q_lower:
        classification["response_type"] = "single_city"
        
        if "population" in q_lower:
            classification["metric"] = "population"
        elif "age" in q_lower:
            classification["metric"] = "median_age"
        elif "household" in q_lower:
            classification["metric"] = "avg_household_size"
    
    return classification
