"""
LLM-powered query classification.
Understands user intent using AI instead of rule-based patterns.
"""

import json
import streamlit as st
from openai import OpenAI


def get_openai_client():
    """Get cached OpenAI client."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception:
        pass
    return None


def classify_query(query: str) -> dict:
    """
    Use LLM to understand user intent and extract query parameters.
    Returns structured JSON for handling by the app.
    """
    
    client = get_openai_client()
    if not client:
        return {"response_type": "general", "can_answer_from_database": True, "error": "No API key"}
    
    prompt = f"""
You are a query classifier for a US Cities database application.

DATABASE SCHEMA:
- Table: cities
- Columns: city, state, state_code, population, median_age, avg_household_size

Analyze this user query and return JSON classification.

User Query: "{query}"

Return ONLY valid JSON (no markdown, no explanation):
{{
    "response_type": "single_city" | "single_state" | "city_list" | "state_list" | "comparison" | "city_profile" | "state_profile" | "aggregate" | "general_question",
    
    "metric": "population" | "median_age" | "avg_household_size" | "all" | null,
    
    "sort_direction": "highest" | "lowest" | null,
    
    "limit": null or number (e.g., 5, 10),
    
    "mentioned_cities": ["city names found in query"],
    
    "mentioned_states": ["state names found in query"],
    
    "comparison_type": "city_vs_city" | "state_vs_state" | null,
    
    "question_type": "factual" | "opinion" | "recommendation" | "explanation",
    
    "can_answer_from_database": true | false,
    
    "needs_ai_insight": true | false
}}

CLASSIFICATION RULES:

1. "Which city has the highest/lowest X?" → response_type: "single_city"
2. "Top 10 cities by X" or "List cities..." → response_type: "city_list"
3. "Compare A and B" or "A vs B" → response_type: "comparison"
4. "Tell me about X" or "City profile for X" or "Life in X" → response_type: "city_profile"
5. "How many cities in X?" or "Total population of X" → response_type: "aggregate"
6. "What is the population of [STATE]?" → response_type: "single_state"
7. Questions about non-US-city topics → can_answer_from_database: false

EXAMPLES:

Query: "Which city has the highest population?"
{{"response_type": "single_city", "metric": "population", "sort_direction": "highest", "limit": 1, "mentioned_cities": [], "mentioned_states": [], "comparison_type": null, "question_type": "factual", "can_answer_from_database": true, "needs_ai_insight": false}}

Query: "Is Miami good for young professionals?"
{{"response_type": "city_profile", "metric": "all", "sort_direction": null, "limit": null, "mentioned_cities": ["Miami"], "mentioned_states": [], "comparison_type": null, "question_type": "opinion", "can_answer_from_database": true, "needs_ai_insight": true}}

Query: "Top 5 largest cities in Texas"
{{"response_type": "city_list", "metric": "population", "sort_direction": "highest", "limit": 5, "mentioned_cities": [], "mentioned_states": ["Texas"], "comparison_type": null, "question_type": "factual", "can_answer_from_database": true, "needs_ai_insight": false}}

Query: "Compare Dallas and Houston"
{{"response_type": "comparison", "metric": "all", "sort_direction": null, "limit": null, "mentioned_cities": ["Dallas", "Houston"], "mentioned_states": [], "comparison_type": "city_vs_city", "question_type": "factual", "can_answer_from_database": true, "needs_ai_insight": true}}

Query: "What is the capital of France?"
{{"response_type": "general_question", "metric": null, "sort_direction": null, "limit": null, "mentioned_cities": [], "mentioned_states": [], "comparison_type": null, "question_type": "factual", "can_answer_from_database": false, "needs_ai_insight": true}}

Now classify the query. Return ONLY the JSON object.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up response (remove markdown if present)
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        return {
            "response_type": "general",
            "can_answer_from_database": True,
            "needs_ai_insight": False,
            "error": str(e)
        }
