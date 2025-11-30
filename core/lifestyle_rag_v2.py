# core/lifestyle_rag_v2.py

import re
import pandas as pd
import streamlit as st
from sqlalchemy import text
from openai import OpenAI
from db_config import get_engine
from difflib import get_close_matches

def fuzzy_match_city_name(input_city: str, cutoff: float = 0.6) -> str:
    """Find closest matching city name from database."""
    if not input_city:
        return None
    
    engine = get_engine()
    
    sql = text("SELECT DISTINCT city FROM dbo.cities")
    
    with engine.connect() as conn:
        result = conn.execute(sql)
        rows = result.fetchall()
    
    city_names = [row[0] for row in rows]
    city_names_lower = [c.lower() for c in city_names]
    
    input_lower = input_city.lower().strip()
    
    # Exact match first
    if input_lower in city_names_lower:
        return city_names[city_names_lower.index(input_lower)]
    
    # Fuzzy match
    matches = get_close_matches(input_lower, city_names_lower, n=1, cutoff=cutoff)
    
    if matches:
        return city_names[city_names_lower.index(matches[0])]
    
    return None
    
# -----------------------------------------
# OpenAI client (cached)
# -----------------------------------------
@st.cache_resource
def get_openai_client():
    """Cached OpenAI client - created once."""
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


# -----------------------------------------
# LIFESTYLE TRIGGERS
# -----------------------------------------
LIFESTYLE_TRIGGERS = [
    "tell me about",
    "what is it like in",
    "what is life like in",
    "life in",
    "living in",
    "live in",
    "how is life in",
    "what's life like in",
]


def _looks_like_lifestyle_query(query: str):
    """Detect general lifestyle intent."""
    q = query.lower()
    return any(p in q for p in LIFESTYLE_TRIGGERS)


# -----------------------------------------
# FAST CITY EXTRACTION (no GPT - regex based)
# -----------------------------------------
def extract_city_fast(query: str):
    """Try to extract city without GPT first - much faster."""
    q = query.lower().strip()
    
    # Common patterns for lifestyle queries
    patterns = [
        r"life in ([a-zA-Z\s]+?)(?:\?|$|,|\.|!)",
        r"living in ([a-zA-Z\s]+?)(?:\?|$|,|\.|!)",
        r"tell me about ([a-zA-Z\s]+?)(?:\?|$|,|\.|!)",
        r"about ([a-zA-Z\s]+?)(?:\?|$|,|\.|!)",
        r"like in ([a-zA-Z\s]+?)(?:\?|$|,|\.|!)",
        r"how is ([a-zA-Z\s]+?)(?:\?|$|,|\.|!)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            city = match.group(1).strip().title()
            # Validate city name length
            if len(city) > 2 and len(city) < 30:
                # Remove common words that aren't cities
                skip_words = ["the", "a", "an", "it", "this", "that"]
                if city.lower() not in skip_words:
                    return city
    
    return None


# -----------------------------------------
# GPT-BASED CITY EXTRACTION (cached, fallback)
# -----------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def extract_city_with_gpt(query: str):
    """
    Uses GPT to extract ONLY the city name from any query.
    Cached for 1 hour to avoid repeated API calls.
    """
    client = get_openai_client()
    if client is None:
        return None

    prompt = f"""
Extract ONLY the city name from this query:

"{query}"

Rules:
- Return ONLY the city name (example: Dallas)
- If no city exists, return empty string
- Do NOT include states, countries, or extra words
- Do NOT add explanations
"""

    try:
        rsp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,  # Limit tokens for speed
        )
        city = rsp.choices[0].message.content.strip()
        return city if city else None

    except Exception:
        return None


# -----------------------------------------
# SMART CITY EXTRACTION (fast first, GPT fallback)
# -----------------------------------------
def extract_city_smart(query: str):
    """Try fast regex extraction first, then fuzzy match, then GPT fallback."""
    # Try fast regex first (instant)
    city = extract_city_fast(query)
    if city:
        # Verify it exists or find closest match
        matched = fuzzy_match_city_name(city, cutoff=0.7)
        if matched:
            return matched
    
    # Try fuzzy matching on query words
    words = query.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_city_name(word, cutoff=0.7)
            if matched:
                return matched
    
    # Fallback to GPT
    city = extract_city_with_gpt(query)
    if city:
        # Verify GPT result with fuzzy matching
        matched = fuzzy_match_city_name(city, cutoff=0.6)
        if matched:
            return matched
    
    return None


# -----------------------------------------
# Database fetch (cached)
# -----------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_city_data_from_db(city_name: str):
    """
    Fetch city data from database.
    Cached for 1 hour to avoid repeated DB queries.
    """
    engine = get_engine()

    sql = text("""
        SELECT TOP 1
            c.city,
            c.state,
            c.population,
            c.median_age,
            c.avg_household_size,
            p.description
        FROM dbo.cities AS c
        LEFT JOIN dbo.city_profiles AS p
            ON c.city = p.city AND c.state = p.state
        WHERE LOWER(c.city) = LOWER(:city)
    """)

    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name})
        rows = result.fetchall()
        cols = result.keys()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=cols)
    return df.iloc[0].to_dict()


# -----------------------------------------
# AI SUMMARY GENERATOR (cached per city)
# -----------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_summary_cached(city_name: str, city_state: str, population, median_age, avg_household_size, description):
    """
    Creates a 2-3 sentence lifestyle summary + bullet points.
    Cached by city data to avoid repeated API calls.
    """
    client = get_openai_client()
    if client is None:
        return "AI lifestyle summary unavailable."

    desc_text = description if description else "No description available."

    prompt = f"""
City: {city_name}, {city_state}
Population: {population}
Median age: {median_age}
Average household size: {avg_household_size}

Lifestyle notes:
{desc_text}

TASK:
Write a friendly 2-3 sentence lifestyle summary describing what it's like to live in this city.
Then add 3 bullet points with key lifestyle highlights.

RULES:
- Use ONLY the information given.
- Do NOT invent crime rates, salaries, or extra facts.
- Keep it friendly and concise.
"""

    try:
        rsp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=300,  # Limit for speed
        )
        return rsp.choices[0].message.content.strip()
    
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"


def generate_ai_summary(user_query: str, city_data: dict):
    """Wrapper that calls cached summary function."""
    return generate_ai_summary_cached(
        city_name=city_data.get("city", ""),
        city_state=city_data.get("state", ""),
        population=city_data.get("population"),
        median_age=city_data.get("median_age"),
        avg_household_size=city_data.get("avg_household_size"),
        description=city_data.get("description"),
    )


# -----------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------
def try_build_lifestyle_card(user_query: str):
    """
    Master handler for lifestyle queries.
    Optimized with caching at every step.
    """

    # Step 0: Check if this looks like a lifestyle query
    if not _looks_like_lifestyle_query(user_query):
        return None

    # Step 1: Extract city using smart extraction (fast regex first, GPT fallback)
    city = extract_city_smart(user_query)
    if not city:
        return None

    # Step 2: Fetch full city data from DB (cached)
    city_data = get_city_data_from_db(city)
    if not city_data:
        return None

    # Step 3: Generate AI summary (cached)
    ai_summary = generate_ai_summary(user_query, city_data)

    # Step 4: Return final card
    return {
        "city": city_data["city"],
        "state": city_data["state"],
        "population": city_data["population"],
        "median_age": city_data["median_age"],
        "avg_household_size": city_data["avg_household_size"],
        "description": city_data.get("description"),
        "ai_summary": ai_summary,
    }
