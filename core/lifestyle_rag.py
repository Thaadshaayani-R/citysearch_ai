import pandas as pd
import streamlit as st
from sqlalchemy import text
from openai import OpenAI
from db_config import get_engine


# -----------------------------------------
# OpenAI client
# -----------------------------------------
def get_openai_client():
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
# GPT-BASED CITY EXTRACTION  (BEST METHOD)
# -----------------------------------------
def extract_city_with_gpt(query: str):
    """
    Uses GPT to extract ONLY the city name from any query.
    Works across all cities, no spaCy required, compatible with Streamlit Cloud.
    """

    client = get_openai_client()
    if client is None:
        return None

    prompt = f"""
Extract ONLY the city name from this query:

"{query}"

Rules:
- Return ONLY the city name (example: Dallas)
- If no city exists, return an empty string
- Do NOT include states, countries, or extra words
- Do NOT add explanations
"""

    try:
        rsp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        city = rsp.choices[0].message.content.strip()
        return city if city else None

    except Exception:
        return None


# -----------------------------------------
# Database fetch
# -----------------------------------------
def get_city_data_from_db(city_name: str):
    """Fetch city profile + stats from the DB."""
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
        df = pd.read_sql(sql, conn, params={"city": city_name})

    if df.empty:
        return None

    row = df.iloc[0]

    return {
        "city": row["city"],
        "state": row["state"],
        "population": int(row["population"]) if row["population"] is not None else None,
        "median_age": float(row["median_age"]) if row["median_age"] is not None else None,
        "avg_household_size": float(row["avg_household_size"]) if row["avg_household_size"] is not None else None,
        "description": row["description"] or "",
    }


# -----------------------------------------
# AI SUMMARY GENERATOR
# -----------------------------------------
def generate_ai_summary(user_query: str, city_data):
    """Creates a 2–3 sentence lifestyle summary + bullet points."""

    client = get_openai_client()
    if client is None:
        return "AI lifestyle summary unavailable."

    prompt = f"""
User question: "{user_query}"

City data:
City: {city_data['city']}
State: {city_data['state']}
Population: {city_data['population']}
Median age: {city_data['median_age']}
Average household size: {city_data['avg_household_size']}

Lifestyle notes:
{city_data['description']}

TASK:
Write a friendly 2–3 sentence lifestyle summary describing what it's like to live in this city.
Then add 3 bullet points with key lifestyle highlights.

RULES:
- Use ONLY the information given.
- Do NOT invent crime rates, salaries, or extra facts.
- Keep it friendly and easy to understand.
"""

    rsp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
    )

    return rsp.choices[0].message.content.strip()


# -----------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------
def try_build_lifestyle_card(user_query: str):
    """Master handler for lifestyle queries."""

    if not _looks_like_lifestyle_query(user_query):
        return None

    # 1) Extract city using GPT
    city = extract_city_with_gpt(user_query)
    if not city:
        return None

    # 2) Fetch full city data
    city_data = get_city_data_from_db(city)
    if not city_data:
        return None

    # 3) Generate AI summary
    ai_summary = generate_ai_summary(user_query, city_data)

    # 4) Return final card
    return {
        "city": city_data["city"],
        "state": city_data["state"],
        "population": city_data["population"],
        "median_age": city_data["median_age"],
        "avg_household_size": city_data["avg_household_size"],
        "description": city_data["description"],
        "ai_summary": ai_summary,
    }
