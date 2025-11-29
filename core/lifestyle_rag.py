# core/lifestyle_rag.py

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
# Query pattern triggers
# -----------------------------------------
LIFESTYLE_TRIGGERS = [
    "tell me about",
    "what is it like in",
    "what is life like in",
    "life in",
    "living in",
    "live in",
]


def _extract_city_from_query(query: str):
    q = query.lower()

    for phrase in LIFESTYLE_TRIGGERS:
        if phrase in q:
            part = q.split(phrase)[-1].strip().rstrip("?.!,")
            return part.title()

    tokens = query.strip().rstrip("?.!,").split()
    if tokens:
        return tokens[-1].title()
    return None


def _looks_like_lifestyle_query(query: str):
    q = query.lower()
    return any(p in q for p in LIFESTYLE_TRIGGERS)


# -----------------------------------------
# Main RAG builder
# -----------------------------------------
def try_build_lifestyle_card(user_query: str):

    if not _looks_like_lifestyle_query(user_query):
        return None

    city = _extract_city_from_query(user_query)
    if not city:
        return None

    # --- SAFE: SQLAlchemy engine ---
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

    # --- Must use engine.connect() ---
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"city": city})

    if df.empty:
        return None

    row = df.iloc[0]
    description = row.get("description") or ""

    # --- RAG ---
    client = get_openai_client()

    if client is None:
        ai_summary = "AI lifestyle summary unavailable."
    else:
        prompt = f"""
User question: "{user_query}"

City data:
City: {row['city']}
State: {row['state']}
Population: {row['population']}
Median age: {row['median_age']}
Average household size: {row['avg_household_size']}

Lifestyle notes:
{description}

Write a friendly 2–3 sentence lifestyle summary + 3 bullet points.
Use only the information above.
"""

        rsp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        ai_summary = rsp.choices[0].message.content.strip()

    return {
        "city": row["city"],
        "state": row["state"],
        "population": int(row["population"]),
        "median_age": float(row["median_age"]),
        "avg_household_size": float(row["avg_household_size"]),
        "description": description,
        "ai_summary": ai_summary,
    }
