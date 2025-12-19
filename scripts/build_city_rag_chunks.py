#scripts/build_city_rag_chunks.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
import streamlit as st
from db_config import get_engine

load_dotenv()


# 2) OpenAI Client

def get_openai_client():
    key = st.secrets["OPENAI_API_KEY"]
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in secrets.toml")
    return OpenAI(api_key=key)



# 3) Load City Profiles

def load_city_profiles():
    engine = get_engine()
    sql = """
        SELECT city, state, description
        FROM dbo.city_profiles
        ORDER BY city, state
    """
    return pd.read_sql(sql, engine)



# 4) GPT Prompt Template

SUMMARY_PROMPT_TEMPLATE = """
You are helping build a city information assistant.

Based ONLY on the description below, create SHORT summaries for these 5 aspects:

1) lifestyle              (overall vibe, culture, activities)
2) jobs_economy           (types of jobs, economy, growth)
3) cost_of_living         (cheap / moderate / expensive, housing)
4) family                 (schools, safety, family-friendly)
5) weather_geography      (climate, seasons, geography)

Rules:
- Max 2 sentences for each aspect.
- Simple language.
- No invented facts.
- Output ONLY valid JSON with EXACT keys:
  "lifestyle", "jobs_economy", "cost_of_living", "family", "weather_geography"

CITY: {city}, {state}

DESCRIPTION:
\"\"\"{description}\"\"\"
"""



# 5) Generate & Insert RAG Chunks

def summarize_city_rows(df: pd.DataFrame):
    engine = get_engine()
    client = get_openai_client()

    # Load already processed cities
    already_df = pd.read_sql(
        "SELECT DISTINCT city, state FROM dbo.city_rag_chunks",
        engine
    )
    processed = set((c.lower(), s.lower())
                    for c, s in zip(already_df["city"], already_df["state"]))

    total = len(df)

    with engine.begin() as conn:
        for idx, row in df.iterrows():
            city = row["city"]
            state = row["state"]
            description = row["description"]

            key = (city.lower(), state.lower())
            if key in processed:
                print(f"Skipping {city}, {state} (already summarized)")
                continue

            print(f"[{idx+1}/{total}] Summarizing {city}, {state}...")

            prompt = SUMMARY_PROMPT_TEMPLATE.format(
                city=city,
                state=state,
                description=description or ""
            )

            # --- Run GPT ---
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    temperature=0.2,
                    messages=[
                        {"role": "system",
                         "content": "You write concise factual summaries."},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = resp.choices[0].message.content.strip()
                # Extract JSON
                js_start = content.find("{")
                js_end = content.rfind("}")
                json_str = content[js_start: js_end + 1]
                data = json.loads(json_str)
            except Exception as e:
                print(f"❌ Error summarizing {city}, {state}: {e}")
                continue

            # --- Insert chunks into DB ---
            insert_sql = text("""
                INSERT INTO dbo.city_rag_chunks (city, state, chunk_type, chunk_text)
                VALUES (:city, :state, :ctype, :text)
            """)

            rows = [
                ("lifestyle", data.get("lifestyle", "").strip()),
                ("jobs_economy", data.get("jobs_economy", "").strip()),
                ("cost_of_living", data.get("cost_of_living", "").strip()),
                ("family", data.get("family", "").strip()),
                ("weather_geography",
                 data.get("weather_geography", "").strip()),
            ]

            for ctype, textval in rows:
                if not textval:
                    continue
                conn.execute(
                    insert_sql,
                    {"city": city, "state": state,
                        "ctype": ctype, "text": textval}
                )

            print(f"✅ Inserted chunks for {city}, {state}")

    print("\n🎉 Finished creating RAG chunks.")



# MAIN

if __name__ == "__main__":
    profiles = load_city_profiles()
    summarize_city_rows(profiles)
