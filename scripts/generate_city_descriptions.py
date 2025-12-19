#scripts/generate_city_description.py

import os
import time
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from openai import OpenAI
from db_config import get_engine

load_dotenv()


# 2) OpenAI Client

def get_openai_client():
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in Streamlit secrets.")
    return OpenAI(api_key=api_key)



# 3) City Description Prompt

DESCRIPTION_PROMPT = """
Generate a rich, meaningful profile for this U.S. city.

Include:
- lifestyle & culture
- affordability
- family-friendliness
- population trends
- age profile
- housing characteristics
- local economy or job market vibe
- who would enjoy living there (families, seniors, students, etc.)

Write in 110–150 words.
Avoid repeating numbers if not required.

City: {city}
State: {state}
Population: {population}
Median Age: {median_age}
Avg Household Size: {avg_household_size}

Write only the description. No headings.
"""



# 4) Main Script

def main():
    engine = get_engine()
    client = get_openai_client()

    # Load cities
    df = None
    with engine.begin() as conn:
        df = conn.execute(
            text("""
                SELECT city, state, population, median_age, avg_household_size
                FROM dbo.cities
                ORDER BY state, city
            """)
        ).fetchall()

    print(f"Found {len(df)} cities. Generating descriptions...")

    for row in df:
        city, state, population, median_age, avg_hh = row

        # Check if already exists
        with engine.begin() as conn:
            exists = conn.execute(
                text("""
                    SELECT COUNT(*) FROM dbo.city_profiles
                    WHERE city = :city AND state = :state
                """),
                {"city": city, "state": state}
            ).scalar()

        if exists > 0:
            print(f"Skipping {city}, already exists.")
            continue

        print(f"Generating description for {city}, {state}...")

        prompt = DESCRIPTION_PROMPT.format(
            city=city,
            state=state,
            population=population,
            median_age=median_age,
            avg_household_size=avg_hh
        )

        # GPT call
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": "You are a helpful city analyst."},
                {"role": "user", "content": prompt}
            ],
        )
        description = response.choices[0].message.content.strip()

        # Insert into SQL
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO dbo.city_profiles (city, state, description)
                    VALUES (:city, :state, :description)
                """),
                {"city": city, "state": state, "description": description}
            )

        time.sleep(0.4)  # reduce rate limits

    print("DONE! All profiles generated.")


if __name__ == "__main__":
    main()
