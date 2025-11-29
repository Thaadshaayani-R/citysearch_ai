#scripts/build_city_embeddings.py

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI
import streamlit as st

load_dotenv()

# ---------------------------------------
# 1) DB Connection (SQLAlchemy + pytds)
# ---------------------------------------
def get_engine():
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    username = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]

    connection_str = (
        f"mssql+pytds://{username}:{password}@{server}:1433/{database}"
        "?charset=utf8&autocommit=True"
    )

    return create_engine(connection_str)


# ---------------------------------------
# 2) OpenAI Client
# ---------------------------------------
def get_openai_client():
    key = st.secrets["OPENAI_API_KEY"]
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in secrets.")
    return OpenAI(api_key=key)


# ---------------------------------------
# 3) Generate Embeddings
# ---------------------------------------
def main():
    engine = get_engine()
    client = get_openai_client()

    # --- Load all profiles from SQL ---
    query = """
        SELECT city, state, description
        FROM dbo.city_profiles
        ORDER BY state, city
    """

    df = pd.read_sql(query, engine)

    print(f"Found {len(df)} descriptions. Generating embeddings...\n")

    # --- Process each row ---
    with engine.begin() as conn:
        for _, row in df.iterrows():
            city = row["city"]
            state = row["state"]
            description = row["description"]

            # Check if already embedded
            exists_sql = text("""
                SELECT COUNT(*) AS cnt
                FROM dbo.city_embeddings
                WHERE city = :city AND state = :state
            """)

            exists = conn.execute(exists_sql, {"city": city, "state": state}).scalar()

            if exists > 0:
                print(f"Skipping {city}, already embedded.")
                continue

            print(f"Embedding {city}, {state}...")

            # Create embedding
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=description
            )

            vector = response.data[0].embedding

            # Insert embedding JSON
            insert_sql = text("""
                INSERT INTO dbo.city_embeddings (city, state, embedding)
                VALUES (:city, :state, :embedding)
            """)

            conn.execute(insert_sql, {
                "city": city,
                "state": state,
                "embedding": json.dumps(vector)
            })

            time.sleep(0.2)  # rate limit safety

    print("\nDONE! All embeddings stored.")


if __name__ == "__main__":
    main()
