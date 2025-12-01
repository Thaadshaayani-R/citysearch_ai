#scripts/build_city_rag_embeddings.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
import streamlit as st
from db_config import get_engine

load_dotenv()


# ---------------------------------------
# OpenAI Client
# ---------------------------------------
def get_openai_client():
    key = st.secrets["OPENAI_API_KEY"]
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit secrets.")
    return OpenAI(api_key=key)


# ---------------------------------------
# Build Embeddings in Batches
# ---------------------------------------
def build_embeddings(batch_size: int = 64):
    engine = get_engine()
    client = get_openai_client()

    # Load rows missing embeddings
    df = pd.read_sql(
        """
        SELECT id, chunk_text
        FROM dbo.city_rag_chunks
        WHERE embedding_json IS NULL
        ORDER BY id
        """,
        engine,
    )

    if df.empty:
        print("No chunks need embeddings.")
        return

    print(f"Embedding {len(df)} chunks...")

    # Streamlit / SQLAlchemy safe DB operations
    with engine.begin() as conn:

        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            texts = batch["chunk_text"].tolist()
            ids = batch["id"].tolist()

            print(f" → Batch {start}–{start + len(batch) - 1}")

            # Create embeddings
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )

            vectors = [item.embedding for item in resp.data]

            # Insert embeddings back into SQL
            update_sql = text("""
                UPDATE dbo.city_rag_chunks
                SET embedding_json = :emb
                WHERE id = :cid
            """)

            for chunk_id, vec in zip(ids, vectors):
                conn.execute(update_sql, {
                    "emb": json.dumps(vec),
                    "cid": chunk_id
                })

            print("   ✓ Committed batch")

    print("🎉 All embeddings created and stored in Azure SQL.")


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    build_embeddings()
