# core/rag_search.py

import os
import json
import math
from typing import List, Dict

import pandas as pd
from openai import OpenAI
import streamlit as st
from db_config import get_engine



# -----------------------------------
# 2) OPENAI CLIENT
# -----------------------------------
def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


# -----------------------------------
# 3) LOAD RAG CHUNKS FROM SQL
# -----------------------------------
def load_all_chunks() -> List[Dict]:
    engine = get_engine()

    query = text("""
        SELECT id, city, state, chunk_type, chunk_text, embedding_json
        FROM dbo.city_rag_chunks
        WHERE embedding_json IS NOT NULL
    """)

    df = pd.read_sql(query, engine)

    records = []
    for _, row in df.iterrows():
        try:
            emb = json.loads(row["embedding_json"])
        except:
            emb = None

        records.append({
            "id": int(row["id"]),
            "city": row["city"],
            "state": row["state"],
            "chunk_type": row["chunk_type"],
            "chunk_text": row["chunk_text"],
            "embedding": emb,
        })

    return records


# -----------------------------------
# 4) COSINE SIMILARITY
# -----------------------------------
def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


# -----------------------------------
# 5) MAIN RAG ANSWER
# -----------------------------------
def rag_answer(user_query: str, top_k: int = 8) -> str:
    client = get_openai_client()
    if client is None:
        return "RAG answer is not available (missing OpenAI key)."

    # Load chunks
    chunks = load_all_chunks()
    if not chunks:
        return "Knowledge base is empty. Please build RAG chunks first."

    # Embed the query
    emb_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[user_query],
    )
    query_emb = emb_resp.data[0].embedding

    # Score
    for ch in chunks:
        ch["score"] = cosine_similarity(query_emb, ch["embedding"])

    # Top K
    chunks_sorted = sorted(chunks, key=lambda x: x["score"], reverse=True)
    top_chunks = chunks_sorted[:top_k]

    # Labels
    type_labels = {
        "lifestyle": "Lifestyle",
        "jobs_economy": "Jobs & Economy",
        "cost_of_living": "Cost of Living",
        "family": "Family-Friendliness",
        "weather_geography": "Weather & Geography",
    }

    # Build context
    context_parts = []
    for ch in top_chunks:
        label = type_labels.get(ch["chunk_type"], ch["chunk_type"])
        context_parts.append(
            f"City: {ch['city']}, {ch['state']} | Aspect: {label}\n"
            f"Summary: {ch['chunk_text']}\n"
        )

    context = "\n---\n".join(context_parts)

    # Final prompt
    prompt = f"""
You are CitySearch AI, explaining US cities to a normal person.

Question:
\"\"\"{user_query}\"\"\"

Here are knowledge snippets from the database:

{context}

Using ONLY this information:
- Give a clear answer in 3–6 short bullet points or small paragraphs.
- If the user asks about one city, focus on that city.
- If they compare cities, highlight differences.
- Use simple language (no technical words).
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a friendly city expert."},
            {"role": "user", "content": prompt},
        ],
    )

    return resp.choices[0].message.content.strip()
