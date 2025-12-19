# core/semantic_search.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text
from openai import OpenAI
from db_config import get_engine


 
# OPENAI CLIENT (cached)
 
@st.cache_resource
def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ OPENAI_API_KEY missing in Streamlit secrets")
    return OpenAI(api_key=key)


 
# EMBEDDING (cached)
 
@st.cache_data(ttl=3600, show_spinner=False)
def embed_query(text_input: str):
    client = get_openai_client()

    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=text_input,
    )

    return np.array(res.data[0].embedding, dtype="float32")


 
# COSINE SIMILARITY
 
def cosine_similarity(a, b):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


 
# LOAD EMBEDDINGS FROM DB (cached)
 
@st.cache_data(ttl=3600, show_spinner=False)
def load_city_embeddings(state_filter: str = None):
    """Load city embeddings from database, cached for 1 hour."""
    engine = get_engine()

    if state_filter:
        sql = text("""
            SELECT city, state, embedding
            FROM dbo.city_embeddings
            WHERE LOWER(state) = LOWER(:state)
        """)
        
        with engine.connect() as conn:
            result = conn.execute(sql, {"state": state_filter})
            rows = result.fetchall()
            cols = result.keys()
    else:
        sql = text("""
            SELECT city, state, embedding
            FROM dbo.city_embeddings
        """)
        
        with engine.connect() as conn:
            result = conn.execute(sql)
            rows = result.fetchall()
            cols = result.keys()

    return pd.DataFrame(rows, columns=cols)


 
# SEMANTIC SEARCH
 
def semantic_city_search(query: str, top_k: int = 10, state_filter: str = None):
    """
    Performs semantic search over SQL Server stored embeddings.
    Returns a list of: [(city, state), ...]
    """

    # Get query embedding (cached)
    query_vec = embed_query(query)
    
    # Load embeddings from DB (cached)
    df = load_city_embeddings(state_filter)
    
    if df.empty:
        return []

    # Compute similarity
    results = []

    for _, row in df.iterrows():
        emb_raw = row["embedding"]

        # Stored embeddings come in two forms:
        #   - JSON string
        #   - Python list
        if isinstance(emb_raw, str):
            emb = json.loads(emb_raw)
        else:
            emb = emb_raw

        score = cosine_similarity(query_vec, emb)
        results.append((row["city"], row["state"], score))

    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)

    return [(c, s) for (c, s, _) in results[:top_k]]
