# core/semantic_search.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from openai import OpenAI
import urllib


# ===============================================================
# DATABASE ENGINE (pyodbc + FreeTDS) — FULLY STREAMLIT SAFE
# ===============================================================
def get_engine():
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    username = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]

    # FreeTDS ODBC connection
    odbc_str = (
        "DRIVER=FreeTDS;"
        f"SERVER={server};"
        "PORT=1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "TDS_Version=7.4;"
    )

    params = urllib.parse.quote_plus(odbc_str)

    # SQLAlchemy engine
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    return engine


# ===============================================================
# OPENAI CLIENT
# ===============================================================
def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ OPENAI_API_KEY missing in Streamlit secrets")
    return OpenAI(api_key=key)


# ===============================================================
# EMBEDDING
# ===============================================================
def embed_query(text):
    client = get_openai_client()

    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )

    return np.array(res.data[0].embedding, dtype="float32")


# ===============================================================
# COSINE SIMILARITY
# ===============================================================
def cosine_similarity(a, b):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ===============================================================
# SEMANTIC SEARCH
# ===============================================================
def semantic_city_search(query, top_k=10, state_filter=None):
    """
    Performs semantic search over SQL Server stored embeddings.
    Returns a list of:  [(city, state), ...]
    """

    query_vec = embed_query(query)
    engine = get_engine()

    # ----------------------------------------
    # Build SQL query
    # ----------------------------------------
    if state_filter:
        sql = text("""
            SELECT city, state, embedding
            FROM dbo.city_embeddings
            WHERE state = :state
        """)
        df = pd.read_sql(sql, engine, params={"state": state_filter})

    else:
        sql = text("""
            SELECT city, state, embedding
            FROM dbo.city_embeddings
        """)
        df = pd.read_sql(sql, engine)

    # ----------------------------------------
    # Compute similarity
    # ----------------------------------------
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
