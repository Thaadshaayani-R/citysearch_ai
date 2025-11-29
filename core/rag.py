# core/rag.py

import os
from typing import List, Dict
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ---------- Embedding client ----------

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)

# ---------- Knowledge base ----------

# Each item is a small "chunk" of knowledge that can help GPT write better SQL.
SCHEMA_KNOWLEDGE: List[Dict] = [
    {
        "id": "table_overview",
        "text": (
            "The main table is dbo.cities. "
            "Each row is a single US city. "
            "Typical queries: filter by state, rank by population, "
            "compute averages per state, and count cities."
        ),
    },
    {
        "id": "col_city",
        "text": (
            "Column city (TEXT): name of the city. "
            "Use it for LIKE searches (starts with, ends with, contains) "
            "or to display the city in SELECT."
        ),
    },
    {
        "id": "col_state",
        "text": (
            "Column state (TEXT): full US state name such as 'Texas', 'Florida'. "
            "Use it in WHERE clauses when the user mentions a state."
        ),
    },
    {
        "id": "col_state_code",
        "text": (
            'Column state_code (TEXT): two-letter state code such as "TX" or "FL". '
            "Sometimes users may say 'CA' or 'TX' instead of 'California' or 'Texas'."
        ),
    },
    {
        "id": "col_population",
        "text": (
            "Column population (INTEGER): city population. "
            "Use it for questions about largest or smallest cities, "
            "cities above or below a population threshold, and totals by state."
        ),
    },
    {
        "id": "col_median_age",
        "text": (
            "Column median_age (FLOAT): median age of residents in the city. "
            "Useful for 'young cities', 'older cities', or thresholds like 'median age > 40'."
        ),
    },
    {
        "id": "col_avg_household_size",
        "text": (
            "Column avg_household_size (FLOAT): average household size. "
            "Useful for family-oriented or household-related questions."
        ),
    },
    {
        "id": "example_top_cities",
        "text": (
            "Example: To get the top 10 cities by population in Texas, use:\n"
            "SELECT TOP (10) city, state, population, median_age, avg_household_size, state_code\n"
            "FROM dbo.cities\n"
            "WHERE state = 'Texas'\n"
            "ORDER BY population DESC;"
        ),
    },
    {
        "id": "example_count_cities",
        "text": (
            "Example: To count cities in California, use:\n"
            "SELECT COUNT(*) AS total_cities\n"
            "FROM dbo.cities\n"
            "WHERE state = 'California';"
        ),
    },
    {
        "id": "example_percentage",
        "text": (
            "Example: To compute percentage of cities with population >= 500000, use:\n"
            "SELECT (100.0 * COUNT(CASE WHEN population >= 500000 THEN 1 END) / COUNT(*)) AS percentage\n"
            "FROM dbo.cities;"
        ),
    },
    {
        "id": "example_groupby_state",
        "text": (
            "Example: To get average population by state, use:\n"
            "SELECT state, AVG(population) AS avg_population\n"
            "FROM dbo.cities\n"
            "GROUP BY state\n"
            "ORDER BY avg_population DESC;"
        ),
    },
]

# ---------- Build embeddings once per process ----------

_embeddings_matrix: np.ndarray = None
_ids: List[str] = None
_texts: List[str] = None


def _build_index():
    global _embeddings_matrix, _ids, _texts
    if _embeddings_matrix is not None:
        return

    client = get_openai_client()

    texts = [chunk["text"] for chunk in SCHEMA_KNOWLEDGE]
    ids = [chunk["id"] for chunk in SCHEMA_KNOWLEDGE]

    # Call embeddings API once with all texts
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    vectors = []
    for item in resp.data:
        vectors.append(item.embedding)

    _embeddings_matrix = np.array(vectors, dtype="float32")
    _ids = ids
    _texts = texts


def retrieve_schema_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Given a user query, return top_k most relevant schema knowledge chunks
    using cosine similarity over OpenAI embeddings.
    """
    _build_index()

    client = get_openai_client()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    query_vec = np.array(resp.data[0].embedding, dtype="float32")

    # cosine similarity
    dot = np.dot(_embeddings_matrix, query_vec)
    norm_matrix = np.linalg.norm(_embeddings_matrix, axis=1)
    norm_query = np.linalg.norm(query_vec)
    sims = dot / (norm_matrix * norm_query + 1e-8)

    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for i in top_idx:
        results.append(
            {
                "id": _ids[i],
                "score": float(sims[i]),
                "text": _texts[i],
            }
        )
    return results
