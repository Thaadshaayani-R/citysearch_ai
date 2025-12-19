# core/rag_search.py

import json
import math
from typing import List, Dict, Optional
import hashlib

import pandas as pd
import numpy as np
from sqlalchemy import text
import streamlit as st

from db_config import get_engine


 
# OPENAI CLIENT
 
def get_openai_client():
    """Get OpenAI client from Streamlit secrets."""
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)


 
# OPTIMIZED: CITY-SPECIFIC CHUNK LOADING
 
@st.cache_data(ttl=3600, show_spinner="Loading city data...")
def load_city_chunks(city_name: str, state_name: str = None) -> List[Dict]:
    """
    Load RAG chunks for a SPECIFIC city only.
    Much faster than loading all 30K chunks.
    """
    engine = get_engine()

    if state_name:
        query = text("""
            SELECT id, city, state, chunk_type, chunk_text, embedding_json
            FROM dbo.city_rag_chunks
            WHERE LOWER(city) = LOWER(:city) 
            AND LOWER(state) = LOWER(:state)
            AND embedding_json IS NOT NULL
        """)
        params = {"city": city_name, "state": state_name}
    else:
        query = text("""
            SELECT id, city, state, chunk_type, chunk_text, embedding_json
            FROM dbo.city_rag_chunks
            WHERE LOWER(city) = LOWER(:city)
            AND embedding_json IS NOT NULL
        """)
        params = {"city": city_name}

    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        print(f"Error loading city chunks: {e}")
        return []

    records = []
    for _, row in df.iterrows():
        try:
            emb = json.loads(row["embedding_json"])
        except:
            continue

        records.append({
            "id": int(row["id"]),
            "city": row["city"],
            "state": row["state"],
            "chunk_type": row["chunk_type"],
            "chunk_text": row["chunk_text"],
            "embedding": emb,
        })

    return records


 
# OPTIMIZED: LAZY LOAD ALL CHUNKS (only when no city filter)
 
@st.cache_data(ttl=7200, show_spinner="Preparing knowledge base...")
def load_all_chunks() -> List[Dict]:
    """
    Load all RAG chunks - ONLY called for queries without city filter.
    Increased TTL since this is expensive.
    """
    engine = get_engine()

    query = text("""
        SELECT id, city, state, chunk_type, chunk_text, embedding_json
        FROM dbo.city_rag_chunks
        WHERE embedding_json IS NOT NULL
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        print(f"Error loading RAG chunks: {e}")
        return []

    records = []
    for _, row in df.iterrows():
        try:
            emb = json.loads(row["embedding_json"])
        except:
            continue

        records.append({
            "id": int(row["id"]),
            "city": row["city"],
            "state": row["state"],
            "chunk_type": row["chunk_type"],
            "chunk_text": row["chunk_text"],
            "embedding": emb,
        })

    return records


 
# OPTIMIZED: CACHED QUERY EMBEDDINGS
 
@st.cache_data(ttl=1800, show_spinner="Analyzing query...")
def get_query_embedding(query_text: str) -> Optional[List[float]]:
    """
    Get embedding for a query - cached to avoid repeated API calls.
    """
    client = get_openai_client()
    if client is None:
        return None

    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query_text],
        )
        return emb_resp.data[0].embedding
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None


 
# COSINE SIMILARITY
 
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors using numpy (10x faster)."""
    if a is None or b is None:
        return -1.0
    
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    
    if norm_a == 0 or norm_b == 0:
        return -1.0
    
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


 
# OPTIMIZED RAG SEARCH
 
def rag_search(user_query: str, top_k: int = 8, city_filter: str = None, state_filter: str = None) -> List[Dict]:
    """
    OPTIMIZED: Search RAG chunks for relevant context.
    
    Args:
        user_query: User's question
        top_k: Number of chunks to return
        city_filter: Optional city name (uses city-specific loading if provided)
        state_filter: Optional state name
        
    Returns:
        List of top K matching chunks with scores
    """
    # Get query embedding (CACHED)
    query_emb = get_query_embedding(user_query)
    if query_emb is None:
        return []

    # OPTIMIZATION: Load city-specific chunks if city is known
    if city_filter:
        chunks = load_city_chunks(city_filter, state_filter)
        if not chunks:
            return []
    else:
        # Fallback: Load all chunks (slower, but necessary for open-ended queries)
        chunks = load_all_chunks()
        if not chunks:
            return []

    # Score all chunks
    for ch in chunks:
        ch["score"] = cosine_similarity(query_emb, ch["embedding"])

    # Sort and return top K
    chunks_sorted = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return chunks_sorted[:top_k]


 
# OPTIMIZED RAG ANSWER
 
def rag_answer(user_query: str, top_k: int = 8, city_filter: str = None, state_filter: str = None) -> str:
    """
    OPTIMIZED: Generate a RAG-powered answer to user's query.
    """
    client = get_openai_client()
    if client is None:
        return "RAG answer is not available (missing OpenAI key)."

    # Get relevant chunks (OPTIMIZED)
    top_chunks = rag_search(user_query, top_k, city_filter, state_filter)
    
    if not top_chunks:
        if city_filter:
            return f"No information found about {city_filter} in our knowledge base."
        return "No relevant information found in knowledge base."

    # Chunk type labels for display
    type_labels = {
        "lifestyle": "Lifestyle",
        "jobs_economy": "Jobs & Economy",
        "cost_of_living": "Cost of Living",
        "family": "Family-Friendliness",
        "weather_geography": "Weather & Geography",
    }

    # Build context from top chunks
    context_parts = []
    for ch in top_chunks:
        label = type_labels.get(ch["chunk_type"], ch["chunk_type"])
        context_parts.append(
            f"City: {ch['city']}, {ch['state']} | Aspect: {label}\n"
            f"Summary: {ch['chunk_text']}\n"
        )

    context = "\n---\n".join(context_parts)

    # Generate answer
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

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a friendly city expert."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating RAG answer: {e}")
        return "Could not generate answer. Please try again."


 
# OPTIMIZED: GET CITY RAG CONTEXT
 
def get_city_rag_context(city_name: str, state_name: str = None) -> Dict:
    """
    OPTIMIZED: Get all RAG context for a specific city using city-specific loading.
    """
    # Use city-specific loading (FAST)
    city_chunks = load_city_chunks(city_name, state_name)

    if not city_chunks:
        return {}

    # Organize by chunk type
    context = {
        "lifestyle": [],
        "jobs_economy": [],
        "cost_of_living": [],
        "family": [],
        "weather_geography": [],
    }

    for ch in city_chunks:
        chunk_type = ch["chunk_type"]
        if chunk_type in context:
            context[chunk_type].append(ch["chunk_text"])

    return context


 
# OPTIMIZED LIFESTYLE ANSWER
 
def get_lifestyle_rag_answer(city_name: str, state_name: str, city_data: Dict) -> str:
    """
    OPTIMIZED: Generate lifestyle answer using city-specific RAG context.
    """
    client = get_openai_client()
    if client is None:
        return None

    # Get RAG context for this city (OPTIMIZED - city-specific query)
    rag_context = get_city_rag_context(city_name, state_name)
    
    # Check if we have ANY RAG context
    has_context = False
    context_parts = []
    
    type_labels = {
        "lifestyle": "Lifestyle & Culture",
        "jobs_economy": "Jobs & Economy",
        "cost_of_living": "Cost of Living",
        "family": "Family Life",
        "weather_geography": "Weather & Geography",
    }
    
    for chunk_type, texts in rag_context.items():
        if texts:
            has_context = True
            label = type_labels.get(chunk_type, chunk_type)
            combined = " ".join(texts[:2])
            context_parts.append(f"**{label}:** {combined}")
    
    # If no RAG context, return None
    if not has_context:
        return None
    
    rag_text = "\n\n".join(context_parts)

    # Build prompt
    pop = city_data.get('population', 'N/A')
    pop_str = f"{pop:,}" if isinstance(pop, (int, float)) else str(pop)
    
    prompt = f"""Summarize life in {city_name}, {state_name} using ONLY the information below.

DATABASE FACTS:
- Population: {pop_str}
- Median Age: {city_data.get('median_age', 'N/A')}
- Avg Household Size: {city_data.get('avg_household_size', 'N/A')}

RETRIEVED KNOWLEDGE (use ONLY this):
{rag_text}

STRICT RULES:
- Use ONLY the information provided above
- Do NOT add any information not explicitly stated above
- Keep response to 2-3 short paragraphs
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a factual assistant. Only use information explicitly provided."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating lifestyle RAG answer: {e}")
        return None