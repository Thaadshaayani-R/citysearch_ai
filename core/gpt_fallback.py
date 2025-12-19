# core/gpt_fallback.py

import os
from core.openai_client import get_openai_client, require_openai_client
from .rag import retrieve_schema_context  # NEW: import RAG retriever


SCHEMA_DESCRIPTION = """
You are an expert SQL assistant. The database has a single main table:

Table: dbo.cities
Columns:
- city (TEXT): City name
- state (TEXT): U.S. state name, e.g. 'Texas', 'California'
- state_code (TEXT): U.S. state code, e.g. 'TX', 'CA'
- population (INTEGER): City population
- median_age (FLOAT): Median age of residents
- avg_household_size (FLOAT): Average household size per home

Rules:
- Always generate T-SQL compatible with SQL Server.
- Never use tables that are not listed.
- Always fully qualify the table as dbo.cities.
- Do NOT use dangerous operations like DELETE, DROP, UPDATE, INSERT.
- Only generate read-only SELECT queries.
"""


def generate_sql_with_gpt(user_query: str) -> str:
    """
    Use GPT to generate a safe SQL query based on natural language and schema,
    enriched with RAG context from the schema knowledge base.
    """
    client = get_openai_client() 
    
    if client is None:
        raise RuntimeError("OpenAI client not configured")
    
    # ---- RAG retrieval step ----
    context_chunks = retrieve_schema_context(user_query, top_k=5)
    context_text = "\n\n".join(
        f"- {chunk['text']}" for chunk in context_chunks
    )

    system_prompt = (
        "You are a senior data engineer that converts natural language into safe, read-only SQL. "
        "You ONLY respond with SQL, no explanation, no markdown. "
        "Use the provided schema and the additional context carefully."
    )

    user_prompt = f"""
User question:
{user_query}

Relevant database context (from knowledge base):
{context_text}

Schema:
{SCHEMA_DESCRIPTION}

Return only ONE valid SQL SELECT statement. No comments, no explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    sql = response.choices[0].message.content.strip()
    return sql
