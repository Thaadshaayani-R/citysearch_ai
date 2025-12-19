# core/openai_client.py (NEW)
"""
Unified OpenAI client for CitySearch AI.
All modules should import from here.

Usage:
    from core.openai_client import get_openai_client, generate_completion, generate_embedding
"""

import os
import streamlit as st
from openai import OpenAI
from typing import List, Optional


 
# CLIENT SINGLETON
 
@st.cache_resource
def get_openai_client() -> Optional[OpenAI]:
    """
    Returns a cached OpenAI client.
    Tries Streamlit secrets first, then environment variables.
    
    Returns:
        OpenAI client or None if no API key found
    """
    # Try Streamlit secrets first
    key = st.secrets.get("OPENAI_API_KEY")
    
    # Fall back to environment variable
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    
    if not key:
        return None
    
    return OpenAI(api_key=key)


def require_openai_client() -> OpenAI:
    """
    Returns OpenAI client or raises error if not configured.
    Use this when OpenAI is required, not optional.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OpenAI API key not configured. "
            "Set OPENAI_API_KEY in Streamlit secrets or environment variables."
        )
    return client


 
# HELPER FUNCTIONS
 
def generate_completion(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int = 500,
    system_prompt: str = None,
) -> Optional[str]:
    """
    Generate a chat completion.
    
    Args:
        prompt: User prompt
        model: Model to use (default: gpt-4.1-mini)
        temperature: Randomness (0-1)
        max_tokens: Max response length
        system_prompt: Optional system message
    
    Returns:
        Generated text or None on error
    """
    client = get_openai_client()
    if client is None:
        return None
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def generate_embedding(
    text: str,
    model: str = "text-embedding-3-small",
) -> Optional[List[float]]:
    """
    Generate embedding for text.
    
    Args:
        text: Text to embed
        model: Embedding model (default: text-embedding-3-small for RAG)
    
    Returns:
        Embedding vector or None on error
    """
    client = get_openai_client()
    if client is None:
        return None
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI embedding error: {e}")
        return None


def generate_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
) -> Optional[List[List[float]]]:
    """
    Generate embeddings for multiple texts in one API call.
    
    Args:
        texts: List of texts to embed
        model: Embedding model
    
    Returns:
        List of embedding vectors or None on error
    """
    client = get_openai_client()
    if client is None:
        return None
    
    try:
        response = client.embeddings.create(
            model=model,
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"OpenAI batch embedding error: {e}")
        return None


 
# MODEL CONSTANTS
 
class Models:
    """Available model names."""
    # Chat models
    GPT_4_MINI = "gpt-4.1-mini"
    GPT_4O_MINI = "gpt-4o-mini"
    
    # Embedding models
    EMBED_SMALL = "text-embedding-3-small"   # 1536 dimensions
    EMBED_LARGE = "text-embedding-3-large"   # 3072 dimensions
