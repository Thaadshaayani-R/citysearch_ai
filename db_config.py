# db_config.py
"""
Unified database configuration for CitySearch AI.
All modules should import from here - DO NOT duplicate this code.

Usage:
    from db_config import get_engine, get_connection
"""

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import urllib.parse


@st.cache_resource
def get_engine():
    """
    Returns a cached SQLAlchemy Engine with connection pooling.
    Uses pymssql for Streamlit Cloud compatibility.
    
    Returns:
        sqlalchemy.Engine: Configured database engine
    """
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    username = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]
    
    # URL encode password to handle special characters
    password_encoded = urllib.parse.quote_plus(password)

    # pymssql connection string (Streamlit Cloud compatible)
    conn_str = (
        f"mssql+pymssql://{username}:{password_encoded}@{server}:1433/{database}"
    )

    return create_engine(
        conn_str,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_connection():
    """
    Returns a connection from the pool.
    
    Usage:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM table", conn)
        conn.close()
    
    Returns:
        sqlalchemy.Connection: Database connection
    """
    engine = get_engine()
    return engine.connect()
