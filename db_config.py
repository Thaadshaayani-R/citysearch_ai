# db_config.py (FIXED VERSION)
"""
Unified database configuration for CitySearch AI.
All modules should import from here - DO NOT duplicate this code.

Usage:
    from db_config import get_engine, get_connection
"""

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool


@st.cache_resource
def get_engine():
    """
    Returns a cached SQLAlchemy Engine with connection pooling.
    Uses pytds for Streamlit Cloud compatibility (pure Python, no ODBC needed).
    
    Returns:
        sqlalchemy.Engine: Configured database engine
    """
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    username = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]

    # pytds connection string (Streamlit Cloud compatible - no system ODBC needed)
    conn_str = (
        f"mssql+pytds://{username}:{password}@{server}:1433/{database}"
        "?charset=utf8&autocommit=True"
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
