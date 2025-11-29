import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import streamlit as st


@st.cache_resource
def get_engine():
    """
    Returns a cached SQLAlchemy Engine with connection pooling.
    """
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]

    username = urllib.parse.quote_plus(st.secrets["SQL_SERVER_USER"])
    password = urllib.parse.quote_plus(st.secrets["SQL_SERVER_PASSWORD"])
    driver = urllib.parse.quote_plus(st.secrets["SQL_SERVER_DRIVER"])

    conn_str = (
        f"mssql+pyodbc://{username}:{password}"
        f"@{server}:1433/{database}"
        f"?driver={driver}"
    )

    return create_engine(
        conn_str,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={"timeout": 30}
    )


def get_connection():
    """Return a connection from the pool."""
    engine = get_engine()
    return engine.connect()
