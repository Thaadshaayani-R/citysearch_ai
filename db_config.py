# db_config.py

import urllib.parse
from sqlalchemy import create_engine
import streamlit as st


def get_engine():
    """
    Returns a SQLAlchemy Engine using pyodbc + ODBC Driver 17.
    Streamlit Cloud compatible.
    """
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]

    username = urllib.parse.quote_plus(st.secrets["SQL_SERVER_USER"])
    password = urllib.parse.quote_plus(st.secrets["SQL_SERVER_PASSWORD"])
    driver   = urllib.parse.quote_plus(st.secrets["SQL_SERVER_DRIVER"])

    conn_str = (
        f"mssql+pyodbc://{username}:{password}"
        f"@{server}:1433/{database}"
        f"?driver={driver}"
    )

    # Debug
    print("🚀 CONNECTION STRING:", conn_str)

    return create_engine(
        conn_str,
        pool_pre_ping=True,
        connect_args={"timeout": 30}
    )


def get_connection():
    """
    MUST return a DB connection, not the engine.
    This is required for pandas.read_sql on Streamlit Cloud.
    """
    engine = get_engine()
    return engine.connect()
