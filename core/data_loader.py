# core/data_loader.py (NEW)
"""
Unified data loading for CitySearch AI.
Single source of truth for loading city data from database.

Usage:
    from core.data_loader import (
        get_cities_df,
        get_city_by_name,
        get_cities_by_state,
        get_city_with_profile,
    )
"""

import pandas as pd
import streamlit as st
from sqlalchemy import text
from db_config import get_engine


 
# CACHED DATA LOADERS
 
@st.cache_data(ttl=3600, show_spinner=False)
def get_cities_df(include_invalid: bool = False) -> pd.DataFrame:
    """
    Load all cities from database.
    Cached for 1 hour.
    
    Args:
        include_invalid: If False, excludes rows with zero/null values
    
    Returns:
        DataFrame with columns: city, state, population, median_age, avg_household_size
    """
    engine = get_engine()
    
    if include_invalid:
        query = text("""
            SELECT city, state, population, median_age, avg_household_size
            FROM dbo.cities
            ORDER BY state, city
        """)
    else:
        query = text("""
            SELECT city, state, population, median_age, avg_household_size
            FROM dbo.cities
            WHERE population > 0
              AND median_age > 0
              AND avg_household_size > 0
            ORDER BY state, city
        """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
        cols = result.keys()
    
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=3600, show_spinner=False)
def get_city_names() -> list:
    """
    Get list of all city names in database.
    Cached for 1 hour.
    
    Returns:
        List of city names
    """
    df = get_cities_df()
    return df["city"].unique().tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def get_state_names() -> list:
    """
    Get list of all state names in database.
    Cached for 1 hour.
    
    Returns:
        List of state names
    """
    df = get_cities_df()
    return df["state"].unique().tolist()


def get_city_by_name(city_name: str) -> dict:
    """
    Get single city data by name.
    
    Args:
        city_name: City name (case-insensitive)
    
    Returns:
        dict with city data or None if not found
    """
    df = get_cities_df()
    row = df[df["city"].str.lower() == city_name.lower()]
    
    if row.empty:
        return None
    
    return row.iloc[0].to_dict()


def get_cities_by_state(state_name: str) -> pd.DataFrame:
    """
    Get all cities in a state.
    
    Args:
        state_name: State name (case-insensitive)
    
    Returns:
        DataFrame of cities in that state
    """
    df = get_cities_df()
    return df[df["state"].str.lower() == state_name.lower()].copy()


@st.cache_data(ttl=3600, show_spinner=False)
def get_city_with_profile(city_name: str) -> dict:
    """
    Get city data with profile description.
    Joins cities and city_profiles tables.
    
    Args:
        city_name: City name (case-insensitive)
    
    Returns:
        dict with city data + description, or None if not found
    """
    engine = get_engine()
    
    query = text("""
        SELECT TOP 1
            c.city,
            c.state,
            c.population,
            c.median_age,
            c.avg_household_size,
            p.description
        FROM dbo.cities AS c
        LEFT JOIN dbo.city_profiles AS p
            ON c.city = p.city AND c.state = p.state
        WHERE LOWER(c.city) = LOWER(:city)
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"city": city_name})
        rows = result.fetchall()
        cols = result.keys()
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=cols)
    return df.iloc[0].to_dict()


@st.cache_data(ttl=3600, show_spinner=False)
def get_state_stats(state_name: str) -> dict:
    """
    Get aggregate statistics for a state.
    
    Args:
        state_name: State name (case-insensitive)
    
    Returns:
        dict with aggregate stats or None if state not found
    """
    engine = get_engine()
    
    query = text("""
        SELECT 
            state,
            COUNT(*) as city_count,
            SUM(population) as total_population,
            AVG(population) as avg_population,
            MIN(population) as min_population,
            MAX(population) as max_population,
            AVG(median_age) as avg_median_age,
            AVG(avg_household_size) as avg_household_size
        FROM dbo.cities
        WHERE LOWER(state) = LOWER(:state)
        GROUP BY state
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"state": state_name})
        rows = result.fetchall()
        cols = result.keys()
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=cols)
    return df.iloc[0].to_dict()


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_cities_in_state(state_name: str, limit: int = 10) -> pd.DataFrame:
    """
    Get top cities by population in a state.
    
    Args:
        state_name: State name (case-insensitive)
        limit: Maximum number of cities to return
    
    Returns:
        DataFrame sorted by population descending
    """
    engine = get_engine()
    
    query = text("""
        SELECT TOP(:limit)
            city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE LOWER(state) = LOWER(:state)
        ORDER BY population DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"state": state_name, "limit": limit})
        rows = result.fetchall()
        cols = result.keys()
    
    return pd.DataFrame(rows, columns=cols)


 
# US STATES CONSTANT (Single Source of Truth)
 
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming"
]

US_STATES_LOWER = [s.lower() for s in US_STATES]


def is_valid_state(state_name: str) -> bool:
    """Check if a string is a valid US state name."""
    return state_name.lower() in US_STATES_LOWER


def normalize_state_name(state_name: str) -> str:
    """Convert state name to proper title case."""
    lower = state_name.lower()
    if lower in US_STATES_LOWER:
        return US_STATES[US_STATES_LOWER.index(lower)]
    return state_name.title()
