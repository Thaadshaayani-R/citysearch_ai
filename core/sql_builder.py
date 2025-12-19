# core/sql_builder.py (NEW - SECURE VERSION)
"""
Secure SQL query builder for CitySearch AI.
Uses parameterized queries to prevent SQL injection.

Usage:
    from core.sql_builder import build_safe_query, SafeQuery
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any


 
# CONSTANTS
 
VALID_COLUMNS = {
    "city", "state", "state_code",
    "population", "median_age", "avg_household_size"
}

VALID_SORT_COLUMNS = {
    "population", "median_age", "avg_household_size", "city", "state"
}

MAX_LIMIT = 100

US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming"
]


 
# SAFE QUERY DATACLASS
 
@dataclass
class SafeQuery:
    """
    Represents a parameterized SQL query.
    
    Attributes:
        sql: SQL string with :param placeholders
        params: Dictionary of parameter values
    """
    sql: str
    params: dict
    
    def __str__(self):
        return self.sql


 
# EXTRACTION HELPERS
 
def extract_limit(text: str) -> int:
    """Extract limit from query like 'top 10'."""
    match = re.search(r"(top|first)\s+(\d+)", text, re.I)
    if match:
        return min(int(match.group(2)), MAX_LIMIT)
    return 10


def extract_state(text: str) -> Optional[str]:
    """Extract and validate state name from query."""
    text_lower = text.lower()
    for state in US_STATES:
        if state in text_lower:
            return state.title()
    return None


def extract_city_pattern(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract city name pattern for LIKE queries.
    
    Returns:
        Tuple of (pattern_type, value) or None
        pattern_type: "starts_with", "ends_with", or "contains"
    """
    text_lower = text.lower()
    
    # "starting with X"
    match = re.search(r"starting with ['\"]?([a-zA-Z]+)['\"]?", text_lower)
    if match:
        return ("starts_with", match.group(1).strip())
    
    # "ending with X"
    match = re.search(r"ending with ['\"]?([a-zA-Z]+)['\"]?", text_lower)
    if match:
        return ("ends_with", match.group(1).strip())
    
    # "containing X"
    match = re.search(r"contain(?:ing|s)? ['\"]?([a-zA-Z]+)['\"]?", text_lower)
    if match:
        return ("contains", match.group(1).strip())
    
    return None


def extract_numeric_comparison(text: str) -> Optional[Tuple[str, str, int]]:
    """
    Extract numeric comparison from query.
    
    Returns:
        Tuple of (column, operator, value) or None
    """
    text_lower = text.lower()
    
    # Map column names
    column_map = {
        "population": "population",
        "median age": "median_age",
        "median_age": "median_age",
        "avg household size": "avg_household_size",
        "average household size": "avg_household_size",
        "household size": "avg_household_size",
    }
    
    # Patterns like "population > 1000000"
    pattern = r"(population|median[_ ]age|(?:avg|average)?\s*household[_ ]size)\s*(>=|<=|>|<|=)\s*(\d+)"
    match = re.search(pattern, text_lower)
    if match:
        col_text = match.group(1).strip()
        col = column_map.get(col_text.replace(" ", "_"), col_text.replace(" ", "_"))
        if col in VALID_COLUMNS:
            return (col, match.group(2), int(match.group(3)))
    
    # Patterns like "population over 1000000"
    pattern = r"(population|median[_ ]age)\s*(over|above|greater than|more than)\s*(\d+)"
    match = re.search(pattern, text_lower)
    if match:
        col_text = match.group(1).strip()
        col = column_map.get(col_text.replace(" ", "_"), col_text.replace(" ", "_"))
        if col in VALID_COLUMNS:
            return (col, ">", int(match.group(3)))
    
    return None


def detect_sort(text: str) -> Tuple[Optional[str], str]:
    """
    Detect sort column and direction.
    
    Returns:
        Tuple of (column, direction) where direction is "ASC" or "DESC"
    """
    text_lower = text.lower()
    
    # Direction
    if any(w in text_lower for w in ["smallest", "lowest", "least"]):
        direction = "ASC"
    else:
        direction = "DESC"
    
    # Column
    if "population" in text_lower:
        return ("population", direction)
    if "age" in text_lower:
        return ("median_age", direction)
    if "household" in text_lower:
        return ("avg_household_size", direction)
    
    return (None, direction)


 
# QUERY TYPE DETECTION
 
def is_count_query(text: str) -> bool:
    """Check if query asks for a count."""
    patterns = [r"how many", r"count of", r"number of", r"total cities"]
    return any(re.search(p, text, re.I) for p in patterns)


def is_percentage_query(text: str) -> bool:
    """Check if query asks for a percentage."""
    patterns = [r"percentage", r"percent", r"what %"]
    return any(re.search(p, text, re.I) for p in patterns)


def is_group_query(text: str) -> bool:
    """Check if query asks for grouping."""
    patterns = [r"by state", r"per state", r"group by", r"each state"]
    return any(re.search(p, text, re.I) for p in patterns)


 
# QUERY BUILDERS (Parameterized - SAFE)
 
def build_count_query(text: str) -> SafeQuery:
    """Build a COUNT query with parameters."""
    state = extract_state(text)
    comparison = extract_numeric_comparison(text)
    
    conditions = []
    params = {}
    
    if state:
        conditions.append("state = :state")
        params["state"] = state
    
    if comparison:
        col, op, val = comparison
        # Note: column names can't be parameterized, but they're validated
        if col in VALID_COLUMNS:
            conditions.append(f"{col} {op} :comp_val")
            params["comp_val"] = val
    
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    sql = f"""
        SELECT COUNT(*) AS total_cities
        FROM dbo.cities
        {where_clause}
    """.strip()
    
    return SafeQuery(sql=sql, params=params)


def build_text_search_query(pattern_type: str, value: str, limit: int) -> SafeQuery:
    """Build a LIKE query with parameters (SAFE!)."""
    params = {"limit": limit}
    
    # Build the LIKE pattern safely
    if pattern_type == "starts_with":
        params["pattern"] = f"{value}%"
    elif pattern_type == "ends_with":
        params["pattern"] = f"%{value}"
    else:  # contains
        params["pattern"] = f"%{value}%"
    
    sql = """
        SELECT TOP(:limit) city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE city LIKE :pattern
        ORDER BY population DESC
    """.strip()
    
    return SafeQuery(sql=sql, params=params)


def build_state_query(state: str, limit: int, sort_col: str = "population", sort_dir: str = "DESC") -> SafeQuery:
    """Build a state filter query with parameters."""
    # Validate sort column
    if sort_col not in VALID_SORT_COLUMNS:
        sort_col = "population"
    if sort_dir not in ("ASC", "DESC"):
        sort_dir = "DESC"
    
    sql = f"""
        SELECT TOP(:limit) city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE state = :state
        ORDER BY {sort_col} {sort_dir}
    """.strip()
    
    return SafeQuery(sql=sql, params={"limit": limit, "state": state})


def build_general_query(text: str) -> SafeQuery:
    """Build a general query based on natural language."""
    limit = extract_limit(text)
    state = extract_state(text)
    comparison = extract_numeric_comparison(text)
    sort_col, sort_dir = detect_sort(text)
    
    conditions = []
    params = {"limit": limit}
    
    if state:
        conditions.append("state = :state")
        params["state"] = state
    
    if comparison:
        col, op, val = comparison
        if col in VALID_COLUMNS:
            conditions.append(f"{col} {op} :comp_val")
            params["comp_val"] = val
    
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    # Default sort
    if not sort_col:
        sort_col = "population"
    
    sql = f"""
        SELECT TOP(:limit) city, state, population, median_age, avg_household_size
        FROM dbo.cities
        {where_clause}
        ORDER BY {sort_col} {sort_dir}
    """.strip()
    
    return SafeQuery(sql=sql, params=params)


 
# MAIN BUILDER
 
def build_safe_query(text: str) -> SafeQuery:
    """
    Build a safe, parameterized query from natural language.
    
    Args:
        text: Natural language query
    
    Returns:
        SafeQuery with SQL and parameters
    """
    text = text.strip()
    
    # Count queries
    if is_count_query(text):
        return build_count_query(text)
    
    # Text search queries
    pattern = extract_city_pattern(text)
    if pattern:
        pattern_type, value = pattern
        limit = extract_limit(text)
        return build_text_search_query(pattern_type, value, limit)
    
    # State-specific queries
    state = extract_state(text)
    if state and not is_group_query(text):
        limit = extract_limit(text)
        sort_col, sort_dir = detect_sort(text)
        return build_state_query(state, limit, sort_col or "population", sort_dir)
    
    # General queries
    return build_general_query(text)


 
# EXECUTION HELPER
 
def execute_safe_query(query: SafeQuery) -> "pd.DataFrame":
    """
    Execute a SafeQuery and return results as DataFrame.
    
    Args:
        query: SafeQuery object
    
    Returns:
        pandas DataFrame with results
    """
    import pandas as pd
    from sqlalchemy import text
    from db_config import get_engine
    
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(text(query.sql), query.params)
        rows = result.fetchall()
        cols = result.keys()
    
    return pd.DataFrame(rows, columns=cols)
