"""
CitySearch AI - Query Handlers (Refactored Option 2: LLM + Light Fallback)
==========================================================================

Cleaner router, no over-detection, no unnecessary rule-based blocking,
LLM classification is always respected FIRST, then light validation fixes gaps.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text
from typing import Optional

from config import DB_TABLE_NAME
from db_config import get_engine

# -----------------------
#  IMPORT HELPERS
# -----------------------
from utils import (
    extract_single_city_fuzzy,
    extract_single_state_fuzzy,
    extract_two_cities_fuzzy,
    extract_two_states_fuzzy,
    fuzzy_match_city,
    fuzzy_match_state,
)

# Your existing UI components (DO NOT MODIFY)
from display_components import (
    show_city_profile_card, show_single_metric_card, show_state_metric_card,
    show_aggregate_card, show_superlative_card, show_city_comparison,
    show_state_comparison, show_recommendation_card, show_cluster_scatter,
    show_lifestyle_card, show_city_table, show_ai_response, show_out_of_scope,
    generate_ai_summary
)

# ML handlers kept as-is
from core.ml_router import (
    run_family_ranking, run_young_ranking, run_retirement_ranking
)

from core.query_router import build_sql_with_fallback
from core.semantic_search import semantic_city_search
from core.lifestyle_rag_v2 import try_build_lifestyle_card

from core.cluster_router import (
    cluster_all,
    cluster_by_state,
    cluster_single_city,
    cluster_similar_to
)

from core.cluster_explain import explain_cluster
from core.cluster_labels import CLUSTER_LABELS
from core.ml_explain import explain_ml_results


# ============================================================================
#  ✨ 1. LIGHTWEIGHT VALIDATION HELPERS
# ============================================================================

def detect_metric_fallback(query: str) -> Optional[str]:
    """Fallback metric extraction if classifier misses it."""
    q = query.lower()

    if "population" in q or "people" in q:
        return "population"

    if "median age" in q or "age" in q:
        return "median_age"

    if "household" in q or "family size" in q:
        return "avg_household_size"

    return None


def fallback_city(query: str, city_list: list) -> Optional[str]:
    """If classifier gave no cities, try fuzzy extract."""
    if not city_list:
        return None

    # simple fuzzy search
    city, score = extract_single_city_fuzzy(query, city_list)
    return city if city else None


def fallback_state(query: str) -> Optional[str]:
    """If classifier gave no states, use fuzzy."""
    state = extract_single_state_fuzzy(query)
    return state if state else None


# ============================================================================
#  ✨ 2. CLEAN CENTRAL ROUTER (OPTION 2)
# ============================================================================

def handle_query(
    query: str,
    classification: dict,
    df_features: pd.DataFrame,
    get_engine_func=None,
    smart_route_func=None,
    lifestyle_rag_func=None,
    classify_intent_func=None,
):
    """
    Clean router. LLM classifier is ALWAYS primary.
    Fallback logic only fills missing values, never overrides LLM intent.
    """

    if get_engine_func is None:
        get_engine_func = get_engine

    city_list = df_features["city"].unique().tolist() if not df_features.empty else []

    # -----------------------
    # Unpack classification
    # -----------------------
    query_type = classification.get("query_type")
    entities = classification.get("entities", {})
    cities = entities.get("cities", [])
    states = entities.get("states", [])
    metric = classification.get("metric")
    direction = classification.get("direction")
    limit = classification.get("limit", 10)
    intent = classification.get("intent")
    filter_info = classification.get("filter", {})

    # -----------------------
    # LIGHT FALLBACKS (FILL ONLY MISSING VALUES)
    # -----------------------
    if not cities:
        fc = fallback_city(query, city_list)
        if fc:
            cities = [fc]

    if not states:
        fs = fallback_state(query)
        if fs:
            states = [fs]

    if not metric:
        metric = detect_metric_fallback(query)

    # -----------------------
    # BEGIN ROUTING
    # -----------------------
    try:

        # 1) SINGLE CITY
        if query_type == "single_city":
            if cities:
                return handle_single_city_router(query, cities[0], df_features, get_engine_func)
            return st.warning("Could not detect city.")

        # 2) SINGLE STATE
        if query_type == "single_state":
            if states:
                return handle_state_router(query, states[0], get_engine_func)
            return st.warning("Could not detect state.")

        # 3) SUPERLATIVE
        if query_type == "superlative":
            return handle_superlative_router(
                query, metric or "population", direction or "highest", get_engine_func, limit
            )

        # 4) COMPARISON (city vs city or state vs state)
        if query_type == "comparison":
            return handle_comparison_router(query, classification, df_features, get_engine_func, city_list)

        # 5) ML RANKING (best for families / young / retirement)
        if query_type == "ranking":
            return handle_ml_ranking_router(query, classification, df_features, get_engine_func)

        # 6) LIFESTYLE QUERIES (“life in X”)
        if query_type == "lifestyle":
            if cities:
                return handle_lifestyle_router(query, cities[0], df_features, get_engine_func)
            return st.warning("City not detected for lifestyle.")

        # 7) FILTERED LIST (“cities in CA with population > 1M”)
        if query_type == "filtered_list":
            return handle_filtered_list_router(query, classification, df_features, get_engine_func, filter_info, limit)

        # 8) AGGREGATE (“how many cities in Texas?”)
        if query_type == "aggregate":
            return handle_aggregate_router(query, classification, df_features, get_engine_func)

        # 9) FALLBACK → SQL or semantic
        return handle_sql_router(query, classification, df_features, get_engine_func, city_list)

    except Exception as e:
        st.error(f"Query error: {str(e)}")
        st.write("Showing fallback sample data:")
        if not df_features.empty:
            show_city_table(df_features.head(10), "Sample Cities", True)

# ============================================================================
#  ✨ SQL BUILDER (Unified)
# ============================================================================

def run_sql(engine, sql, params=None):
    """Safe SQL wrapper."""
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=cols)


def build_basic_query(metric="population", direction="DESC", limit=10, state=None):
    """Universal SQL builder for superlatives, listing, basic queries."""
    
    base = f"SELECT TOP {limit} * FROM {DB_TABLE_NAME}"
    
    if state:
        base += " WHERE LOWER(state) = LOWER(:state)"

    order = f" ORDER BY {metric} {direction}"
    return base + order

# ============================================================================
#  ✨ SINGLE CITY
# ============================================================================

def handle_single_city_router(query, city_name, df_features, get_engine_func):
    engine = get_engine_func()

    # Query exact city
    sql = f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city)=LOWER(:c)"
    df = run_sql(engine, sql, {"c": city_name})

    # Fuzzy fallback
    if df.empty:
        city_list = df_features["city"].tolist()
        match, score = fuzzy_match_city(city_name, city_list)
        if score > 70:
            df = run_sql(engine, sql, {"c": match})

    if df.empty:
        st.warning(f"City '{city_name}' not found.")
        return

    row = df.iloc[0]
    show_city_profile_card(row["city"], row["state"], row)

# ============================================================================
#  ✨ SINGLE STATE
# ============================================================================

def handle_state_router(query, state_name, get_engine_func):
    engine = get_engine_func()

    sql = f"""
        SELECT 
            state,
            COUNT(*) AS city_count,
            SUM(population) AS total_population,
            AVG(median_age) AS avg_median_age,
            AVG(avg_household_size) AS avg_household_size
        FROM {DB_TABLE_NAME}
        WHERE LOWER(state)=LOWER(:state)
        GROUP BY state
    """

    df = run_sql(engine, sql, {"state": state_name})
    if df.empty:
        st.warning(f"State '{state_name}' not found.")
        return

    row = df.iloc[0]
    total_pop = int(row["total_population"])
    city_count = int(row["city_count"])

    show_state_metric_card(
        state_name,
        "Population",
        total_pop,
        city_count,
        None  # you may add top cities
    )

# ============================================================================
#  ✨ SINGLE STATE
# ============================================================================

def handle_state_router(query, state_name, get_engine_func):
    engine = get_engine_func()

    sql = f"""
        SELECT 
            state,
            COUNT(*) AS city_count,
            SUM(population) AS total_population,
            AVG(median_age) AS avg_median_age,
            AVG(avg_household_size) AS avg_household_size
        FROM {DB_TABLE_NAME}
        WHERE LOWER(state)=LOWER(:state)
        GROUP BY state
    """

    df = run_sql(engine, sql, {"state": state_name})
    if df.empty:
        st.warning(f"State '{state_name}' not found.")
        return

    row = df.iloc[0]
    total_pop = int(row["total_population"])
    city_count = int(row["city_count"])

    show_state_metric_card(
        state_name,
        "Population",
        total_pop,
        city_count,
        None  # you may add top cities
    )

# ============================================================================
#  ✨ SUPERLATIVE
# ============================================================================

def handle_superlative_router(query, metric, direction, get_engine_func, limit):
    engine = get_engine_func()

    direction_sql = "DESC" if direction == "highest" else "ASC"
    metric_sql = metric

    sql = build_basic_query(metric_sql, direction_sql, limit)
    df = run_sql(engine, sql)

    if df.empty:
        st.warning("No results found.")
        return

    top = df.iloc[0]
    runners = df.iloc[1:5]

    label = f"{direction.capitalize()} {metric.replace('_',' ').title()}"
    show_superlative_card(
        top["city"], top["state"], top[metric_sql], label, runners, None
    )

# ============================================================================
#  ✨ AGGREGATE
# ============================================================================

def handle_aggregate_router(query, classification, df_features, get_engine_func):
    engine = get_engine_func()
    state = classification["entities"].get("states", [None])[0]

    if state:
        sql = f"SELECT COUNT(*) AS count FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s)"
        df = run_sql(engine, sql, {"s": state})
        show_aggregate_card(f"Cities in {state}", df.iloc[0]["count"], "cities")
    else:
        sql = f"SELECT COUNT(*) AS count FROM {DB_TABLE_NAME}"
        df = run_sql(engine, sql)
        show_aggregate_card("Total Cities", df.iloc[0]["count"], "cities")


# ============================================================================
#  ✨ COMPARISON
# ============================================================================

def handle_comparison_router(query, classification, df_features, get_engine_func, city_list):
    engine = get_engine_func()

    # CITY vs CITY
    if classification.get("comparison_type") == "city_vs_city":
        cities = classification["entities"].get("cities", [])

        # fallback fuzzy
        if len(cities) < 2:
            c1, c2 = extract_two_cities_fuzzy(query, city_list)
            cities = [c1, c2]

        sql = f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE LOWER(city) IN (LOWER(:c1), LOWER(:c2))
        """

        df = run_sql(engine, sql, {"c1": cities[0], "c2": cities[1]})
        if len(df) == 2:
            r1 = df.iloc[0]
            r2 = df.iloc[1]
            return show_city_comparison(r1, r2, query)

        st.warning("Could not compare cities.")
        return

    # STATE vs STATE
    states = classification["entities"].get("states", [])
    if len(states) < 2:
        s1, s2 = extract_two_states_fuzzy(query)
        states = [s1, s2]

    sql = f"""
        SELECT state, COUNT(*) AS cnt, SUM(population) AS pop, AVG(median_age) AS age
        FROM {DB_TABLE_NAME}
        WHERE LOWER(state) IN (LOWER(:s1), LOWER(:s2))
        GROUP BY state
    """

    df = run_sql(engine, sql, {"s1": states[0], "s2": states[1]})
    if len(df) == 2:
        show_state_comparison(states[0], df.iloc[0], states[1], df.iloc[1], query)
    else:
        st.warning("Could not compare states.")

# ============================================================================
#  ✨ ML RANKING
# ============================================================================

def handle_ml_ranking_router(query, classification, df_features, get_engine_func):
    intent = classification.get("intent")

    func_map = {
        "families": run_family_ranking,
        "young_professionals": run_young_ranking,
        "retirement": run_retirement_ranking,
    }

    func = func_map.get(intent)
    if not func:
        st.warning("Ranking intent missing.")
        return

    try:
        df = func()  # call ML function normally
    except:
        st.warning("ML failed; using fallback ranking.")
        df = df_features.copy()
        df["score"] = df["population"] / 10000
        df = df.sort_values("score", ascending=False).head(10)

    show_recommendation_card(df.iloc[0], intent, df, query)
    if explain_ml_results:
        insights = explain_ml_results(query, df)
        if insights:
            st.markdown("### Why These Cities?")
            st.markdown(insights)

    show_city_table(df, "Ranking Results", True)

# ============================================================================
#  ✨ LIFESTYLE
# ============================================================================

def handle_lifestyle_router(query, city, df_features, get_engine_func):
    r = try_build_lifestyle_card(query)
    if r:
        return show_lifestyle_card(
            r["city"], r["state"], r["population"],
            r["median_age"], r["household_size"],
            r["description"], r["ai_summary"]
        )

    return handle_single_city_router(query, city, df_features, get_engine_func)

# ============================================================================
#  ✨ FILTERED LIST
# ============================================================================

def handle_filtered_list_router(query, classification, df_features, get_engine_func, filter_info, limit):
    engine = get_engine_func()

    sql = f"SELECT * FROM {DB_TABLE_NAME} WHERE 1=1"
    params = {}

    # Filter state
    state = filter_info.get("state")
    if state:
        sql += " AND LOWER(state)=LOWER(:state)"
        params["state"] = state

    # Population filters
    if "min_population" in filter_info:
        sql += " AND population >= :min_pop"
        params["min_pop"] = filter_info["min_population"]

    if "max_population" in filter_info:
        sql += " AND population <= :max_pop"
        params["max_pop"] = filter_info["max_population"]

    sql += f" ORDER BY population DESC OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"

    df = run_sql(engine, sql, params)
    show_city_table(df, "Filtered Cities", True)

# ============================================================================
#  ✨ FALLBACK SQL ROUTER
# ============================================================================

def handle_sql_router(query, classification, df_features, get_engine_func, city_list):
    engine = get_engine_func()

    # Basic fallback: top 10 by population
    sql = build_basic_query(limit=10)
    df = run_sql(engine, sql)

    show_city_table(df, "Top Cities", True)


def handle_cluster(query, classification, df_features, get_engine_func, city_list):
    """Handle queries like 'which cluster is Denver in' or 'cluster of Austin'."""
    
    # Get city from classifier or fuzzy match
    cities = classification.get("entities", {}).get("cities", [])
    city = cities[0] if cities else extract_single_city_fuzzy(query, city_list)[0]
    
    if not city:
        st.warning("Could not identify the city for clustering.")
        return

    # Get cluster info using cluster_single_city()
    info = cluster_single_city(city)

    if not info:
        st.warning(f"No cluster information available for {city}.")
        return

    # Display cluster details
    st.markdown(f"### 🧠 Cluster for {info['city']}, {info['state']}")
    st.markdown(f"**Cluster:** {info['cluster_name']} (ID: {info['cluster']})")

    if info.get("cluster_summary"):
        st.markdown(f"**Summary:** {info['cluster_summary']}")

    # Show similar cities
    similar_df = cluster_similar_to(city)

    if similar_df is not None and not similar_df.empty:
        st.markdown("### 🔍 Cities similar to this one")
        show_city_table(similar_df, "Similar Cities", True)


