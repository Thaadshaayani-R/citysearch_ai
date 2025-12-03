"""
CitySearch AI - Query Handlers
===============================
Routes classified queries to appropriate core modules and display components.

This module integrates:
- Rule-based + LLM hybrid classification
- All core modules (semantic search, RAG, ML ranking, etc.)
- Display components for UI rendering
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text

# Configuration
from config import DB_TABLE_NAME

# Database
from db_config import get_engine

# Core Modules - YOUR EXISTING CODE
from core.intent_classifier import classify_query_intent
from core.semantic_search import semantic_city_search
from core.lifestyle_rag_v2 import try_build_lifestyle_card
from core.ml_router import run_family_ranking, run_young_ranking, run_retirement_ranking, run_single_city_prediction
from core.ml_explain import explain_ml_results
from core.score_translate import to_level
from core.query_router import build_sql_with_fallback
from core.cluster_router import get_cluster_for_city, get_all_clusters
from core.cluster_explain import explain_cluster
from core.cluster_labels import CLUSTER_LABELS
from core.rag_search import search_city_rag

# Display Components
from display_components import (
    show_city_profile_card, show_single_metric_card, show_state_metric_card,
    show_aggregate_card, show_superlative_card, show_city_comparison,
    show_state_comparison, show_recommendation_card, show_cluster_scatter,
    show_lifestyle_card, show_city_table, show_ai_response, show_out_of_scope,
    generate_ai_summary
)

# Utilities
from utils import (
    extract_single_city_fuzzy, extract_two_cities_fuzzy,
    extract_single_state_fuzzy, extract_two_states_fuzzy,
    fuzzy_match_city, fuzzy_match_state, format_population
)


# =============================================================================
# MAIN QUERY HANDLER
# =============================================================================

def handle_query(query: str, classification: dict, df_features: pd.DataFrame, 
                 get_engine_func=None, smart_route_func=None, 
                 lifestyle_rag_func=None, classify_intent_func=None):
    """
    Main query handler - routes to appropriate handler based on classification.
    
    Args:
        query: User's query string
        classification: Dict from hybrid_classifier with response_type, etc.
        df_features: DataFrame with city data
        get_engine_func: Function to get database engine
        smart_route_func: smart_route function (optional)
        lifestyle_rag_func: try_build_lifestyle_card function (optional)
        classify_intent_func: classify_query_intent function (optional)
    """
    
    if get_engine_func is None:
        get_engine_func = get_engine
    
    # Get city list for fuzzy matching
    city_list = df_features["city"].unique().tolist() if not df_features.empty else []
    
    # -------------------------------------------------------------------------
    # CHECK: Is this city-related?
    # -------------------------------------------------------------------------
    if not classification.get("is_city_related", True):
        show_out_of_scope()
        return
    
    # -------------------------------------------------------------------------
    # CHECK: Should we use GPT general knowledge?
    # -------------------------------------------------------------------------
    if classification.get("use_gpt_knowledge", False):
        handle_gpt_knowledge_fallback(query)
        return
    
    # -------------------------------------------------------------------------
    # ROUTE: Based on response_type or original_mode
    # -------------------------------------------------------------------------
    response_type = classification.get("response_type", "city_list")
    original_mode = classification.get("original_mode", response_type)
    
    # Map original modes to handlers
    mode_handlers = {
        # Rule-based modes (from your intent_classifier)
        "sql": handle_sql_query,
        "semantic": handle_semantic_search,
        "hybrid": handle_hybrid_search,
        "ml_family": handle_ml_family,
        "ml_young": handle_ml_young,
        "ml_retirement": handle_ml_retirement,
        "ml_compare_cities": handle_city_comparison,
        "ml_single_city": handle_single_city_score,
        
        # LLM response types
        "sql_query": handle_sql_query,
        "semantic_search": handle_semantic_search,
        "hybrid_search": handle_hybrid_search,
        "single_city": handle_single_city,
        "single_state": handle_single_state,
        "city_list": handle_city_list,
        "state_list": handle_state_list,
        "comparison": handle_comparison,
        "city_profile": handle_city_profile,
        "state_profile": handle_state_profile,
        "aggregate": handle_aggregate,
        "recommendation": handle_recommendation,
        "cluster": handle_cluster,
        "similar_cities": handle_similar_cities,
        "lifestyle": handle_lifestyle,
        "general_question": handle_gpt_knowledge_fallback,
    }
    
    # Try original_mode first, then response_type
    handler = mode_handlers.get(original_mode) or mode_handlers.get(response_type)
    
    if handler:
        handler(query, classification, df_features, get_engine_func, city_list)
    else:
        # Default fallback
        handle_sql_query(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# HANDLER: SQL Query (Rule-Based + GPT Fallback)
# =============================================================================

def handle_sql_query(query: str, classification: dict, df_features: pd.DataFrame,
                     get_engine_func, city_list: list):
    """
    Handle SQL queries using your query_router.py
    Rule-based first, GPT fallback if needed.
    """
    
    try:
        # Use YOUR query router (rule-based → GPT fallback)
        sql = build_sql_with_fallback(query, use_gpt=True)
        
        # Execute SQL
        engine = get_engine_func()
        df = pd.read_sql(sql, engine)
        
        if df.empty:
            st.warning("No results found for your query.")
            return
        
        # Determine how to display
        if len(df) == 1:
            # Single result
            row = df.iloc[0]
            city_name = row.get("city", "Unknown")
            state_name = row.get("state", "")
            
            metric = classification.get("metric")
            if metric and metric != "all":
                # Single metric display
                metric_value = row.get(metric, "N/A")
                show_single_metric_card(city_name, state_name, metric, metric_value, row)
            else:
                # Full profile
                show_city_profile_card(city_name, state_name, row)
        else:
            # Multiple results - show table
            title = "Query Results"
            
            # Check for superlative (highest/lowest)
            sort_dir = classification.get("sort_direction")
            if sort_dir:
                metric = classification.get("metric", "population")
                top_city = df.iloc[0]
                runners_up = df.iloc[1:4] if len(df) > 1 else pd.DataFrame()
                
                rank_label = f"City with {'Highest' if sort_dir == 'highest' else 'Lowest'} {metric.replace('_', ' ').title()}"
                show_superlative_card(
                    top_city.get("city", "Unknown"),
                    top_city.get("state", ""),
                    top_city.get(metric, "N/A"),
                    rank_label,
                    runners_up,
                    insights=None
                )
                
                # Show remaining results
                if len(df) > 4:
                    show_city_table(df, "All Results", show_download=True)
            else:
                show_city_table(df, title, show_download=True)
        
        # Add ML insights if applicable
        if classification.get("needs_ai_summary", False) and len(df) > 1:
            try:
                insights = explain_ml_results(query, df)
                if insights:
                    st.markdown("### 🧠 AI Insights")
                    st.markdown(insights)
            except Exception:
                pass
                
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")


# =============================================================================
# HANDLER: Semantic Search (Vector Embeddings)
# =============================================================================

def handle_semantic_search(query: str, classification: dict, df_features: pd.DataFrame,
                           get_engine_func, city_list: list):
    """
    Handle semantic search using your semantic_search.py
    Vector embeddings to find similar/matching cities.
    """
    
    state_filter = classification.get("state_filter")
    
    try:
        # Use YOUR semantic search
        results = semantic_city_search(query, top_k=10, state_filter=state_filter)
        
        if not results:
            st.warning("No matching cities found.")
            return
        
        # Convert results to DataFrame
        city_data = []
        engine = get_engine_func()
        
        for city, state in results:
            # Fetch full city data
            sql = text(f"""
                SELECT city, state, state_code, population, median_age, avg_household_size
                FROM {DB_TABLE_NAME}
                WHERE LOWER(city) = LOWER(:city) AND LOWER(state) = LOWER(:state)
            """)
            with engine.connect() as conn:
                result = conn.execute(sql, {"city": city, "state": state})
                row = result.fetchone()
                if row:
                    city_data.append(dict(row._mapping))
        
        if city_data:
            df = pd.DataFrame(city_data)
            
            # Show results
            st.markdown("### 🔍 Semantic Search Results")
            st.markdown(f"*Cities matching: \"{query}\"*")
            
            show_city_table(df, "Matching Cities", show_download=True)
            
            # Try to get RAG context for top result
            if city_data:
                top_city = city_data[0]
                try:
                    rag_context = search_city_rag(top_city["city"])
                    if rag_context:
                        st.markdown("### 📚 About " + top_city["city"])
                        st.markdown(rag_context)
                except Exception:
                    pass
        else:
            st.warning("Could not fetch details for matching cities.")
            
    except Exception as e:
        st.error(f"Semantic search error: {str(e)}")


# =============================================================================
# HANDLER: Hybrid Search (Semantic + State Filter)
# =============================================================================

def handle_hybrid_search(query: str, classification: dict, df_features: pd.DataFrame,
                         get_engine_func, city_list: list):
    """
    Handle hybrid search - semantic search with state filter.
    """
    # Hybrid search is just semantic search with state filter
    handle_semantic_search(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# HANDLER: ML Family Ranking
# =============================================================================

def handle_ml_family(query: str, classification: dict, df_features: pd.DataFrame,
                     get_engine_func, city_list: list):
    """
    Handle ML family ranking using your ml_router.py
    """
    
    state_filter = classification.get("state_filter")
    
    try:
        # Use YOUR ML ranking
        df = run_family_ranking(state=state_filter)
        
        if df.empty:
            st.warning("No results found.")
            return
        
        # Show top city as recommendation card
        top_city = df.iloc[0]
        show_recommendation_card(top_city, "families", df)
        
        # Add ML explanation
        try:
            insights = explain_ml_results(query, df)
            if insights:
                st.markdown("### 🧠 Why These Cities?")
                st.markdown(insights)
        except Exception:
            pass
        
        # Show full results table with scores
        st.markdown("### 📊 All Ranked Cities")
        
        # Add score level translation
        if "score" in df.columns:
            df["Rating"] = df["score"].apply(to_level)
        
        show_city_table(df, "Family-Friendly Cities", show_download=True)
        
    except Exception as e:
        st.error(f"ML ranking error: {str(e)}")
        # Fallback to basic display
        _handle_recommendation_fallback(query, "families", df_features, get_engine_func)


# =============================================================================
# HANDLER: ML Young Professionals Ranking
# =============================================================================

def handle_ml_young(query: str, classification: dict, df_features: pd.DataFrame,
                    get_engine_func, city_list: list):
    """
    Handle ML young professionals ranking using your ml_router.py
    """
    
    state_filter = classification.get("state_filter")
    
    try:
        # Use YOUR ML ranking
        df = run_young_ranking(state=state_filter)
        
        if df.empty:
            st.warning("No results found.")
            return
        
        # Show top city as recommendation card
        top_city = df.iloc[0]
        show_recommendation_card(top_city, "young_professionals", df)
        
        # Add ML explanation
        try:
            insights = explain_ml_results(query, df)
            if insights:
                st.markdown("### 🧠 Why These Cities?")
                st.markdown(insights)
        except Exception:
            pass
        
        # Show full results
        st.markdown("### 📊 All Ranked Cities")
        
        if "score" in df.columns:
            df["Rating"] = df["score"].apply(to_level)
        
        show_city_table(df, "Cities for Young Professionals", show_download=True)
        
    except Exception as e:
        st.error(f"ML ranking error: {str(e)}")
        _handle_recommendation_fallback(query, "young_professionals", df_features, get_engine_func)


# =============================================================================
# HANDLER: ML Retirement Ranking
# =============================================================================

def handle_ml_retirement(query: str, classification: dict, df_features: pd.DataFrame,
                         get_engine_func, city_list: list):
    """
    Handle ML retirement ranking using your ml_router.py
    """
    
    state_filter = classification.get("state_filter")
    
    try:
        # Use YOUR ML ranking
        df = run_retirement_ranking(state=state_filter)
        
        if df.empty:
            st.warning("No results found.")
            return
        
        # Show top city as recommendation card
        top_city = df.iloc[0]
        show_recommendation_card(top_city, "retirement", df)
        
        # Add ML explanation
        try:
            insights = explain_ml_results(query, df)
            if insights:
                st.markdown("### 🧠 Why These Cities?")
                st.markdown(insights)
        except Exception:
            pass
        
        # Show full results
        st.markdown("### 📊 All Ranked Cities")
        
        if "score" in df.columns:
            df["Rating"] = df["score"].apply(to_level)
        
        show_city_table(df, "Retirement Cities", show_download=True)
        
    except Exception as e:
        st.error(f"ML ranking error: {str(e)}")
        _handle_recommendation_fallback(query, "retirement", df_features, get_engine_func)


# =============================================================================
# HANDLER: Recommendation (Routes to appropriate ML handler)
# =============================================================================

def handle_recommendation(query: str, classification: dict, df_features: pd.DataFrame,
                          get_engine_func, city_list: list):
    """
    Route recommendation queries to appropriate ML handler.
    """
    
    specific_intent = classification.get("specific_intent", "general")
    
    if specific_intent == "families":
        handle_ml_family(query, classification, df_features, get_engine_func, city_list)
    elif specific_intent == "young_professionals":
        handle_ml_young(query, classification, df_features, get_engine_func, city_list)
    elif specific_intent == "retirement":
        handle_ml_retirement(query, classification, df_features, get_engine_func, city_list)
    else:
        # General recommendation - use family as default
        handle_ml_family(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# HANDLER: Single City
# =============================================================================

def handle_single_city(query: str, classification: dict, df_features: pd.DataFrame,
                       get_engine_func, city_list: list):
    """
    Handle single city queries.
    """
    
    # Extract city from query or classification
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if mentioned_cities:
        city_name = mentioned_cities[0]
    else:
        city_name, _ = extract_single_city_fuzzy(query, city_list)
    
    if not city_name:
        # Check for superlative (highest/lowest)
        sort_dir = classification.get("sort_direction")
        if sort_dir:
            _handle_superlative(query, classification, df_features, get_engine_func)
            return
        
        st.warning("Could not identify which city you're asking about.")
        return
    
    # Fetch city data
    engine = get_engine_func()
    sql = text(f"""
        SELECT * FROM {DB_TABLE_NAME}
        WHERE LOWER(city) = LOWER(:city)
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"city": city_name})
            row = result.fetchone()
        
        if row:
            row_dict = dict(row._mapping)
            state_name = row_dict.get("state", "")
            metric = classification.get("metric")
            
            if metric and metric != "all":
                metric_value = row_dict.get(metric, "N/A")
                show_single_metric_card(city_name, state_name, metric, metric_value, pd.Series(row_dict))
            else:
                show_city_profile_card(city_name, state_name, pd.Series(row_dict))
        else:
            st.warning(f"City '{city_name}' not found in database.")
            
    except Exception as e:
        st.error(f"Error fetching city data: {str(e)}")


# =============================================================================
# HANDLER: Single City Score
# =============================================================================

def handle_single_city_score(query: str, classification: dict, df_features: pd.DataFrame,
                             get_engine_func, city_list: list):
    """
    Handle single city ML score prediction.
    """
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if mentioned_cities:
        city_name = mentioned_cities[0]
    else:
        city_name, _ = extract_single_city_fuzzy(query, city_list)
    
    if not city_name:
        st.warning("Could not identify which city you're asking about.")
        return
    
    try:
        # Use YOUR single city prediction
        scores = run_single_city_prediction(city_name)
        
        if scores:
            st.markdown(f"### 🎯 ML Scores for {city_name}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                family_score = scores.get("family", 0)
                st.metric("👨‍👩‍👧 Family Score", f"{family_score:.1f}", to_level(family_score))
            
            with col2:
                young_score = scores.get("young", 0)
                st.metric("💼 Young Pro Score", f"{young_score:.1f}", to_level(young_score))
            
            with col3:
                retire_score = scores.get("retirement", 0)
                st.metric("🏖️ Retirement Score", f"{retire_score:.1f}", to_level(retire_score))
        else:
            st.warning(f"Could not calculate scores for {city_name}.")
            
    except Exception as e:
        st.error(f"Error calculating scores: {str(e)}")


# =============================================================================
# HANDLER: Single State
# =============================================================================

def handle_single_state(query: str, classification: dict, df_features: pd.DataFrame,
                        get_engine_func, city_list: list):
    """
    Handle single state queries.
    """
    
    mentioned_states = classification.get("mentioned_states", [])
    
    if mentioned_states:
        state_name = mentioned_states[0]
    else:
        state_name = extract_single_state_fuzzy(query)
    
    if not state_name:
        st.warning("Could not identify which state you're asking about.")
        return
    
    # Fetch state data
    engine = get_engine_func()
    sql = text(f"""
        SELECT 
            state,
            COUNT(*) as city_count,
            SUM(population) as total_population,
            AVG(median_age) as avg_median_age,
            AVG(avg_household_size) as avg_household_size
        FROM {DB_TABLE_NAME}
        WHERE LOWER(state) = LOWER(:state)
        GROUP BY state
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"state": state_name})
            row = result.fetchone()
        
        if row:
            row_dict = dict(row._mapping)
            metric = classification.get("metric", "population")
            
            # Get metric value
            if metric == "population":
                metric_value = row_dict.get("total_population", 0)
            elif metric == "median_age":
                metric_value = row_dict.get("avg_median_age", 0)
            else:
                metric_value = row_dict.get("total_population", 0)
            
            # Fetch top cities in state
            sql_cities = text(f"""
                SELECT TOP 5 city, state, population, median_age, avg_household_size
                FROM {DB_TABLE_NAME}
                WHERE LOWER(state) = LOWER(:state)
                ORDER BY population DESC
            """)
            
            with engine.connect() as conn:
                cities_result = conn.execute(sql_cities, {"state": state_name})
                cities_df = pd.DataFrame(cities_result.fetchall(), columns=cities_result.keys())
            
            show_state_metric_card(
                state_name, 
                metric.replace("_", " ").title(), 
                metric_value, 
                row_dict.get("city_count", 0),
                cities_df
            )
        else:
            st.warning(f"State '{state_name}' not found in database.")
            
    except Exception as e:
        st.error(f"Error fetching state data: {str(e)}")


# =============================================================================
# HANDLER: City List
# =============================================================================

def handle_city_list(query: str, classification: dict, df_features: pd.DataFrame,
                     get_engine_func, city_list: list):
    """
    Handle city list queries (top N, filters, etc.)
    """
    
    # Use SQL query handler for lists
    handle_sql_query(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# HANDLER: State List
# =============================================================================

def handle_state_list(query: str, classification: dict, df_features: pd.DataFrame,
                      get_engine_func, city_list: list):
    """
    Handle state list queries.
    """
    
    engine = get_engine_func()
    
    sql = text(f"""
        SELECT 
            state,
            COUNT(*) as city_count,
            SUM(population) as total_population,
            AVG(median_age) as avg_median_age
        FROM {DB_TABLE_NAME}
        GROUP BY state
        ORDER BY total_population DESC
    """)
    
    try:
        df = pd.read_sql(sql, engine)
        
        if df.empty:
            st.warning("No state data found.")
            return
        
        show_city_table(df, "States Overview", show_download=True)
        
    except Exception as e:
        st.error(f"Error fetching state list: {str(e)}")


# =============================================================================
# HANDLER: Comparison
# =============================================================================

def handle_comparison(query: str, classification: dict, df_features: pd.DataFrame,
                      get_engine_func, city_list: list):
    """
    Handle comparison queries (city vs city, state vs state).
    """
    
    comparison_type = classification.get("comparison_type", "city_vs_city")
    
    if comparison_type == "state_vs_state":
        handle_state_comparison(query, classification, df_features, get_engine_func, city_list)
    else:
        handle_city_comparison(query, classification, df_features, get_engine_func, city_list)


def handle_city_comparison(query: str, classification: dict, df_features: pd.DataFrame,
                           get_engine_func, city_list: list):
    """
    Handle city vs city comparison.
    """
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if len(mentioned_cities) >= 2:
        city1, city2 = mentioned_cities[0], mentioned_cities[1]
    else:
        result = extract_two_cities_fuzzy(query, city_list)
        if result and len(result) >= 2:
            city1, city2 = result[0], result[1]
        else:
            st.warning("Could not identify two cities to compare. Please mention both city names.")
            return
    
    # Fetch city data
    engine = get_engine_func()
    
    sql = text(f"""
        SELECT * FROM {DB_TABLE_NAME}
        WHERE LOWER(city) IN (LOWER(:city1), LOWER(:city2))
    """)
    
    try:
        df = pd.read_sql(sql, engine, params={"city1": city1, "city2": city2})
        
        if len(df) < 2:
            st.warning(f"Could not find both cities in database. Found: {list(df['city'])}")
            return
        
        city1_row = df[df["city"].str.lower() == city1.lower()].iloc[0]
        city2_row = df[df["city"].str.lower() == city2.lower()].iloc[0]
        
        show_city_comparison(city1_row, city2_row, query)
        
    except Exception as e:
        st.error(f"Error comparing cities: {str(e)}")


def handle_state_comparison(query: str, classification: dict, df_features: pd.DataFrame,
                            get_engine_func, city_list: list):
    """
    Handle state vs state comparison.
    """
    
    mentioned_states = classification.get("mentioned_states", [])
    
    if len(mentioned_states) >= 2:
        state1, state2 = mentioned_states[0], mentioned_states[1]
    else:
        result = extract_two_states_fuzzy(query)
        if result and len(result) >= 2:
            state1, state2 = result[0], result[1]
        else:
            st.warning("Could not identify two states to compare. Please mention both state names.")
            return
    
    engine = get_engine_func()
    
    # Get stats for both states
    def get_state_stats(state):
        sql = text(f"""
            SELECT 
                state,
                COUNT(*) as city_count,
                SUM(population) as total_population,
                AVG(population) as avg_population,
                AVG(median_age) as avg_median_age,
                AVG(avg_household_size) as avg_household_size
            FROM {DB_TABLE_NAME}
            WHERE LOWER(state) = LOWER(:state)
            GROUP BY state
        """)
        with engine.connect() as conn:
            result = conn.execute(sql, {"state": state})
            row = result.fetchone()
            return dict(row._mapping) if row else {}
    
    def get_top_cities(state, limit=5):
        sql = text(f"""
            SELECT TOP {limit} city, state, population, median_age, avg_household_size
            FROM {DB_TABLE_NAME}
            WHERE LOWER(state) = LOWER(:state)
            ORDER BY population DESC
        """)
        return pd.read_sql(sql, engine, params={"state": state})
    
    try:
        stats1 = get_state_stats(state1)
        stats2 = get_state_stats(state2)
        cities1 = get_top_cities(state1)
        cities2 = get_top_cities(state2)
        
        if not stats1 or not stats2:
            st.warning("Could not find both states in database.")
            return
        
        show_state_comparison(state1, stats1, cities1, state2, stats2, cities2, query)
        
    except Exception as e:
        st.error(f"Error comparing states: {str(e)}")


# =============================================================================
# HANDLER: City Profile
# =============================================================================

def handle_city_profile(query: str, classification: dict, df_features: pd.DataFrame,
                        get_engine_func, city_list: list):
    """
    Handle city profile queries using RAG.
    """
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if mentioned_cities:
        city_name = mentioned_cities[0]
    else:
        city_name, _ = extract_single_city_fuzzy(query, city_list)
    
    if not city_name:
        st.warning("Could not identify which city you're asking about.")
        return
    
    # Fetch city data
    engine = get_engine_func()
    sql = text(f"""
        SELECT * FROM {DB_TABLE_NAME}
        WHERE LOWER(city) = LOWER(:city)
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"city": city_name})
            row = result.fetchone()
        
        if row:
            row_dict = dict(row._mapping)
            state_name = row_dict.get("state", "")
            
            # Show profile card
            show_city_profile_card(city_name, state_name, pd.Series(row_dict))
            
            # Try to get RAG context
            try:
                rag_context = search_city_rag(city_name)
                if rag_context:
                    st.markdown("### 📚 More About " + city_name)
                    st.markdown(rag_context)
            except Exception:
                pass
        else:
            st.warning(f"City '{city_name}' not found in database.")
            
    except Exception as e:
        st.error(f"Error fetching city profile: {str(e)}")


# =============================================================================
# HANDLER: State Profile
# =============================================================================

def handle_state_profile(query: str, classification: dict, df_features: pd.DataFrame,
                         get_engine_func, city_list: list):
    """
    Handle state profile queries.
    """
    handle_single_state(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# HANDLER: Aggregate
# =============================================================================

def handle_aggregate(query: str, classification: dict, df_features: pd.DataFrame,
                     get_engine_func, city_list: list):
    """
    Handle aggregate queries (count, sum, average).
    """
    
    state_filter = classification.get("state_filter") or (
        classification.get("mentioned_states", [None])[0]
    )
    
    engine = get_engine_func()
    q_lower = query.lower()
    
    try:
        if "how many" in q_lower or "count" in q_lower:
            # Count query
            if state_filter:
                sql = text(f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
                with engine.connect() as conn:
                    result = conn.execute(sql, {"state": state_filter}).fetchone()
                value = result[0] if result else 0
                show_aggregate_card(f"Cities in {state_filter}", str(value), "cities in database")
            else:
                sql = text(f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME}")
                with engine.connect() as conn:
                    result = conn.execute(sql).fetchone()
                value = result[0] if result else 0
                show_aggregate_card("Total Cities", str(value), "in our database")
        
        elif "total" in q_lower or "sum" in q_lower:
            # Sum query (population)
            if state_filter:
                sql = text(f"SELECT SUM(population) as total FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
                with engine.connect() as conn:
                    result = conn.execute(sql, {"state": state_filter}).fetchone()
                value = result[0] if result else 0
                show_aggregate_card(f"Total Population in {state_filter}", format_population(value), "people")
            else:
                sql = text(f"SELECT SUM(population) as total FROM {DB_TABLE_NAME}")
                with engine.connect() as conn:
                    result = conn.execute(sql).fetchone()
                value = result[0] if result else 0
                show_aggregate_card("Total Population", format_population(value), "in all cities")
        
        elif "average" in q_lower or "avg" in q_lower:
            # Average query
            metric = classification.get("metric", "population")
            if state_filter:
                sql = text(f"SELECT AVG({metric}) as average FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
                with engine.connect() as conn:
                    result = conn.execute(sql, {"state": state_filter}).fetchone()
                value = result[0] if result else 0
                label = f"Average {metric.replace('_', ' ').title()} in {state_filter}"
            else:
                sql = text(f"SELECT AVG({metric}) as average FROM {DB_TABLE_NAME}")
                with engine.connect() as conn:
                    result = conn.execute(sql).fetchone()
                value = result[0] if result else 0
                label = f"Average {metric.replace('_', ' ').title()}"
            
            formatted = f"{value:,.2f}" if value else "N/A"
            show_aggregate_card(label, formatted, "")
        
        else:
            # Default to count
            sql = text(f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME}")
            with engine.connect() as conn:
                result = conn.execute(sql).fetchone()
            value = result[0] if result else 0
            show_aggregate_card("Total Cities", str(value), "in our database")
            
    except Exception as e:
        st.error(f"Error executing aggregate query: {str(e)}")


# =============================================================================
# HANDLER: Cluster
# =============================================================================

def handle_cluster(query: str, classification: dict, df_features: pd.DataFrame,
                   get_engine_func, city_list: list):
    """
    Handle cluster queries using your cluster modules.
    """
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if mentioned_cities:
        city_name = mentioned_cities[0]
    else:
        city_name, _ = extract_single_city_fuzzy(query, city_list)
    
    if city_name:
        # Get cluster for specific city
        try:
            cluster_id = get_cluster_for_city(city_name)
            
            if cluster_id is not None:
                # Get cluster explanation
                cluster_info = explain_cluster(cluster_id, return_dict=True)
                
                st.markdown(f"### 🏙️ {city_name} - Cluster Analysis")
                
                cluster_name = cluster_info.get("cluster_name", f"Cluster {cluster_id}")
                st.markdown(f"**Cluster:** {cluster_name}")
                
                if "detailed_summary" in cluster_info:
                    st.markdown(cluster_info["detailed_summary"])
                
                if "best_for" in cluster_info:
                    st.markdown(f"**Best For:** {cluster_info['best_for']}")
            else:
                st.warning(f"Could not determine cluster for {city_name}.")
                
        except Exception as e:
            st.error(f"Error getting cluster info: {str(e)}")
    else:
        # Show all clusters
        try:
            clusters_df = get_all_clusters()
            
            if not clusters_df.empty:
                st.markdown("### 🏙️ City Clusters")
                show_cluster_scatter(clusters_df)
                
                # Show cluster descriptions
                st.markdown("### 📋 Cluster Descriptions")
                for cluster_id, label in CLUSTER_LABELS.items():
                    with st.expander(f"Cluster {cluster_id}: {label}"):
                        info = explain_cluster(cluster_id, return_dict=True)
                        if "detailed_summary" in info:
                            st.markdown(info["detailed_summary"])
            else:
                st.warning("No cluster data available.")
                
        except Exception as e:
            st.error(f"Error loading clusters: {str(e)}")


# =============================================================================
# HANDLER: Similar Cities
# =============================================================================

def handle_similar_cities(query: str, classification: dict, df_features: pd.DataFrame,
                          get_engine_func, city_list: list):
    """
    Handle similar cities queries using semantic search or clustering.
    """
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if mentioned_cities:
        city_name = mentioned_cities[0]
    else:
        city_name, _ = extract_single_city_fuzzy(query, city_list)
    
    if not city_name:
        st.warning("Could not identify which city you want to find similar cities for.")
        return
    
    try:
        # Use semantic search to find similar cities
        results = semantic_city_search(f"cities like {city_name}", top_k=10)
        
        if results:
            # Filter out the original city
            results = [(c, s) for c, s in results if c.lower() != city_name.lower()]
            
            if results:
                # Fetch full data for similar cities
                engine = get_engine_func()
                city_data = []
                
                for city, state in results[:10]:
                    sql = text(f"""
                        SELECT city, state, population, median_age, avg_household_size
                        FROM {DB_TABLE_NAME}
                        WHERE LOWER(city) = LOWER(:city) AND LOWER(state) = LOWER(:state)
                    """)
                    with engine.connect() as conn:
                        row = conn.execute(sql, {"city": city, "state": state}).fetchone()
                        if row:
                            city_data.append(dict(row._mapping))
                
                if city_data:
                    df = pd.DataFrame(city_data)
                    
                    st.markdown(f"### 🔍 Cities Similar to {city_name}")
                    show_city_table(df, "Similar Cities", show_download=True)
                else:
                    st.warning("Could not fetch details for similar cities.")
            else:
                st.info(f"No cities similar to {city_name} found.")
        else:
            st.warning("Semantic search returned no results.")
            
    except Exception as e:
        st.error(f"Error finding similar cities: {str(e)}")


# =============================================================================
# HANDLER: Lifestyle
# =============================================================================

def handle_lifestyle(query: str, classification: dict, df_features: pd.DataFrame,
                     get_engine_func, city_list: list):
    """
    Handle lifestyle queries using your lifestyle_rag_v2.py
    """
    
    try:
        # Use YOUR lifestyle RAG
        result = try_build_lifestyle_card(query)
        
        if result:
            show_lifestyle_card(
                city=result.get("city", "Unknown"),
                state=result.get("state", ""),
                population=result.get("population", "N/A"),
                median_age=result.get("median_age", "N/A"),
                household_size=result.get("household_size", "N/A"),
                description=result.get("description", ""),
                ai_summary=result.get("ai_summary", "")
            )
        else:
            # Fallback to city profile
            mentioned_cities = classification.get("mentioned_cities", [])
            if mentioned_cities:
                classification["mentioned_cities"] = mentioned_cities
                handle_city_profile(query, classification, df_features, get_engine_func, city_list)
            else:
                st.warning("Could not generate lifestyle information for this query.")
                
    except Exception as e:
        st.error(f"Error generating lifestyle info: {str(e)}")
        # Fallback
        handle_city_profile(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# HANDLER: GPT Knowledge Fallback
# =============================================================================

def handle_gpt_knowledge_fallback(query: str, classification: dict = None, 
                                   df_features: pd.DataFrame = None,
                                   get_engine_func = None, city_list: list = None):
    """
    Handle queries that need GPT general knowledge.
    Shows disclaimer that this is from general knowledge, not database.
    """
    
    # Check if query is city-related
    from hybrid_classifier import is_city_related_query
    
    if not is_city_related_query(query):
        show_out_of_scope()
        return
    
    # Use GPT to answer
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""
        The user asked about US cities: "{query}"
        
        Provide a helpful, informative answer based on your general knowledge.
        Focus on factual information about US cities.
        Keep your response concise (2-3 paragraphs max).
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Show with disclaimer
        show_ai_response(answer, is_fallback=True)
        
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")


# =============================================================================
# HELPER: Handle Superlative (Highest/Lowest)
# =============================================================================

def _handle_superlative(query: str, classification: dict, df_features: pd.DataFrame,
                        get_engine_func):
    """
    Handle superlative queries (highest/lowest population, etc.)
    """
    
    metric = classification.get("metric", "population")
    sort_dir = classification.get("sort_direction", "highest")
    state_filter = classification.get("state_filter") or (
        classification.get("mentioned_states", [None])[0]
    )
    
    engine = get_engine_func()
    
    order = "DESC" if sort_dir == "highest" else "ASC"
    
    if state_filter:
        sql = text(f"""
            SELECT TOP 5 * FROM {DB_TABLE_NAME}
            WHERE LOWER(state) = LOWER(:state)
            ORDER BY {metric} {order}
        """)
        df = pd.read_sql(sql, engine, params={"state": state_filter})
    else:
        sql = text(f"""
            SELECT TOP 5 * FROM {DB_TABLE_NAME}
            ORDER BY {metric} {order}
        """)
        df = pd.read_sql(sql, engine)
    
    if df.empty:
        st.warning("No results found.")
        return
    
    top_city = df.iloc[0]
    runners_up = df.iloc[1:] if len(df) > 1 else pd.DataFrame()
    
    rank_label = f"{'Highest' if sort_dir == 'highest' else 'Lowest'} {metric.replace('_', ' ').title()}"
    if state_filter:
        rank_label += f" in {state_filter}"
    
    show_superlative_card(
        top_city.get("city", "Unknown"),
        top_city.get("state", ""),
        top_city.get(metric, "N/A"),
        rank_label,
        runners_up,
        insights=None
    )


# =============================================================================
# HELPER: Recommendation Fallback
# =============================================================================

def _handle_recommendation_fallback(query: str, intent: str, df_features: pd.DataFrame,
                                    get_engine_func):
    """
    Fallback recommendation using basic metrics when ML models unavailable.
    """
    
    if df_features.empty:
        st.warning("No data available for recommendations.")
        return
    
    df = df_features.copy()
    
    # Simple scoring based on intent
    if intent == "families":
        # Higher household size, moderate age
        df["score"] = (df["avg_household_size"] * 20 + 
                      (50 - abs(df["median_age"] - 35)) +
                      df["population"] / 100000)
    elif intent == "young_professionals":
        # Lower age, higher population
        df["score"] = ((45 - df["median_age"]) * 3 + 
                      df["population"] / 50000)
    elif intent == "retirement":
        # Higher age, lower household size
        df["score"] = (df["median_age"] * 2 + 
                      (4 - df["avg_household_size"]) * 10)
    else:
        df["score"] = df["population"] / 10000
    
    df = df.sort_values("score", ascending=False).head(10)
    
    top_city = df.iloc[0]
    show_recommendation_card(top_city, intent, df)
