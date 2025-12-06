"""
CitySearch AI - Query Handlers
===============================
Routes classified queries to appropriate handlers based on LLM classification.

Architecture:
    User Query → LLM Classification → Query Handler → Display Component

Query Types Supported:
    - single_city: "Population of Denver"
    - single_state: "Population of Texas"
    - superlative: "Which city has highest population?"
    - comparison: "Compare Miami and Austin"
    - ranking: "Best cities for families"
    - aggregate: "How many cities in Texas?"
    - lifestyle: "Life in Dallas"
    - city_list: Default fallback
"""
import re
import streamlit as st
import pandas as pd
from sqlalchemy import text

from config import DB_TABLE_NAME
from db_config import get_engine


# =============================================================================
# SAFE IMPORTS WITH FALLBACKS
# =============================================================================

try:
    from core.intent_classifier import classify_query_intent
except ImportError:
    classify_query_intent = None

try:
    from core.semantic_search import semantic_city_search
except ImportError:
    semantic_city_search = None

try:
    from core.lifestyle_rag_v2 import try_build_lifestyle_card
except ImportError:
    try_build_lifestyle_card = None

try:
    from core.ml_router import run_family_ranking, run_young_ranking, run_retirement_ranking
except ImportError:
    run_family_ranking = None
    run_young_ranking = None
    run_retirement_ranking = None

try:
    from core.ml_explain import explain_ml_results
except ImportError:
    explain_ml_results = None

try:
    from core.score_translate import to_level
except ImportError:
    def to_level(score):
        if score is None:
            return "Unknown"
        try:
            score = float(score)
        except:
            return "Unknown"
        if score < 5:
            return "Low"
        elif score < 15:
            return "Medium"
        elif score < 25:
            return "High"
        return "Excellent"

try:
    from core.cluster_router import get_cluster_for_city, get_all_clusters
except ImportError:
    get_cluster_for_city = None
    get_all_clusters = None

try:
    from core.cluster_labels import CLUSTER_LABELS
except ImportError:
    CLUSTER_LABELS = {}

from display_components import (
    show_city_profile_card, show_single_metric_card, show_state_metric_card,
    show_aggregate_card, show_superlative_card, show_city_comparison,
    show_state_comparison, show_recommendation_card, show_cluster_scatter,
    show_lifestyle_card, show_city_table, show_ai_response, show_out_of_scope,
    generate_ai_summary
)

from utils import (
    extract_single_city_fuzzy, extract_two_cities_fuzzy,
    extract_single_state_fuzzy, extract_two_states_fuzzy,
    fuzzy_match_city, fuzzy_match_state, format_population
)


# =============================================================================
# MAIN QUERY HANDLER (ROUTER)
# =============================================================================

def handle_query(query, classification, df_features, get_engine_func=None, 
                 smart_route_func=None, lifestyle_rag_func=None, classify_intent_func=None):
    # Check if LLM generated SQL directly
    if classification.get("has_sql") and classification.get("sql"):
        handle_llm_sql_query(query, classification, get_engine_func)
        return
    """
    Main query router - routes queries to appropriate handlers based on LLM classification.
    
    Args:
        query: User's natural language query
        classification: Dict from hybrid_classifier with query_type, metric, etc.
        df_features: DataFrame with city features
        get_engine_func: Database engine function
    """
    if get_engine_func is None:
        get_engine_func = get_engine
    
    city_list = df_features["city"].unique().tolist() if not df_features.empty else []
    
    # Check if city-related
    if not classification.get("is_city_related", True):
        show_out_of_scope()
        return
    
    # Extract classification details
    query_type = classification.get("query_type", "city_list")
    metric = classification.get("metric")
    direction = classification.get("direction")
    limit = classification.get("limit", 10) or 10
    cities = classification.get("cities", [])
    states = classification.get("states", [])
    intent = classification.get("intent", "general")
    
    try:
        # Route based on query type
        if query_type == "single_city":
            _route_single_city(query, cities, metric, df_features, get_engine_func)
        
        elif query_type == "single_state":
            _route_single_state(query, states, metric, classification, df_features, get_engine_func, city_list)
        
        elif query_type == "superlative":
            handle_superlative_query(query, metric or "population", direction or "highest", get_engine_func, limit, states)
        
        elif query_type == "comparison":
            handle_comparison(query, classification, df_features, get_engine_func, city_list)
        
        elif query_type == "ranking":
            _route_ranking(query, intent, classification, df_features, get_engine_func, city_list)
        
        elif query_type == "aggregate":
            handle_sql_query(query, classification, df_features, get_engine_func, city_list)
        
        elif query_type == "lifestyle":
            city_name = cities[0] if cities else None
            if city_name:
                handle_lifestyle_query(query, city_name, classification, df_features, get_engine_func, city_list)
            else:
                handle_sql_query(query, classification, df_features, get_engine_func, city_list)
                
        elif query_type == "filter":
            handle_filter_query(query, classification, df_features, get_engine_func)

        elif query_type == "filter_range":
            handle_filter_range_query(query, classification, df_features, get_engine_func)
        
        elif query_type == "similar_cities":
            city_name = cities[0] if cities else None
            if city_name:
                handle_similar_cities_query(query, city_name, df_features, get_engine_func)
            else:
                handle_sql_query(query, classification, df_features, get_engine_func, city_list)

        elif query_type == "general_knowledge" or classification.get("needs_gpt_knowledge"):
            handle_general_knowledge_query(query, classification)
                
        else:
            # Default fallback
            handle_sql_query(query, classification, df_features, get_engine_func, city_list)
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        _show_fallback_data(df_features)


def _route_single_city(query, cities, metric, df_features, get_engine_func):
    """Route single city queries."""
    city_name = cities[0] if cities else None
    if city_name:
        handle_single_city_query(query, city_name, df_features, get_engine_func, metric)
    else:
        st.warning("Could not identify city from your query.")
        _show_fallback_data(df_features)


def _route_single_state(query, states, metric, classification, df_features, get_engine_func, city_list):
    """Route single state queries."""
    state_name = states[0] if states else None
    if state_name and metric == "population":
        handle_state_population_query(query, state_name, get_engine_func)
    elif state_name:
        handle_single_state(query, classification, df_features, get_engine_func, city_list)
    else:
        st.warning("Could not identify state from your query.")
        _show_fallback_data(df_features)


def _route_ranking(query, intent, classification, df_features, get_engine_func, city_list):
    """Route ranking queries to appropriate ML handler."""
    intent_handlers = {
        "families": "families",
        "young_professionals": "young_professionals",
        "retirement": "retirement"
    }
    
    handler_intent = intent_handlers.get(intent, "general")
    if handler_intent != "general":
        handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, handler_intent)
    else:
        handle_sql_query(query, classification, df_features, get_engine_func, city_list)


# =============================================================================
# SINGLE CITY HANDLER
# =============================================================================

def handle_single_city_query(query, city_name, df_features, get_engine_func, metric=None):
    """
    Handle queries about a specific city.
    
    Examples:
        - "Population of Denver" → Shows population metric card
        - "Tell me about Austin" → Shows full city profile
    """
    engine = get_engine_func()
    q_lower = query.lower()
    
    # Find the city
    city_data = _find_city(city_name, df_features, engine)
    
    if not city_data:
        st.warning(f"City '{city_name}' not found in our database.")
        handle_gpt_knowledge_fallback(query)
        return
    
    city = city_data.get("city", city_name)
    state = city_data.get("state", "")
    
    # Determine metric from query or parameter
    metric_info = _detect_metric(q_lower, metric)
    
    if metric_info:
        _display_city_metric_card(query, city, state, city_data, metric_info)
    else:
        # Show full city profile
        show_city_profile_card(city, state, pd.Series(city_data))


def _find_city(city_name, df_features, engine):
    """Find city in database with fuzzy matching fallback."""
    # Try exact match
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    if result:
        return dict(result._mapping)
    
    # Try fuzzy match
    city_list = df_features["city"].unique().tolist() if not df_features.empty else []
    matched_city, score = fuzzy_match_city(city_name, city_list)
    
    if matched_city and score > 70:
        sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
        with engine.connect() as conn:
            result = conn.execute(sql, {"city": matched_city}).fetchone()
        if result:
            return dict(result._mapping)
    
    return None


def _detect_metric(query_lower, metric_param):
    """Detect which metric the user is asking about."""
    metric_keywords = {
        "population": ("Population", "population"),
        "median age": ("Median Age", "median_age"),
        "age": ("Median Age", "median_age"),
        "household": ("Avg Household Size", "avg_household_size"),
        "family size": ("Avg Household Size", "avg_household_size"),
    }
    
    # Check parameter first
    if metric_param:
        display_name = metric_param.replace("_", " ").title()
        return (display_name, metric_param)
    
    # Check query keywords
    for keyword, info in metric_keywords.items():
        if keyword in query_lower:
            return info
    
    return None


def _display_city_metric_card(query, city, state, city_data, metric_info):
    """Display a single metric card for a city."""
    metric_name, metric_column = metric_info
    metric_value = city_data.get(metric_column, "N/A")
    
    # Format value
    formatted_value = _format_metric_value(metric_value)
    
    # Display card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">
            {city}, {state}
        </div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.5rem;">
            {metric_name}
        </div>
        <div style="font-size: 3.5rem; font-weight: 700;">
            {formatted_value}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate AI insight
    _generate_metric_insight(query, city, state, city_data, metric_name, formatted_value)


def _format_metric_value(value):
    """Format a metric value for display."""
    if isinstance(value, (int, float)) and value > 1000:
        return f"{int(value):,}"
    elif isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _generate_metric_insight(query, city, state, city_data, metric_name, formatted_value):
    """Generate AI insight for a metric."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        pop = city_data.get('population', 'N/A')
        pop_str = f"{pop:,}" if isinstance(pop, (int, float)) else str(pop)
        
        prompt = f"""The user asked: "{query}"
        
        {city}, {state} has a {metric_name.lower()} of {formatted_value}.
        
        Additional context:
        - Population: {pop_str}
        - Median Age: {city_data.get('median_age', 'N/A')}
        - Avg Household Size: {city_data.get('avg_household_size', 'N/A')}
        
        Give 2-3 short sentences explaining what this {metric_name.lower()} means for {city}. 
        Include context like comparisons (is it large/small for a US city?) and what it suggests about the city's character.
        Be concise and insightful."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <div style="font-size: 0.85rem; color: #667eea; margin-bottom: 0.5rem;">💡 Insight</div>
            {summary}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass


# =============================================================================
# STATE POPULATION HANDLER
# =============================================================================

def handle_state_population_query(query, state_name, get_engine_func):
    """
    Handle queries about state population.
    
    Example: "Population of Texas"
    """
    engine = get_engine_func()
    
    # Get state statistics
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
    
    with engine.connect() as conn:
        result = conn.execute(sql, {"state": state_name}).fetchone()
    
    if not result:
        st.warning(f"State '{state_name}' not found in our database.")
        handle_gpt_knowledge_fallback(query)
        return
    
    state_data = dict(result._mapping)
    total_pop = state_data.get("total_population", 0)
    city_count = state_data.get("city_count", 0)
    formatted_pop = _format_metric_value(total_pop)
    
    # Display state card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">
            {state_name}
        </div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.5rem;">
            Total Population
        </div>
        <div style="font-size: 3.5rem; font-weight: 700;">
            {formatted_pop}
        </div>
        <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
            Across {city_count} cities in our database
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI insight
    _generate_state_insight(query, state_name, state_data, formatted_pop, city_count)
    
    # Top cities
    _show_top_cities_in_state(state_name, engine)


def _generate_state_insight(query, state_name, state_data, formatted_pop, city_count):
    """Generate AI insight for state population."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        avg_age = state_data.get('avg_median_age', 0)
        avg_household = state_data.get('avg_household_size', 0)
        
        prompt = f"""The user asked about the population of {state_name}.
        
        Based on our database:
        - Total population: {formatted_pop}
        - Number of cities: {city_count}
        - Average median age: {avg_age:.1f} years
        - Average household size: {avg_household:.2f}
        
        Give 2-3 short sentences about {state_name}'s population. Include context about its ranking among US states and what the demographics suggest about the state.
        Be concise and insightful."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        
        st.markdown(f"""
        <div style="
            background: rgba(17, 153, 142, 0.1);
            border-left: 4px solid #11998e;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <div style="font-size: 0.85rem; color: #11998e; margin-bottom: 0.5rem;">💡 Insight</div>
            {summary}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass


def _show_top_cities_in_state(state_name, engine):
    """Show top 5 cities in a state."""
    sql = text(f"SELECT TOP 5 city, population FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state) ORDER BY population DESC")
    with engine.connect() as conn:
        result = conn.execute(sql, {"state": state_name})
        top_cities = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if not top_cities.empty:
        st.markdown("#### 🏙️ Largest Cities")
        top_cities["population"] = top_cities["population"].apply(lambda x: f"{int(x):,}")
        st.dataframe(top_cities, use_container_width=True, hide_index=True)


# =============================================================================
# SUPERLATIVE HANDLER
# =============================================================================

def handle_superlative_query(query, metric_text, direction, get_engine_func, limit=5, states=None):
    """
    Handle superlative questions.
    
    Examples:
        - "Which city has the highest population?"
        - "Top 5 most populated cities"
        - "City with lowest median age"
    """
    engine = get_engine_func()
    
    # Map metric text to column name
    metric_column = _map_metric_to_column(metric_text)
    order = "DESC" if direction == "highest" else "ASC"
    
    # Query database (with optional state filter)
    state_filter = states[0] if states else None
    
    if state_filter:
        sql = text(f"SELECT TOP {limit} * FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state) ORDER BY {metric_column} {order}")
        with engine.connect() as conn:
            result = conn.execute(sql, {"state": state_filter})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        sql = text(f"SELECT TOP {limit} * FROM {DB_TABLE_NAME} ORDER BY {metric_column} {order}")
        with engine.connect() as conn:
            result = conn.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        st.warning("No results found.")
        return
    
    # Display results
    _display_superlative_card(query, df, metric_column, direction)


def _map_metric_to_column(metric_text):
    """Map metric text to database column name."""
    metric_map = {
        "population": "population",
        "people": "population",
        "residents": "population",
        "median age": "median_age",
        "median_age": "median_age",
        "age": "median_age",
        "oldest": "median_age",
        "youngest": "median_age",
        "household size": "avg_household_size",
        "avg_household_size": "avg_household_size",
        "household": "avg_household_size",
        "family size": "avg_household_size",
    }
    
    if metric_text:
        metric_lower = str(metric_text).lower()
        for key, col in metric_map.items():
            if key in metric_lower:
                return col
    
    return "population"  # default


def _display_superlative_card(query, df, metric_column, direction):
    """Display superlative result card with runners-up."""
    top = df.iloc[0]
    city = top.get("city", "Unknown")
    state = top.get("state", "")
    value = top.get(metric_column, "N/A")
    formatted_value = _format_metric_value(value)
    
    metric_display = metric_column.replace("_", " ").title()
    rank_label = f"{'Highest' if direction == 'highest' else 'Lowest'} {metric_display}"
    
    # Add state to label if filtered
    # Extract state from query if present
    state_match = re.search(r"\bin\s+([A-Za-z\s]+?)(?:\?|$)", query.lower())
    if state_match:
        state_name = state_match.group(1).strip().title()
        rank_label = f"{rank_label} in {state_name}"
    
    # Main card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
            🏆 {rank_label}
        </div>
        <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.25rem;">
            {city}, {state}
        </div>
        <div style="font-size: 3rem; font-weight: 700;">
            {formatted_value}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # If limit > 1, show as table instead of card
    if limit > 1:
        # Show count header
        state_label = f" in {states[0]}" if states else ""
        metric_label = metric_column.replace("_", " ").title()
        direction_label = "Highest" if direction == "highest" else "Lowest"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            color: white;
            text-align: center;
        ">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">
                🏆 Top {limit} {direction_label} {metric_label}{state_label}
            </div>
            <div style="font-size: 2.5rem; font-weight: 700;">
                {len(df)} cities
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show full table
        show_city_table(df, f"Top {limit} Cities by {metric_label}", True)
    else:
        # Single result - show card format
        _display_superlative_card(df.iloc[0], metric_column, direction, query)
        
        # Show runners up (for single queries like "largest city")
        if len(df) > 1:
            _display_runners_up(df.iloc[1:5], metric_column)
    
    # AI insight
    _generate_superlative_insight(query, city, state, direction, metric_display, formatted_value)


def _display_runners_up(runners_df, metric_column):
    """Display runners-up for superlative query."""
    st.markdown("#### 🥈 Runners Up")
    
    for _, row in runners_df.iterrows():
        city = row.get("city", "")
        state = row.get("state", "")
        value = row.get(metric_column, "N/A")
        formatted = _format_metric_value(value)
        
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            margin-bottom: 0.5rem;
        ">
            <span>{city}, {state}</span>
            <span style="font-weight: 600;">{formatted}</span>
        </div>
        """, unsafe_allow_html=True)


def _generate_superlative_insight(query, city, state, direction, metric_display, formatted_value):
    """Generate AI insight for superlative query."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""The user asked: "{query}"
        
        Answer: {city}, {state} has the {direction} {metric_display.lower()} at {formatted_value}.
        
        Give 1-2 sentences of insight about why {city} has this ranking and what it means.
        Be concise."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        summary = response.choices[0].message.content.strip()
        
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <div style="font-size: 0.85rem; color: #667eea; margin-bottom: 0.5rem;">💡 Insight</div>
            {summary}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass


# =============================================================================
# ML RANKING HANDLER
# =============================================================================

def handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, intent):
    """
    Handle ML-based ranking queries.
    
    Examples:
        - "Best cities for families"
        - "Best places for young professionals"
        - "Best retirement cities"
    """
    state_filter = classification.get("state_filter") or \
                   (classification.get("states", [None])[0] if classification.get("states") else None)
    
    # Get ML ranking function
    ml_functions = {
        "families": run_family_ranking,
        "young_professionals": run_young_ranking,
        "retirement": run_retirement_ranking
    }
    
    func = ml_functions.get(intent)
    df = None
    
    # Try ML model
    if func:
        try:
            df = func()
            if state_filter and df is not None and not df.empty and "state" in df.columns:
                filtered = df[df["state"].str.lower() == state_filter.lower()]
                if not filtered.empty:
                    df = filtered
        except Exception as e:
            st.warning(f"ML model error: {e}")
    
    # Fallback to basic ranking
    if df is None or df.empty:
        df = _fallback_ranking(df_features, intent, state_filter)
    
    # Display results
    if df is not None and not df.empty:
        show_recommendation_card(df.iloc[0], intent, df, query)
        
        # ML explanation
        if explain_ml_results:
            try:
                insights = explain_ml_results(query, df)
                if insights:
                    st.markdown("### 🧠 Why These Cities?")
                    st.markdown(insights)
            except:
                pass
        
        show_city_table(df, f"Best for {intent.replace('_', ' ').title()}", True)
    else:
        st.warning("Could not generate rankings.")


def _fallback_ranking(df_features, intent, state_filter=None):
    """Generate fallback ranking when ML models unavailable."""
    if df_features.empty:
        return None
    
    df = df_features.copy()
    
    if state_filter:
        df = df[df["state"].str.lower() == state_filter.lower()]
    
    if df.empty:
        return None
    
    # Calculate scores based on intent
    if intent == "families":
        df["score"] = df["avg_household_size"] * 20 + (50 - abs(df["median_age"] - 35)) + df["population"] / 100000
    elif intent == "young_professionals":
        df["score"] = (45 - df["median_age"]) * 3 + df["population"] / 50000
    elif intent == "retirement":
        df["score"] = df["median_age"] * 2 + (4 - df["avg_household_size"]) * 10
    else:
        df["score"] = df["population"] / 10000
    
    return df.sort_values("score", ascending=False).head(10)


# =============================================================================
# COMPARISON HANDLER
# =============================================================================

def handle_comparison(query, classification, df_features, get_engine_func, city_list):
    """
    Handle comparison queries.
    
    Examples:
        - "Compare Miami and Austin"
        - "Compare Texas and California"
    """
    comparison_type = classification.get("comparison_type", "city_vs_city")
    engine = get_engine_func()
    
    if comparison_type == "state_vs_state":
        _handle_state_comparison(query, classification, engine)
    else:
        _handle_city_comparison(query, classification, city_list, engine)


def _handle_city_comparison(query, classification, city_list, engine):
    """Handle city vs city comparison."""
    cities = classification.get("cities", [])
    
    if len(cities) < 2:
        result = extract_two_cities_fuzzy(query, city_list)
        if result and len(result) >= 2:
            cities = [result[0], result[1]]
    
    if len(cities) < 2:
        st.warning("Could not find two cities to compare.")
        return
    
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) IN (LOWER(:c1), LOWER(:c2))")
    with engine.connect() as conn:
        result = conn.execute(sql, {"c1": cities[0], "c2": cities[1]})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if len(df) >= 2:
        r1 = df[df["city"].str.lower() == cities[0].lower()].iloc[0]
        r2 = df[df["city"].str.lower() == cities[1].lower()].iloc[0]
        show_city_comparison(r1, r2, query)
    else:
        st.warning("Could not find both cities in database.")


def _handle_state_comparison(query, classification, engine):
    """Handle state vs state comparison."""
    states = classification.get("states", [])
    
    if len(states) < 2:
        result = extract_two_states_fuzzy(query)
        if result and len(result) >= 2:
            states = [result[0], result[1]]
    
    if len(states) < 2:
        st.warning("Could not find two states to compare.")
        return
    
    def get_state_stats(state):
        sql = text(f"""
            SELECT state, COUNT(*) as cnt, SUM(population) as pop, AVG(median_age) as age 
            FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) GROUP BY state
        """)
        with engine.connect() as conn:
            r = conn.execute(sql, {"s": state}).fetchone()
        return dict(r._mapping) if r else {}
    
    def get_state_cities(state):
        sql = text(f"SELECT TOP 5 * FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) ORDER BY population DESC")
        with engine.connect() as conn:
            result = conn.execute(sql, {"s": state})
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    
    s1, s2 = get_state_stats(states[0]), get_state_stats(states[1])
    c1, c2 = get_state_cities(states[0]), get_state_cities(states[1])
    
    if s1 and s2:
        show_state_comparison(states[0], s1, c1, states[1], s2, c2, query)
    else:
        st.warning("Could not find both states.")


# =============================================================================
# LIFESTYLE HANDLER
# =============================================================================

def handle_lifestyle_query(query, city_name, classification, df_features, get_engine_func, city_list):
    """
    Handle lifestyle queries.
    
    Example: "Life in Dallas"
    """
    engine = get_engine_func()
    
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    if not result:
        st.warning(f"'{city_name.title()}' not found in our database. Showing general information.")
        handle_gpt_knowledge_fallback(query)
        return
    
    city_data = dict(result._mapping)
    
    # Show city profile
    show_city_profile_card(city_data.get("city", city_name), city_data.get("state", ""), pd.Series(city_data))
    
    # Generate lifestyle summary
    st.markdown("### 🏙️ What's Life Like?")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""Describe what life is like in {city_name}. 
        Population: {city_data.get('population', 'N/A')}, 
        Median Age: {city_data.get('median_age', 'N/A')}, 
        Avg Household Size: {city_data.get('avg_household_size', 'N/A')}. 
        Give a brief, engaging 2-3 sentence description of the lifestyle, culture, and what it's like to live there."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        st.markdown(response.choices[0].message.content.strip())
        
    except Exception:
        pop = city_data.get('population', 'N/A')
        age = city_data.get('median_age', 'N/A')
        if isinstance(pop, (int, float)):
            st.info(f"A city with population {pop:,} and median age {age}.")
        else:
            st.info(f"A city with population {pop} and median age {age}.")


# =============================================================================
# SQL HANDLER (DEFAULT)
# =============================================================================

def handle_sql_query(query, classification, df_features, get_engine_func, city_list):
    """Handle SQL-based queries (aggregate, list, etc.)."""
    try:
        engine = get_engine_func()
        df = _build_basic_sql(query, classification, engine)
        
        if df is None or df.empty:
            st.warning("No results found.")
            _show_fallback_data(df_features)
            return
        
        _display_sql_results(query, classification, df)
        
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        _show_fallback_data(df_features)


def _build_basic_sql(query, classification, engine):
    """Build and execute SQL query."""
    q = query.lower()
    states = classification.get("states", [])
    state_filter = states[0] if states else None
    metric = classification.get("metric", "population")
    direction = classification.get("direction")
    limit = classification.get("limit", 10) or 10
    
    # Count queries
    if "how many" in q or "count" in q:
        return _execute_count_query(state_filter, engine)
    
    # List queries
    if direction == "highest":
        order = f"ORDER BY {metric} DESC"
    elif direction == "lowest":
        order = f"ORDER BY {metric} ASC"
    else:
        order = "ORDER BY population DESC"
    
    if state_filter:
        sql = text(f"SELECT TOP {limit} * FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state) {order}")
        with engine.connect() as conn:
            result = conn.execute(sql, {"state": state_filter})
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        sql = text(f"SELECT TOP {limit} * FROM {DB_TABLE_NAME} {order}")
        with engine.connect() as conn:
            result = conn.execute(sql)
            return pd.DataFrame(result.fetchall(), columns=result.keys())


def _execute_count_query(state_filter, engine):
    """Execute count query."""
    if state_filter:
        sql = text(f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
        with engine.connect() as conn:
            r = conn.execute(sql, {"state": state_filter}).fetchone()
        return pd.DataFrame([{"count": r[0], "state": state_filter}])
    else:
        sql = text(f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME}")
        with engine.connect() as conn:
            r = conn.execute(sql).fetchone()
        return pd.DataFrame([{"count": r[0]}])


def _display_sql_results(query, classification, df):
    """Display SQL query results."""
    # Count result
    if "count" in df.columns:
        val = df["count"].iloc[0]
        state = df.get("state", pd.Series([None])).iloc[0]
        label = f"Cities in {state}" if state else "Total Cities"
        show_aggregate_card(label, str(val), "cities")
        return
    
    # Single city result
    if len(df) == 1:
        row = df.iloc[0]
        show_city_profile_card(row.get("city", ""), row.get("state", ""), row)
        return
    
    # List result
    show_city_table(df, "Results", True)


# =============================================================================
# ADDITIONAL HANDLERS
# =============================================================================

def handle_single_state(query, classification, df_features, get_engine_func, city_list):
    """Handle single state queries (non-population specific)."""
    states = classification.get("states", [])
    state = states[0] if states else extract_single_state_fuzzy(query)
    
    if not state:
        st.warning("Could not identify state.")
        return
    
    engine = get_engine_func()
    
    sql = text(f"""
        SELECT state, COUNT(*) as city_count, SUM(population) as total_pop, AVG(median_age) as avg_age
        FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) GROUP BY state
    """)
    
    with engine.connect() as conn:
        r = conn.execute(sql, {"s": state}).fetchone()
    
    if r:
        d = dict(r._mapping)
        sql2 = text(f"SELECT TOP 5 * FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) ORDER BY population DESC")
        with engine.connect() as conn:
            result = conn.execute(sql2, {"s": state})
            cities = pd.DataFrame(result.fetchall(), columns=result.keys())
        show_state_metric_card(state, "Population", d.get("total_pop", 0), d.get("city_count", 0), cities)
    else:
        st.warning(f"State '{state}' not found.")


def handle_gpt_knowledge_fallback(query, classification=None, df_features=None, get_engine_func=None, city_list=None):
    """Fallback to GPT knowledge when data not in database."""
    from hybrid_classifier import is_city_related_query
    
    if not is_city_related_query(query):
        show_out_of_scope()
        return
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Answer about US cities: {query}"}],
            max_tokens=500
        )
        show_ai_response(response.choices[0].message.content, is_fallback=True)
        
    except Exception as e:
        st.error(str(e))


def _show_fallback_data(df_features):
    """Show fallback data when query fails."""
    st.markdown("### Available Data")
    if not df_features.empty:
        show_city_table(df_features.head(10), "Sample Cities", True)

# =============================================================================
# FILTER HANDLER
# =============================================================================

def handle_filter_query(query, classification, df_features, get_engine_func):
    """
    Handle filter queries like "cities with population > 1000000".
    """
    engine = get_engine_func()
    
    metric = classification.get("metric", "population")
    filter_op = classification.get("filter_op", "gt")
    filter_value = classification.get("filter_value", 0)
    
    # Build SQL based on filter operation
    if filter_op == "gt":
        op_sql = ">"
        op_label = "greater than"
    elif filter_op == "lt":
        op_sql = "<"
        op_label = "less than"
    elif filter_op == "gte":
        op_sql = ">="
        op_label = "at least"
    elif filter_op == "lte":
        op_sql = "<="
        op_label = "at most"
    else:
        op_sql = ">"
        op_label = "greater than"
    
    states = classification.get("states", [])
    state_filter = states[0] if states else None
    
    if state_filter:
        sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE {metric} {op_sql} :value AND LOWER(state) = LOWER(:state) ORDER BY {metric} DESC")
        with engine.connect() as conn:
            result = conn.execute(sql, {"value": filter_value, "state": state_filter})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE {metric} {op_sql} :value ORDER BY {metric} DESC")
        with engine.connect() as conn:
            result = conn.execute(sql, {"value": filter_value})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        st.warning(f"No cities found with {metric.replace('_', ' ')} {op_label} {filter_value:,}")
        return
    
    # Format the value for display
    formatted_value = f"{filter_value:,}" if filter_value > 1000 else str(filter_value)
    metric_display = metric.replace("_", " ").title()
    
    # Show count card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">
            Cities with {metric_display} {op_label} {formatted_value}
        </div>
        <div style="font-size: 3rem; font-weight: 700;">
            {len(df)}
        </div>
        <div style="font-size: 0.85rem; opacity: 0.8;">cities found</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show results table
    show_city_table(df, f"Cities with {metric_display} {op_label} {formatted_value}", True)


# =============================================================================
# SIMILAR CITIES HANDLER
# =============================================================================

def handle_similar_cities_query(query, city_name, df_features, get_engine_func):
    """
    Handle queries like "cities similar to Chicago".
    """
    engine = get_engine_func()
    
    # First, get the reference city data
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    if not result:
        st.warning(f"City '{city_name}' not found in our database.")
        return
    
    ref_city = dict(result._mapping)
    ref_pop = ref_city.get("population", 0)
    ref_age = ref_city.get("median_age", 0)
    ref_household = ref_city.get("avg_household_size", 0)
    
    # Show reference city card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">Reference City</div>
        <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;">
            {ref_city.get('city')}, {ref_city.get('state')}
        </div>
        <div style="display: flex; gap: 2rem; font-size: 0.9rem;">
            <div>Population: {ref_pop:,}</div>
            <div>Median Age: {ref_age}</div>
            <div>Household: {ref_household:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Try semantic search first if available
    # if semantic_city_search:
    #     try:
    #         results = semantic_city_search(f"cities like {city_name}", top_k=11)
    #         # Filter out the reference city
    #         results = [(c, s) for c, s in results if c.lower() != city_name.lower()][:10]
            
    #         if results:
    #             data = []
    #             for c, s in results:
    #                 sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city)=LOWER(:c) AND LOWER(state)=LOWER(:s)")
    #                 with engine.connect() as conn:
    #                     r = conn.execute(sql, {"c": c, "s": s}).fetchone()
    #                     if r:
    #                         data.append(dict(r._mapping))
                
    #             if data:
    #                 st.markdown("### 🔍 Similar Cities (Semantic Match)")
    #                 show_city_table(pd.DataFrame(data), f"Cities Similar to {city_name}", True)
    #                 return
    #     except Exception as e:
    #         pass  # Fall back to demographic similarity
    
    # Fallback: Find cities with similar demographics (excluding same state for variety)
    pop_range_low = ref_pop * 0.5
    pop_range_high = ref_pop * 2.0
    age_range_low = ref_age - 5
    age_range_high = ref_age + 5
    
    # First try: Different states with similar demographics
    sql = text(f"""
        SELECT TOP 10 * FROM {DB_TABLE_NAME} 
        WHERE population BETWEEN :pop_low AND :pop_high
        AND median_age BETWEEN :age_low AND :age_high
        AND LOWER(city) != LOWER(:city)
        AND LOWER(state) != LOWER(:state)
        ORDER BY ABS(population - :ref_pop) + ABS(median_age - :ref_age) * 100000
    """)
    
    with engine.connect() as conn:
        result = conn.execute(sql, {
            "pop_low": pop_range_low,
            "pop_high": pop_range_high,
            "age_low": age_range_low,
            "age_high": age_range_high,
            "city": city_name,
            "state": ref_city.get("state", ""),
            "ref_pop": ref_pop,
            "ref_age": ref_age
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    # If not enough results, include same state
    if len(df) < 5:
        sql = text(f"""
            SELECT TOP 10 * FROM {DB_TABLE_NAME} 
            WHERE population BETWEEN :pop_low AND :pop_high
            AND median_age BETWEEN :age_low AND :age_high
            AND LOWER(city) != LOWER(:city)
            ORDER BY ABS(population - :ref_pop) + ABS(median_age - :ref_age) * 100000
        """)
        
        with engine.connect() as conn:
            result = conn.execute(sql, {
                "pop_low": pop_range_low,
                "pop_high": pop_range_high,
                "age_low": age_range_low,
                "age_high": age_range_high,
                "city": city_name,
                "ref_pop": ref_pop,
                "ref_age": ref_age
            })
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        st.info(f"No cities found with similar demographics to {city_name}.")
        return
    
    st.markdown("### 🔍 Similar Cities (Demographic Match)")
    show_city_table(df.head(10), f"Cities Similar to {city_name}", True)
    
    # AI insight
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        similar_cities = df.head(5)['city'].tolist()
        
        prompt = f"""The user asked for cities similar to {city_name}.
        
        Reference: {city_name} has population {ref_pop:,}, median age {ref_age}, household size {ref_household:.2f}.
        
        Similar cities found: {', '.join(similar_cities)}
        
        Give 1-2 sentences explaining what these cities have in common with {city_name}.
        Be concise."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <div style="font-size: 0.85rem; color: #667eea; margin-bottom: 0.5rem;">💡 Insight</div>
            {response.choices[0].message.content.strip()}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass

# =============================================================================
# GENERAL KNOWLEDGE HANDLER
# =============================================================================

def handle_general_knowledge_query(query, classification):
    """
    Handle city/state questions that need GPT general knowledge.
    
    Examples:
        - "Why is Texas so big?"
        - "What makes New York expensive?"
        - "History of Chicago"
    """
    cities = classification.get("cities", [])
    states = classification.get("states", [])
    
    # Determine the subject
    if cities:
        subject = f"{cities[0]} (city)"
    elif states:
        subject = f"{states[0]} (state)"
    else:
        subject = "US cities/states"
    
    # Show info banner
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    ">
        ℹ️ This answer is from AI general knowledge, not from our database.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        prompt = f"""The user asked about US cities/states: "{query}"

Please provide a helpful, informative answer about {subject}.

Guidelines:
- Be informative and educational
- Include relevant facts, history, or context
- Keep the response concise (3-5 paragraphs max)
- Focus on what makes this place unique or interesting
- If discussing size, population, or demographics, provide context"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in US cities and states. Provide helpful, accurate information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Display the answer
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            color: white;
        ">
            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                🤖 AI Response
            </div>
            <div style="font-size: 0.85rem; opacity: 0.8;">
                About: {subject}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(answer)
        
        # If we have related data in database, show it
        if cities or states:
            _show_related_database_info(cities, states)
        
    except Exception as e:
        st.error(f"Could not generate response: {str(e)}")


def _show_related_database_info(cities, states):
    """Show related data from database if available."""
    try:
        engine = get_engine()
        
        if cities:
            city = cities[0]
            sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
            with engine.connect() as conn:
                result = conn.execute(sql, {"city": city}).fetchone()
            
            if result:
                data = dict(result._mapping)
                st.markdown("---")
                st.markdown("#### 📊 From Our Database")
                col1, col2, col3 = st.columns(3)
                with col1:
                    pop = data.get('population', 'N/A')
                    st.metric("Population", f"{pop:,}" if isinstance(pop, (int, float)) else pop)
                with col2:
                    st.metric("Median Age", data.get('median_age', 'N/A'))
                with col3:
                    st.metric("Household Size", data.get('avg_household_size', 'N/A'))
        
        elif states:
            state = states[0]
            sql = text(f"""
                SELECT COUNT(*) as city_count, SUM(population) as total_pop, AVG(median_age) as avg_age
                FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)
            """)
            with engine.connect() as conn:
                result = conn.execute(sql, {"state": state}).fetchone()
            
            if result and result[0] > 0:
                st.markdown("---")
                st.markdown("#### 📊 From Our Database")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cities in DB", result[0])
                with col2:
                    pop = result[1]
                    st.metric("Total Population", f"{int(pop):,}" if pop else "N/A")
                with col3:
                    st.metric("Avg Median Age", f"{result[2]:.1f}" if result[2] else "N/A")
    
    except Exception:
        pass  # Silently fail if database query fails

def handle_filter_range_query(query, classification, df_features, get_engine_func):
    """
    Handle range filter queries like "cities with population between 100k and 300k".
    """
    engine = get_engine_func()
    
    metric = classification.get("metric", "population")
    min_val = classification.get("filter_min", 0)
    max_val = classification.get("filter_max", 999999999)
    states = classification.get("states", [])
    state_filter = states[0] if states else None
    
    metric_display = metric.replace("_", " ").title()
    
    # Build SQL
    if state_filter:
        sql = text(f"""
            SELECT * FROM {DB_TABLE_NAME} 
            WHERE {metric} BETWEEN :min_val AND :max_val 
            AND LOWER(state) = LOWER(:state)
            ORDER BY {metric} DESC
        """)
        with engine.connect() as conn:
            result = conn.execute(sql, {"min_val": min_val, "max_val": max_val, "state": state_filter})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        sql = text(f"""
            SELECT * FROM {DB_TABLE_NAME} 
            WHERE {metric} BETWEEN :min_val AND :max_val 
            ORDER BY {metric} DESC
        """)
        with engine.connect() as conn:
            result = conn.execute(sql, {"min_val": min_val, "max_val": max_val})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        st.warning(f"No cities found with {metric_display} between {min_val:,} and {max_val:,}")
        return
    
    # Show count card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">
            Cities with {metric_display} between {min_val:,} and {max_val:,}
        </div>
        <div style="font-size: 3rem; font-weight: 700;">
            {len(df)}
        </div>
        <div style="font-size: 0.85rem; opacity: 0.8;">cities found</div>
    </div>
    """, unsafe_allow_html=True)
    
    show_city_table(df, f"Cities with {metric_display} {min_val:,} - {max_val:,}", True)

# =============================================================================
# LLM SQL HANDLER - Execute SQL generated by LLM
# =============================================================================

def handle_llm_sql_query(query, classification, get_engine_func):
    """
    Execute SQL query generated by LLM classification.
    This is the core Text-to-SQL handler.
    """
    engine = get_engine_func()
    sql_query = classification.get("sql", "")
    explanation = classification.get("explanation", "")
    query_type = classification.get("query_type", "filter")
    
    if not sql_query:
        st.error("No SQL query generated. Please try rephrasing your question.")
        return
    
    # Show the generated SQL (collapsible)
    with st.expander("🔍 View Generated SQL", expanded=False):
        st.code(sql_query, language="sql")
        if explanation:
            st.caption(f"💡 {explanation}")
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            st.warning("No results found for your query.")
            return
        
        # Different display based on query type
        if query_type == "aggregate":
            _display_aggregate_result(df, query)
        elif query_type == "group_by":
            _display_group_by_result(df, query)
        elif query_type == "comparison":
            cities = classification.get("cities", [])
            if len(cities) >= 2:
                _display_comparison_result(df, cities, query)
            else:
                show_city_table(df, "Comparison Results", True)
        else:
            # Filter, superlative, pattern_match - show as table
            _display_filter_result(df, query, classification)
    
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        st.code(sql_query, language="sql")


def _display_aggregate_result(df, query):
    """Display aggregate results (COUNT, SUM, AVG) as metrics."""
    st.markdown("### 📊 Results")
    
    cols = st.columns(len(df.columns))
    for i, col_name in enumerate(df.columns):
        value = df.iloc[0][col_name]
        label = col_name.replace("_", " ").title()
        
        # Format the value
        if isinstance(value, float):
            if value > 1000:
                formatted = f"{value:,.0f}"
            else:
                formatted = f"{value:.2f}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)
        
        with cols[i]:
            st.metric(label, formatted)


def _display_group_by_result(df, query):
    """Display GROUP BY results as a table with chart."""
    st.markdown("### 📊 Results by Group")
    
    # Show count card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 2rem; font-weight: 700;">{len(df)}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">groups found</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Show bar chart if applicable
    if len(df) <= 20 and len(df.columns) >= 2:
        try:
            st.bar_chart(df.set_index(df.columns[0])[df.columns[1]])
        except:
            pass


def _display_comparison_result(df, cities, query):
    """Display comparison between cities."""
    st.markdown("### ⚖️ Comparison")
    
    if len(df) < 2:
        st.warning("Could not find both cities for comparison.")
        show_city_table(df, "Results", True)
        return
    
    # Display side by side
    cols = st.columns(len(df))
    for i, (idx, row) in enumerate(df.iterrows()):
        with cols[i]:
            pop = row.get('population', 0)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {'#667eea' if i == 0 else '#764ba2'} 0%, {'#764ba2' if i == 0 else '#667eea'} 100%);
                border-radius: 16px;
                padding: 1.5rem;
                color: white;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 700;">{row.get('city', 'N/A')}</div>
                <div style="font-size: 0.9rem; opacity: 0.8;">{row.get('state', '')}</div>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 1rem 0;">
                <div style="font-size: 0.85rem;">Population</div>
                <div style="font-size: 1.3rem; font-weight: 600;">{pop:,}</div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">Median Age: {row.get('median_age', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Determine winner
    if len(df) >= 2:
        city1, city2 = df.iloc[0], df.iloc[1]
        winner = city1['city'] if city1['population'] > city2['population'] else city2['city']
        diff = abs(city1['population'] - city2['population'])
        st.markdown(f"""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <strong>{winner}</strong> is larger by <strong>{diff:,}</strong> people.
        </div>
        """, unsafe_allow_html=True)


def _display_filter_result(df, query, classification):
    """Display filter/superlative results."""
    query_type = classification.get("query_type", "filter")
    
    # Show count card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 2rem; font-weight: 700;">{len(df)}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">cities found</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show table
    show_city_table(df, "Results", True)
