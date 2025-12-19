#query_handlers.py

import re
import streamlit as st
import pandas as pd
from sqlalchemy import text
from typing import Optional
from unsupported_data_handler import handle_unsupported_data_query

from config import DB_TABLE_NAME
from db_config import get_engine


 
# SAFE IMPORTS WITH FALLBACKS
 
"""
# NEW: RAG imports for enhanced answers
try:
    from core.rag_search import rag_answer, rag_search, get_lifestyle_rag_answer, get_city_rag_context
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    rag_answer = None
    rag_search = None
    get_lifestyle_rag_answer = None
    get_city_rag_context = None

try:
    from core.lifestyle_rag_v2 import try_build_lifestyle_card, get_city_profile, display_lifestyle_card
    LIFESTYLE_RAG_AVAILABLE = True
except ImportError:
    LIFESTYLE_RAG_AVAILABLE = False
    try_build_lifestyle_card = None
    get_city_profile = None
    display_lifestyle_card = None
"""
try:
    from core.intent_classifier import classify_query_intent
except ImportError:
    classify_query_intent = None

try:
    from core.semantic_search import semantic_city_search
except ImportError:
    semantic_city_search = None

try:
    from core.rag_search import rag_answer, rag_search, get_lifestyle_rag_answer, get_city_rag_context
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    rag_answer = None
    rag_search = None
    get_lifestyle_rag_answer = None
    get_city_rag_context = None

# Lifestyle RAG (city profiles)
try:
    from core.lifestyle_rag_v2 import try_build_lifestyle_card, get_city_profile
    LIFESTYLE_RAG_AVAILABLE = True
except ImportError:
    LIFESTYLE_RAG_AVAILABLE = False
    try_build_lifestyle_card = None
    get_city_profile = None

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


 
# RAG CONTEXT CACHING (Prevents redundant calls within same request)
 

def get_cached_rag_context(city: str, state: str) -> dict:
    """
    Get RAG context with request-level caching.
    Prevents multiple RAG calls for the same city in one query.
    """
    if not city:
        return {}
    
    cache_key = f"rag_ctx_{city.lower()}_{state.lower() if state else ''}"
    
    # Check session state cache first
    if cache_key not in st.session_state:
        if RAG_AVAILABLE and get_city_rag_context:
            try:
                st.session_state[cache_key] = get_city_rag_context(city, state) or {}
            except Exception:
                st.session_state[cache_key] = {}
        else:
            st.session_state[cache_key] = {}
    
    return st.session_state[cache_key]


def clear_rag_cache():
    """Clear RAG context cache (call at start of new query if needed)."""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("rag_ctx_")]
    for key in keys_to_remove:
        del st.session_state[key]


def show_styled_insight(content: str, color_scheme: str = "purple"):
    """
    Display a beautifully styled insight card.
    
    Args:
        content: The insight text
        color_scheme: "purple" (default), "blue" (for data-only)
    """
    schemes = {
        "purple": {
            "gradient": "linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.08) 100%)",
            "border": "#667eea",
            "icon_color": "#a78bfa",
            "shadow": "rgba(102, 126, 234, 0.15)",
            "border_glow": "rgba(102, 126, 234, 0.2)"
        },
        "blue": {
            "gradient": "linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.08) 100%)",
            "border": "#3b82f6",
            "icon_color": "#60a5fa",
            "shadow": "rgba(59, 130, 246, 0.15)",
            "border_glow": "rgba(59, 130, 246, 0.2)"
        }
    }
    
    # Map "green" to "purple" for consistency
    if color_scheme == "green":
        color_scheme = "purple"
    
    s = schemes.get(color_scheme, schemes["purple"])
    
    st.markdown(f"""
    <div style="
        background: {s['gradient']};
        border-left: 4px solid {s['border']};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px {s['shadow']};
        border: 1px solid {s['border_glow']};
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        ">
            <span style="font-size: 1rem;"></span>
            <span style="
                font-size: 0.7rem; 
                color: {s['icon_color']}; 
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-weight: 700;
            ">Insight</span>
        </div>
        <div style="color: #e2e8f0; line-height: 1.7; font-size: 0.95rem;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def generate_enhanced_insight(query: str, city_data: dict, context_type: str = "general", rag_context: dict = None) -> Optional[str]:
    """
    Generate rich, contextual AI insight using RAG context.
    
    Args:
        query: User's original query
        city_data: Dict with city info (must have 'city' and 'state')
        context_type: Type of insight needed - 'profile', 'comparison', 'ranking', 'similarity', etc.
        rag_context: Optional pre-fetched RAG context (avoids redundant calls)
        
    Returns:
        Rich AI-generated insight or None
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        city = city_data.get('city', '')
        state = city_data.get('state', '')
        
        if not city:
            return None
        
        # Use passed context or fetch with caching
        if rag_context is None:
            rag_context = get_cached_rag_context(city, state)
        
        if not rag_context or not any(rag_context.values()):
            return None
        
        # Build rich context from RAG
        context_parts = []
        type_labels = {
            "lifestyle": "Lifestyle & Culture",
            "jobs_economy": "Jobs & Economy", 
            "cost_of_living": "Cost of Living",
            "family": "Family Life",
            "weather_geography": "Climate & Geography"
        }
        
        for chunk_type, texts in rag_context.items():
            if texts:
                label = type_labels.get(chunk_type, chunk_type)
                # Take first 2 chunks for depth
                combined = " ".join(texts[:2])
                context_parts.append(f"{label}: {combined}")
        
        if not context_parts:
            return None
        
        # Build database facts
        pop = city_data.get('population', 'N/A')
        pop_str = f"{pop:,}" if isinstance(pop, (int, float)) else str(pop)
        age = city_data.get('median_age', 'N/A')
        household = city_data.get('avg_household_size', 'N/A')
        
        # Create context-specific prompts
        if context_type == "profile":
            prompt = f"""Create a rich profile summary of {city}, {state}.

DATABASE METRICS:
- Population: {pop_str}
- Median Age: {age}
- Household Size: {household}

LIFESTYLE CONTEXT:
{chr(10).join(context_parts)}

Write 3-4 sentences that paint a vivid picture of what makes {city} unique. Include:
- What the city is known for (culture, economy, lifestyle)
- Who would thrive here (demographics, lifestyle fit)
- Key characteristics from the context above

Be engaging and specific. Use the context to go beyond just numbers."""

        elif context_type == "ranking":
            prompt = f"""Explain why {city}, {state} ranks highly for this query: "{query}"

DATABASE METRICS:
- Population: {pop_str}
- Median Age: {age}
- Household Size: {household}

LIFESTYLE CONTEXT:
{chr(10).join(context_parts)}

Write 2-3 sentences explaining:
- What specific factors make {city} great for this demographic/need
- Use the lifestyle context to support your answer
- Be specific about amenities, culture, or economic factors

Be persuasive and use concrete details from the context."""

        elif context_type == "similarity":
            prompt = f"""Explain what makes {city}, {state} distinctive and what similar cities might share.

DATABASE METRICS:
- Population: {pop_str}
- Median Age: {age}
- Household Size: {household}

LIFESTYLE CONTEXT:
{chr(10).join(context_parts)}

Write 2-3 sentences covering:
- {city}'s defining characteristics (culture, economy, lifestyle)
- What type of cities would be similar
- Key factors that define its character

Use the lifestyle context to go beyond demographics."""

        else:  # general
            prompt = f"""Provide an insightful summary about {city}, {state} relevant to: "{query}"

DATABASE METRICS:
- Population: {pop_str}
- Median Age: {age}
- Household Size: {household}

LIFESTYLE CONTEXT:
{chr(10).join(context_parts)}

Write 2-3 engaging sentences using the lifestyle context above.
Focus on what makes {city} interesting and relevant to the question.
Go beyond basic demographics - use culture, economy, lifestyle details."""

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable city expert who provides rich, contextual insights using specific details."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.6
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Enhanced insight error: {e}")
        return None


def display_results_container(query: str, classification: dict, content_func):
    """
    Unified results container for consistent UI across all query types.
    
    Args:
        query: The user's query
        classification: Classification result
        content_func: Function that renders the specific content
    """
    query_type = classification.get("query_type", "")
    explanation = classification.get("explanation", "")
    
    # Results header
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    ">
        <div style="font-size: 0.8rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px;">
            Query Results
        </div>
        <div style="font-size: 1.1rem; color: #e2e8f0; margin-top: 0.25rem;">
            {query}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render specific content
    content_func()
    
    # Query explanation footer (if available)
    if explanation:
        st.markdown(f"""
        <div style="
            font-size: 0.8rem;
            color: #a0aec0;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        ">
            {explanation}
        </div>
        """, unsafe_allow_html=True)

def show_sql_expander(sql_query: str, params: dict = None):
    """Display the executed SQL in a collapsible expander."""
    with st.expander("View Generated SQL", expanded=False):
        # Format SQL for display
        sql_display = str(sql_query)
        
        # Replace parameters with actual values for clarity
        if params:
            for key, value in params.items():
                if isinstance(value, str):
                    sql_display = sql_display.replace(f":{key}", f"'{value}'")
                else:
                    sql_display = sql_display.replace(f":{key}", str(value))
        
        st.code(sql_display, language="sql")

 
# MAIN QUERY HANDLER (ROUTER)
 
    
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
        if classification.get("query_type") == "unsupported_data":
            handle_unsupported_data_query(query, classification)
            return
        # Route based on query type
        if query_type == "single_city":
            _route_single_city(query, cities, metric, df_features, get_engine_func)
        
        elif query_type == "single_best":
            # Execute SQL to get the result first
            sql = classification.get("sql")
            if sql:
                try:
                    engine = get_engine_func()
                    with engine.connect() as conn:
                        result = conn.execute(text(sql))
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    # Show SQL expander
                    show_sql_expander(sql)
                    
                    handle_single_best_query(query, classification, df)
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
            else:
                st.warning("Could not generate query for this request.")
        
        elif query_type == "single_state":
            _route_single_state(query, states, metric, classification, df_features, get_engine_func, city_list)

        elif query_type == "superlative":
            handle_superlative_query(query, classification, get_engine_func)
        
        elif query_type == "comparison":
            handle_comparison(query, classification, df_features, get_engine_func, city_list)
        
        elif query_type == "ranking":
            _route_ranking(query, intent, classification, df_features, get_engine_func, city_list)
        
        # elif query_type == "aggregate":
        #     handle_sql_query(query, classification, df_features, get_engine_func, city_list)

        elif query_type == "aggregate":
            handle_aggregate_query(query, classification, get_engine_func)
        
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
    if not cities or len(cities) == 0:
        st.warning("Could not identify city from your query.")
        _show_fallback_data(df_features)
        return
    
    city_name = cities[0]
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


 
# SINGLE CITY HANDLER
 

def handle_single_city_query(query, city_name, df_features, get_engine_func, metric=None):
    """
    Handle queries about a specific city.
    
    Examples:
        - "Population of Denver" → Shows population metric card
        - "Tell me about Austin" → Shows full city profile
    """
    engine = get_engine_func()
    q_lower = query.lower()
    
    # Find the city AND get RAG context in one call
    city_data = _find_city(city_name, df_features, engine, include_rag=True)
    
    if not city_data:
        st.warning(f"City '{city_name}' not found in our database.")
        handle_gpt_knowledge_fallback(query)
        return
    
    city = city_data.get("city", city_name)
    state = city_data.get("state", "")
    rag_context = city_data.pop('rag_context', None)  # Extract and remove from dict
    
    # Determine metric from query or parameter
    metric_info = _detect_metric(q_lower, metric)
    
    if metric_info:
        _display_city_metric_card(query, city, state, city_data, metric_info, rag_context)
    else:
        # Show full city profile
        show_city_profile_card(city, state, pd.Series(city_data))


def _find_city(city_name, df_features, engine, include_rag: bool = False):
    """
    Find city in database with fuzzy matching fallback.
    
    Args:
        city_name: Name of city to find
        df_features: DataFrame with city features
        engine: Database engine
        include_rag: If True, also fetch RAG context (reduces redundant calls)
        
    Returns:
        dict with city data, or None if not found
        If include_rag=True, adds 'rag_context' key to the dict
    """
    # Try exact match
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    if result:
        city_data = dict(result._mapping)
        if include_rag:
            city_data['rag_context'] = get_cached_rag_context(
                city_data.get('city', city_name), 
                city_data.get('state', '')
            )
        return city_data
    
    # Try fuzzy match
    city_list = df_features["city"].unique().tolist() if not df_features.empty else []
    matched_city, score = fuzzy_match_city(city_name, city_list)
    
    if matched_city and score > 70:
        sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
        with engine.connect() as conn:
            result = conn.execute(sql, {"city": matched_city}).fetchone()
        if result:
            city_data = dict(result._mapping)
            if include_rag:
                city_data['rag_context'] = get_cached_rag_context(
                    city_data.get('city', matched_city), 
                    city_data.get('state', '')
                )
            return city_data
    
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


def _display_city_metric_card(query, city, state, city_data, metric_info, rag_context=None):
    """Display a single metric card for a city with ENHANCED insight."""
    if not metric_info or not isinstance(metric_info, (tuple, list)) or len(metric_info) < 2:
        show_city_profile_card(city, state, pd.Series(city_data))
        return
    
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
    
    # Generate ENHANCED insight (pass RAG context to avoid re-fetching)
    enhanced_insight = generate_enhanced_insight(query, city_data, context_type="profile", rag_context=rag_context)
    
    if enhanced_insight:
        show_styled_insight(enhanced_insight, color_scheme="purple")
        st.caption("Based on retrieved knowledge")
    else:
        # Fallback to basic insight
        _generate_rag_city_insight(query, city, state, city_data, metric_name)


def _format_metric_value(value):
    """Format a metric value for display."""
    if isinstance(value, (int, float)) and value > 1000:
        return f"{int(value):,}"
    elif isinstance(value, float):
        return f"{value:.1f}"
    return str(value)

def _generate_rag_city_insight(query, city, state, city_data, metric_name):
    """Generate insight using RAG context only."""
    if not RAG_AVAILABLE or not get_city_rag_context:
        # No RAG - just show the data without AI interpretation
        return
    
    try:
        rag_context = get_city_rag_context(city, state)
        if not rag_context:
            return
        
        context_parts = []
        for chunk_type, texts in rag_context.items():
            if texts:
                context_parts.append(f"{chunk_type}: {texts[0][:150]}")
        
        if not context_parts:
            return
        
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        pop = city_data.get('population', 'N/A')
        pop_str = f"{pop:,}" if isinstance(pop, (int, float)) else str(pop)
        
        prompt = f"""Based ONLY on this retrieved information about {city}, {state}:

Database: Population {pop_str}, Median Age {city_data.get('median_age')}, Household Size {city_data.get('avg_household_size')}

Retrieved Knowledge:
{chr(10).join(context_parts)}

The user asked about {metric_name}.

Give 1-2 sentences using ONLY the information above. Do not add external knowledge."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        show_styled_insight(summary, color_scheme="purple")
        
        st.markdown("""
        <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.25rem;">
            Based on retrieved knowledge
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass  # Silently fail - no AI insight if RAG fails

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
        
        show_styled_insight(summary, color_scheme="purple")  # or "green" or "blue"
        
    except Exception:
        pass


 
# STATE POPULATION HANDLER
 

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        
        prompt = f"""Based ONLY on this database data about {state_name}:

        - Total population: {formatted_pop}
        - Number of cities: {city_count}
        - Average median age: {avg_age:.1f} years
        - Average household size: {avg_household:.2f}

        Summarize what this data shows about {state_name} in 1-2 sentences.
        Use ONLY the numbers provided. Do not add external knowledge about rankings or comparisons."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        
        show_styled_insight(summary, color_scheme="purple")  # or "green" or "blue"
        
    except Exception:
        pass


def _show_top_cities_in_state(state_name, engine):
    """Show top 5 cities in a state."""
    sql = text(f"SELECT TOP 5 city, population FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state) ORDER BY population DESC")
    with engine.connect() as conn:
        result = conn.execute(sql, {"state": state_name})
        top_cities = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if not top_cities.empty:
        st.markdown("""
        <div style="
            font-size: 1.1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            bborder-bottom: 2px solid rgba(102, 126, 234, 0.3);
        ">
            Largest Cities
        </div>
        """, unsafe_allow_html=True)
        
        # Rename columns for display
        display_df = top_cities.copy()
        display_df.columns = ['City', 'Population']
        display_df["Population"] = display_df["Population"].apply(lambda x: f"{int(x):,}")
        
        st.dataframe(
            display_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "City": st.column_config.TextColumn("City", width="medium"),
                "Population": st.column_config.TextColumn("Population", width="medium"),
            }
        )


 
# SUPERLATIVE HANDLER
 

def handle_superlative_query(query, classification, get_engine_func):
    """
    Handle superlative queries like "top 10 largest cities in Florida".
    """
    engine = get_engine_func()
    
    # Extract from classification
    metric = classification.get("metric", "population")
    direction = classification.get("direction", "highest")
    states = classification.get("states", [])
    query_limit = classification.get("limit", 10) or 10  # Default to 10
    
    # Map metric to column
    metric_map = {
        "population": "population",
        "median_age": "median_age",
        "age": "median_age",
        "household": "avg_household_size",
        "avg_household_size": "avg_household_size",
    }
    metric_column = metric_map.get(metric, "population")
    
    # Determine order
    order = "DESC" if direction == "highest" else "ASC"
    
    # Build and execute query
    state_filter = states[0] if states else None
    
    try:
        if state_filter:
            sql = text(f"SELECT TOP {query_limit} * FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state) ORDER BY {metric_column} {order}")
            with engine.connect() as conn:
                result = conn.execute(sql, {"state": state_filter})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
        else:
            sql = text(f"SELECT TOP {query_limit} * FROM {DB_TABLE_NAME} ORDER BY {metric_column} {order}")
            with engine.connect() as conn:
                result = conn.execute(sql)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
        # Show SQL
        if state_filter:
            show_sql_expander(f"SELECT TOP {query_limit} * FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER('{state_filter}') ORDER BY {metric_column} {order}")
        else:
            show_sql_expander(f"SELECT TOP {query_limit} * FROM {DB_TABLE_NAME} ORDER BY {metric_column} {order}")
            
        if df.empty:
            st.warning("No results found.")
            return
        
        # Display based on limit
        metric_label = metric_column.replace("_", " ").title()
        direction_label = "Highest" if direction == "highest" else "Lowest"
        state_label = f" in {state_filter}" if state_filter else ""
        if query_limit > 1:
            # Show as table for multiple results
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
                     Top {query_limit} {direction_label} {metric_label}{state_label}
                </div>
                <div style="font-size: 2.5rem; font-weight: 700;">
                    {len(df)} cities
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show full table
            show_city_table(df, f"Top {query_limit} Cities by {metric_label}", True)
        else:
            # Single result - show card format
            winner = df.iloc[0]
            city_name = winner.get('city', 'Unknown')
            state_name = winner.get('state', '')
            value = winner.get(metric_column, 0)
            
            # Format value
            if isinstance(value, (int, float)) and value > 1000:
                formatted_value = f"{int(value):,}"
            else:
                formatted_value = str(value)
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                padding: 2rem;
                margin-bottom: 1.5rem;
                color: white;
                text-align: center;
            ">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">
                     {direction_label.upper()} {metric_label.upper()}{state_label.upper()}
                </div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {city_name}, {state_name}
                </div>
                <div style="font-size: 2.5rem; font-weight: 700;">
                    {formatted_value}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show runners up
            if len(df) > 1:
                st.markdown("### Runners Up")
                for idx, row in df.iloc[1:5].iterrows():
                    row_value = row.get(metric_column, 0)
                    if isinstance(row_value, (int, float)) and row_value > 1000:
                        row_formatted = f"{int(row_value):,}"
                    else:
                        row_formatted = str(row_value)
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.05);
                        border-radius: 8px;
                        padding: 0.75rem 1rem;
                        margin-bottom: 0.5rem;
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span>{row.get('city', '')}, {row.get('state', '')}</span>
                        <span style="font-weight: 600;">{row_formatted}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error in superlative query: {str(e)}")


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
    show_styled_insight(summary, color_scheme="purple")
    
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
                 Top {limit} {direction_label} {metric_label}{state_label}
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
    
    # # AI insight
    # _generate_superlative_insight(query, city, state, direction, metric_display, formatted_value)


def _display_runners_up(runners_df, metric_column):
    """Display runners-up for superlative query."""
    st.markdown("#### Runners Up")
    
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
        
        show_styled_insight(summary, color_scheme="purple")  # or "green" or "blue"
        
    except Exception:
        pass


 
# ML RANKING HANDLER
 

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
                    st.markdown("### Why These Cities?")
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


 
# COMPARISON HANDLER
 

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


 
# LIFESTYLE HANDLER
 

def handle_lifestyle_query(query, city_name, classification, df_features, get_engine_func, city_list=None):
    """
    Handle lifestyle queries with RAG enhancement.
    
    Example: "Life in Dallas"
    
    NEW: Uses RAG to provide richer, more detailed answers!
    """
    engine = get_engine_func()
    
    # Get basic city data from database
    sql = text(f"SELECT * FROM dbo.cities WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    if not result:
        st.warning(f"'{city_name.title()}' not found in our database. Showing general information.")
        handle_gpt_knowledge_fallback(query)
        return
    
    city_data = dict(result._mapping)
    city = city_data.get("city", city_name)
    state = city_data.get("state", "")
    
    # Show city profile card with database metrics
    show_city_profile_card(city, state, pd.Series(city_data))
    
    # =========================================================================
    # NEW: Try RAG-powered lifestyle answer
    # =========================================================================
    st.markdown("### What's Life Like?")
    
    # Method 1: Try full RAG answer from city_rag_chunks (30K embeddings)
    if RAG_AVAILABLE and rag_answer:
        try:
            # Get RAG-powered answer focused on this city
            rag_response = get_lifestyle_rag_answer(city, state, city_data)
            
            if rag_response:
                st.markdown(rag_response)
                
                # Show RAG source indicator
                st.markdown("""
                <div style="
                    font-size: 0.75rem;
                    color: #a0aec0;
                    margin-top: 0.5rem;
                    padding-top: 0.5rem;
                    border-top: 1px solid rgba(255,255,255,0.1);
                ">
                    Enhanced with knowledge from our city database
                </div>
                """, unsafe_allow_html=True)
                return
                
        except Exception as e:
            print(f"RAG lifestyle error: {e}")
    
    # Method 2: Try lifestyle_rag with city_profiles table
    if LIFESTYLE_RAG_AVAILABLE and try_build_lifestyle_card:
        try:
            card_data = try_build_lifestyle_card(query)
            if card_data and card_data.get("ai_summary"):
                st.markdown(card_data["ai_summary"])
                return
        except Exception as e:
            print(f"Lifestyle RAG error: {e}")
    
    # Method 3: Fallback to database-only info (no general knowledge)
    pop = city_data.get('population', 'N/A')
    age = city_data.get('median_age', 'N/A')
    household = city_data.get('avg_household_size', 'N/A')

    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
    ">
        <div style="color: #a0aec0;">
            <strong>{city}</strong> has a population of <strong>{pop:,}</strong>, 
            median age of <strong>{age}</strong>, and average household size of <strong>{household}</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Detailed lifestyle information requires enhanced knowledge base data.")



 
# SQL HANDLER (DEFAULT)
 

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


 
# ADDITIONAL HANDLERS
 

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
    """
    Fallback handler - now shows data only, no general knowledge.
    """
    from hybrid_classifier import is_city_related_query
    
    if not is_city_related_query(query):
        show_out_of_scope()
        return
    
    # Try RAG first
    if RAG_AVAILABLE and rag_answer:
        try:
            rag_response = rag_answer(query)
            if rag_response:
                st.markdown(rag_response)
                st.markdown("""
                <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.5rem;">
                    Based on retrieved knowledge from our database
                </div>
                """, unsafe_allow_html=True)
                return
        except:
            pass
    
    # No RAG - show message
    st.info("This query requires data we don't currently have in our knowledge base. Try asking about city demographics, population, or lifestyle.")


def _show_fallback_data(df_features):
    """Show fallback data when query fails."""
    st.markdown("### Available Data")
    if not df_features.empty:
        show_city_table(df_features.head(10), "Sample Cities", True)

 
# FILTER HANDLER
 

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

    # Show SQL
    if state_filter:
        show_sql_expander(f"SELECT * FROM {DB_TABLE_NAME} WHERE {metric} {op_sql} {filter_value} AND LOWER(state) = LOWER('{state_filter}') ORDER BY {metric} DESC")
    else:
        show_sql_expander(f"SELECT * FROM {DB_TABLE_NAME} WHERE {metric} {op_sql} {filter_value} ORDER BY {metric} DESC")
    
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


 
# SIMILAR CITIES HANDLER
 

def handle_similar_cities_query(query, city_name, df_features, get_engine_func):
    """
    OPTIMIZED: Handle queries like "cities similar to Chicago".
    """
    engine = get_engine_func()
    
    # Get reference city data
    sql = text(f"SELECT * FROM dbo.cities WHERE LOWER(city) = LOWER(:city)")
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
    
    # =========================================================================
    # OPTIMIZED: Use demographic similarity FIRST (fast), RAG as enhancement
    # =========================================================================
    
    # Step 1: Get demographically similar cities (FAST - pure SQL)
    pop_range_low = ref_pop * 0.5
    pop_range_high = ref_pop * 2.0
    age_range_low = ref_age - 5
    age_range_high = ref_age + 5
    
    sql = text(f"""
        SELECT TOP 15 * FROM dbo.cities 
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
    
    # Step 2: OPTIONAL - Enhance top results with RAG similarity scores
    # Only process top 15 instead of searching all 30K
    if RAG_AVAILABLE and rag_search:
        try:
            # Get RAG context for reference city
            from core.rag_search import get_city_rag_context
            ref_context = get_city_rag_context(city_name, ref_city.get("state"))
            
            if ref_context:
                # Build a query from the reference city's characteristics
                context_summary = []
                for chunk_type, texts in ref_context.items():
                    if texts:
                        context_summary.append(texts[0][:100])
                
                if context_summary:
                    # Use the reference city's context as the query
                    # This is MUCH faster than open-ended search
                    search_query = f"{city_name} lifestyle: " + " ".join(context_summary[:2])
                    
                    # Search with the reference city's STATE as hint
                    # This narrows the search space
                    rag_results = rag_search(
                        search_query, 
                        top_k=20,
                        state_filter=None  # Keep open to find cities in other states
                    )
                    
                    # Create a RAG score boost map
                    rag_boost = {}
                    for chunk in rag_results:
                        chunk_city = chunk.get("city", "").lower()
                        if chunk_city and chunk_city != city_name.lower():
                            rag_boost[chunk_city] = rag_boost.get(chunk_city, 0) + chunk.get("score", 0)
                    
                    # Add RAG score to dataframe
                    df["rag_score"] = df["city"].str.lower().map(rag_boost).fillna(0)
                    
                    # Re-rank combining demographic + RAG similarity
                    df["combined_score"] = df["rag_score"] * 100  # Weight RAG heavily
                    df = df.sort_values("combined_score", ascending=False)
                    
                    st.caption("Results ranked by demographic and lifestyle similarity")
        except Exception as e:
            print(f"RAG enhancement error: {e}")
            st.caption("Results ranked by demographic similarity")
    else:
        st.caption("Results ranked by demographic similarity")
    
    # Show results
    st.markdown("### Similar Cities")
    show_city_table(df.head(10), f"Cities Similar to {city_name}", True)
    
    # Generate ENHANCED insight using reference city
    ref_rag_context = get_cached_rag_context(ref_city.get('city', ''), ref_city.get('state', ''))
    enhanced_insight = generate_enhanced_insight(
        query, 
        ref_city, 
        context_type="similarity",
        rag_context=ref_rag_context
    )
    
    if enhanced_insight:
        show_styled_insight(enhanced_insight, color_scheme="purple")
        st.caption("Enhanced with lifestyle context")
    elif len(df) >= 3:
        # Fallback to basic summary
        top_3 = df.head(3)
        city_names = ", ".join(top_3["city"].tolist())
        avg_pop = top_3["population"].mean()
        
        summary = f"Based on demographics, cities most similar to {city_name} include {city_names}. These cities share similar population sizes (avg: {avg_pop:,.0f}) and age distributions."
        show_styled_insight(summary, color_scheme="purple")

 
# GENERAL KNOWLEDGE HANDLER
 

def handle_general_knowledge_query(query, classification):
    """
    Handle city/state questions - now uses RAG or shows data only.
    General knowledge is disabled.
    """
    cities = classification.get("cities", [])
    states = classification.get("states", [])
    
    # Try RAG first
    if RAG_AVAILABLE and rag_answer:
        try:
            rag_response = rag_answer(query)
            if rag_response:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 16px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    color: white;
                ">
                    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                        From Our Knowledge Base
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(rag_response)
                
                # Show related database info
                if cities or states:
                    _show_related_database_info(cities, states)
                return
        except Exception as e:
            print(f"RAG error: {e}")
    
    # No RAG available - show database data only
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    ">
        <div style="font-size: 0.9rem; color: #a0aec0;">
            ℹ️ Showing available database information for this query.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show related database info only
    if cities or states:
        _show_related_database_info(cities, states)
    else:
        st.info("Please try a more specific query about a city or state.")


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
                st.markdown("#### From Our Database")
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
                st.markdown("#### From Our Database")
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

    # Show SQL
    if state_filter:
        show_sql_expander(f"SELECT * FROM {DB_TABLE_NAME} WHERE {metric} BETWEEN {min_val} AND {max_val} AND LOWER(state) = LOWER('{state_filter}') ORDER BY {metric} DESC")
    else:
        show_sql_expander(f"SELECT * FROM {DB_TABLE_NAME} WHERE {metric} BETWEEN {min_val} AND {max_val} ORDER BY {metric} DESC")
    
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

 
# LLM SQL HANDLER - Execute SQL generated by LLM
 

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
    with st.expander("View Generated SQL", expanded=False):
        st.code(sql_query, language="sql")
        if explanation:
            st.caption(f"{explanation}")
    
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
    st.markdown("### Results")
    
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
    st.markdown("### Results by Group")
    
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
    
    # ADD THIS: Generate AI insight
    _generate_group_by_insight(query, df)


def _generate_group_by_insight(query, df):
    """Generate AI insight for GROUP BY results."""
    if df is None or df.empty:
        return
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        # Get top and bottom entries
        top_3 = df.head(3).to_dict('records')
        bottom_3 = df.tail(3).to_dict('records') if len(df) > 3 else []
        
        prompt = f"""The user asked: "{query}"

Results show {len(df)} groups.
Top 3: {top_3}
Bottom 3: {bottom_3}

Write 1-2 sentences highlighting the key insight from this data.
What's the most interesting pattern or finding?"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        show_styled_insight(summary, color_scheme="purple")
        
    except Exception:
        pass


def _display_comparison_result(df, cities, query):
    """Display comparison between cities."""
    st.markdown("### Comparison")
    
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
    """Display filter/superlative results with consistent UI structure."""
    query_type = classification.get("query_type", "filter")
    explanation = classification.get("explanation", "")
    intent = classification.get("intent", "general")
    
    # SECTION 1: Count Card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 3rem; font-weight: 700;">{len(df)}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">cities found</div>
    </div>
    """, unsafe_allow_html=True)
    
    # SECTION 2: Results Table
    show_city_table(df, "Results", True)
    
    # SECTION 3: AI Summary (RAG-only)
    st.markdown("### Insights")
    _generate_rag_only_summary(query, df, classification)
    
def _generate_rag_only_summary(query, df, classification):
    """
    Generate AI summary using ONLY RAG context.
    No general knowledge fallback - if RAG has no context, show data-only summary.
    """
    if df is None or df.empty:
        return
    
    explanation = classification.get("explanation", "")
    intent = classification.get("intent", "general")
    
    # Build context from results data
    top_cities = df.head(5)
    cities_info = []
    for _, row in top_cities.iterrows():
        city = row.get('city', 'Unknown')
        state = row.get('state', '')
        pop = row.get('population', 0)
        age = row.get('median_age', 0)
        household = row.get('avg_household_size', 0)
        cities_info.append(f"- {city}, {state}: Population {pop:,}, Median Age {age}, Household Size {household}")
    
    cities_context = "\n".join(cities_info)
    
    # Try to get RAG context
    rag_context = ""
    rag_available = False
    
    if RAG_AVAILABLE and get_city_rag_context:
        try:
            top_city = df.iloc[0].get('city', '')
            top_state = df.iloc[0].get('state', '')
            rag_data = get_city_rag_context(top_city, top_state)
            if rag_data:
                for chunk_type, texts in rag_data.items():
                    if texts:
                        rag_context += f"\n{chunk_type}: {texts[0][:200]}..."
                rag_available = True
        except Exception as e:
            print(f"RAG context error: {e}")
    
    # Only generate AI summary if RAG context is available
    if rag_available and rag_context:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
            
            prompt = f"""Based ONLY on the following retrieved data, provide a brief summary.

Query: "{query}"
Query Logic: {explanation}

Database Results:
{cities_context}

Retrieved Context:
{rag_context}

RULES:
- Use ONLY the information provided above
- Do NOT add any external knowledge
- Keep to 2-3 sentences
- Focus on patterns in the data"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3  # Lower temperature for more factual responses
            )
            
            summary = response.choices[0].message.content.strip()
            
            show_styled_insight(summary, color_scheme="purple")
            
            st.markdown("""
            <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.5rem;">
                Based on retrieved knowledge from our city database
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            # Show data-only summary on error
            _show_data_only_summary(df, explanation)
    else:
        # No RAG context available - show data-only summary
        _show_data_only_summary(df, explanation)


def _show_data_only_summary(df, explanation):
    """Show summary based only on database data (no AI generation)."""
    if df is None or df.empty:
        return
    
    top_city = df.iloc[0]
    city_name = top_city.get('city', 'Unknown')
    state_name = top_city.get('state', '')
    pop = top_city.get('population', 0)
    
    summary = f"Top result: {city_name}, {state_name} with population {pop:,}."
    if explanation:
        summary += f" Query logic: {explanation}"
    
    show_styled_insight(summary, color_scheme="blue")


def _generate_results_ai_summary(query, df, classification):
    """Generate AI summary for query results using available data."""
    
    if df is None or df.empty:
        return
    
    explanation = classification.get("explanation", "")
    intent = classification.get("intent", "general")
    query_type = classification.get("query_type", "filter")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        # Build context from results
        top_cities = df.head(5)
        cities_info = []
        for _, row in top_cities.iterrows():
            city = row.get('city', 'Unknown')
            state = row.get('state', '')
            pop = row.get('population', 0)
            age = row.get('median_age', 0)
            household = row.get('avg_household_size', 0)
            cities_info.append(f"- {city}, {state}: Pop {pop:,}, Age {age}, Household {household}")
        
        cities_context = "\n".join(cities_info)
        
        # Try to get RAG context if available
        rag_context = ""
        if RAG_AVAILABLE and get_city_rag_context:
            try:
                # Get RAG context for top city
                top_city = df.iloc[0].get('city', '')
                top_state = df.iloc[0].get('state', '')
                rag_data = get_city_rag_context(top_city, top_state)
                if rag_data:
                    for chunk_type, texts in rag_data.items():
                        if texts:
                            rag_context += f"\n{chunk_type}: {texts[0][:150]}..."
            except:
                pass
        
        prompt = f"""The user asked: "{query}"

Query Logic: {explanation}

Top Results:
{cities_context}
{f"Additional Context:{rag_context}" if rag_context else ""}

Write 2-3 sentences summarizing why these cities match the query. 
Be specific about what makes these cities stand out for "{intent if intent != 'general' else query_type}" queries.
Focus on the data patterns you see."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Display the summary
        show_styled_insight(summary, color_scheme="purple")  # or "green" or "blue"
        
        # Show RAG indicator if used
        if rag_context:
            st.markdown("""
            <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.5rem;">
                Enhanced with knowledge from our city database
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        # Fallback: show the explanation if available
        if explanation:
            st.markdown(f"""
            <div style="
                background: rgba(102, 126, 234, 0.1);
                border-left: 4px solid #667eea;
                border-radius: 8px;
                padding: 1rem;
                margin-top: 1rem;
            ">
                <div style="font-size: 0.85rem; color: #667eea; margin-bottom: 0.5rem;">Query Logic</div>
                <div style="color: #a0aec0;">{explanation}</div>
            </div>
            """, unsafe_allow_html=True)

 
# AGGREGATE HANDLER
 

def handle_aggregate_query(query, classification, get_engine_func):
    """
    Handle aggregate queries like:
    - "total population of Texas"
    - "how many cities in California"
    - "average median age in Ohio"
    """
    engine = get_engine_func()
    
    states = classification.get("states", [])
    metric = classification.get("metric", "count")
    state_filter = states[0] if states else None
    
    try:
        if metric == "total_population":
            if state_filter:
                sql = text(f"SELECT SUM(population) as total_population, COUNT(*) as city_count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
                with engine.connect() as conn:
                    result = conn.execute(sql, {"state": state_filter}).fetchone()
            else:
                sql = text(f"SELECT SUM(population) as total_population, COUNT(*) as city_count FROM {DB_TABLE_NAME}")
                with engine.connect() as conn:
                    result = conn.execute(sql).fetchone()
            # Show SQL
            if state_filter:
                show_sql_expander(f"SELECT SUM(population) as total_population, COUNT(*) as city_count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER('{state_filter}')")
            else:
                show_sql_expander(f"SELECT SUM(population) as total_population, COUNT(*) as city_count FROM {DB_TABLE_NAME}")
                
            if result:
                total_pop = result[0] or 0
                city_count = result[1] or 0
                
                state_label = f" in {state_filter}" if state_filter else ""
                
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
                        Total Population{state_label}
                    </div>
                    <div style="font-size: 3rem; font-weight: 700;">
                        {total_pop:,}
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
                        across {city_count} cities
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        elif metric == "count":
            if state_filter:
                sql = text(f"SELECT COUNT(*) as city_count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
                with engine.connect() as conn:
                    result = conn.execute(sql, {"state": state_filter}).fetchone()
            else:
                sql = text(f"SELECT COUNT(*) as city_count FROM {DB_TABLE_NAME}")
                with engine.connect() as conn:
                    result = conn.execute(sql).fetchone()

            # Show SQL
            if state_filter:
                show_sql_expander(f"SELECT COUNT(*) as city_count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER('{state_filter}')")
            else:
                show_sql_expander(f"SELECT COUNT(*) as city_count FROM {DB_TABLE_NAME}")
            
            if result:
                city_count = result[0] or 0
                state_label = f" in {state_filter}" if state_filter else ""
                
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
                        Cities{state_label}
                    </div>
                    <div style="font-size: 3rem; font-weight: 700;">
                        {city_count}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        elif metric in ["average_age", "avg_median_age"]:
            if state_filter:
                sql = text(f"SELECT AVG(median_age) as avg_age, COUNT(*) as city_count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state)")
                with engine.connect() as conn:
                    result = conn.execute(sql, {"state": state_filter}).fetchone()
            else:
                sql = text(f"SELECT AVG(median_age) as avg_age, COUNT(*) as city_count FROM {DB_TABLE_NAME}")
                with engine.connect() as conn:
                    result = conn.execute(sql).fetchone()

            # Show SQL
            if state_filter:
                show_sql_expander(f"SELECT AVG(median_age) as avg_age, COUNT(*) as city_count FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER('{state_filter}')")
            else:
                show_sql_expander(f"SELECT AVG(median_age) as avg_age, COUNT(*) as city_count FROM {DB_TABLE_NAME}")
            
            if result:
                avg_age = result[0] or 0
                city_count = result[1] or 0
                state_label = f" in {state_filter}" if state_filter else ""
                
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
                        Average Median Age{state_label}
                    </div>
                    <div style="font-size: 3rem; font-weight: 700;">
                        {avg_age:.1f}
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">
                        across {city_count} cities
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Default to count
            handle_sql_query(query, classification, None, get_engine_func, None)
    
    except Exception as e:
        st.error(f"Error processing aggregate query: {str(e)}")

def _generate_similarity_insight(city_name, similar_cities):
    """Generate insight about similar cities using ONLY database data."""
    if not similar_cities:
        return
    
    city_names = [c.get('city', '') for c in similar_cities[:5]]
    avg_pop = sum(c.get('population', 0) for c in similar_cities) / len(similar_cities)
    avg_age = sum(c.get('median_age', 0) for c in similar_cities) / len(similar_cities)
    
    summary = f"Found {len(similar_cities)} cities similar to {city_name}. These include {', '.join(city_names[:3])}. Average population: {avg_pop:,.0f}, average median age: {avg_age:.1f}."
    
    show_styled_insight(summary, color_scheme="purple")


 
# STEP 4: ADD RAG-ENHANCED CITY PROFILE (Optional enhancement for single_city)
 

def _get_rag_enhanced_insight(city_name, state, city_data, query):
    """
    Get RAG-enhanced insight for a city.
    Call this after showing the basic city card.
    """
    if not RAG_AVAILABLE or not get_city_rag_context:
        return None
    
    try:
        # Get RAG context for this city
        rag_context = get_city_rag_context(city_name, state)
        
        if not rag_context:
            return None
        
        # Build context string
        context_parts = []
        for chunk_type, texts in rag_context.items():
            if texts:
                context_parts.append(f"{chunk_type}: {texts[0][:200]}...")
        
        if not context_parts:
            return None
        
        # Generate enhanced insight
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        pop = city_data.get('population', 'N/A')
        pop_str = f"{pop:,}" if isinstance(pop, (int, float)) else str(pop)
        
        prompt = f"""Based on this information about {city_name}, {state}:

Database: Population {pop_str}, Median Age {city_data.get('median_age')}, Household Size {city_data.get('avg_household_size')}

Knowledge Base:
{chr(10).join(context_parts)}

User asked: "{query}"

Give a 2-3 sentence insightful answer. Use only the information provided."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"RAG insight error: {e}")
        return None

def handle_single_best_query(query, classification, df):
    """Handle 'best city for X' queries with ENHANCED insight."""
    
    intent = classification.get("intent", "general")
    explanation = classification.get("explanation", "")
    
    if df is not None and len(df) > 0:
        city_row = df.iloc[0]
        city_name = city_row.get("city", "Unknown")
        state_name = city_row.get("state", "Unknown")
        
        # Display the city card
        st.markdown(f"### Best City for {intent.replace('_', ' ').title()}")
        st.markdown(f"## {city_name}, {state_name}")
        
        # Show stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Population", f"{city_row.get('population', 0):,}")
        col2.metric("Median Age", f"{city_row.get('median_age', 0)}")
        col3.metric("Household Size", f"{city_row.get('avg_household_size', 0)}")
        
        # Generate ENHANCED insight
        city_data = dict(city_row)
        rag_context = get_cached_rag_context(city_data.get('city', ''), city_data.get('state', ''))
        enhanced_insight = generate_enhanced_insight(query, city_data, context_type="ranking", rag_context=rag_context)
        
        if enhanced_insight:
            show_styled_insight(enhanced_insight, color_scheme="purple")
            st.caption("Based on retrieved knowledge")
        else:
            # Fallback
            _display_ai_summary(query, city_name, state_name, city_row, intent, explanation)
        
def _display_ai_summary(query, city_name, state_name, city_data, intent, explanation):
    """Display summary using ONLY database data - no GPT generation."""
    
    pop = city_data.get('population', 0)
    age = city_data.get('median_age', 0)
    household = city_data.get('avg_household_size', 0)
    
    if intent == "families":
        reason = f"with household size of {household:.2f}" if household > 2.5 else "based on demographic profile"
    elif intent == "retirement":
        reason = f"with median age of {age}" if age > 38 else "based on population characteristics"
    elif intent == "young_professionals":
        reason = f"with younger median age of {age}" if age < 35 else "based on urban opportunities"
    else:
        reason = "based on population and demographic data"
    
    summary = f"{city_name}, {state_name} ranks as a top choice {reason}. Population: {pop:,}, Median Age: {age}, Household Size: {household:.2f}."
    
    show_styled_insight(summary, color_scheme="purple")
    
def show_unified_results_header(query: str, classification: dict):
    """Show consistent results header for all query types."""
    query_type = classification.get("query_type", "")
    source = classification.get("source", "")
    explanation = classification.get("explanation", "")
    
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    ">
        <div style="font-size: 0.75rem; color: #667eea; text-transform: uppercase; letter-spacing: 1px;">
            Results
        </div>
        <div style="font-size: 1rem; color: #e2e8f0; margin-top: 0.25rem;">
            {query}
        </div>
        {f'<div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.5rem;">{explanation}</div>' if explanation else ''}
    </div>
    """, unsafe_allow_html=True)


def show_unified_insight_box(title: str, content: str, is_rag: bool = False):
    """Show consistent insight box."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
        border-left: 4px solid #667eea;
    ">
        <div style="font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">{title}</div>
        <div style="color: #e2e8f0; line-height: 1.6;">{content}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if is_rag:
        st.markdown("""
        <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.5rem;">
            Based on retrieved knowledge from our database
        </div>
        """, unsafe_allow_html=True)


def show_data_only_summary(df: pd.DataFrame, explanation: str = ""):
    """Show data-only summary when RAG is unavailable."""
    if df is None or df.empty:
        return
    
    top_city = df.iloc[0]
    city_name = top_city.get('city', 'Unknown')
    state_name = top_city.get('state', '')
    pop = top_city.get('population', 0)
    
    content = f"Top result: <strong>{city_name}, {state_name}</strong> with population {pop:,}."
    if explanation:
        content += f"<br>Query logic: {explanation}"
    
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    ">
        <div style="font-size: 0.85rem; color: #667eea; margin-bottom: 0.5rem;">Data Summary</div>
        <div style="color: #a0aec0;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

