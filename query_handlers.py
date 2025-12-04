"""
CitySearch AI - Query Handlers
===============================
Routes classified queries to appropriate core modules and display components.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text

from config import DB_TABLE_NAME
from db_config import get_engine

# Safe imports with fallbacks
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
        if score is None: return "Unknown"
        try: score = float(score)
        except: return "Unknown"
        if score < 5: return "Low"
        elif score < 15: return "Medium"
        elif score < 25: return "High"
        return "Excellent"

try:
    from core.query_router import build_sql_with_fallback
except ImportError:
    build_sql_with_fallback = None

try:
    from core.cluster_router import get_cluster_for_city, get_all_clusters
except ImportError:
    get_cluster_for_city = None
    get_all_clusters = None

try:
    from core.cluster_explain import explain_cluster
except ImportError:
    explain_cluster = None

try:
    from core.cluster_labels import CLUSTER_LABELS
except ImportError:
    CLUSTER_LABELS = {}

try:
    from core.rag_search import search_city_rag
except ImportError:
    search_city_rag = None

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


def handle_query(query, classification, df_features, get_engine_func=None, 
                 smart_route_func=None, lifestyle_rag_func=None, classify_intent_func=None):
    """Main query handler."""
    if get_engine_func is None:
        get_engine_func = get_engine
    
    city_list = df_features["city"].unique().tolist() if not df_features.empty else []
    
    if not classification.get("is_city_related", True):
        show_out_of_scope()
        return
    
    # ========================================================
    # SAFETY NET: Override for misclassified queries
    # ========================================================
    q_lower = query.lower()
    original_mode = classification.get("original_mode", "")

    # ========================================================
    # SAFETY NET 0: Single city questions (population of X, tell me about X)
    # ========================================================
    import re
    
    # Pattern: "population of [city]" or "what is the population of [city]"
    pop_match = re.search(r"(?:what is the |what's the )?population (?:of |in )(.+?)(?:\?|$)", q_lower)
    if pop_match:
        city_name = pop_match.group(1).strip().rstrip("?.,!")
        # Find and show just this city
        handle_single_city_query(query, city_name, df_features, get_engine_func)
        return
    
    # Pattern: "tell me about [city]" or "info on [city]"
    about_match = re.search(r"(?:tell me about|info on|information about|details about)\s+(.+?)(?:\?|$)", q_lower)
    if about_match:
        city_name = about_match.group(1).strip().rstrip("?.,!")
        handle_single_city_query(query, city_name, df_features, get_engine_func)
        return
    
    # ========================================================
    # SAFETY NET 1: "Life in X" queries should show lifestyle/profile
    # ========================================================
    if "life in" in q_lower or "living in" in q_lower or "what is it like in" in q_lower:
        import re
        match = re.search(r"(?:life in|living in|what is it like in)\s+(.+?)(?:\?|$)", q_lower)
        if match:
            city_name = match.group(1).strip().rstrip("?.,!")
            handle_lifestyle_query(query, city_name, classification, df_features, get_engine_func, city_list)
            return
    
    # ========================================================
    # SAFETY NET 2: "Best" queries should go to ML ranking
    # ========================================================
    if "best" in q_lower and original_mode not in ["ml_family", "ml_young", "ml_retirement"]:
        # Family keywords
        if any(word in q_lower for word in ["family", "families", "kids", "children", "child", "kid"]):
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, "families")
            return
        
        # Young professionals keywords
        if any(word in q_lower for word in ["young", "professional", "adults", "adult", "career", "millennials"]):
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, "young_professionals")
            return
        
        # Retirement keywords
        if any(word in q_lower for word in ["retire", "retirement", "senior", "seniors", "elderly", "retirees"]):
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, "retirement")
            return
    
    if classification.get("use_gpt_knowledge", False):
        handle_gpt_knowledge_fallback(query)
        return
    
    response_type = classification.get("response_type", "city_list")
    original_mode = classification.get("original_mode", response_type)
    
    try:
        if original_mode in ["sql", "sql_query"]:
            handle_sql_query(query, classification, df_features, get_engine_func, city_list)
        elif original_mode in ["semantic", "semantic_search"]:
            handle_semantic_search(query, classification, df_features, get_engine_func, city_list)
        elif original_mode in ["hybrid", "hybrid_search"]:
            handle_semantic_search(query, classification, df_features, get_engine_func, city_list)
        elif original_mode == "ml_family":
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, "families")
        elif original_mode == "ml_young":
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, "young_professionals")
        elif original_mode == "ml_retirement":
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, "retirement")
        elif original_mode in ["ml_compare_cities", "comparison"]:
            handle_comparison(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "single_city":
            handle_single_city(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "single_state":
            handle_single_state(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "city_list":
            handle_sql_query(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "city_profile":
            handle_city_profile(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "aggregate":
            handle_sql_query(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "recommendation":
            intent = classification.get("specific_intent", "families")
            handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, intent)
        elif response_type == "cluster":
            handle_cluster(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "similar_cities":
            handle_similar_cities(query, classification, df_features, get_engine_func, city_list)
        elif response_type == "lifestyle":
            handle_lifestyle(query, classification, df_features, get_engine_func, city_list)
        else:
            handle_sql_query(query, classification, df_features, get_engine_func, city_list)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        _show_fallback_data(df_features)


def handle_sql_query(query, classification, df_features, get_engine_func, city_list):
    """Handle SQL queries."""
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
    """Build SQL from classification."""
    q = query.lower()
    state_filter = classification.get("state_filter") or (classification.get("mentioned_states", [None])[0] if classification.get("mentioned_states") else None)
    metric = classification.get("metric", "population")
    sort_dir = classification.get("sort_direction")
    limit = classification.get("limit", 10) or 10
    
    if "how many" in q or "count" in q:
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
    
    order = "ORDER BY population DESC"
    if sort_dir:
        order = f"ORDER BY {metric} {'DESC' if sort_dir == 'highest' else 'ASC'}"
    
    # Use engine.connect() and execute with text() for parameterized queries
    if state_filter:
        sql = text(f"SELECT TOP {limit} * FROM {DB_TABLE_NAME} WHERE LOWER(state) = LOWER(:state) {order}")
        with engine.connect() as conn:
            result = conn.execute(sql, {"state": state_filter})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    else:
        sql = text(f"SELECT TOP {limit} * FROM {DB_TABLE_NAME} {order}")
        with engine.connect() as conn:
            result = conn.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df


def _display_sql_results(query, classification, df):
    """Display SQL results."""
    if "count" in df.columns:
        val = df["count"].iloc[0]
        state = df.get("state", pd.Series([None])).iloc[0]
        label = f"Cities in {state}" if state else "Total Cities"
        show_aggregate_card(label, str(val), "cities")
        return
    
    if len(df) == 1:
        row = df.iloc[0]
        show_city_profile_card(row.get("city", ""), row.get("state", ""), row)
        return
    
    sort_dir = classification.get("sort_direction")
    if sort_dir:
        metric = classification.get("metric", "population")
        top = df.iloc[0]
        runners = df.iloc[1:4] if len(df) > 1 else pd.DataFrame()
        label = f"{'Highest' if sort_dir == 'highest' else 'Lowest'} {metric.replace('_',' ').title()}"
        show_superlative_card(top.get("city",""), top.get("state",""), top.get(metric,"N/A"), label, runners, None)
        if len(df) > 4:
            show_city_table(df, "All Results", True)
    else:
        show_city_table(df, "Results", True)


def handle_semantic_search(query, classification, df_features, get_engine_func, city_list):
    """Handle semantic search."""
    if not semantic_city_search:
        handle_sql_query(query, classification, df_features, get_engine_func, city_list)
        return
    try:
        results = semantic_city_search(query, top_k=10, state_filter=classification.get("state_filter"))
        if not results:
            st.warning("No matching cities found.")
            return
        engine = get_engine_func()
        data = []
        for city, state in results:
            sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city)=LOWER(:c) AND LOWER(state)=LOWER(:s)")
            with engine.connect() as conn:
                r = conn.execute(sql, {"c": city, "s": state}).fetchone()
                if r: data.append(dict(r._mapping))
        if data:
            show_city_table(pd.DataFrame(data), "Semantic Results", True)
    except Exception as e:
        st.error(f"Semantic search error: {e}")


def handle_ml_ranking(query, classification, df_features, get_engine_func, city_list, intent):
    """Handle ML ranking."""
    state = classification.get("state_filter") or (classification.get("mentioned_states", [None])[0] if classification.get("mentioned_states") else None)
    
    func = None
    if intent == "families" and run_family_ranking: func = run_family_ranking
    elif intent == "young_professionals" and run_young_ranking: func = run_young_ranking
    elif intent == "retirement" and run_retirement_ranking: func = run_retirement_ranking
    
    df = None
    if func:
        try:
            # Call function without any parameters
            df = func()
            # Filter by state manually if needed
            if state and df is not None and not df.empty and "state" in df.columns:
                filtered = df[df["state"].str.lower() == state.lower()]
                if not filtered.empty:
                    df = filtered
        except Exception as e:
            st.warning(f"ML error: {e}")
    
    # If ML failed or returned empty, use fallback
    if df is None or df.empty:
        df = _fallback_ranking(df_features, intent, state)
    
    if df is not None and not df.empty:
        show_recommendation_card(df.iloc[0], intent, df, query)
        if explain_ml_results:
            try:
                insights = explain_ml_results(query, df)
                if insights:
                    st.markdown("### 🧠 Why These Cities?")
                    st.markdown(insights)
            except: pass
        show_city_table(df, f"Best for {intent.replace('_',' ').title()}", True)
    else:
        st.warning("Could not generate rankings.")


def _fallback_ranking(df_features, intent, state_filter=None):
    """Fallback ranking."""
    if df_features.empty: return None
    df = df_features.copy()
    if state_filter:
        df = df[df["state"].str.lower() == state_filter.lower()]
    if df.empty: return None
    
    if intent == "families":
        df["score"] = df["avg_household_size"]*20 + (50-abs(df["median_age"]-35)) + df["population"]/100000
    elif intent == "young_professionals":
        df["score"] = (45-df["median_age"])*3 + df["population"]/50000
    elif intent == "retirement":
        df["score"] = df["median_age"]*2 + (4-df["avg_household_size"])*10
    else:
        df["score"] = df["population"]/10000
    return df.sort_values("score", ascending=False).head(10)


def handle_single_city(query, classification, df_features, get_engine_func, city_list):
    """Handle single city."""
    cities = classification.get("mentioned_cities", [])
    city = cities[0] if cities else extract_single_city_fuzzy(query, city_list)[0]
    
    if classification.get("sort_direction") and not city:
        _handle_superlative(query, classification, get_engine_func)
        return
    
    if not city:
        st.warning("Could not identify city.")
        return
    
    engine = get_engine_func()
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city)=LOWER(:c)")
    with engine.connect() as conn:
        r = conn.execute(sql, {"c": city}).fetchone()
    
    if r:
        d = dict(r._mapping)
        show_city_profile_card(city, d.get("state",""), pd.Series(d))
    else:
        st.warning(f"City '{city}' not found.")


def handle_single_state(query, classification, df_features, get_engine_func, city_list):
    """Handle single state."""
    states = classification.get("mentioned_states", [])
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
        show_state_metric_card(state, "Population", d.get("total_pop",0), d.get("city_count",0), cities)
    else:
        st.warning(f"State '{state}' not found.")


def handle_comparison(query, classification, df_features, get_engine_func, city_list):
    """Handle comparison."""
    ctype = classification.get("comparison_type", "city_vs_city")
    engine = get_engine_func()
    
    if ctype == "state_vs_state":
        states = classification.get("mentioned_states", [])
        if len(states) < 2:
            r = extract_two_states_fuzzy(query)
            if r and len(r) >= 2:
                states = [r[0], r[1]]
        if len(states) < 2:
            st.warning("Could not find two states to compare.")
            return
        
        def get_state_stats(s):
            sql = text(f"SELECT state, COUNT(*) as cnt, SUM(population) as pop, AVG(median_age) as age FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) GROUP BY state")
            with engine.connect() as conn:
                r = conn.execute(sql, {"s": s}).fetchone()
            return dict(r._mapping) if r else {}
        
        def get_state_cities(s):
            sql = text(f"SELECT TOP 5 * FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) ORDER BY population DESC")
            with engine.connect() as conn:
                result = conn.execute(sql, {"s": s})
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        
        s1, s2 = get_state_stats(states[0]), get_state_stats(states[1])
        c1, c2 = get_state_cities(states[0]), get_state_cities(states[1])
        
        if s1 and s2:
            show_state_comparison(states[0], s1, c1, states[1], s2, c2, query)
        else:
            st.warning("Could not find both states.")
    else:
        # City vs City comparison
        cities = classification.get("mentioned_cities", [])
        if len(cities) < 2:
            r = extract_two_cities_fuzzy(query, city_list)
            if r and len(r) >= 2:
                cities = [r[0], r[1]]
        if len(cities) < 2:
            st.warning("Could not find two cities to compare.")
            return
        
        # Fixed SQL execution
        sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) IN (LOWER(:c1), LOWER(:c2))")
        with engine.connect() as conn:
            result = conn.execute(sql, {"c1": cities[0], "c2": cities[1]})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if len(df) >= 2:
            r1 = df[df["city"].str.lower() == cities[0].lower()].iloc[0]
            r2 = df[df["city"].str.lower() == cities[1].lower()].iloc[0]
            show_city_comparison(r1, r2, query)
        else:
            st.warning("Could not find both cities.")

def handle_city_profile(query, classification, df_features, get_engine_func, city_list):
    """Handle city profile."""
    handle_single_city(query, classification, df_features, get_engine_func, city_list)


def handle_cluster(query, classification, df_features, get_engine_func, city_list):
    """Handle cluster."""
    if not get_cluster_for_city:
        st.warning("Cluster not available.")
        return
    cities = classification.get("mentioned_cities", [])
    city = cities[0] if cities else extract_single_city_fuzzy(query, city_list)[0]
    if city:
        try:
            cid = get_cluster_for_city(city)
            if cid is not None:
                st.markdown(f"### {city} - Cluster {cid}")
                st.markdown(CLUSTER_LABELS.get(cid, f"Cluster {cid}"))
        except Exception as e:
            st.error(str(e))


def handle_similar_cities(query, classification, df_features, get_engine_func, city_list):
    """Handle similar cities."""
    if not semantic_city_search:
        st.warning("Similar cities not available.")
        return
    cities = classification.get("mentioned_cities", [])
    city = cities[0] if cities else extract_single_city_fuzzy(query, city_list)[0]
    if not city:
        st.warning("Could not identify city.")
        return
    try:
        results = semantic_city_search(f"cities like {city}", top_k=10)
        results = [(c,s) for c,s in results if c.lower() != city.lower()]
        if results:
            engine = get_engine_func()
            data = []
            for c,s in results[:10]:
                sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city)=LOWER(:c) AND LOWER(state)=LOWER(:s)")
                with engine.connect() as conn:
                    r = conn.execute(sql, {"c": c, "s": s}).fetchone()
                    if r: data.append(dict(r._mapping))
            if data:
                show_city_table(pd.DataFrame(data), f"Cities Similar to {city}", True)
    except Exception as e:
        st.error(str(e))


def handle_lifestyle(query, classification, df_features, get_engine_func, city_list):
    """Handle lifestyle."""
    if try_build_lifestyle_card:
        try:
            r = try_build_lifestyle_card(query)
            if r:
                show_lifestyle_card(r.get("city",""), r.get("state",""), r.get("population",""), r.get("median_age",""), r.get("household_size",""), r.get("description",""), r.get("ai_summary",""))
                return
        except: pass
    handle_city_profile(query, classification, df_features, get_engine_func, city_list)


def handle_gpt_knowledge_fallback(query, classification=None, df_features=None, get_engine_func=None, city_list=None):
    """GPT fallback."""
    from hybrid_classifier import is_city_related_query
    if not is_city_related_query(query):
        show_out_of_scope()
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Answer about US cities: {query}"}],
            max_tokens=500
        )
        show_ai_response(r.choices[0].message.content, is_fallback=True)
    except Exception as e:
        st.error(str(e))


def _handle_superlative(query, classification, get_engine_func):
    """Handle superlative."""
    metric = classification.get("metric", "population")
    sort_dir = classification.get("sort_direction", "highest")
    state = classification.get("state_filter") or (classification.get("mentioned_states", [None])[0] if classification.get("mentioned_states") else None)
    
    engine = get_engine_func()
    order = "DESC" if sort_dir == "highest" else "ASC"
    
    if state:
        sql = text(f"SELECT TOP 5 * FROM {DB_TABLE_NAME} WHERE LOWER(state)=LOWER(:s) ORDER BY {metric} {order}")
        with engine.connect() as conn:
            result = conn.execute(sql, {"s": state})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        sql = text(f"SELECT TOP 5 * FROM {DB_TABLE_NAME} ORDER BY {metric} {order}")
        with engine.connect() as conn:
            result = conn.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    if df.empty:
        st.warning("No results.")
        return
    
    top = df.iloc[0]
    runners = df.iloc[1:] if len(df) > 1 else pd.DataFrame()
    label = f"{'Highest' if sort_dir=='highest' else 'Lowest'} {metric.replace('_',' ').title()}"
    if state: label += f" in {state}"
    show_superlative_card(top.get("city",""), top.get("state",""), top.get(metric,""), label, runners, None)


def _show_fallback_data(df_features):
    """Show fallback."""
    st.markdown("### Available Data")
    if not df_features.empty:
        show_city_table(df_features.head(10), "Sample Cities", True)


def handle_lifestyle_query(query, city_name, classification, df_features, get_engine_func, city_list):
    """Handle 'Life in X' queries with city profile and AI summary."""
    from sqlalchemy import text
    
    engine = get_engine_func()
    
    # Try to find the city
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    if result:
        city_data = dict(result._mapping)
        
        # Show city profile card
        show_city_profile_card(city_data.get("city", city_name), city_data.get("state", ""), pd.Series(city_data))
        
        # Generate AI summary about life in this city
        st.markdown("### 🏙️ What's Life Like?")
        
        try:
            # Use OpenAI directly for lifestyle summary
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
            summary = response.choices[0].message.content.strip()
            st.markdown(summary)
        except Exception as e:
            pop = city_data.get('population', 'N/A')
            age = city_data.get('median_age', 'N/A')
            if isinstance(pop, (int, float)):
                st.info(f"A city with population {pop:,} and median age {age}.")
            else:
                st.info(f"A city with population {pop} and median age {age}.")
        
    else:
        # City not in database - use GPT knowledge
        st.warning(f"'{city_name.title()}' not found in our database. Showing general information.")
        handle_gpt_knowledge_fallback(query)

def handle_single_city_query(query, city_name, df_features, get_engine_func):
    """Handle queries about a specific city with metric detection."""
    from sqlalchemy import text
    
    engine = get_engine_func()
    q_lower = query.lower()
    
    # Try exact match first
    sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
    with engine.connect() as conn:
        result = conn.execute(sql, {"city": city_name}).fetchone()
    
    # If not found, try fuzzy match
    if not result:
        city_list = df_features["city"].unique().tolist() if not df_features.empty else []
        matched_city, score = fuzzy_match_city(city_name, city_list)
        if matched_city and score > 70:
            sql = text(f"SELECT * FROM {DB_TABLE_NAME} WHERE LOWER(city) = LOWER(:city)")
            with engine.connect() as conn:
                result = conn.execute(sql, {"city": matched_city}).fetchone()
    
    if not result:
        st.warning(f"City '{city_name}' not found in our database.")
        handle_gpt_knowledge_fallback(query)
        return
    
    city_data = dict(result._mapping)
    city = city_data.get("city", city_name)
    state = city_data.get("state", "")
    
    # Detect if asking about a specific metric
    metric_name = None
    metric_value = None
    
    if "population" in q_lower:
        metric_name = "Population"
        metric_value = city_data.get("population", "N/A")
    elif "median age" in q_lower or "age" in q_lower:
        metric_name = "Median Age"
        metric_value = city_data.get("median_age", "N/A")
    elif "household" in q_lower or "family size" in q_lower:
        metric_name = "Avg Household Size"
        metric_value = city_data.get("avg_household_size", "N/A")
    
    # If specific metric requested, show metric card
    if metric_name and metric_value:
        # Format value
        if isinstance(metric_value, (int, float)) and metric_value > 1000:
            formatted_value = f"{int(metric_value):,}"
        elif isinstance(metric_value, float):
            formatted_value = f"{metric_value:.1f}"
        else:
            formatted_value = str(metric_value)
        
        # Display metric card
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
        
        # Generate AI summary about this specific metric
        try:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
            
            prompt = f"""The user asked: "{query}"
            
            {city}, {state} has a {metric_name.lower()} of {formatted_value}.
            
            Additional context:
            - Population: {city_data.get('population', 'N/A'):,}
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
            
        except Exception as e:
            pass  # Skip AI summary if it fails
    
    else:
        # No specific metric - show full city profile
        show_city_profile_card(city, state, pd.Series(city_data))
