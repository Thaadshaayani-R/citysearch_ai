"""
CitySearch AI - Query Handlers
===============================
Main query routing and handling logic.
Routes queries to appropriate handlers based on LLM classification.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text

from config import LLM_MODEL, LLM_MAX_TOKENS_SUMMARY
from utils import (
    extract_single_city_fuzzy, extract_single_state_fuzzy,
    extract_two_cities_fuzzy, extract_two_states_fuzzy,
    convert_df_to_csv, is_city_related_query
)
from display_components import (
    show_city_profile_card, show_single_metric_card, show_state_metric_card,
    show_aggregate_card, show_superlative_card, show_city_comparison,
    show_state_comparison, show_recommendation_card, show_cluster_scatter,
    show_lifestyle_card, show_city_table, show_ai_response, show_out_of_scope,
    generate_ai_summary, get_openai_client
)


# -------------------------------------------------
# MAIN QUERY HANDLER
# -------------------------------------------------
def handle_query(query: str, classification: dict, df_features: pd.DataFrame, 
                 get_engine_func, smart_route_func=None, lifestyle_rag_func=None,
                 classify_intent_func=None):
    """
    Main query router - directs to appropriate handler based on LLM classification.
    
    Args:
        query: User's query string
        classification: LLM classification result
        df_features: DataFrame with city features
        get_engine_func: Function to get database engine
        smart_route_func: Optional function for ML routing (from core.smart_router)
        lifestyle_rag_func: Optional function for lifestyle RAG (from core.lifestyle_rag_v2)
        classify_intent_func: Optional function for intent classification (from core.intent_classifier)
    """
    
    response_type = classification.get("response_type", "city_list")
    can_answer = classification.get("can_answer_from_database", True)
    is_city_related = classification.get("is_city_related", True)
    
    # -------------------------------------------------
    # CASE 1: Not city-related at all
    # -------------------------------------------------
    if not is_city_related:
        show_out_of_scope()
        return
    
    # -------------------------------------------------
    # CASE 2: City-related but not in database → GPT fallback
    # -------------------------------------------------
    if not can_answer and is_city_related:
        handle_gpt_fallback(query)
        return
    
    # -------------------------------------------------
    # CASE 3: Route to specific handler
    # -------------------------------------------------
    city_list = df_features["city"].unique().tolist()
    
    handlers = {
        "single_city": lambda: handle_single_city(query, classification, df_features, city_list, get_engine_func),
        "single_state": lambda: handle_single_state(query, classification, df_features, get_engine_func),
        "city_list": lambda: handle_city_list(query, classification, df_features, get_engine_func),
        "state_list": lambda: handle_state_list(query, classification, df_features, get_engine_func),
        "comparison": lambda: handle_comparison(query, classification, df_features, city_list, get_engine_func),
        "city_profile": lambda: handle_city_profile(query, classification, df_features, city_list),
        "state_profile": lambda: handle_state_profile(query, classification, df_features, get_engine_func),
        "aggregate": lambda: handle_aggregate(query, classification, df_features, get_engine_func),
        "recommendation": lambda: handle_recommendation(query, classification, df_features, smart_route_func),
        "cluster": lambda: handle_cluster(query, classification, df_features, smart_route_func),
        "similar_cities": lambda: handle_similar_cities(query, classification, df_features, smart_route_func),
        "lifestyle": lambda: handle_lifestyle(query, classification, df_features, city_list, lifestyle_rag_func),
        "general_question": lambda: handle_gpt_fallback(query),
    }
    
    handler = handlers.get(response_type, lambda: handle_default(query, classification, df_features, get_engine_func))
    handler()


# -------------------------------------------------
# SINGLE CITY HANDLER
# -------------------------------------------------
def handle_single_city(query: str, classification: dict, df_features: pd.DataFrame, 
                       city_list: list, get_engine_func):
    """Handle queries about a single city."""
    
    mentioned_cities = classification.get("mentioned_cities", [])
    metric = classification.get("metric")
    sort_direction = classification.get("sort_direction")
    
    # If specific city mentioned
    if mentioned_cities:
        city_name = mentioned_cities[0]
        city_row = df_features[df_features["city"].str.lower() == city_name.lower()]
        
        if not city_row.empty:
            row = city_row.iloc[0]
            
            # If asking for specific metric
            if metric and metric != "all":
                metric_col = _get_metric_column(metric)
                metric_value = row.get(metric_col, "N/A")
                show_single_metric_card(
                    city_name=row["city"],
                    state_name=row["state"],
                    metric_name=metric,
                    metric_value=metric_value,
                    row=row
                )
            else:
                # Show full profile
                show_city_profile_card(row["city"], row["state"], row)
            return
    
    # If asking for highest/lowest (superlative)
    if sort_direction:
        _handle_superlative_city(query, classification, df_features)
        return
    
    # Try to extract city from query
    detected_city = extract_single_city_fuzzy(query, city_list)
    if detected_city:
        city_row = df_features[df_features["city"].str.lower() == detected_city.lower()]
        if not city_row.empty:
            row = city_row.iloc[0]
            
            if metric and metric != "all":
                metric_col = _get_metric_column(metric)
                metric_value = row.get(metric_col, "N/A")
                show_single_metric_card(row["city"], row["state"], metric, metric_value, row)
            else:
                show_city_profile_card(row["city"], row["state"], row)
            return
    
    # Fallback: show as superlative if no city found
    if sort_direction:
        _handle_superlative_city(query, classification, df_features)
    else:
        st.warning("Could not find the city you're asking about. Please check the spelling.")


def _handle_superlative_city(query: str, classification: dict, df_features: pd.DataFrame):
    """Handle 'highest/lowest' city queries."""
    
    metric = classification.get("metric", "population")
    sort_direction = classification.get("sort_direction", "highest")
    mentioned_states = classification.get("mentioned_states", [])
    
    # Filter by state if mentioned
    data = df_features.copy()
    state_context = ""
    if mentioned_states:
        state = mentioned_states[0]
        data = data[data["state"].str.lower() == state.lower()]
        state_context = f" in {state}"
    
    if data.empty:
        st.warning("No cities found matching your criteria.")
        return
    
    metric_col = _get_metric_column(metric)
    
    # Sort and get results
    if sort_direction == "lowest":
        sorted_data = data.nsmallest(5, metric_col)
        direction_word = "Lowest"
    else:
        sorted_data = data.nlargest(5, metric_col)
        direction_word = "Highest"
    
    top_city = sorted_data.iloc[0]
    runners_up = sorted_data.iloc[1:5]["city"].tolist() if len(sorted_data) > 1 else []
    
    # Generate insights
    insights = _generate_superlative_insights(top_city, metric_col, sorted_data)
    
    # Metric label
    metric_labels = {
        "population": "Population",
        "median_age": "Median Age",
        "avg_household_size": "Household Size"
    }
    metric_label = metric_labels.get(metric_col, metric_col.title())
    rank_label = f"{direction_word} {metric_label}{state_context}"
    
    show_superlative_card(
        city_name=top_city["city"],
        state_name=top_city["state"],
        metric_value=top_city[metric_col],
        rank_label=rank_label,
        runners_up=runners_up,
        insights=insights
    )


def _generate_superlative_insights(top_city: pd.Series, metric_col: str, sorted_data: pd.DataFrame) -> list:
    """Generate insights for superlative results."""
    insights = []
    
    if len(sorted_data) > 1:
        second_city = sorted_data.iloc[1]
        
        if metric_col == "population":
            ratio = top_city["population"] / second_city["population"] if second_city["population"] > 0 else 1
            insights.append(f"#1 most populated metro in this selection")
            if ratio > 1.5:
                insights.append(f"Nearly {ratio:.1f}x larger than {second_city['city']}")
            else:
                insights.append(f"Slightly larger than {second_city['city']}")
        
        elif metric_col == "median_age":
            insights.append(f"Median age: {top_city['median_age']:.1f} years")
            diff = abs(top_city['median_age'] - second_city['median_age'])
            insights.append(f"{diff:.1f} years difference from {second_city['city']}")
        
        else:
            insights.append(f"Value: {top_city[metric_col]:.2f}")
            insights.append(f"Compared to {second_city['city']}: {second_city[metric_col]:.2f}")
    
    return insights


# -------------------------------------------------
# SINGLE STATE HANDLER
# -------------------------------------------------
def handle_single_state(query: str, classification: dict, df_features: pd.DataFrame, get_engine_func):
    """Handle queries about a single state."""
    
    mentioned_states = classification.get("mentioned_states", [])
    metric = classification.get("metric", "population")
    
    if not mentioned_states:
        # Try to extract from query
        detected_state = extract_single_state_fuzzy(query)
        if detected_state:
            mentioned_states = [detected_state]
    
    if not mentioned_states:
        st.warning("Could not identify the state. Please specify a state name.")
        return
    
    state = mentioned_states[0]
    state_data = df_features[df_features["state"].str.lower() == state.lower()]
    
    if state_data.empty:
        st.warning(f"No data found for {state}.")
        return
    
    metric_col = _get_metric_column(metric)
    city_count = len(state_data)
    
    # Calculate aggregate
    if metric_col == "population":
        value = state_data["population"].sum()
    else:
        value = state_data[metric_col].mean()
    
    show_state_metric_card(
        state_name=state,
        metric_name=metric,
        metric_value=value,
        city_count=city_count,
        state_data=state_data
    )


# -------------------------------------------------
# CITY LIST HANDLER
# -------------------------------------------------
def handle_city_list(query: str, classification: dict, df_features: pd.DataFrame, get_engine_func):
    """Handle queries expecting multiple cities."""
    
    metric = classification.get("metric", "population")
    sort_direction = classification.get("sort_direction", "highest")
    limit = classification.get("limit") or 10
    mentioned_states = classification.get("mentioned_states", [])
    needs_summary = classification.get("needs_ai_summary", False)
    summary_style = classification.get("summary_style", "brief")
    
    # Filter by state if mentioned
    data = df_features.copy()
    if mentioned_states:
        state = mentioned_states[0]
        data = data[data["state"].str.lower() == state.lower()]
    
    if data.empty:
        st.warning("No cities found matching your criteria.")
        return
    
    metric_col = _get_metric_column(metric)
    
    # Sort
    if sort_direction == "lowest":
        sorted_data = data.nsmallest(limit, metric_col)
    else:
        sorted_data = data.nlargest(limit, metric_col)
    
    # Show AI summary if needed
    if needs_summary:
        summary = generate_ai_summary(sorted_data, query, summary_style)
        if summary:
            show_ai_response(summary)
    
    # Show table
    show_city_table(sorted_data)


# -------------------------------------------------
# STATE LIST HANDLER
# -------------------------------------------------
def handle_state_list(query: str, classification: dict, df_features: pd.DataFrame, get_engine_func):
    """Handle queries expecting multiple states."""
    
    # Group by state and calculate aggregates
    state_stats = df_features.groupby("state").agg({
        "population": "sum",
        "median_age": "mean",
        "avg_household_size": "mean",
        "city": "count"
    }).reset_index()
    
    state_stats.columns = ["State", "Total Population", "Avg Median Age", "Avg Household Size", "City Count"]
    
    metric = classification.get("metric", "population")
    sort_direction = classification.get("sort_direction", "highest")
    limit = classification.get("limit") or 10
    
    # Sort
    sort_col = "Total Population" if metric == "population" else "Avg Median Age"
    ascending = sort_direction == "lowest"
    sorted_data = state_stats.sort_values(sort_col, ascending=ascending).head(limit)
    
    show_city_table(sorted_data, title="State Statistics")


# -------------------------------------------------
# COMPARISON HANDLER
# -------------------------------------------------
def handle_comparison(query: str, classification: dict, df_features: pd.DataFrame, 
                      city_list: list, get_engine_func):
    """Handle city vs city or state vs state comparisons."""
    
    comparison_type = classification.get("comparison_type")
    mentioned_cities = classification.get("mentioned_cities", [])
    mentioned_states = classification.get("mentioned_states", [])
    
    # City comparison
    if comparison_type == "city_vs_city" or len(mentioned_cities) >= 2:
        if len(mentioned_cities) < 2:
            # Try to extract from query
            cities = extract_two_cities_fuzzy(query, city_list)
            if cities:
                mentioned_cities = cities
        
        if len(mentioned_cities) >= 2:
            city1, city2 = mentioned_cities[0], mentioned_cities[1]
            
            city1_data = df_features[df_features["city"].str.lower() == city1.lower()]
            city2_data = df_features[df_features["city"].str.lower() == city2.lower()]
            
            if not city1_data.empty and not city2_data.empty:
                show_city_comparison(city1_data.iloc[0], city2_data.iloc[0], query)
                return
    
    # State comparison
    if comparison_type == "state_vs_state" or len(mentioned_states) >= 2:
        if len(mentioned_states) < 2:
            # Try to extract from query
            states = extract_two_states_fuzzy(query)
            if states:
                mentioned_states = states
        
        if len(mentioned_states) >= 2:
            state1, state2 = mentioned_states[0], mentioned_states[1]
            
            stats1 = _get_state_stats(state1, df_features, get_engine_func)
            stats2 = _get_state_stats(state2, df_features, get_engine_func)
            
            cities1 = _get_top_cities_in_state(state1, df_features)
            cities2 = _get_top_cities_in_state(state2, df_features)
            
            if stats1 and stats2:
                show_state_comparison(state1, stats1, cities1, state2, stats2, cities2, query)
                return
    
    st.warning("Could not find the cities or states to compare. Please check the names.")


def _get_state_stats(state_name: str, df_features: pd.DataFrame, get_engine_func) -> dict:
    """Get comprehensive stats for a state."""
    
    state_data = df_features[df_features["state"].str.lower() == state_name.lower()]
    
    if state_data.empty:
        return None
    
    return {
        "state": state_name,
        "city_count": len(state_data),
        "total_population": state_data["population"].sum(),
        "avg_population": state_data["population"].mean(),
        "avg_median_age": state_data["median_age"].mean(),
        "avg_household_size": state_data["avg_household_size"].mean(),
    }


def _get_top_cities_in_state(state_name: str, df_features: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Get top cities by population in a state."""
    
    state_data = df_features[df_features["state"].str.lower() == state_name.lower()]
    return state_data.nlargest(limit, "population")[["city", "state", "population", "median_age", "avg_household_size"]]


# -------------------------------------------------
# CITY PROFILE HANDLER
# -------------------------------------------------
def handle_city_profile(query: str, classification: dict, df_features: pd.DataFrame, city_list: list):
    """Handle city profile queries."""
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if not mentioned_cities:
        # Try to extract from query
        detected_city = extract_single_city_fuzzy(query, city_list)
        if detected_city:
            mentioned_cities = [detected_city]
    
    if mentioned_cities:
        city = mentioned_cities[0]
        city_data = df_features[df_features["city"].str.lower() == city.lower()]
        
        if not city_data.empty:
            row = city_data.iloc[0]
            show_city_profile_card(row["city"], row["state"], row)
            return
    
    st.warning("Could not find the city. Please check the spelling.")


# -------------------------------------------------
# STATE PROFILE HANDLER
# -------------------------------------------------
def handle_state_profile(query: str, classification: dict, df_features: pd.DataFrame, get_engine_func):
    """Handle state profile queries."""
    
    mentioned_states = classification.get("mentioned_states", [])
    
    if not mentioned_states:
        detected_state = extract_single_state_fuzzy(query)
        if detected_state:
            mentioned_states = [detected_state]
    
    if mentioned_states:
        state = mentioned_states[0]
        state_data = df_features[df_features["state"].str.lower() == state.lower()]
        
        if not state_data.empty:
            city_count = len(state_data)
            total_pop = state_data["population"].sum()
            
            show_state_metric_card(
                state_name=state,
                metric_name="population",
                metric_value=total_pop,
                city_count=city_count,
                state_data=state_data
            )
            return
    
    st.warning("Could not find the state. Please check the spelling.")


# -------------------------------------------------
# AGGREGATE HANDLER
# -------------------------------------------------
def handle_aggregate(query: str, classification: dict, df_features: pd.DataFrame, get_engine_func):
    """Handle aggregate queries (count, total, average)."""
    
    mentioned_states = classification.get("mentioned_states", [])
    
    # Filter by state if mentioned
    data = df_features.copy()
    context = "in the U.S."
    if mentioned_states:
        state = mentioned_states[0]
        data = data[data["state"].str.lower() == state.lower()]
        context = f"in {state}"
    
    if data.empty:
        st.warning("No data found.")
        return
    
    q_lower = query.lower()
    
    if "how many" in q_lower or "count" in q_lower or "number of" in q_lower:
        count = len(data)
        show_aggregate_card(f"Total Cities {context}", f"{count:,}", "")
    
    elif "total" in q_lower and "population" in q_lower:
        total = data["population"].sum()
        show_aggregate_card(f"Total Population {context}", f"{int(total):,}", f"Across {len(data)} cities")
    
    elif "average" in q_lower or "avg" in q_lower:
        if "age" in q_lower:
            avg = data["median_age"].mean()
            show_aggregate_card(f"Average Median Age {context}", f"{avg:.1f} years", "")
        elif "population" in q_lower:
            avg = data["population"].mean()
            show_aggregate_card(f"Average City Population {context}", f"{int(avg):,}", "")
        else:
            avg = data["avg_household_size"].mean()
            show_aggregate_card(f"Average Household Size {context}", f"{avg:.2f}", "")
    
    else:
        # Default: show count
        count = len(data)
        show_aggregate_card(f"Total Cities {context}", f"{count:,}", "")


# -------------------------------------------------
# RECOMMENDATION HANDLER
# -------------------------------------------------
def handle_recommendation(query: str, classification: dict, df_features: pd.DataFrame, smart_route_func):
    """Handle recommendation queries (best for families, etc.)."""
    
    specific_intent = classification.get("specific_intent", "general")
    mentioned_states = classification.get("mentioned_states", [])
    needs_summary = classification.get("needs_ai_summary", True)
    summary_style = classification.get("summary_style", "recommendation")
    
    # Try to use smart_route if available
    if smart_route_func:
        try:
            ml_mode, ml_target = smart_route_func(query)
            
            if ml_target is not None and not (isinstance(ml_target, pd.DataFrame) and ml_target.empty):
                if isinstance(ml_target, pd.DataFrame) and not ml_target.empty:
                    df = ml_target
                    
                    # Filter by state if mentioned
                    if mentioned_states and "state" in df.columns:
                        state = mentioned_states[0]
                        df_filtered = df[df["state"].str.lower() == state.lower()]
                        if not df_filtered.empty:
                            df = df_filtered
                    
                    # Show top city card
                    top_city = df.iloc[0]
                    show_recommendation_card(top_city, specific_intent, df)
                    
                    # AI Summary
                    if needs_summary:
                        summary = generate_ai_summary(df, query, summary_style)
                        if summary:
                            st.markdown("<div class='section-header'>AI Insight</div>", unsafe_allow_html=True)
                            show_ai_response(summary)
                    
                    # Show full results table
                    show_city_table(df, title="All Results")
                    return
        except Exception as e:
            pass  # Fall through to fallback
    
    # Fallback: basic ranking
    _handle_recommendation_fallback(query, classification, df_features, specific_intent, mentioned_states)


def _handle_recommendation_fallback(query: str, classification: dict, df_features: pd.DataFrame,
                                     specific_intent: str, mentioned_states: list):
    """Fallback recommendation handler using basic metrics."""
    
    data = df_features.copy()
    
    # Filter by state
    if mentioned_states:
        state = mentioned_states[0]
        data = data[data["state"].str.lower() == state.lower()]
    
    if data.empty:
        st.warning("No cities found matching your criteria.")
        return
    
    # Score based on intent
    if specific_intent == "families":
        # Prefer larger households, moderate age
        data["score"] = (
            data["avg_household_size"].rank(pct=True) * 0.5 +
            (1 - abs(data["median_age"] - 35) / 20).clip(0, 1) * 0.3 +
            data["population"].rank(pct=True) * 0.2
        )
        title = "Best Cities for Families"
    
    elif specific_intent == "young_professionals":
        # Prefer younger age, larger population
        data["score"] = (
            (1 - data["median_age"].rank(pct=True)) * 0.5 +
            data["population"].rank(pct=True) * 0.5
        )
        title = "Best Cities for Young Professionals"
    
    elif specific_intent == "retirement":
        # Prefer older age, smaller households
        data["score"] = (
            data["median_age"].rank(pct=True) * 0.6 +
            (1 - data["avg_household_size"].rank(pct=True)) * 0.4
        )
        title = "Best Cities for Retirement"
    
    else:
        # General: balance of all factors
        data["score"] = (
            data["population"].rank(pct=True) * 0.4 +
            (1 - abs(data["median_age"] - 35) / 20).clip(0, 1) * 0.3 +
            data["avg_household_size"].rank(pct=True) * 0.3
        )
        title = "Top Recommended Cities"
    
    # Sort and display
    top_cities = data.nlargest(10, "score")
    
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
    
    # Top city card
    top_city = top_cities.iloc[0]
    show_recommendation_card(top_city, specific_intent, top_cities)
    
    # AI insight
    summary = generate_ai_summary(top_cities, query, "recommendation")
    if summary:
        st.markdown("<div class='section-header'>AI Insight</div>", unsafe_allow_html=True)
        show_ai_response(summary)
    
    # Full table
    show_city_table(top_cities, title="All Results")


# -------------------------------------------------
# CLUSTER HANDLER
# -------------------------------------------------
def handle_cluster(query: str, classification: dict, df_features: pd.DataFrame, smart_route_func):
    """Handle cluster queries."""
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if smart_route_func:
        try:
            ml_mode, ml_target = smart_route_func(query)
            
            if ml_mode == "cluster_single" and ml_target:
                city_info = ml_target
                
                # Import cluster explain if available
                try:
                    from core.cluster_explain import explain_cluster
                    cid = city_info["cluster"]
                    profile = explain_cluster(return_dict=True, cluster_id=cid)
                    
                    header_html = f"""
                    <div class="insight-card">
                        <div class="insight-label">{city_info['city']} belongs to</div>
                        <div class="insight-title">Cluster {cid} — {profile['cluster_name']}</div>
                        <div class="insight-text">{profile['cluster_summary']}</div>
                    </div>
                    """
                    st.markdown(header_html, unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-header'>Lifestyle Summary</div>", unsafe_allow_html=True)
                    st.markdown(profile["detailed_summary"])
                    return
                except ImportError:
                    pass
            
            elif ml_mode in ["cluster_all", "cluster_state"] and isinstance(ml_target, pd.DataFrame):
                df_clusters = ml_target
                
                st.markdown("<div class='section-header'>City Clusters</div>", unsafe_allow_html=True)
                show_city_table(df_clusters, title="Cluster Results")
                show_cluster_scatter(df_clusters)
                return
                
        except Exception as e:
            pass
    
    # Fallback: show cluster info if available in df_features
    if "cluster_label" in df_features.columns:
        if mentioned_cities:
            city = mentioned_cities[0]
            city_data = df_features[df_features["city"].str.lower() == city.lower()]
            if not city_data.empty:
                row = city_data.iloc[0]
                cluster_id = row.get("cluster_label", "N/A")
                st.info(f"{city} belongs to Cluster {cluster_id}")
                return
    
    st.warning("Cluster information not available.")


# -------------------------------------------------
# SIMILAR CITIES HANDLER
# -------------------------------------------------
def handle_similar_cities(query: str, classification: dict, df_features: pd.DataFrame, smart_route_func):
    """Handle similar cities queries."""
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    if smart_route_func:
        try:
            ml_mode, ml_target = smart_route_func(query)
            
            if ml_mode == "cluster_similar" and isinstance(ml_target, pd.DataFrame):
                df_similar = ml_target
                
                st.markdown("<div class='section-header'>Cities in the Same Cluster</div>", unsafe_allow_html=True)
                
                # AI summary
                summary = generate_ai_summary(df_similar, query, "brief")
                if summary:
                    show_ai_response(summary)
                
                show_city_table(df_similar)
                return
        except Exception as e:
            pass
    
    # Fallback: find similar by cluster
    if "cluster_label" in df_features.columns and mentioned_cities:
        city = mentioned_cities[0]
        city_data = df_features[df_features["city"].str.lower() == city.lower()]
        
        if not city_data.empty:
            cluster_id = city_data.iloc[0]["cluster_label"]
            similar = df_features[df_features["cluster_label"] == cluster_id]
            similar = similar[similar["city"].str.lower() != city.lower()]
            
            if not similar.empty:
                st.markdown(f"<div class='section-header'>Cities Similar to {city}</div>", unsafe_allow_html=True)
                show_city_table(similar.head(10))
                return
    
    st.warning("Could not find similar cities.")


# -------------------------------------------------
# LIFESTYLE HANDLER
# -------------------------------------------------
def handle_lifestyle(query: str, classification: dict, df_features: pd.DataFrame, 
                     city_list: list, lifestyle_rag_func):
    """Handle lifestyle queries."""
    
    mentioned_cities = classification.get("mentioned_cities", [])
    
    # Try RAG function first
    if lifestyle_rag_func:
        try:
            lifestyle_card = lifestyle_rag_func(query)
            
            if lifestyle_card:
                show_lifestyle_card(
                    city=lifestyle_card["city"],
                    state=lifestyle_card["state"],
                    population=lifestyle_card.get("population"),
                    median_age=lifestyle_card.get("median_age"),
                    household_size=lifestyle_card.get("avg_household_size"),
                    description=lifestyle_card.get("description", ""),
                    ai_summary=lifestyle_card.get("ai_summary", "")
                )
                return
        except Exception as e:
            pass
    
    # Fallback to city profile
    if not mentioned_cities:
        detected_city = extract_single_city_fuzzy(query, city_list)
        if detected_city:
            mentioned_cities = [detected_city]
    
    if mentioned_cities:
        city = mentioned_cities[0]
        city_data = df_features[df_features["city"].str.lower() == city.lower()]
        
        if not city_data.empty:
            row = city_data.iloc[0]
            show_city_profile_card(row["city"], row["state"], row)
            return
    
    st.warning("Could not find lifestyle information for this city.")


# -------------------------------------------------
# GPT FALLBACK HANDLER
# -------------------------------------------------
def handle_gpt_fallback(query: str):
    """Handle queries that can't be answered from database using GPT."""
    
    client = get_openai_client()
    if not client:
        st.warning("This question requires AI, but the API is not configured.")
        return
    
    with st.spinner("Searching for information..."):
        prompt = f"""
The user asked a question about US cities that we couldn't answer from our database.
Our database contains: city name, state, population, median_age, avg_household_size.

Question: {query}

Please provide a helpful answer based on your general knowledge about US cities.
If this is about a specific city or state, share relevant information.
Keep it concise and helpful.
"""
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=LLM_MAX_TOKENS_SUMMARY
            )
            
            answer = response.choices[0].message.content.strip()
            show_ai_response(answer, is_fallback=True)
            
        except Exception as e:
            st.error(f"Could not generate response: {e}")


# -------------------------------------------------
# DEFAULT HANDLER (SQL-based)
# -------------------------------------------------
def handle_default(query: str, classification: dict, df_features: pd.DataFrame, get_engine_func):
    """Default handler - tries SQL query."""
    
    needs_summary = classification.get("needs_ai_summary", False)
    summary_style = classification.get("summary_style", "brief")
    sql_hint = classification.get("sql_hint")
    
    try:
        # Try to use query router if available
        from core.query_router import build_sql_with_fallback
        
        with st.spinner("Running query..."):
            sql = build_sql_with_fallback(query, use_gpt=True)
            
            engine = get_engine_func()
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn)
        
        if df.empty:
            st.warning("No results found.")
            return
        
        # AI summary if needed
        if needs_summary:
            summary = generate_ai_summary(df, query, summary_style)
            if summary:
                show_ai_response(summary)
        
        # Show results
        show_city_table(df)
        
    except Exception as e:
        # Ultimate fallback
        st.warning(f"Could not process query. Please try rephrasing.")
        
        # Show df_features as fallback
        st.markdown("<div class='section-header'>Available Data</div>", unsafe_allow_html=True)
        show_city_table(df_features.head(20), title="Sample Cities")


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def _get_metric_column(metric: str) -> str:
    """Map metric name to column name."""
    metric_map = {
        "population": "population",
        "median_age": "median_age",
        "avg_household_size": "avg_household_size",
        "age": "median_age",
        "household": "avg_household_size",
        "all": "population"
    }
    return metric_map.get(metric, "population")
