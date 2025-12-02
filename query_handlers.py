"""
Query handlers for each response type.
"""

import streamlit as st
import pandas as pd
from llm_classifier import get_openai_client
from display_components import (
    show_single_city_card,
    show_city_profile_card,
    show_comparison_card,
    show_aggregate_card,
    show_city_table,
    show_ai_response,
    show_insight_bullets
)


def handle_query(query: str, intent: dict, df_features: pd.DataFrame):
    """
    Main query router - directs to appropriate handler based on intent.
    """
    
    response_type = intent.get("response_type", "general")
    
    # Check if we can answer from database
    if not intent.get("can_answer_from_database", True):
        handle_fallback_query(query)
        return
    
    # Route to appropriate handler
    handlers = {
        "single_city": handle_single_city,
        "city_list": handle_city_list,
        "comparison": handle_comparison,
        "city_profile": handle_city_profile,
        "single_state": handle_single_state,
        "state_list": handle_state_list,
        "state_profile": handle_state_profile,
        "aggregate": handle_aggregate,
        "general_question": handle_fallback_query
    }
    
    handler = handlers.get(response_type, handle_fallback_query)
    
    if response_type == "general_question":
        handler(query)
    else:
        handler(query, intent, df_features)


def handle_single_city(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle queries expecting a single city answer."""
    
    metric = intent.get("metric", "population")
    direction = intent.get("sort_direction", "highest")
    mentioned_cities = intent.get("mentioned_cities", [])
    mentioned_states = intent.get("mentioned_states", [])
    
    # Map metric names
    metric_map = {
        "population": "population",
        "median_age": "median_age",
        "avg_household_size": "avg_household_size",
        "age": "median_age",
        "household": "avg_household_size",
        "all": "population"
    }
    metric_col = metric_map.get(metric, "population")
    
    # Filter data
    data = df_features.copy()
    state_context = ""
    
    if mentioned_states:
        state = mentioned_states[0]
        data = data[data["state"].str.lower() == state.lower()]
        state_context = f" in {state}"
    
    if data.empty:
        st.warning("No cities found matching your criteria.")
        return
    
    # If specific city mentioned, show that city
    if mentioned_cities:
        city = mentioned_cities[0]
        city_data = data[data["city"].str.lower() == city.lower()]
        if not city_data.empty:
            row = city_data.iloc[0]
            show_city_profile_card(row["city"], row["state"], row)
            return
    
    # Sort and get top result
    if direction == "lowest":
        sorted_data = data.nsmallest(5, metric_col)
        direction_word = "Lowest"
    else:
        sorted_data = data.nlargest(5, metric_col)
        direction_word = "Highest"
    
    top_city = sorted_data.iloc[0]
    runners_up = sorted_data.iloc[1:5]["city"].tolist() if len(sorted_data) > 1 else []
    
    # Metric label
    metric_labels = {
        "population": "Population",
        "median_age": "Median Age",
        "avg_household_size": "Household Size"
    }
    metric_label = metric_labels.get(metric_col, metric_col.title())
    rank_label = f"{direction_word} {metric_label}{state_context}"
    
    show_single_city_card(
        city_name=top_city["city"],
        state_name=top_city["state"],
        metric_value=top_city[metric_col],
        metric_label=metric_label,
        rank_label=rank_label,
        runners_up=runners_up
    )
    
    # Add AI insight if needed
    if intent.get("needs_ai_insight"):
        add_ai_insight(query, top_city, metric_col)


def handle_city_list(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle queries expecting multiple cities."""
    
    metric = intent.get("metric", "population")
    direction = intent.get("sort_direction", "highest")
    limit = intent.get("limit") or 10
    mentioned_states = intent.get("mentioned_states", [])
    
    # Map metric
    metric_map = {
        "population": "population",
        "median_age": "median_age",
        "avg_household_size": "avg_household_size",
        "age": "median_age",
        "household": "avg_household_size",
        "all": "population"
    }
    metric_col = metric_map.get(metric, "population")
    
    # Filter data
    data = df_features.copy()
    
    if mentioned_states:
        state = mentioned_states[0]
        data = data[data["state"].str.lower() == state.lower()]
    
    if data.empty:
        st.warning("No cities found matching your criteria.")
        return
    
    # Sort
    if direction == "lowest":
        sorted_data = data.nsmallest(limit, metric_col)
    else:
        sorted_data = data.nlargest(limit, metric_col)
    
    show_city_table(sorted_data)
    
    # Add AI insight if needed
    if intent.get("needs_ai_insight"):
        add_list_insight(query, sorted_data)


def handle_comparison(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle city or state comparisons."""
    
    mentioned_cities = intent.get("mentioned_cities", [])
    mentioned_states = intent.get("mentioned_states", [])
    
    if intent.get("comparison_type") == "city_vs_city" and len(mentioned_cities) >= 2:
        city1, city2 = mentioned_cities[0], mentioned_cities[1]
        
        city1_data = df_features[df_features["city"].str.lower() == city1.lower()]
        city2_data = df_features[df_features["city"].str.lower() == city2.lower()]
        
        if not city1_data.empty and not city2_data.empty:
            show_comparison_card(city1_data.iloc[0], city2_data.iloc[0])
            
            # Add AI comparison insight
            if intent.get("needs_ai_insight"):
                add_comparison_insight(query, city1_data.iloc[0], city2_data.iloc[0])
            return
    
    st.warning("Could not find the cities to compare. Please check the city names.")


def handle_city_profile(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle city profile queries."""
    
    mentioned_cities = intent.get("mentioned_cities", [])
    
    if mentioned_cities:
        city = mentioned_cities[0]
        city_data = df_features[df_features["city"].str.lower() == city.lower()]
        
        if not city_data.empty:
            row = city_data.iloc[0]
            show_city_profile_card(row["city"], row["state"], row)
            return
    
    st.warning("Could not find the city you're asking about. Please check the spelling.")


def handle_single_state(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle single state metric queries."""
    
    metric = intent.get("metric", "population")
    mentioned_states = intent.get("mentioned_states", [])
    
    if not mentioned_states:
        st.warning("Please specify a state.")
        return
    
    state = mentioned_states[0]
    state_data = df_features[df_features["state"].str.lower() == state.lower()]
    
    if state_data.empty:
        st.warning(f"No data found for {state}.")
        return
    
    # Calculate aggregate
    if metric == "population":
        value = state_data["population"].sum()
        label = f"{state} — Total Population"
        formatted = f"{int(value):,}"
    elif metric == "median_age":
        value = state_data["median_age"].mean()
        label = f"{state} — Average Median Age"
        formatted = f"{value:.1f} years"
    else:
        value = state_data["avg_household_size"].mean()
        label = f"{state} — Average Household Size"
        formatted = f"{value:.2f}"
    
    city_count = len(state_data)
    show_aggregate_card(label, formatted, f"Across {city_count} cities in our database")
    
    # Show top cities
    st.markdown("<h4 style='margin-top: 1.5rem;'>🏙️ Top Cities</h4>", unsafe_allow_html=True)
    show_city_table(state_data.nlargest(5, "population"), "")


def handle_state_list(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle state list queries."""
    handle_city_list(query, intent, df_features)


def handle_state_profile(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle state profile queries."""
    handle_single_state(query, intent, df_features)


def handle_aggregate(query: str, intent: dict, df_features: pd.DataFrame):
    """Handle aggregate queries (count, total, average)."""
    
    mentioned_states = intent.get("mentioned_states", [])
    
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


def handle_fallback_query(query: str):
    """Handle queries that can't be answered from database."""
    
    client = get_openai_client()
    if not client:
        st.warning("This question requires AI, but the API is not configured.")
        return
    
    with st.spinner("Thinking..."):
        prompt = f"""
        The user asked a question that cannot be answered from our US cities database.
        Our database only has: city name, state, population, median_age, avg_household_size.
        
        Question: {query}
        
        Provide a helpful, accurate response. If uncertain, say so. Be concise.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            show_ai_response(answer, is_fallback=True)
            
        except Exception as e:
            st.error(f"Could not generate response: {e}")


# -------------------------------------------------
# AI Insight Helpers
# -------------------------------------------------

def add_ai_insight(query: str, city_row: pd.Series, metric: str):
    """Add AI insight for single city result."""
    
    client = get_openai_client()
    if not client:
        return
    
    prompt = f"""
    User asked: "{query}"
    Answer: {city_row['city']}, {city_row['state']}
    Data: Population {city_row['population']:,}, Median Age {city_row['median_age']}, Household Size {city_row['avg_household_size']}
    
    Give 2 brief insights (8-12 words each). No intro, just 2 bullet points.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        insights = response.choices[0].message.content.strip().split('\n')
        show_insight_bullets(insights)
    except Exception:
        pass


def add_list_insight(query: str, df: pd.DataFrame):
    """Add AI insight for list results."""
    
    client = get_openai_client()
    if not client:
        return
    
    data_summary = df.head(5).to_string()
    
    prompt = f"""
    User asked: "{query}"
    Top results:
    {data_summary}
    
    Write a 1-2 sentence summary of these results. Be concise.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        summary = response.choices[0].message.content.strip()
        st.info(summary)
    except Exception:
        pass


def add_comparison_insight(query: str, city1: pd.Series, city2: pd.Series):
    """Add AI insight for comparison."""
    
    client = get_openai_client()
    if not client:
        return
    
    prompt = f"""
    User compared {city1['city']} vs {city2['city']}.
    
    {city1['city']}: Pop {city1['population']:,}, Age {city1['median_age']}, HH {city1['avg_household_size']}
    {city2['city']}: Pop {city2['population']:,}, Age {city2['median_age']}, HH {city2['avg_household_size']}
    
    Write 2 sentences comparing these cities. Be insightful.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        show_ai_response(summary)
    except Exception:
        pass
