# display_components.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from config import LLM_MODEL, LLM_TEMPERATURE_SUMMARY, LLM_MAX_TOKENS_SUMMARY, LLM_MAX_TOKENS_COMPARISON
from utils import convert_markdown_to_html, format_population, format_age, format_household_size, convert_df_to_csv


def get_openai_client():
    """Get OpenAI client from secrets."""
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception:
        pass
    return None


# CITY PROFILE CARD
def show_city_profile_card(city_name: str, state_name: str, row: pd.Series):
    """
    Display a city profile card with RAG-sourced or data-only information.
    No general GPT knowledge.
    """
    
    # Get values safely
    population = row.get('population', 'N/A')
    median_age = row.get('median_age', 'N/A')
    household_size = row.get('avg_household_size', 'N/A')
    state_code = row.get('state_code', state_name[:2].upper() if state_name else 'N/A')
    
    # Format population
    if isinstance(population, (int, float)):
        population_display = f"{int(population):,}"
    else:
        population_display = str(population)
    
    # 1. City Overview Card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
    ">
        <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">{city_name}, {state_name}</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 2rem; margin-top: 1.5rem;">
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Population</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{population_display}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Median Age</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{median_age}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Household Size</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{household_size}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">State</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{state_code}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Try to get RAG context for this city
    try:
        from core.rag_search import get_city_rag_context
        rag_context = get_city_rag_context(city_name, state_name)
        
        if rag_context:
            # Show RAG-sourced summary
            context_parts = []
            for chunk_type, texts in rag_context.items():
                if texts:
                    context_parts.append(texts[0][:200])
            
            if context_parts:
                st.markdown(f"""
                <div style="
                    background: rgba(102, 126, 234, 0.1);
                    border-left: 4px solid #667eea;
                    border-radius: 8px;
                    padding: 1.25rem;
                    margin-bottom: 1.5rem;
                    line-height: 1.6;
                ">
                    {context_parts[0]}...
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="font-size: 0.75rem; color: #a0aec0;">
                    From our knowledge base
                </div>
                """, unsafe_allow_html=True)
                return
    except ImportError:
        pass


 
# SINGLE METRIC CARD (City)
 
def show_single_metric_card(city_name: str, state_name: str, metric_name: str, metric_value, row: pd.Series):
    """
    Display a single metric card - data only, no GPT generation.
    """
    
    # Format the metric value
    if isinstance(metric_value, (int, float)):
        if metric_value > 1000:
            formatted_value = f"{int(metric_value):,}"
        else:
            formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, float) else str(metric_value)
    else:
        formatted_value = str(metric_value)
    
    # Get metric display name
    metric_display = metric_name.replace("_", " ").title()
    
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
        <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">
            {city_name}, {state_name} — {metric_display}
        </div>
        <div style="font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">
            {formatted_value}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show other metrics as context (no AI interpretation)
    population = row.get('population', 'N/A')
    median_age = row.get('median_age', 'N/A')
    household_size = row.get('avg_household_size', 'N/A')
    
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    ">
        <div style="font-size: 0.85rem; color: #667eea; margin-bottom: 0.5rem;">📊 Other Metrics</div>
        <div style="color: #a0aec0; display: flex; gap: 2rem; flex-wrap: wrap;">
            <span>Population: <strong>{population:,}</strong></span>
            <span>Median Age: <strong>{median_age}</strong></span>
            <span>Household Size: <strong>{household_size}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


 
# STATE METRIC CARD
 
def show_state_metric_card(state_name: str, metric_name: str, metric_value, city_count: int, state_data: pd.DataFrame):
    """
    Display a beautiful state-level metric card with context.
    """
    
    # Format the metric value
    if isinstance(metric_value, (int, float)):
        if metric_value > 1000:
            formatted_value = f"{int(metric_value):,}"
        else:
            formatted_value = f"{metric_value:.1f}"
    else:
        formatted_value = str(metric_value)
    
    # Get metric display name
    metric_display = metric_name.replace("_", " ").title()
    
    # Add label based on metric type
    if "population" in metric_name.lower():
        metric_label = "Total Population"
        sub_label = f"Across {city_count} cities in our database"
    elif "age" in metric_name.lower():
        metric_label = "Average Median Age"
        sub_label = f"Averaged across {city_count} cities"
    else:
        metric_label = f"Average {metric_display}"
        sub_label = f"Averaged across {city_count} cities"
    
    # Main card (green gradient for states)
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
            {state_name} — {metric_label}
        </div>
        <div style="font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">
            {formatted_value}
        </div>
        <div style="font-size: 0.85rem; opacity: 0.8;">
            {sub_label}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show top cities for this metric
    if "population" in metric_name.lower():
        top_cities = state_data.nlargest(5, "population")[["city", "population"]]
        st.markdown("""
        <div style="margin-top: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 0.75rem;">Largest Cities</h4>
        </div>
        """, unsafe_allow_html=True)
    elif "age" in metric_name.lower():
        top_cities = state_data.nlargest(5, "median_age")[["city", "median_age"]]
        st.markdown("""
        <div style="margin-top: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 0.75rem;">Highest Median Age</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        top_cities = state_data.nlargest(5, "avg_household_size")[["city", "avg_household_size"]]
        st.markdown("""
        <div style="margin-top: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 0.75rem;">Largest Households</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Display top cities
    for _, city_row in top_cities.iterrows():
        city_name = city_row.iloc[0]
        city_value = city_row.iloc[1]
        
        if isinstance(city_value, (int, float)) and city_value > 1000:
            display_value = f"{int(city_value):,}"
        elif isinstance(city_value, float):
            display_value = f"{city_value:.1f}"
        else:
            display_value = str(city_value)
        
        st.markdown(f"""
        <div style="
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(17, 153, 142, 0.2);
        ">
            <span>{city_name}</span>
            <span style="font-weight: 600;">{display_value}</span>
        </div>
        """, unsafe_allow_html=True)


 
# AGGREGATE CARD
 
def show_aggregate_card(label: str, value: str, sub_label: str = ""):
    """Display an aggregate result card."""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    ">
        <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 3rem; font-weight: 700;">{value}</div>
        <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">{sub_label}</div>
    </div>
    """, unsafe_allow_html=True)


 
# SUPERLATIVE CARD (Highest/Lowest)
 
def show_superlative_card(city_name: str, state_name: str, metric_value, rank_label: str, 
                          runners_up: list = None, insights: list = None):
    """Display a single answer card for superlative questions."""
    
    # Format value
    if isinstance(metric_value, (int, float)):
        if metric_value > 1000:
            formatted_value = f"{int(metric_value):,}"
        else:
            formatted_value = f"{metric_value:.1f}"
    else:
        formatted_value = str(metric_value)
    
    # Get state code
    state_code = state_name[:2].upper() if state_name else ""
    
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
            {rank_label}
        </div>
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;">
            {city_name}, {state_code}
        </div>
        <div style="font-size: 2.5rem; font-weight: 700;">
            {formatted_value}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Insights
    if insights:
        for insight in insights[:2]:
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; gap: 0.5rem; padding: 0.4rem 0;">
                <span style="color: #667eea;">•</span>
                <span>{insight}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Runners up
    if runners_up and len(runners_up) > 0:
        runners_text = ", ".join(runners_up[:4])
        st.markdown(f"""
        <div style="
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            font-size: 0.9rem;
        ">
            <span style="opacity: 0.7;">Other top cities:</span> {runners_text}
        </div>
        """, unsafe_allow_html=True)


 
# CITY COMPARISON CARD
 
def show_city_comparison(city1_row: pd.Series, city2_row: pd.Series, query: str = ""):
    """Display comprehensive city comparison."""
    
    df1, df2 = city1_row, city2_row
    
    # Header
    st.markdown(
        f"<div class='section-header'>City Comparison: {df1['city']} vs {df2['city']}</div>",
        unsafe_allow_html=True
    )
    
    # Side-by-side City Profiles
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">City Profile</div>
                <div class="insight-title">{df1['city']}, {df1['state']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("Population", f"{int(df1['population']):,}")
        st.metric("Median Age", f"{df1['median_age']:.1f} years")
        st.metric("Avg Household Size", f"{df1['avg_household_size']:.2f}")
        if 'cluster_label' in df1.index and pd.notna(df1.get('cluster_label')):
            st.metric("Cluster", f"{int(df1['cluster_label'])}")
        if 'lifestyle_score' in df1.index and pd.notna(df1.get('lifestyle_score')):
            st.metric("Lifestyle Score", f"{df1['lifestyle_score']:.3f}")
    
    with col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">City Profile</div>
                <div class="insight-title">{df2['city']}, {df2['state']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("Population", f"{int(df2['population']):,}")
        st.metric("Median Age", f"{df2['median_age']:.1f} years")
        st.metric("Avg Household Size", f"{df2['avg_household_size']:.2f}")
        if 'cluster_label' in df2.index and pd.notna(df2.get('cluster_label')):
            st.metric("Cluster", f"{int(df2['cluster_label'])}")
        if 'lifestyle_score' in df2.index and pd.notna(df2.get('lifestyle_score')):
            st.metric("Lifestyle Score", f"{df2['lifestyle_score']:.3f}")
    
    # Visual Comparison Chart
    st.markdown("<div class='section-header'>Visual Comparison</div>", unsafe_allow_html=True)
    
    max_pop = max(df1['population'], df2['population'])
    
    comparison_data = pd.DataFrame({
        "Metric": ["Population (scaled)", "Median Age", "Household Size (x10)"],
        df1['city']: [
            (df1['population'] / max_pop) * 100,
            df1['median_age'],
            df1['avg_household_size'] * 10
        ],
        df2['city']: [
            (df2['population'] / max_pop) * 100,
            df2['median_age'],
            df2['avg_household_size'] * 10
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=df1['city'],
        x=comparison_data["Metric"],
        y=comparison_data[df1['city']],
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        name=df2['city'],
        x=comparison_data["Metric"],
        y=comparison_data[df2['city']],
        marker_color='#764ba2'
    ))
    fig.update_layout(
        barmode='group',
        title="City Metrics Comparison",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar Chart for Lifestyle Indexes (if available)
    if all(col in df1.index for col in ['opportunity_index', 'youth_index', 'family_index']):
        st.markdown("<div class='section-header'>Lifestyle Profile Comparison</div>", unsafe_allow_html=True)
        
        categories = ["Opportunity", "Youth", "Family"]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[df1.get('opportunity_index', 0), df1.get('youth_index', 0), df1.get('family_index', 0)],
            theta=categories,
            fill='toself',
            name=df1['city'],
            line_color='#667eea'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[df2.get('opportunity_index', 0), df2.get('youth_index', 0), df2.get('family_index', 0)],
            theta=categories,
            fill='toself',
            name=df2['city'],
            line_color='#764ba2'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Lifestyle Index Comparison",
            paper_bgcolor="white",
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # AI Analysis & Recommendation
    _show_comparison_ai_analysis(df1, df2, query)
    
    # Download comparison data
    comparison_df = pd.DataFrame([
        {
            "City": df1['city'],
            "State": df1['state'],
            "Population": df1['population'],
            "Median Age": df1['median_age'],
            "Avg Household Size": df1['avg_household_size'],
        },
        {
            "City": df2['city'],
            "State": df2['state'],
            "Population": df2['population'],
            "Median Age": df2['median_age'],
            "Avg Household Size": df2['avg_household_size'],
        }
    ])
    
    csv = convert_df_to_csv(comparison_df)
    st.download_button(
        label="📥 Download Comparison Data",
        data=csv,
        file_name=f"{df1['city']}_vs_{df2['city']}_comparison.csv",
        mime="text/csv",
    )


def _show_comparison_ai_analysis(df1: pd.Series, df2: pd.Series, query: str):
    """Show comparison analysis using RAG context or data-only summary."""
    
    st.markdown("<div class='section-header'>Analysis</div>", unsafe_allow_html=True)
    
    # Calculate differences
    pop_diff = abs(int(df1['population']) - int(df2['population']))
    age_diff = abs(df1['median_age'] - df2['median_age'])
    household_diff = abs(df1['avg_household_size'] - df2['avg_household_size'])
    
    larger_city = df1['city'] if df1['population'] > df2['population'] else df2['city']
    older_city = df1['city'] if df1['median_age'] > df2['median_age'] else df2['city']
    larger_household = df1['city'] if df1['avg_household_size'] > df2['avg_household_size'] else df2['city']
    
    # Data Comparison Card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    ">
        <div style="font-size: 0.7rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; font-weight: 600;">
            Data Comparison
        </div>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 0.5rem;">
            <strong>Population:</strong> {larger_city} is larger by {pop_diff:,} people.
        </p>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 0.5rem;">
            <strong>Demographics:</strong> {older_city} has an older population (difference: {age_diff:.1f} years).
        </p>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
            <strong>Household Size:</strong> {larger_household} has larger households (difference: {household_diff:.2f}).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to get RAG context
    try:
        from core.rag_search import get_city_rag_context
        
        ctx1 = get_city_rag_context(df1['city'], df1['state'])
        ctx2 = get_city_rag_context(df2['city'], df2['state'])
        
        has_rag = False
        
        if ctx1 or ctx2:
            st.markdown("#### From Knowledge Base")
            
            if ctx1:
                for chunk_type, texts in ctx1.items():
                    if texts:
                        text = texts[0][:150]
                        if '.' in text[50:]:
                            text = text[:text.rfind('.', 50) + 1]
                        st.info(f"**{df1['city']}:** {text}")
                        has_rag = True
                        break
            
            if ctx2:
                for chunk_type, texts in ctx2.items():
                    if texts:
                        text = texts[0][:150]
                        if '.' in text[50:]:
                            text = text[:text.rfind('.', 50) + 1]
                        st.info(f"**{df2['city']}:** {text}")
                        has_rag = True
                        break
            
            if has_rag:
                st.caption("Based on retrieved knowledge")
    except:
        pass


def _extract_specific_question(query: str, city1: str, city2: str) -> str:
    """Extract any specific question beyond the comparison request."""
    q_lower = query.lower()
    c1, c2 = city1.lower(), city2.lower()
    
    remove_phrases = [
        f"compare {c1} and {c2}", f"compare {c2} and {c1}",
        f"{c1} vs {c2}", f"{c2} vs {c1}",
        f"{c1} versus {c2}", f"{c2} versus {c1}",
        f"{c1} or {c2}", f"{c2} or {c1}",
        f"which is best {c1} or {c2}", f"which is better {c1} or {c2}",
    ]
    
    remaining = q_lower
    for phrase in remove_phrases:
        remaining = remaining.replace(phrase, "")
    
    remaining = remaining.strip().strip(".").strip(",").strip()
    
    if len(remaining) > 10:
        return remaining
    return None


 
# STATE COMPARISON CARD
 
def show_state_comparison(state1: str, stats1: dict, cities1: pd.DataFrame,
                          state2: str, stats2: dict, cities2: pd.DataFrame, query: str = ""):
    """Display comprehensive state comparison."""
    
    # Header
    st.markdown(
        f"<div class='section-header'>State Comparison: {state1} vs {state2}</div>",
        unsafe_allow_html=True
    )
    
    # Side-by-side stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">State Profile</div>
                <div class="insight-title">📍 {state1}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("Cities in Database", f"{int(stats1['city_count']):,}")
        st.metric("Total Population", f"{int(stats1['total_population']):,}")
        st.metric("Avg City Population", f"{int(stats1['avg_population']):,}")
        st.metric("Avg Median Age", f"{stats1['avg_median_age']:.1f} years")
        st.metric("Avg Household Size", f"{stats1['avg_household_size']:.2f}")
    
    with col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-label">State Profile</div>
                <div class="insight-title">{state2}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("Cities in Database", f"{int(stats2['city_count']):,}")
        st.metric("Total Population", f"{int(stats2['total_population']):,}")
        st.metric("Avg City Population", f"{int(stats2['avg_population']):,}")
        st.metric("Avg Median Age", f"{stats2['avg_median_age']:.1f} years")
        st.metric("Avg Household Size", f"{stats2['avg_household_size']:.2f}")
    
    # Top Cities Comparison
    st.markdown("<div class='section-header'>Top Cities Comparison</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Top 5 Cities in {state1}**")
        st.dataframe(
            cities1[["city", "population", "median_age"]].rename(
                columns={"city": "City", "population": "Population", "median_age": "Median Age"}
            ),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown(f"**Top 5 Cities in {state2}**")
        st.dataframe(
            cities2[["city", "population", "median_age"]].rename(
                columns={"city": "City", "population": "Population", "median_age": "Median Age"}
            ),
            use_container_width=True,
            hide_index=True
        )
    
    # Visual Comparison Chart
    st.markdown("<div class='section-header'>Visual Comparison</div>", unsafe_allow_html=True)
    
    comparison_data = pd.DataFrame({
        "Metric": ["Total Population (M)", "Avg City Population (K)", "Avg Median Age", "Avg Household Size"],
        state1: [
            stats1['total_population'] / 1_000_000,
            stats1['avg_population'] / 1_000,
            stats1['avg_median_age'],
            stats1['avg_household_size'] * 10
        ],
        state2: [
            stats2['total_population'] / 1_000_000,
            stats2['avg_population'] / 1_000,
            stats2['avg_median_age'],
            stats2['avg_household_size'] * 10
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name=state1, x=comparison_data["Metric"], y=comparison_data[state1], marker_color='#667eea'))
    fig.add_trace(go.Bar(name=state2, x=comparison_data["Metric"], y=comparison_data[state2], marker_color='#764ba2'))
    fig.update_layout(
        barmode='group',
        title="State Metrics Comparison",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Analysis
    _show_state_comparison_ai_analysis(state1, stats1, cities1, state2, stats2, cities2, query)
    
    # Download combined data
    combined_df = pd.DataFrame([
        {"State": state1, **{k: v for k, v in stats1.items() if k != 'state'}},
        {"State": state2, **{k: v for k, v in stats2.items() if k != 'state'}},
    ])
    
    csv = convert_df_to_csv(combined_df)
    st.download_button(
        label="📥 Download Comparison Data",
        data=csv,
        file_name=f"{state1}_vs_{state2}_comparison.csv",
        mime="text/csv",
    )


def _show_state_comparison_ai_analysis(state1: str, stats1: dict, cities1: pd.DataFrame,
                                        state2: str, stats2: dict, cities2: pd.DataFrame, query: str):
    """Show state comparison analysis using data only."""
    
    st.markdown("<div class='section-header'>Analysis</div>", unsafe_allow_html=True)
    
    # Calculate differences
    pop_diff = abs(int(stats1['total_population']) - int(stats2['total_population']))
    larger_state = state1 if stats1['total_population'] > stats2['total_population'] else state2
    
    age_diff = abs(stats1['avg_median_age'] - stats2['avg_median_age'])
    older_state = state1 if stats1['avg_median_age'] > stats2['avg_median_age'] else state2
    
    more_cities = state1 if stats1['city_count'] > stats2['city_count'] else state2
    city_diff = abs(int(stats1['city_count']) - int(stats2['city_count']))
    
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-label">Data Comparison</div>
        <div class="insight-text">
            <p><strong>Population:</strong> {larger_state} has a larger total population by {pop_diff:,}.</p>
            <p><strong>Demographics:</strong> {older_state} has an older average population (difference: {age_diff:.1f} years).</p>
            <p><strong>Urbanization:</strong> {more_cities} has {city_diff} more cities in our database.</p>
            <p><strong>Top Cities in {state1}:</strong> {', '.join(cities1['city'].head(3).tolist())}</p>
            <p><strong>Top Cities in {state2}:</strong> {', '.join(cities2['city'].head(3).tolist())}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


 
# RECOMMENDATION CARD
 
def _generate_dynamic_title(query: str, intent: str) -> str:
    """Generate title dynamically based on user's actual query."""
    import re
    
    q = query.lower().strip()
    
    if not q:
        # No query provided, use intent-based title
        intent_titles = {
            "families": "Best City for Families",
            "young_professionals": "Best City for Young Professionals",
            "retirement": "Best City for Retirement",
            "general": "Top Recommended City"
        }
        return intent_titles.get(intent, "Top Recommended City")
    
    # Try to extract what user asked for
    if "best" in q:
        # Pattern: "best city for X" or "best cities for X"
        match = re.search(r"best\s+(?:city|cities|place|places)\s+(?:for|to live for)\s+(.+?)(?:\?|$)", q)
        if match:
            subject = match.group(1).strip().rstrip("?.,!")
            return f"Best City for {subject.title()}"
        
        # Pattern: "best X city" or "best X cities"
        match = re.search(r"best\s+(.+?)\s+(?:city|cities|place|places)", q)
        if match:
            subject = match.group(1).strip()
            return f"Best {subject.title()} City"
        
        # Pattern: "which is best for X" or "what is best for X"
        match = re.search(r"(?:which|what)\s+(?:is|are)\s+(?:the\s+)?best\s+(?:city\s+)?(?:for\s+)?(.+?)(?:\?|$)", q)
        if match:
            subject = match.group(1).strip().rstrip("?.,!")
            if subject and len(subject) > 1:
                return f"Best City for {subject.title()}"
    
    # Fallback to intent-based titles
    intent_titles = {
        "families": "Best City for Families",
        "young_professionals": "Best City for Young Professionals",
        "retirement": "Best City for Retirement",
        "general": "Top Recommended City"
    }
    return intent_titles.get(intent, "Top Recommended City")


def show_recommendation_card(top_city, intent: str, df: pd.DataFrame, query: str = ""):
    """Display recommendation card using native Streamlit components."""
    
    city_name = top_city.get("city", "Unknown")
    state_name = top_city.get("state", "")
    raw_score = top_city.get("score", 0)
    
    # Get all scores for percentile calculation
    all_scores = df["score"].values if "score" in df.columns else None
    
    # Calculate normalized score
    try:
        from core.score_translate import format_score_display
        score_info = format_score_display(raw_score, all_scores)
        score_display = score_info["score_100"]
        label = score_info["label"]
        emoji = score_info["emoji"]
    except Exception:
        if all_scores is not None and len(all_scores) > 0:
            import numpy as np
            score_display = round((np.sum(all_scores < raw_score) / len(all_scores)) * 100, 1)
        else:
            score_display = round(min(100, raw_score * 10), 1) if raw_score < 10 else round(raw_score, 1)
        
        if score_display >= 90:
            label, emoji = "Excellent Match", "⭐"
        elif score_display >= 75:
            label, emoji = "Great Match", "🟢"
        elif score_display >= 60:
            label, emoji = "Good Match", "🔵"
        elif score_display >= 40:
            label, emoji = "Average Match", "🟡"
        else:
            label, emoji = "Below Average", "🟠"
    
    # Dynamic title based on query
    title = _generate_dynamic_title(query, intent)
    
    # Display using native Streamlit
    st.subheader(title)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"### {city_name}")
        st.caption(f"📍 {state_name}")
    
    with col2:
        st.metric(
            label="Match Score",
            value=f"{score_display:.0f}/100",
            delta=f"{emoji} {label}"
        )
    
    # Progress bar
    st.progress(int(min(100, score_display)) / 100)
    st.caption(f"This city ranks in the top {100 - score_display:.0f}% for {intent.replace('_', ' ')}")
    
    st.divider()
    
 
# CLUSTER SCATTER PLOT
 
def show_cluster_scatter(df_clusters: pd.DataFrame):
    """Display cluster scatter plot with PCA."""
    
    required = ["ml_vector_population", "ml_vector_age", "ml_vector_household"]
    if not all(col in df_clusters.columns for col in required):
        st.caption("Scatter plot unavailable (missing ML feature columns).")
        return
    
    X = df_clusters[required].values
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    
    plot_df = df_clusters.copy()
    plot_df["pc1"] = comps[:, 0]
    plot_df["pc2"] = comps[:, 1]
    
    color_col = "cluster_label" if "cluster_label" in plot_df.columns else "cluster_id"
    
    fig = px.scatter(
        plot_df,
        x="pc1",
        y="pc2",
        color=color_col,
        hover_data=["city", "state"],
        title="City Clusters — PCA Projection",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


 
# LIFESTYLE CARD
 
def show_lifestyle_card(city: str, state: str, population, median_age, household_size, 
                        description: str, ai_summary: str):
    """Display lifestyle profile card."""
    
    card_html = f"""
    <div class="insight-card">
      <div class="insight-label">Lifestyle Profile Generated with RAG</div>
      <div class="insight-title">Life in {city}, {state}</div>
      <div class="insight-text">{ai_summary}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if population is not None:
            st.metric("Population", f"{population:,}")
    with col2:
        if median_age is not None:
            st.metric("Median age", f"{median_age:.1f} years")
    with col3:
        if household_size is not None:
            st.metric("Avg household size", f"{household_size:.2f}")
    
    if description:
        st.markdown("<div class='section-header'>Dataset Description</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-container'>{description}</div>", unsafe_allow_html=True)


 
# CITY TABLE
 
COLUMN_DISPLAY_NAMES = {
    'city': 'City',
    'state': 'State',
    'population': 'Population',
    'median_age': 'Median Age',
    'avg_household_size': 'Household Size',
    'state_code': 'State Code',
    'Score (0-100)': 'Score',
    'Rating': 'Rating',
    'score': 'Score',
    'total_population': 'Total Population',
    'city_count': 'City Count',
    'avg_age': 'Average Age',
    'total_pop': 'Total Population',
    'cnt': 'Count',
    'pop': 'Population',
    'age': 'Age'
}


def show_city_table(df: pd.DataFrame, title: str = "Results", show_download: bool = True):
    """Display a formatted table of cities with improved score display."""
    
    if df is None or df.empty:
        st.info("No data to display.")
        return
    
    # Create display copy
    display_df = df.copy()
    
    # Rename columns to clean display names
    display_df = display_df.rename(columns={k: v for k, v in COLUMN_DISPLAY_NAMES.items() if k in display_df.columns})
    
    # Format score column if present
    if "score" in display_df.columns:
        all_scores = display_df["score"].values
        
        try:
            from core.score_translate import normalize_to_100, to_level
            
            # Add normalized score
            display_df["Score (0-100)"] = display_df["score"].apply(
                lambda x: normalize_to_100(x, all_scores)
            )
            
            # Add rating label
            display_df["Rating"] = display_df["Score (0-100)"].apply(
                lambda x: to_level(x)[0]
            )
            
            # Remove raw score column
            display_df = display_df.drop(columns=["score"], errors="ignore")
            
        except Exception:
            # Fallback: simple normalization
            display_df["Score (0-100)"] = display_df["score"].apply(
                lambda x: round(min(100, x * 10), 1) if x < 10 else round(x, 1)
            )
            display_df = display_df.drop(columns=["score"], errors="ignore")
    
    # Format population if present
    if "population" in display_df.columns:
        display_df["population"] = display_df["population"].apply(
            lambda x: f"{x:,}" if pd.notna(x) else "N/A"
        )
    
    st.markdown(f"#### {title}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    if show_download:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )


 
# AI RESPONSE CARD
 
def show_ai_response(message: str, is_fallback: bool = False):
    """Display an AI-generated response."""
    
    if is_fallback:
        label = "ℹThis answer is from AI knowledge (not from our verified database)"
        bg_color = "rgba(237, 137, 54, 0.1)"
        border_color = "#ed8936"
    else:
        label = ""
        bg_color = "rgba(102, 126, 234, 0.1)"
        border_color = "#667eea"
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border-left: 4px solid {border_color};
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    ">
        {"<div style='font-size: 0.8rem; color: " + border_color + "; margin-bottom: 0.5rem;'>" + label + "</div>" if label else ""}
        <div>{convert_markdown_to_html(message)}</div>
    </div>
    """, unsafe_allow_html=True)


 
# OUT OF SCOPE CARD
 
def show_out_of_scope():
    """Display out of scope message for non-city questions."""
    
    st.markdown(
        """
        <div class='insight-card' style="background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);">
            <div class='insight-label'>Out of Scope</div>
            <div class='insight-title'>This platform provides US city insights only</div>
            <div class='insight-text'>
                Your question doesn't appear to be about US cities or states.<br><br>
                You can ask about:<br>
                • City populations and demographics<br>
                • Best cities for families, professionals, or retirement<br>
                • City and state comparisons<br>
                • Lifestyle profiles and similar cities<br><br>
                Please ask something related to <b>US cities</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


 
# AI SUMMARY GENERATOR
 
def generate_ai_summary(df: pd.DataFrame, query: str, summary_style: str = "brief") -> str:
    """Generate summary based on data patterns only - no GPT calls."""
    
    if df is None or df.empty:
        return None
    
    # Build data-based summary
    num_results = len(df)
    
    if 'population' in df.columns:
        avg_pop = df['population'].mean()
        max_pop = df['population'].max()
        top_city = df.loc[df['population'].idxmax(), 'city'] if 'city' in df.columns else "Top result"
        return f"Found {num_results} results. {top_city} leads with population {int(max_pop):,}. Average population: {int(avg_pop):,}."
    
    if 'city' in df.columns:
        cities = df['city'].head(3).tolist()
        return f"Found {num_results} results including: {', '.join(cities)}."
    
    return f"Found {num_results} results matching your query."
