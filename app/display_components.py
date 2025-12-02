"""
UI display components for CitySearch AI.
All cards, tables, and visual elements.
"""

import streamlit as st
import pandas as pd
from llm_classifier import get_openai_client


def show_single_city_card(city_name: str, state_name: str, metric_value, metric_label: str, rank_label: str, runners_up: list = None):
    """Display a single city answer card."""
    
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
    
    # Runners up
    if runners_up and len(runners_up) > 0:
        st.markdown(f"""
        <div style="
            padding: 0.75rem 1rem;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        ">
            <span style="opacity: 0.7;">Other top cities:</span> {", ".join(runners_up[:4])}
        </div>
        """, unsafe_allow_html=True)


def show_city_profile_card(city_name: str, state_name: str, row: pd.Series):
    """Display a beautiful city profile card with AI-generated summary."""
    
    population = row.get('population', 'N/A')
    median_age = row.get('median_age', 'N/A')
    household_size = row.get('avg_household_size', 'N/A')
    
    # Format population
    if isinstance(population, (int, float)):
        population_display = f"{int(population):,}"
    else:
        population_display = str(population)
    
    # Main card
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
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Summary
    client = get_openai_client()
    if client:
        with st.spinner("Generating city profile..."):
            prompt = f"""
            Write a 2-3 sentence profile for {city_name}, {state_name}.
            Population: {population_display}, Median Age: {median_age}, Household Size: {household_size}
            Focus on lifestyle, culture, and what makes this city unique. Be concise.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                summary = response.choices[0].message.content.strip()
                
                st.markdown(f"""
                <div style="
                    background: rgba(102, 126, 234, 0.1);
                    border-left: 4px solid #667eea;
                    border-radius: 8px;
                    padding: 1.25rem;
                    margin-bottom: 1.5rem;
                    font-style: italic;
                ">
                    "{summary}"
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass
        
        # Highlights
        with st.spinner("Generating highlights..."):
            prompt = f"""
            Give exactly 3 short highlights about {city_name}, {state_name}.
            Each highlight: 5-8 words maximum. No bullets or numbers.
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=100
                )
                highlights = response.choices[0].message.content.strip().split('\n')
                highlights = [h.strip().lstrip('•-*0123456789. ') for h in highlights if h.strip()][:3]
                
                st.markdown("<h4 style='color: #667eea;'>✨ Highlights</h4>", unsafe_allow_html=True)
                
                for h in highlights:
                    st.markdown(f"<div style='padding: 0.3rem 0;'><span style='color: #667eea;'>✓</span> {h}</div>", unsafe_allow_html=True)
            except Exception:
                pass


def show_comparison_card(city1_row: pd.Series, city2_row: pd.Series):
    """Display city comparison."""
    
    metrics = [
        ("Population", "population", "{:,}"),
        ("Median Age", "median_age", "{:.1f}"),
        ("Household Size", "avg_household_size", "{:.2f}")
    ]
    
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
        <div style="flex: 1; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 1.5rem; color: white; text-align: center;">
            <h3 style="margin: 0;">{city1_row['city']}, {city1_row['state']}</h3>
        </div>
        <div style="display: flex; align-items: center; font-size: 1.5rem; font-weight: bold; color: #667eea;">VS</div>
        <div style="flex: 1; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 16px; padding: 1.5rem; color: white; text-align: center;">
            <h3 style="margin: 0;">{city2_row['city']}, {city2_row['state']}</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    for label, col, fmt in metrics:
        val1 = city1_row.get(col, 0)
        val2 = city2_row.get(col, 0)
        
        formatted1 = fmt.format(val1) if isinstance(val1, (int, float)) else str(val1)
        formatted2 = fmt.format(val2) if isinstance(val2, (int, float)) else str(val2)
        
        # Determine winner
        if val1 > val2:
            style1, style2 = "font-weight: 700; color: #667eea;", ""
        elif val2 > val1:
            style1, style2 = "", "font-weight: 700; color: #11998e;"
        else:
            style1, style2 = "", ""
        
        st.markdown(f"""
        <div style="display: flex; padding: 0.75rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
            <div style="flex: 1; text-align: center; {style1}">{formatted1}</div>
            <div style="flex: 1; text-align: center; font-weight: 600; opacity: 0.7;">{label}</div>
            <div style="flex: 1; text-align: center; {style2}">{formatted2}</div>
        </div>
        """, unsafe_allow_html=True)


def show_aggregate_card(label: str, value: str, sub_label: str = ""):
    """Display an aggregate result card."""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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


def show_city_table(df: pd.DataFrame, title: str = "Results"):
    """Display a city table with download option."""
    
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=400)
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="city_results.csv",
        mime="text/csv"
    )


def show_ai_response(message: str, is_fallback: bool = False):
    """Display an AI-generated response."""
    
    label = "ℹ️ This answer is from AI (not from our database)" if is_fallback else ""
    
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    ">
        {"<div style='font-size: 0.8rem; color: #667eea; margin-bottom: 0.5rem;'>" + label + "</div>" if label else ""}
        <div>{message}</div>
    </div>
    """, unsafe_allow_html=True)


def show_insight_bullets(insights: list):
    """Display insight bullet points."""
    
    for insight in insights[:3]:
        insight = insight.strip().lstrip('•-*0123456789. ')
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; gap: 0.5rem; padding: 0.4rem 0;">
            <span style="color: #667eea;">•</span>
            <span>{insight}</span>
        </div>
        """, unsafe_allow_html=True)