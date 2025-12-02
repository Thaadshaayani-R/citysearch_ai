"""
CitySearch AI - Main Application
A smart AI-powered search engine for US cities.
"""

import streamlit as st
import pandas as pd
from db_config import get_engine
from config import APP_TITLE, APP_SUBTITLE
from llm_classifier import classify_query
from query_handlers import handle_query


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="CitySearch AI",
    page_icon="🏙️",
    layout="wide"
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_city_data():
    """Load city data from database."""
    engine = get_engine()
    query = """
        SELECT city, state, state_code, population, median_age, avg_household_size
        FROM dbo.cities
    """
    return pd.read_sql(query, engine)


df_features = load_city_data()


# -------------------------------------------------
# STYLES
# -------------------------------------------------
st.markdown("""
<style>
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("### Choose View")
    mode = st.radio("", ["Search", "MLOps Dashboard"], label_visibility="collapsed")
    
    st.markdown("### Quick Examples")
    examples = [
        "Which city has the highest population?",
        "Top 10 cities in Texas",
        "Compare Miami and Dallas",
        "Life in Denver",
        "Best cities for families",
        "How many cities in California?"
    ]
    
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state["query"] = example
            st.rerun()


# -------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------
if mode == "Search":
    
    # Hero Section
    st.markdown(f"""
    <div class='hero-section'>
        <div class='hero-title'>{APP_TITLE}</div>
        <div class='hero-subtitle'>{APP_SUBTITLE}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Box
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input(
            "",
            value=st.session_state.get("query", ""),
            placeholder="Ask anything about US cities...",
            label_visibility="collapsed"
        )
    with col2:
        search_clicked = st.button("Search", use_container_width=True)
    
    # Handle Query
    if search_clicked and user_query.strip():
        query = user_query.strip()
        
        # Clear previous query from session
        if "query" in st.session_state:
            del st.session_state["query"]
        
        # Classify query with LLM
        with st.spinner("Understanding your question..."):
            intent = classify_query(query)
        
        # Handle the query based on intent
        handle_query(query, intent, df_features)

else:
    # MLOps Dashboard
    st.markdown("""
    <div class='hero-section'>
        <div class='hero-title'>MLOps Dashboard</div>
        <div class='hero-subtitle'>Monitor model health, drift, and retraining.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("MLOps dashboard coming soon...")
