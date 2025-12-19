#city.py - Main Streamlit Application

import streamlit as st
import pandas as pd


# PAGE CONFIG (must be first Streamlit command)

st.set_page_config(
    page_title="CitySearch AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# IMPORTS

from config import APP_TITLE, APP_SUBTITLE, QUICK_EXAMPLES
from styles import get_custom_css
from utils import correct_query_spelling, is_nonsense_query, is_world_query
from hybrid_classifier import classify_query_hybrid  # NEW: Hybrid classification
from query_handlers import handle_query
# from mlops_dashboard import render_mlops_dashboard

# Import database config
from db_config import get_engine

# Import core modules (from your existing codebase)
try:
    from core.smart_router import smart_route
except ImportError:
    smart_route = None

try:
    from core.lifestyle_rag_v2 import try_build_lifestyle_card
except ImportError:
    try_build_lifestyle_card = None

try:
    from core.intent_classifier import classify_query_intent
except ImportError:
    classify_query_intent = None

try:
    from core.ml_utils import load_trained_model, load_feature_data
except ImportError:
    load_trained_model = None
    load_feature_data = None



# CACHED DATA LOADING

@st.cache_resource
def load_models():
    """Load ML models once and cache."""
    if load_trained_model:
        try:
            model, metadata = load_trained_model("city_clusters")
            return model, metadata
        except Exception:
            pass
    return None, None


@st.cache_data(ttl=3600)
def load_features():
    """Load feature data once and cache."""
    if load_feature_data:
        try:
            return load_feature_data()
        except Exception:
            pass
    
    # Fallback: load from database
    try:
        engine = get_engine()
        query = """
            SELECT city, state, state_code, population, median_age, avg_household_size
            FROM dbo.cities
        """
        # Use connection context manager for SQLAlchemy 2.0 compatibility
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


# Load data
model, metadata = load_models()
df_features = load_features()



# SESSION STATE INITIALIZATION

if "current_query" not in st.session_state:
    st.session_state["current_query"] = ""

if "auto_search" not in st.session_state:
    st.session_state["auto_search"] = False

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"



# APPLY CUSTOM CSS

st.markdown(get_custom_css(st.session_state["theme"]), unsafe_allow_html=True)

with st.sidebar:

    # Always Search mode — no radio buttons
    mode = "Search"

    # Quick examples
    st.markdown("### Quick Examples")
    
    for example in QUICK_EXAMPLES:
        if st.button(example, key=f"btn_{example}", use_container_width=True):
            st.session_state["current_query"] = example
            st.session_state["auto_search"] = True
            st.rerun()
    
    # st.markdown("<hr style='margin: 0.3rem 0;'>", unsafe_allow_html=True)
    
    # # Data info with no extra margin
    # if not df_features.empty:
    #     st.markdown(f"""
    #     <div style="font-size: 0.8rem; color: #a0aec0; line-height: 1.1; margin-bottom: 0;">
    #         <strong>{len(df_features):,}</strong> cities loaded<br>
    #         <strong>{df_features['state'].nunique()}</strong> states covered
    #     </div>
    #     """, unsafe_allow_html=True)

    

# HERO SECTION

st.markdown(
    f"""
    <div class='hero-section'>
        <div class='hero-title'>{APP_TITLE}</div>
        <div class='hero-subtitle'>{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# SEARCH INPUT

left_col, clear_col, right_col = st.columns([4, 0.3, 1])

with left_col:
    user_query = st.text_input(
        "",
        value=st.session_state.get("current_query", ""),
        placeholder="Ask anything about US cities... (e.g., 'Best cities for families in Texas')",
        label_visibility="collapsed"
    )
    
with clear_col:
    if st.button("✕ ", key="clear_btn", help="Clear search"):
        st.session_state["current_query"] = ""
        st.rerun()

with right_col:
    search_clicked = st.button("Search", use_container_width=True)

# Auto-search trigger (from sidebar examples)
if st.session_state.get("auto_search", False):
    search_clicked = True
    st.session_state["auto_search"] = False


# QUERY PROCESSING

if search_clicked and user_query.strip():
    query_original = user_query.strip()
    
    # # Clear the stored query to prevent re-runs
    # st.session_state["current_query"] = ""
    
    
    # STEP 1: Spelling Correction
    
    city_list = df_features["city"].unique().tolist() if not df_features.empty else []
    query, corrections = correct_query_spelling(query_original, city_list)
    
    if corrections:
        correction_text = ", ".join([f'"{old}" → "{new}"' for old, new in corrections])
        st.info(f"Auto-corrected: {correction_text}")
    
    
    # STEP 2: Safety Checks
    
    if is_nonsense_query(query):
        st.error("I couldn't understand that question. Try asking about US cities.")
        st.stop()
    
    if is_world_query(query):
        st.error("This system only supports US cities. Please ask about cities in the United States.")
        st.stop()
    
    
    # STEP 3: Hybrid Classification (Rule-based first, LLM fallback)
    
    with st.spinner("Understanding your question..."):
        classification = classify_query_hybrid(query)
    
    # we need to hide this
    # Show classification source (for debugging - can be removed)
    source = classification.get("source", "unknown")
    confidence = classification.get("confidence", "unknown")
    if source == "rule_based":
        source = classification.get("source", "unknown")
        source_labels = {
            "cache": "Cached result",
            "pattern": "Pattern match",
            "llm": "LLM classification",
            "rule_based": "Rule-based fallback"
        }
        st.caption(source_labels.get(source, f"Classified ({source})"))
    else:
        # Display classification source
        source = classification.get("source", "unknown")
        source_labels = {
            "cache": "Cached result",
            "pattern": "Pattern match",
            "llm": "LLM classified",
            "rule_based": "Rule-based"
        }
        query_type = classification.get("query_type", "")
        st.caption(f"{source_labels.get(source, 'Classified')} → {query_type}")
    
    # Debug: Show classification (can be removed in production)
    # with st.expander("🔍 Debug: Query Classification", expanded=False):
    #     st.json(classification)
    
    
    # STEP 4: Handle Query
    
    handle_query(
        query=query,
        classification=classification,
        df_features=df_features,
        get_engine_func=get_engine,
        smart_route_func=smart_route,
        lifestyle_rag_func=try_build_lifestyle_card,
        classify_intent_func=classify_query_intent
    )


# EMPTY STATE

elif not search_clicked:
    # Feature highlights - at the top with minimal spacing
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 15rem; margin-bottom: 0.5rem;">
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;"></div>
            <div style="font-weight: 600; font-size: 0.95rem;">City Data</div>
            <div style="font-size: 0.8rem; color: #a0aec0;">Population, demographics, and more</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;"></div>
            <div style="font-weight: 600; font-size: 0.95rem;">AI-Powered</div>
            <div style="font-size: 0.8rem; color: #a0aec0;">Smart recommendations and insights</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;"></div>
            <div style="font-weight: 600; font-size: 0.95rem;">Instant Results</div>
            <div style="font-size: 0.8rem; color: #a0aec0;">Fast answers to any city question</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# FOOTER

st.markdown("<hr style='margin: 0.5rem 0; border-color: #2d3748;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 0.75rem; color: #a0aec0; padding: 0.25rem 0; margin-bottom: 0rem;">
    CitySearch AI - Powered by GPT-4 and ML Clustering<br>
    <span style="opacity: 0.7;">Built by Thaadshaayani Rasanehru</span>
</div>
""", unsafe_allow_html=True)

