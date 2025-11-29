#dash.py

import os
import io
import json
from db_config import get_engine, get_connection
import pandas as pd
import streamlit as st
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# core imports
from core.query_router import build_sql_with_fallback
from core.semantic_search import semantic_city_search
from core.intent_classifier import classify_query_intent
from core.smart_router import smart_route
from core.ml_explain import explain_ml_results
from core.score_translate import to_level

from mlops.retrain import retrain

# clustering helpers
from core.cluster_router import (
    cluster_all,
    cluster_by_state,
    cluster_single_city,
    cluster_similar_to,
)
from core.cluster_explain import explain_cluster

# Lifestyle RAG
from core.lifestyle_rag_v2 import try_build_lifestyle_card

# ML utilities
from core.ml_utils import load_trained_model, load_feature_data

# MLOps imports
from mlops.monitoring import run_monitoring
from mlops.retrain import retrain

import time

# Add this helper at the top of app.py
def log_time(label, start):
    elapsed = time.time() - start
    print(f"⏱️ {label}: {elapsed:.2f}s")
    return time.time()

if search_clicked and user_query.strip():
    t = time.time()
    q = user_query.strip()
    
    # Safety checks
    if is_nonsense_query(q):
        respond_nonsense()
    t = log_time("Safety check", t)

    if is_world_query(q):
        respond_world_not_supported()

    # Two-city comparison
    two_cities = extract_two_cities(q)
    t = log_time("Two-city extraction", t)

# -------------------------------------------------
# CACHED MODEL & DATA LOADING
# -------------------------------------------------
@st.cache_resource
def load_models():
    """Load ML models once and cache."""
    from core.ml_utils import load_trained_model
    model, metadata = load_trained_model("city_clusters")
    return model, metadata


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_features():
    """Load feature data once and cache."""
    from core.ml_utils import load_feature_data
    return load_feature_data()


# Use cached versions
model, metadata = load_models()
df_features = load_features()


# -------------------------------------------------
# SAFETY FILTERS (BLOCK 2)
# -------------------------------------------------
import re

def is_nonsense_query(q: str):
    if len(q.strip()) < 2:
        return True
    if re.fullmatch(r"[0-9]+", q.strip()):
        return True
    return False

def is_world_query(q: str):
    forbidden = [
        "canada","india","europe","uk","england",
        "china","japan","australia","germany","mexico",
        "france","spain","africa","dubai","uae"
    ]
    q = q.lower()
    return any(f in q for f in forbidden)

def respond_nonsense():
    st.error("❌ I couldn't understand that question. Try asking about US cities.")
    st.stop()

def respond_world_not_supported():
    st.error("❌ This system only supports US cities.")
    st.stop()


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="CitySearch AI", layout="wide", initial_sidebar_state="expanded")

# -------------------------------------------------
# INIT CHAT HISTORY (must run before anything else)
# -------------------------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# -------------------------------------------------
# THEME TOGGLE
# -------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# -------------------------------------------------
# GLOBAL ML OBJECTS
# -------------------------------------------------
model, metadata = load_trained_model("city_clusters")  # New line
df_features = load_feature_data()


# Dynamic theme styles
if st.session_state.theme == 'dark':
    bg_main = "#0f1419"
    bg_card = "#1a202c"
    bg_input = "#1a202c"
    text_primary = "#ffffff"
    text_secondary = "#e2e8f0"
    text_muted = "#a0aec0"
    border_color = "#2d3748"
    gradient_start = "#3b4a6b"
    gradient_end = "#4a5f7f"
    accent_gradient_start = "#5a67d8"
    accent_gradient_end = "#7c3aed"
else:
    bg_main = "#f7fafc"
    bg_card = "#ffffff"
    bg_input = "#ffffff"
    text_primary = "#1a202c"
    text_secondary = "#2d3748"
    text_muted = "#718096"
    border_color = "#e2e8f0"
    gradient_start = "#667eea"
    gradient_end = "#764ba2"
    accent_gradient_start = "#667eea"
    accent_gradient_end = "#764ba2"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
        
    .main {{
        background-color: {bg_main};
        padding: 1rem 2rem;
    }}
    
    /* Remove Streamlit's global top padding */
    .css-18e3th9 {{
        padding-top: 0 !important;
        margin-top: 3rem !important;
    }}

    /* Remove padding from main content block */
    .block-container {{
        padding-top: 0rem !important;
        margin-top: 3rem !important;
    }}
    
    /* Make "Choose view", "Search", "MLOps Dashboard" smaller */
    [data-testid="stSidebar"] label {{
        font-size: 0.2rem !important;    /* try 0.75 or 0.70 if you want smaller */
        line-height: 1.1 !important;
    }}

    /* Tighten spacing between the two radio options */
    [data-testid="stSidebar"] div[role="radiogroup"] {{
        row-gap: 0.2rem !important;
        margin-bottom: -0.5rem !important;
    }}

    /* Shrink the radio circle a bit */
    [data-testid="stSidebar"] input[type="radio"] {{
        transform: scale(0.8) !important;
    }}



    .hero-section {{
    background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
    padding: 1rem 1.5rem;     /* ↓ reduced from 1.5rem 2rem */
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Force Quick Example to stay on ONE LINE */
    [data-testid="stSidebar"] .stButton>button {{
        font-size: 0.55rem !important;
        padding: 0.4rem 0.6rem !important;
        line-height: 1.0 !important;
        white-space: nowrap !important;        /* force one line */
        overflow: hidden !important;           /* hide overflow */
        text-overflow: ellipsis !important;    /* add ... if too long */
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
    }}

    /* Also target the inner div */
    [data-testid="stSidebar"] .stButton>button > div {{
        font-size: 0.55rem !important;
        white-space: nowrap !important;
        text-align: left !important;
    }}

    /* Also target the inner span */
    [data-testid="stSidebar"] .stButton>button span {{
        font-size: 0.55rem !important;
        white-space: nowrap !important;
        text-align: left !important;
    }}
    
    /* Reduce spacing between cards */
    [data-testid="stSidebar"] .stButton {{
        margin-bottom: -0.6rem !important;
    }}


    .hero-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }}
    
    .hero-subtitle {{
        font-size: 0.8rem;
        color: #e2e8f0;
        font-weight: 400;
        line-height: 1.5;
        max-width: 1000px;
    }}
    
    .data-container {{
        background-color: {bg_card};
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid {border_color};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .sql-container {{
        background-color: {bg_card};
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        border: 1px solid {border_color};
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        border: none;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.25);
        margin-top: 1rem;
        color: white;
    }}

    .metric-header {{
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .metric-grid {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        text-align: center;
        width: 100%;
    }}

    .metric-label {{
        font-size: 0.75rem;
        opacity: 0.85;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }}

    .metric-value {{
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }}

    
    /* --- Optimized Result Card --- */
    .result-card {{
        background: linear-gradient(135deg, {accent_gradient_start} 0%, {accent_gradient_end} 100%);
        padding: 1rem 1.25rem;                     /* reduced height */
        border-radius: 12px;                       /* smoother rounding */
        margin: 0.8rem 0;                          /* tighter spacing */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18); /* softer shadow */
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #ffffff;
    }}

    /* Title (left side) */
    .result-card-title {{
        font-size: 1rem;            /* reduced from 1.25rem */
        font-weight: 600;
        margin-bottom: 0.6rem;      /* tighter spacing */
        color: #f1f1f1;
    }}

    /* Metric label (TOTAL_CITIES) */
    .result-card-subtitle {{
        font-size: 0.8rem; 
        opacity: 0.9;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }}

    /* Value (57) */
    .result-card-value {{
        font-size: 1.8rem;          /* reduced from 2rem */
        font-weight: 800;
        color: #ffffff;
    }}

    
    .insight-card {{
        background: linear-gradient(135deg, {accent_gradient_start} 0%, {accent_gradient_end} 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .insight-label {{
        font-size: 0.65rem;
        color: rgba(255, 255, 255, 0.75);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }}
    
    .insight-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }}
    
    .insight-text {{
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.6;
    }}
    
    .lifestyle-card {{
        background: linear-gradient(135deg, {accent_gradient_start} 0%, {accent_gradient_end} 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .section-header {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {text_primary};
        margin: 0.3rem 0 0.3rem 0 !important;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid {border_color};
        letter-spacing: -0.01em;
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, {accent_gradient_start} 0%, {accent_gradient_end} 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.25rem;
        font-weight: 700;
        font-size: 0.85rem;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        letter-spacing: 0.02em;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    
    .sidebar .stButton>button {{
        background: {bg_card};
        color: {text_secondary};
        border: 1px solid {border_color};
        font-weight: 500;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
    }}
    
    .sidebar .stButton>button:hover {{
        background: {border_color};
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {text_primary};
    }}
    
    div[data-testid="stMetricLabel"] {{
        font-size: 0.7rem;
        color: {text_muted};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {bg_card};
        padding: 1rem;
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {text_secondary};
    }}
    
    .stTextInput input {{
        background-color: {bg_input};
        color: {text_primary};
        border: 1px solid {border_color};
        border-radius: 6px;
        font-size: 0.9rem;
        padding: 0.6rem 1rem;
    }}
    
    .stTextInput input:focus {{
        border-color: {accent_gradient_start};
        box-shadow: 0 0 0 1px {accent_gradient_start};
    }}
    
    .stCheckbox {{
        color: {text_secondary};
    }}
    
    .dataframe {{
        font-size: 0.85rem;
    }}
    
    .stAlert {{
        background-color: {bg_card};
        border: 1px solid {border_color};
        color: {text_secondary};
        border-radius: 6px;
    }}
    
    .download-btn {{
        background: {bg_card};
        border: 1px solid {border_color};
        padding: 0.5rem 1rem;
        border-radius: 6px;
        color: {text_primary};
        font-weight: 600;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }}
    
    .download-btn:hover {{
        background: {border_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# DB HELPERS
# -------------------------------------------------
def get_db_connection():
    server = os.getenv("SQL_SERVER_HOST")
    database = os.getenv("SQL_SERVER_DB")
    username = os.getenv("SQL_SERVER_USER")
    password = os.getenv("SQL_SERVER_PASSWORD")
    driver = os.getenv("SQL_SERVER_DRIVER", "{ODBC Driver 17 for SQL Server}")

    return pyodbc.connect(
        f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    )


def run_sql_query(sql):
    import pandas as pd

    # open connection
    conn = get_connection()
    raw_conn = conn.connection  # raw pyodbc

    df = pd.read_sql(sql, raw_conn)
    
    conn.close()
    return df

# -------------------------------------------------
# LOG USER QUERY INTO SQL SERVER
# -------------------------------------------------
from db_config import get_connection
from sqlalchemy import text

def log_user_query(query_text, detected_mode, ml_mode):
    try:
        engine = get_engine()

        # use SQLAlchemy text() for parameterized INSERT
        insert_sql = text("""
            INSERT INTO user_queries (query_text, detected_mode, ml_mode)
            VALUES (:query_text, :detected_mode, :ml_mode)
        """)

        with engine.begin() as conn:
            conn.execute(
                insert_sql,
                {
                    "query_text": query_text,
                    "detected_mode": detected_mode,
                    "ml_mode": ml_mode
                }
            )

    except Exception as e:
        print("Query Log Error:", e)





def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def summarize_results(df: pd.DataFrame, user_query: str) -> str:
    client = get_openai_client()
    if client is None or df.empty:
        return "AI summary unavailable."

    sample = df.head(10).to_markdown(index=False)

    prompt = f"""
User asked: "{user_query}"

Here are the first 10 rows of the results:
{sample}

Give 3–5 short, simple insights in plain English.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a concise data analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------
# CSV DOWNLOAD HELPER
# -------------------------------------------------
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


# -------------------------------------------------
# ADVANCED CITY DETECTION
# -------------------------------------------------
def detect_city_in_query(q: str) -> str | None:
    import re

    q_low = q.lower()
    city_values = df_features["city"].str.lower().values

    # 1) anything inside single or double quotes
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", q)
    for grp in quoted:
        value = (grp[0] or grp[1]).strip().lower()
        if value in city_values:
            return value

    # 2) pattern city = something
    m = re.search(r"city\s*=\s*([a-zA-Z\s]+)", q_low)
    if m:
        val = m.group(1).strip()
        for kw in [" and", " or", " where", ";"]:
            idx = val.find(kw)
            if idx != -1:
                val = val[:idx]
        val = val.strip().lower()
        if val in city_values:
            return val

    # 3) fallback: natural language full-word match
    matches = []
    padded_q = f" {q_low} "
    for cname in df_features["city"].str.lower().unique():
        if f" {cname} " in padded_q:
            matches.append(cname)

    if len(matches) == 1:
        return matches[0]

    return None


# -------------------------------------------------
# ML — CITY INSIGHTS
# -------------------------------------------------
def show_city_insights(city_name: str):
    row = df_features[df_features["city"].str.lower() == str(city_name).lower()]

    if row.empty:
        st.info("ML insights unavailable — this city is not in the ML feature table.")
        return

    r = row.iloc[0]

    city = r["city"]
    state = r["state"]
    lifestyle_score = float(r.get("lifestyle_score", 0.0))
    lifestyle_rank = int(r.get("lifestyle_rank", 0))
    cluster_label = r.get("cluster_label", None)

    opp = float(r.get("opportunity_index", 0.0))
    youth = float(r.get("youth_index", 0.0))
    family = float(r.get("family_index", 0.0))

    cluster_display = (
        str(int(cluster_label)) if cluster_label is not None else "N/A"
    )

    card_html = f"""
    <div class="lifestyle-card">
      <div class="insight-label">ML-Powered Lifestyle Score</div>
      <div class="insight-title">{city}, {state}</div>
      <div style="display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap;">
        <div>
          <div style="font-size: 0.65rem; color: rgba(255, 255, 255, 0.8); font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">Lifestyle Score</div>
          <div style="font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-top: 0.25rem;">{lifestyle_score:.3f}</div>
        </div>
        <div>
          <div style="font-size: 0.65rem; color: rgba(255, 255, 255, 0.8); font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">Rank</div>
          <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin-top: 0.25rem;">#{lifestyle_rank}</div>
        </div>
        <div>
          <div style="font-size: 0.65rem; color: rgba(255, 255, 255, 0.8); font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">Cluster</div>
          <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin-top: 0.25rem;">{cluster_display}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    if lifestyle_score > 0:
        st.progress(min(max(lifestyle_score, 0.0), 1.0))

    st.markdown("<div class='section-header'>Lifestyle Profile</div>", unsafe_allow_html=True)
    categories = ["Opportunity", "Youthfulness", "Family-friendliness"]
    values = [opp, youth, family]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Lifestyle profile",
            line_color="#667eea",
            fillcolor="rgba(102, 126, 234, 0.3)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0")
        ),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Score Breakdown</div>", unsafe_allow_html=True)
    st.json(
        {
            "Opportunity Index": opp,
            "Youth Index": youth,
            "Family Index": family,
        }
    )


# -------------------------------------------------
# ML — SIMILAR CITIES
# -------------------------------------------------
def show_similar_cities(city_name: str):
    if "cluster_label" not in df_features.columns:
        st.caption("Similar cities not available — cluster labels missing in ML dataset.")
        return

    st.markdown("<div class='section-header'>Similar Cities in the Same Cluster</div>", unsafe_allow_html=True)

    row = df_features[df_features["city"].str.lower() == str(city_name).lower()]
    if row.empty:
        st.info("No ML cluster found for this city.")
        return

    cluster = int(row.iloc[0]["cluster_label"])

    similar = df_features[df_features["cluster_label"] == cluster].copy()
    similar = similar[similar["city"].str.lower() != str(city_name).lower()]

    if similar.empty:
        st.info("No other cities in this cluster.")
        return

    st.markdown(
        f"These cities share a similar demographic profile as **{city_name}** (Cluster {cluster}):"
    )

    top_similar = similar.sort_values("lifestyle_score", ascending=False).head(10)

    display_df = top_similar[
        ["city", "state", "lifestyle_score"]
    ].rename(
        columns={
            "city": "City",
            "state": "State",
            "lifestyle_score": "Lifestyle Score",
        }
    )
    st.dataframe(display_df, use_container_width=True, height=300)

    fig = px.bar(
        display_df,
        x="City",
        y="Lifestyle Score",
        title="Similar Cities — Lifestyle Score",
        color_discrete_sequence=["#667eea"]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# ML — CLUSTER SCATTER PLOT
# -------------------------------------------------
def plot_cluster_scatter(df_clusters: pd.DataFrame):
    required = [
        "ml_vector_population",
        "ml_vector_age",
        "ml_vector_household",
    ]
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


# -------------------------------------------------
# MLOps Helper — Load Registry
# -------------------------------------------------
def load_registry_dashboard():
    try:
        with open("mlops/registry.json", "r") as f:
            return json.load(f)
    except Exception:
        return None


# -------------------------------------------------
# SIDEBAR MODE SWITCH
# -------------------------------------------------
mode = st.sidebar.radio(
    "Choose view:",
    ["Search", "MLOps Dashboard"],
    index=0,
)

# Quick examples only for Search mode
if mode == "Search":
    st.sidebar.title("Quick Examples")
    examples = [
        "cities with population > 1000000",
        "how many cities are in Texas",
        "top 10 largest cities in Florida",
        "Which is best city for families",
        "best cities for young professionals",
        "best retirement city",
        "Life in Dallas",
        "City profile for Denver",
        "Cities similar to Chicago",
        "which cluster is Miami in",
    ]
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""
    if "auto_search" not in st.session_state:
        st.session_state["auto_search"] = False

    for ex in examples:
        if st.sidebar.button(ex, use_container_width=True, key=f"btn_{ex}"):
            st.session_state["current_query"] = ex
            st.session_state["auto_search"] = True
            st.rerun()

    st.sidebar.markdown("---")


# -------------------------------------------------
# HERO + SEARCH BOX (for Search mode)
# -------------------------------------------------
if mode == "Search":
    st.markdown(
        """
        <div class='hero-section'>
            <div class='hero-title'>CitySearch AI</div>
            <div class='hero-subtitle'>
                Ask anything about US cities. CitySearch AI finds the right data and gives clear insights instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([4, 1])
    with left:
        user_query = st.text_input(
            "",
            value=st.session_state.get("current_query", ""),
            placeholder="Example: Tell me about Miami",
            label_visibility="collapsed"
        )
    with right:
        search_clicked = st.button("Search", use_container_width=True)

    enable_summary = st.checkbox("Enable AI Summary", value=False)

    # Auto-search trigger
    if st.session_state.get("auto_search", False):
        search_clicked = True
        st.session_state["auto_search"] = False

else:
    # MLOps hero header
    st.markdown(
        """
        <div class='hero-section'>
            <div class='hero-title'>MLOps Dashboard</div>
            <div class='hero-subtitle'>
                Monitor model health, drift, and retraining for the city clustering engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# MLOps DASHBOARD
# -------------------------------------------------
if mode == "MLOps Dashboard":
    registry = load_registry_dashboard()

    if not registry:
        st.error("No registry.json found in mlops/. Please create it first.")
    else:
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Name", registry.get("model_name", "N/A"))
        with col2:
            st.metric("Version", registry.get("version", "N/A"))
        with col3:
            st.metric("Silhouette Score", f"{registry.get('silhouette_score', 0):.3f}")
        with col4:
            st.metric("Num Cities", registry.get("num_cities", 0))

        st.markdown("<div class='section-header'>Training Information</div>", unsafe_allow_html=True)
        st.write(
            {
                "Trained On": registry.get("trained_on", "N/A"),
                "Model Path": registry.get("model_path", "N/A"),
                "Notes": registry.get("notes", ""),
            }
        )

        st.markdown("---")

        # Buttons row
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            monitor_clicked = st.button("Run Monitoring", use_container_width=True)
        with btn_col2:
            retrain_clicked = st.button("Run Retrain", use_container_width=True)

        # Results containers
        if monitor_clicked:
            with st.spinner("Running monitoring..."):
                result = run_monitoring()
            st.markdown("<div class='section-header'>Drift Report</div>", unsafe_allow_html=True)
            st.json(result["drift_report"])
            st.info(f"Retraining Required? {'YES' if result['retrain_needed'] else 'NO'}")
            st.success(f"New silhouette (temp): {result['new_silhouette']:.3f}")

        if retrain_clicked:
            with st.spinner("Retraining model..."):
                result = retrain()
            if result["status"] == "accepted":
                st.success(
                    f"New model accepted. Version: {result['new_version']}, "
                    f"Silhouette: {result['new_silhouette']:.3f}"
                )
            else:
                st.warning(
                    f"New model rejected. Old silhouette: {result['old_silhouette']:.3f}, "
                    f"New: {result['new_silhouette']:.3f}"
                )

            # Reload registry after retrain
            registry = load_registry_dashboard()
            st.markdown("<div class='section-header'>Updated Registry</div>", unsafe_allow_html=True)
            st.json(registry)

    st.stop()


# -------------------------------------------------
# MAIN LOGIC — SEARCH MODE
# -------------------------------------------------
if mode == "Search":
    if search_clicked and user_query.strip():
        q = user_query.strip()
        
        # Safety checks
        if is_nonsense_query(q):
            respond_nonsense()

        if is_world_query(q):
            respond_world_not_supported()

        # -------------------------------------------------
        # SIMPLE TWO-CITY COMPARISON (OPTION A)
        # -------------------------------------------------
        import re

        def extract_two_cities(q):
            """Extract exactly two city names from the query."""
            q_low = q.lower()
            city_values = df_features["city"].str.lower().unique()

            found = []
            for cname in city_values:
                if cname in q_low:
                    found.append(cname)

            found = list(set(found))
            if len(found) == 2:
                return found

            return None


        two_cities = extract_two_cities(q)

        if two_cities:
            city1, city2 = two_cities

            # Get from ML feature dataset (NOT SQL)
            df1 = df_features[df_features["city"].str.lower() == city1].iloc[0]
            df2 = df_features[df_features["city"].str.lower() == city2].iloc[0]

            st.markdown("<div class='section-header'>City Comparison</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            # ---- CITY 1 ----
            with c1:
                st.subheader(df1["city"])
                st.metric("Population", f"{df1['population']:,}")
                st.metric("Median Age", f"{df1['median_age']:.1f}")
                st.metric("Cluster", f"{df1.get('cluster_label', 'N/A')}")


            # ---- CITY 2 ----
            with c2:
                st.subheader(df2["city"])
                st.metric("Population", f"{df2['population']:,}")
                st.metric("Median Age", f"{df2['median_age']:.1f}")
                st.metric("Cluster", f"{df2.get('cluster_label', 'N/A')}")

            # ----------------------------------------------------
            # ADD AI SUMMARY FOR COMPARISON
            # ----------------------------------------------------
            if enable_summary:
                st.markdown("<div class='section-header'>AI Insight</div>", unsafe_allow_html=True)

                ai_prompt_df = pd.DataFrame([
                    {
                        "city": df1["city"],
                        "population": df1["population"],
                        "median_age": df1["median_age"],
                        "cluster": df1.get("cluster_label", "N/A"),
                    },
                    {
                        "city": df2["city"],
                        "population": df2["population"],
                        "median_age": df2["median_age"],
                        "cluster": df2.get("cluster_label", "N/A"),
                    }
                ])

                st.info(summarize_results(ai_prompt_df, q))

            st.stop()




        # -------------------------------------------------
        # NON-CITY QUERY GUARD (prevents nonsense questions)
        # -------------------------------------------------
        q_low = q.lower()

        # If user asks anything NOT related to cities, states, clusters, lifestyle, etc.
        allowed_keywords = [
            "city", "cities", "state", "population", "cluster", "compare",
            "texas", "florida", "california", "best", "similar", "lifestyle",
            "profile", "retirement", "family", "professionals"
        ]

        # --- NEW FIX ---
        # First check if lifestyle RAG can answer
        _life_test = try_build_lifestyle_card(q)
        
        # --- NEW FIX ---
        # First check if lifestyle RAG can answer
        _life_test = try_build_lifestyle_card(q)

        if _life_test is None:
            # Only run the out-of-scope guard when lifestyle RAG can't handle it
            if not any(word in q_low for word in allowed_keywords):
                st.markdown(
                    """
                    <div class='insight-card'>
                        <div class='insight-label'>Out of Scope</div>
                        <div class='insight-title'>I'm here to help with US cities</div>
                        <div class='insight-text'>
                            Your question doesn't match city-related topics.<br><br>
                            You can ask about:<br>
                            • population<br>
                            • lifestyle or city profiles<br>
                            • clusters or similar cities<br>
                            • best cities for families, professionals, retirement<br><br>
                            Please ask something related to <b>US cities</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.stop()
        else:
            # Lifestyle RAG recognized the query — render the card
            lifestyle_card = _life_test
            city = lifestyle_card["city"]
            state = lifestyle_card["state"]
            pop = lifestyle_card["population"]
            median_age = lifestyle_card["median_age"]
            hh_size = lifestyle_card["avg_household_size"]
            desc = lifestyle_card["description"]
            insight = lifestyle_card["ai_summary"]

            card_html = f"""
            <div class="insight-card">
              <div class="insight-label">Lifestyle Profile Generated with RAG</div>
              <div class="insight-title">Life in {city}, {state}</div>
              <div class="insight-text">{insight}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if pop is not None:
                    st.metric("Population", f"{pop:,}")
            with col2:
                if median_age is not None:
                    st.metric("Median age", f"{median_age:.1f} years")
            with col3:
                if hh_size is not None:
                    st.metric("Avg household size", f"{hh_size:.2f}")

            if desc:
                st.markdown("<div class='section-header'>Dataset Description</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='data-container'>{desc}</div>",
                    unsafe_allow_html=True,
                )

            st.stop()



        # Initial log entry
        log_user_query(
            query_text=q,
            detected_mode="received",
            ml_mode="pending"
        )


        # ---------------------------------------------
        # STEP 0: Lifestyle RAG
        # ---------------------------------------------
        # Trigger the new lifestyle card builder here
        lifestyle_card = try_build_lifestyle_card(q)
        if lifestyle_card is not None:
            city = lifestyle_card["city"]
            state = lifestyle_card["state"]
            pop = lifestyle_card["population"]
            median_age = lifestyle_card["median_age"]
            hh_size = lifestyle_card["avg_household_size"]
            desc = lifestyle_card["description"]
            insight = lifestyle_card["ai_summary"]

            # Render the lifestyle profile in your UI
            card_html = f"""
            <div class="insight-card">
              <div class="insight-label">Lifestyle Profile Generated with RAG</div>
              <div class="insight-title">Life in {city}, {state}</div>
              <div class="insight-text">{insight}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if pop is not None:
                    st.metric("Population", f"{pop:,}")
            with col2:
                if median_age is not None:
                    st.metric("Median age", f"{median_age:.1f} years")
            with col3:
                if hh_size is not None:
                    st.metric("Avg household size", f"{hh_size:.2f}")

            if desc:
                st.markdown("<div class='section-header'>Dataset Description</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='data-container'>{desc}</div>",
                    unsafe_allow_html=True,
                )

            st.stop()

        # ---------------------------------------------
        # INTENT CLASSIFICATION + ML ROUTING
        # ---------------------------------------------
        mode_intent, state_filter = classify_query_intent(q)
        ml_mode, ml_target = smart_route(q)
        
        # Log query
        log_user_query(
            query_text=q,
            detected_mode=mode_intent,
            ml_mode=ml_mode
            )

        # -------------------------------------------------
        # ML RANKING MODES (supports list + single city mode)
        # -------------------------------------------------
        if ml_mode in [
            "ml_family_list", "ml_family_single",
            "ml_young_list", "ml_young_single",
            "ml_retirement_list", "ml_retirement_single",
        ]:

            df = ml_target

            # Safety
            if df is None or df.empty:
                st.warning("No city results found.")
                st.stop()

            # Detect if single city mode
            is_single = ml_mode.endswith("_single")

            # Header selection
            if ml_mode.startswith("ml_family"):
                title = "Best City for Families" if is_single else "Best Cities for Families"
            elif ml_mode.startswith("ml_young"):
                title = "Best City for Young Professionals" if is_single else "Best Cities for Young Professionals"
            else:
                title = "Best City for Retirement" if is_single else "Best Cities for Retirement"

            st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

            # Always pick top row as card
            top_city = df.iloc[0]

            # ==== YOUR ORIGINAL CARD – UNCHANGED ====
            card_html = f"""
            <div class="result-card">
              <div class="result-card-title">Top City</div>
              <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div style="text-align: center;">
                  <div style="font-size: 0.75rem; opacity: 0.9; margin-bottom: 0.25rem;">CITY</div>
                  <div class="result-card-value">{top_city["city"]}</div>
                </div>
                <div style="text-align: center;">
                  <div style="font-size: 0.75rem; opacity: 0.9; margin-bottom: 0.25rem;">STATE</div>
                  <div class="result-card-value">{top_city["state"]}</div>
                </div>
                <div style="text-align: center;">
                  <div style="font-size: 0.75rem; opacity: 0.9; margin-bottom: 0.25rem;">SCORE</div>
                  <div class="result-card-value">{top_city.get('score', 0):.3f}</div>
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # AI Summary before table
            if enable_summary:
                st.markdown("<div class='section-header'>AI Insight</div>", unsafe_allow_html=True)
                st.info(explain_ml_results(df, df))

            # ===== TABLE TITLE =====
            table_title = "Selected City" if is_single else "All Results"
            st.markdown(f"<div class='section-header'>{table_title}</div>", unsafe_allow_html=True)

            # ==== TABLE (1 row for single, all 10 rows for list mode) ====
            display_df = df.head(1) if is_single else df

            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.dataframe(display_df, use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV Download for exactly what is shown
            csv = convert_df_to_csv(display_df)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="city_results.csv",
                mime="text/csv",
                use_container_width=False
            )

            st.stop()


        # -------------------------------------------------
        # ML SINGLE CITY PROFILE
        # -------------------------------------------------
        if ml_mode == "ml_single_city":
            city_obj = ml_target

            st.markdown(
                f"<div class='section-header'>AI Lifestyle Profile — {city_obj['city']}, {city_obj['state']}</div>",
                unsafe_allow_html=True
            )

            st.write(
                {
                    "Population": city_obj["population"],
                    "Median Age": city_obj["median_age"],
                    "Avg Household Size": city_obj["avg_household_size"],
                    "Family Friendliness": to_level(city_obj["family_score"]),
                    "Young Professional Appeal": to_level(city_obj["young_score"]),
                    "Retirement Appeal": to_level(city_obj["retirement_score"]),
                }
            )

            st.markdown("<div class='section-header'>AI Insight</div>", unsafe_allow_html=True)
            df_single = pd.DataFrame([city_obj])
            st.info(explain_ml_results(df_single, df_single))

            st.stop()

        # -------------------------------------------------
        # CLUSTER MODES
        # -------------------------------------------------
        if ml_mode == "cluster_single":
            city_info = ml_target
            if city_info is None:
                st.error("City not found in dataset.")
                st.stop()

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

            st.markdown("<div class='section-header'>City Snapshot</div>", unsafe_allow_html=True)
            st.json(
                {
                    "City": city_info["city"],
                    "State": city_info["state"],
                    "Cluster ID": cid,
                    "Cluster Name": profile["cluster_name"],
                    "Cluster Summary": profile["cluster_summary"],
                }
            )

            st.markdown("<div class='section-header'>Lifestyle Summary</div>", unsafe_allow_html=True)
            st.markdown(profile["detailed_summary"])

            st.markdown("<div class='section-header'>How to Interpret This Cluster</div>", unsafe_allow_html=True)
            st.markdown(profile["how_to_read"])

            st.stop()

        if ml_mode == "cluster_all":
            df_clusters = ml_target
            st.markdown("<div class='section-header'>City Clusters — All Cities</div>", unsafe_allow_html=True)

            # AI Summary before table
            if enable_summary:
                st.markdown("<div class='section-header'>AI Summary</div>", unsafe_allow_html=True)
                st.info(summarize_results(df_clusters, q))

            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.dataframe(df_clusters, use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV Download
            csv = convert_df_to_csv(df_clusters)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="clusters.csv",
                mime="text/csv",
                use_container_width=False
            )

            plot_cluster_scatter(df_clusters)

            st.stop()

        if ml_mode == "cluster_state":
            df_clusters = ml_target
            if df_clusters is None or df_clusters.empty:
                st.warning("No cities found.")
                st.stop()

            st.markdown(
                f"<div class='section-header'>City Clusters in {df_clusters['state'].iloc[0]}</div>",
                unsafe_allow_html=True
            )

            # AI Summary before table
            if enable_summary:
                st.markdown("<div class='section-header'>AI Summary</div>", unsafe_allow_html=True)
                st.info(summarize_results(df_clusters, q))

            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.dataframe(df_clusters, use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV Download
            csv = convert_df_to_csv(df_clusters)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="state_clusters.csv",
                mime="text/csv",
                use_container_width=False
            )

            plot_cluster_scatter(df_clusters)

            st.stop()

        if ml_mode == "cluster_similar":
            df_clusters = ml_target
            if df_clusters is None or df_clusters.empty:
                st.warning("No similar cities found.")
                st.stop()

            cid = int(df_clusters.iloc[0]["cluster_id"])

            st.markdown("<div class='section-header'>Cities in the Same Cluster</div>", unsafe_allow_html=True)
            
            # AI Summary before table
            if enable_summary:
                st.markdown("<div class='section-header'>AI Summary</div>", unsafe_allow_html=True)
                st.info(summarize_results(df_clusters, q))

            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.dataframe(df_clusters, use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV Download
            csv = convert_df_to_csv(df_clusters)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="similar_cities.csv",
                mime="text/csv",
                use_container_width=False
            )

            profile = explain_cluster(return_dict=True, cluster_id=cid)
            st.markdown("<div class='section-header'>Cluster Profile</div>", unsafe_allow_html=True)
            st.markdown(profile["detailed_summary"])

            st.stop()

        # -------------------------------------------------
        # DEFAULT SQL MODE (WITH CITY OVERRIDE + ML)
        # -------------------------------------------------
        if mode_intent == "sql":
            with st.spinner("Running SQL..."):
                sql = build_sql_with_fallback(q, use_gpt=True)

                detected_city = detect_city_in_query(q)
                if detected_city:
                    sql = f"SELECT * FROM dbo.cities WHERE LOWER(city) = '{detected_city}'"

                # Run SQL quietly
                df = run_sql_query(sql)

                # If single row → card mode
                if len(df) == 1 and len(df.columns) <= 3:
                    row = df.iloc[0]
                    card_html = "<div class='result-card'><div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 1rem;'>"

                    
                    for col in df.columns:
                        label = col.replace("_", " ").title()
                        value = row[col]

                        if isinstance(value, (int, float)):
                            display_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                        else:
                            display_value = str(value)

                        card_html += f"""
                            <div style="text-align: center;">
                                <div style="font-size: 0.75rem; opacity: 0.9; margin-bottom: 0.25rem;">{label}</div>
                                <div class="result-card-value">{display_value}</div>
                            </div>
                        """

                    card_html += "</div></div>"
                    st.markdown(card_html, unsafe_allow_html=True)

                else:
                    # AI Summary
                    if enable_summary:
                        st.markdown("<div class='section-header'>AI Summary</div>", unsafe_allow_html=True)
                        st.info(summarize_results(df, q))

                    st.markdown("<div class='section-header'>Results</div>", unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True, height=400)

                    csv = convert_df_to_csv(df)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )

                # ML city insights
                if len(df) == 1 and "city" in [c.lower() for c in df.columns]:
                    city_col = [c for c in df.columns if c.lower() == "city"][0]
                    city_name = df.iloc[0][city_col]

                    if str(city_name).lower() in df_features["city"].str.lower().values:
                        show_city_insights(city_name)
                        show_similar_cities(city_name)

            st.stop()

        # -------------------------------------------------
        # SEMANTIC & HYBRID MODES
        # -------------------------------------------------
        if mode_intent == "semantic":
            with st.spinner("Semantic search..."):
                results = semantic_city_search(q, top_k=10)

            df = pd.DataFrame(results, columns=["City", "State"])

            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV Download
            csv = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="semantic_results.csv",
                mime="text/csv",
                use_container_width=False
            )

            if len(df) == 1:
                show_city_insights(df.iloc[0]["City"])
                show_similar_cities(df.iloc[0]["City"])

            if enable_summary:
                st.info(summarize_results(df, q))

            st.stop()

        if mode_intent == "hybrid":
            with st.spinner("Hybrid semantic search..."):
                results = semantic_city_search(q, top_k=10, state_filter=state_filter)

            df = pd.DataFrame(results, columns=["City", "State"])

            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV Download
            csv = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="hybrid_results.csv",
                mime="text/csv",
                use_container_width=False
            )

            if len(df) == 1:
                show_city_insights(df.iloc[0]["City"])
                show_similar_cities(df.iloc[0]["City"])

            if enable_summary:
                st.info(summarize_results(df, q))

            st.stop()

    else:
        st.caption("Try an example from the sidebar to get started.")
        
