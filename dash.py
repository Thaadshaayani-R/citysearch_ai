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
from core.openai_client import get_openai_client
from core.data_loader import US_STATES, is_valid_state, normalize_state_name

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


# -------------------------------------------------
# MARKDOWN TO HTML CONVERTER
# -------------------------------------------------
def convert_markdown_to_html(text: str) -> str:
    """Convert common Markdown formatting to HTML."""
    import re
    
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert *italic* to <em>italic</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert bullet points (lines starting with - or •)
    lines = text.split('\n')
    converted_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('• '):
            if not in_list:
                converted_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            item_text = stripped[2:].strip()
            converted_lines.append(f'<li style="margin: 0.25rem 0;">{item_text}</li>')
        elif stripped.startswith('* ') and not stripped.startswith('**'):
            if not in_list:
                converted_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            item_text = stripped[2:].strip()
            converted_lines.append(f'<li style="margin: 0.25rem 0;">{item_text}</li>')
        else:
            if in_list:
                converted_lines.append('</ul>')
                in_list = False
            if stripped:
                converted_lines.append(f'<p style="margin: 0.5rem 0;">{stripped}</p>')
    
    if in_list:
        converted_lines.append('</ul>')
    
    return '\n'.join(converted_lines)

# -------------------------------------------------
# GLOBAL FUZZY MATCHING FOR STATES AND CITIES
# -------------------------------------------------
from difflib import get_close_matches

US_STATES_FULL = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming"
]


def fuzzy_match_state(input_state: str, cutoff: float = 0.6) -> str:
    """Find the closest matching state name."""
    if not input_state:
        return None
    
    input_lower = input_state.lower().strip()
    
    # Exact match first
    for state in US_STATES_FULL:
        if state.lower() == input_lower:
            return state
    
    # Fuzzy match
    state_names_lower = [s.lower() for s in US_STATES_FULL]
    matches = get_close_matches(input_lower, state_names_lower, n=1, cutoff=cutoff)
    
    if matches:
        matched_index = state_names_lower.index(matches[0])
        return US_STATES_FULL[matched_index]
    
    return None


def fuzzy_match_city(input_city: str, cutoff: float = 0.6) -> str:
    """Find the closest matching city name from df_features."""
    if not input_city:
        return None
    
    input_lower = input_city.lower().strip()
    
    city_names = df_features["city"].unique().tolist()
    city_names_lower = [c.lower() for c in city_names]
    
    # Exact match first
    if input_lower in city_names_lower:
        matched_index = city_names_lower.index(input_lower)
        return city_names[matched_index]
    
    # Fuzzy match
    matches = get_close_matches(input_lower, city_names_lower, n=1, cutoff=cutoff)
    
    if matches:
        matched_index = city_names_lower.index(matches[0])
        return city_names[matched_index]
    
    return None


def correct_query_spelling(query: str) -> tuple:
    """
    Correct spelling mistakes in the query for states and cities.
    Returns: (corrected_query, list_of_corrections)
    
    IMPORTANT: Only corrects words that look like misspelled city/state names,
    NOT common English words.
    """
    
    # Words that should NEVER be auto-corrected (common English words)
    PROTECTED_WORDS = {
        # Query keywords
        "best", "worst", "good", "bad", "top", "bottom", "most", "least",
        "compare", "comparison", "versus", "between", "which", "what", "where",
        "city", "cities", "state", "states", "country", "countries",
        "population", "median", "average", "household", "size", "age",
        
        # Demographics
        "young", "old", "older", "younger", "family", "families", "senior", "seniors",
        "professional", "professionals", "student", "students", "retired", "retiree",
        "retirement", "children", "kids", "adults", "people", "person",
        
        # Descriptors
        "large", "small", "big", "little", "high", "low", "cheap", "expensive",
        "affordable", "safe", "dangerous", "friendly", "beautiful", "nice",
        "growing", "declining", "urban", "rural", "suburban", "coastal",
        
        # Weather/Climate
        "weather", "climate", "warm", "cold", "hot", "cool", "sunny", "rainy",
        "humid", "dry", "snow", "beach", "mountain", "desert",
        
        # Economy/Jobs
        "jobs", "job", "work", "employment", "economy", "economic", "business",
        "industry", "tech", "technology", "healthcare", "education",
        
        # Lifestyle
        "life", "living", "lifestyle", "live", "move", "moving", "relocate",
        "similar", "like", "cluster", "group", "category", "type",
        
        # Actions
        "show", "find", "search", "list", "give", "tell", "about", "information",
        "score", "rank", "ranking", "rated", "rating",
        
        # Common words
        "the", "and", "for", "with", "from", "into", "over", "under",
        "how", "many", "much", "more", "less", "than", "that", "this",
        "are", "is", "was", "were", "been", "being", "have", "has", "had",
        "all", "any", "some", "every", "each", "both", "other", "another",
    }
    
    corrections = []
    corrected_query = query
    words = query.split()
    
    city_names = df_features["city"].unique().tolist()
    city_names_lower = [c.lower() for c in city_names]
    state_names_lower = [s.lower() for s in US_STATES_FULL]
    
    i = 0
    while i < len(words):
        word = words[i]
        word_clean = word.lower().strip(".,?!;:'\"")
        
        # Skip short words
        if len(word_clean) <= 3:
            i += 1
            continue
        
        # Skip protected words
        if word_clean in PROTECTED_WORDS:
            i += 1
            continue
        
        # Skip if it's already a valid city or state (exact match)
        if word_clean in city_names_lower or word_clean in state_names_lower:
            i += 1
            continue
        
        # Check two-word combinations first (New York, North Carolina, etc.)
        if i < len(words) - 1:
            next_word = words[i+1].lower().strip(".,?!;:'\"")
            two_word = f"{word_clean} {next_word}"
            
            # Skip if either word is protected
            if word_clean in PROTECTED_WORDS or next_word in PROTECTED_WORDS:
                pass  # Don't try two-word match
            else:
                matched_state = fuzzy_match_state(two_word, cutoff=0.85)  # Higher cutoff
                if matched_state and matched_state.lower() != two_word:
                    original = f"{word} {words[i+1]}"
                    corrections.append((original.strip(".,?!;:'\""), matched_state))
                    corrected_query = corrected_query.replace(original.strip(".,?!;:'\""), matched_state, 1)
                    i += 2
                    continue
        
        # Only try fuzzy matching if the word LOOKS like a city/state name
        # (starts with capital letter in original, or is reasonably long)
        should_try_fuzzy = (
            word[0].isupper() or  # Capitalized
            len(word_clean) >= 6   # Long enough to be a place name
        )
        
        if not should_try_fuzzy:
            i += 1
            continue
        
        # Try single word - state match (high cutoff)
        matched_state = fuzzy_match_state(word_clean, cutoff=0.8)
        if matched_state and matched_state.lower() != word_clean:
            corrections.append((word_clean, matched_state))
            corrected_query = corrected_query.replace(word, matched_state, 1)
            i += 1
            continue
        
        # Try single word - city match (high cutoff)
        matched_city = fuzzy_match_city(word_clean, cutoff=0.8)
        if matched_city and matched_city.lower() != word_clean:
            corrections.append((word_clean, matched_city))
            corrected_query = corrected_query.replace(word, matched_city, 1)
            i += 1
            continue
        
        i += 1
    
    return corrected_query, corrections



def extract_two_states_fuzzy(q: str):
    """Extract two state names from query with fuzzy matching."""
    q_lower = q.lower()
    found = []
    
    # First pass: exact matches
    for state in US_STATES_FULL:
        if state.lower() in q_lower:
            found.append(state)
    
    if len(set(found)) == 2:
        return list(set(found))[:2]
    
    # Second pass: fuzzy matching
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    # Try two-word combinations first
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        matched = fuzzy_match_state(two_word, cutoff=0.7)
        if matched and matched not in found:
            found.append(matched)
    
    # Then single words
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_state(word, cutoff=0.7)
            if matched and matched not in found:
                found.append(matched)
    
    found = list(set(found))
    if len(found) >= 2:
        return found[:2]
    
    return None


def extract_two_cities_fuzzy(q: str):
    """Extract two city names from query with fuzzy matching."""
    q_lower = q.lower()
    city_names = df_features["city"].unique().tolist()
    city_names_lower = [c.lower() for c in city_names]
    
    found_original = []
    
    # First pass: exact matches
    for i, city_lower in enumerate(city_names_lower):
        if city_lower in q_lower and city_names[i] not in found_original:
            found_original.append(city_names[i])
    
    if len(found_original) == 2:
        return found_original
    
    # Second pass: fuzzy matching
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_city(word, cutoff=0.7)
            if matched and matched not in found_original:
                found_original.append(matched)
    
    if len(found_original) >= 2:
        return found_original[:2]
    
    return None


def is_state_comparison_query_fuzzy(q: str) -> bool:
    """Check if query is comparing states with fuzzy matching."""
    q_lower = q.lower()
    comparison_words = ["best", "better", "compare", "vs", "versus", "or", "comparison", "between"]
    has_comparison = any(word in q_lower for word in comparison_words)
    
    states_found = extract_two_states_fuzzy(q)
    return has_comparison and states_found is not None


def extract_single_city_fuzzy(q: str):
    """Extract a single city name from query with fuzzy matching."""
    q_lower = q.lower()
    
    # Common query words to skip (NOT hardcoded state names)
    QUERY_SKIP_WORDS = {
        "cities", "city", "population", "median", "average", "household", 
        "size", "age", "total", "count", "how", "many", "what", "which",
        "best", "worst", "top", "bottom", "largest", "smallest", "biggest",
        "highest", "lowest", "greater", "less", "more", "than", "over", 
        "under", "above", "below", "with", "from", "the", "and", "for", 
        "are", "this", "that", "number", "percent", "percentage"
    }
    
    # Get state names dynamically from existing list
    state_names_lower = {s.lower() for s in US_STATES_FULL}
    
    # Combine skip words
    all_skip_words = QUERY_SKIP_WORDS | state_names_lower
    
    city_names = df_features["city"].unique().tolist()
    city_names_lower = [c.lower() for c in city_names]
    
    # Exact match first - only if the city name appears as a complete word
    for i, city_lower in enumerate(city_names_lower):
        # Skip if city name is also a state name (e.g., "New York")
        if city_lower in state_names_lower:
            continue
            
        # Check for exact word match using word boundaries
        pattern = r'\b' + re.escape(city_lower) + r'\b'
        if re.search(pattern, q_lower):
            return city_names[i]
    
    # Fuzzy match - but be very careful
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    for word in words:
        # Skip short words
        if len(word) <= 3:
            continue
            
        # Skip if this word is in our skip list
        if word in all_skip_words:
            continue
        
        # Only do fuzzy matching with high cutoff (0.85 = very similar)
        matched = fuzzy_match_city(word, cutoff=0.85)
        if matched and matched.lower() not in all_skip_words:
            return matched
    
    return None


def extract_single_state_fuzzy(q: str):
    """Extract a single state name from query with fuzzy matching."""
    q_lower = q.lower()
    
    # Exact match first
    for state in US_STATES_FULL:
        if state.lower() in q_lower:
            return state
    
    # Fuzzy match
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    # Two-word combinations first
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        matched = fuzzy_match_state(two_word, cutoff=0.7)
        if matched:
            return matched
    
    # Single words
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_state(word, cutoff=0.7)
            if matched:
                return matched
    
    return None



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

def show_city_profile_card(city_name: str, state_name: str, row: pd.Series):
    """
    Display a beautiful city profile card with AI-generated summary.
    """
    
    # Get values safely (handle missing columns)
    population = row.get('population', 'N/A')
    median_age = row.get('median_age', 'N/A')
    household_size = row.get('avg_household_size', 'N/A')
    state_code = row.get('state_code', state_name[:2].upper() if state_name else 'N/A')
    
    # Format population with commas if it's a number
    if isinstance(population, (int, float)):
        population_display = f"{int(population):,}"
    else:
        population_display = str(population)
    
    # 1. City Overview Card (NO TABLE)
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
    
    # 2. AI-Generated Profile Summary
    client = get_openai_client()
    if client:
        with st.spinner("Generating city profile..."):
            prompt = f"""
            Write a 2-3 sentence profile for {city_name}, {state_name}.
            Population: {population_display}
            Median Age: {median_age}
            Avg Household Size: {household_size}
            
            Focus on lifestyle, culture, and what makes this city unique.
            Be concise and engaging. No bullet points.
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
                    line-height: 1.6;
                ">
                    "{summary}"
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning("Could not generate AI summary.")
    
    # 3. Highlights Section (AI-Generated)
    if client:
        with st.spinner("Generating highlights..."):
            prompt = f"""
            Give exactly 3 short highlights about {city_name}, {state_name}.
            Each highlight should be 5-8 words maximum.
            Format: Just the 3 highlights, one per line, no numbers or bullets.
            
            Example format:
            Strong outdoor recreation culture
            Growing tech and business opportunities
            Young and active population
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=100
                )
                highlights = response.choices[0].message.content.strip().split('\n')
                highlights = [h.strip() for h in highlights if h.strip()][:3]
                
                st.markdown("""
                <div style="margin-bottom: 1rem;">
                    <h4 style="color: #667eea; margin-bottom: 0.75rem;">✨ Highlights</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for highlight in highlights:
                    # Remove any leading bullets or numbers
                    highlight = highlight.lstrip('•-*0123456789. ')
                    st.markdown(f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        padding: 0.5rem 0;
                    ">
                        <span style="color: #667eea;">✓</span>
                        <span>{highlight}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                pass


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
        q_original = user_query.strip()
        
        # -------------------------------------------------
        # GLOBAL SPELLING CORRECTION
        # -------------------------------------------------
        q, corrections = correct_query_spelling(q_original)
        
        if corrections:
            correction_text = ", ".join([f'"{old}" → "{new}"' for old, new in corrections])
            st.info(f"🔄 Auto-corrected: {correction_text}")

        
        # Safety checks
        if is_nonsense_query(q):
            respond_nonsense()

        if is_world_query(q):
            respond_world_not_supported()

        # -------------------------------------------------
        # SIMPLE TWO-CITY COMPARISON (OPTION A)
        # -------------------------------------------------
        import re

        def extract_two_cities_fuzzy(q):
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


        # -------------------------------------------------
        # TWO-CITY COMPARISON (e.g., "Denver vs Dallas")
        # -------------------------------------------------
        two_cities = extract_two_cities_fuzzy(q)

        if two_cities:
            city1, city2 = two_cities

            # Get city data from features DataFrame
            df1_row = df_features[df_features["city"].str.lower() == city1.lower()]
            df2_row = df_features[df_features["city"].str.lower() == city2.lower()]

            if df1_row.empty or df2_row.empty:
                st.warning("Could not find data for one or both cities.")
                st.stop()

            df1 = df1_row.iloc[0]
            df2 = df2_row.iloc[0]

            # Header
            st.markdown(
                f"<div class='section-header'>🏆 City Comparison: {df1['city']} vs {df2['city']}</div>",
                unsafe_allow_html=True
            )

            # Side-by-side City Profiles
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-label">City Profile</div>
                        <div class="insight-title">📍 {df1['city']}, {df1['state']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.metric("Population", f"{int(df1['population']):,}")
                st.metric("Median Age", f"{df1['median_age']:.1f} years")
                st.metric("Avg Household Size", f"{df1['avg_household_size']:.2f}")
                if 'cluster_label' in df1 and pd.notna(df1.get('cluster_label')):
                    st.metric("Cluster", f"{int(df1['cluster_label'])}")
                if 'lifestyle_score' in df1 and pd.notna(df1.get('lifestyle_score')):
                    st.metric("Lifestyle Score", f"{df1['lifestyle_score']:.3f}")

            with col2:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-label">City Profile</div>
                        <div class="insight-title">📍 {df2['city']}, {df2['state']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.metric("Population", f"{int(df2['population']):,}")
                st.metric("Median Age", f"{df2['median_age']:.1f} years")
                st.metric("Avg Household Size", f"{df2['avg_household_size']:.2f}")
                if 'cluster_label' in df2 and pd.notna(df2.get('cluster_label')):
                    st.metric("Cluster", f"{int(df2['cluster_label'])}")
                if 'lifestyle_score' in df2 and pd.notna(df2.get('lifestyle_score')):
                    st.metric("Lifestyle Score", f"{df2['lifestyle_score']:.3f}")

            # Visual Comparison Chart
            st.markdown("<div class='section-header'>📊 Visual Comparison</div>", unsafe_allow_html=True)

            # Normalize values for comparison
            max_pop = max(df1['population'], df2['population'])
            
            comparison_data = pd.DataFrame({
                "Metric": ["Population (scaled)", "Median Age", "Household Size (x10)"],
                df1['city']: [
                    (df1['population'] / max_pop) * 100,  # Scale to percentage
                    df1['median_age'],
                    df1['avg_household_size'] * 10  # Scale for visibility
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
                st.markdown("<div class='section-header'>🎯 Lifestyle Profile Comparison</div>", unsafe_allow_html=True)
                
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

            # -------------------------------------------------
            # AI Analysis & Recommendation (with user's specific question)
            # -------------------------------------------------
            st.markdown("<div class='section-header'>🤖 AI Analysis & Recommendation</div>", unsafe_allow_html=True)

            with st.spinner("Generating AI analysis..."):
                
                # Extract any specific question from the user's query
                def extract_specific_question(query: str, city1: str, city2: str) -> str:
                    """Extract any specific question beyond the comparison request."""
                    q_lower = query.lower()
                    
                    # Remove common comparison phrases to find the specific question
                    remove_phrases = [
                        f"compare {city1.lower()} and {city2.lower()}",
                        f"compare {city2.lower()} and {city1.lower()}",
                        f"{city1.lower()} vs {city2.lower()}",
                        f"{city2.lower()} vs {city1.lower()}",
                        f"{city1.lower()} versus {city2.lower()}",
                        f"{city2.lower()} versus {city1.lower()}",
                        f"{city1.lower()} or {city2.lower()}",
                        f"{city2.lower()} or {city1.lower()}",
                        f"which is best {city1.lower()} or {city2.lower()}",
                        f"which is better {city1.lower()} or {city2.lower()}",
                    ]
                    
                    remaining = q_lower
                    for phrase in remove_phrases:
                        remaining = remaining.replace(phrase, "")
                    
                    # Clean up
                    remaining = remaining.strip().strip(".").strip(",").strip()
                    
                    # If something meaningful remains, it's a specific question
                    if len(remaining) > 10:
                        return remaining
                    return None

                specific_question = extract_specific_question(q, df1['city'], df2['city'])
                
                # Build the prompt
                specific_question_section = ""
                if specific_question:
                    specific_question_section = f"""

**USER'S SPECIFIC QUESTION:**
The user specifically asked: "{specific_question}"
Please address this question directly at the beginning of your response.
"""

                city_comparison_prompt = f"""
Compare these two U.S. cities based on the data provided:

**{df1['city']}, {df1['state']}:**
- Population: {int(df1['population']):,}
- Median Age: {df1['median_age']:.1f} years
- Average Household Size: {df1['avg_household_size']:.2f}

**{df2['city']}, {df2['state']}:**
- Population: {int(df2['population']):,}
- Median Age: {df2['median_age']:.1f} years
- Average Household Size: {df2['avg_household_size']:.2f}
{specific_question_section}

TASK:
1. {"First, directly answer the user's specific question about " + specific_question if specific_question else "Write a 2-3 sentence summary comparing both cities"}
2. List 3 key differences as bullet points
3. Provide recommendations:
   - Which city is better for families?
   - Which city is better for young professionals?
   - Which city is better for retirement?
4. End with a balanced conclusion

Keep it concise, friendly, and actionable.
Use your general knowledge about these cities (weather, culture, economy, etc.) to provide a complete answer.
"""
                client = get_openai_client()
                if client:
                    try:
                        rsp = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[{"role": "user", "content": city_comparison_prompt}],
                            temperature=0.4,
                            max_tokens=600,  # Increased for longer response
                        )
                        city_insight = rsp.choices[0].message.content.strip()
                    except Exception as e:
                        city_insight = f"AI analysis unavailable: {str(e)}"
                else:
                    city_insight = "AI analysis unavailable - OpenAI client not configured."

            # Show what question was detected (optional - for debugging)
            if specific_question:
                st.markdown(
                    f"<div style='font-size: 0.85rem; color: #a0aec0; margin-bottom: 0.5rem;'>📝 Detected specific question: <em>{specific_question}</em></div>",
                    unsafe_allow_html=True
                )


            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-label">AI-Powered Comparison</div>
                    <div class="insight-text">{convert_markdown_to_html(city_insight)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )



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

            st.stop()


        # -------------------------------------------------
        # STATE COMPARISON FEATURE
        # -------------------------------------------------
        US_STATES_LIST = [
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
            "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
            "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
            "maine", "maryland", "massachusetts", "michigan", "minnesota",
            "mississippi", "missouri", "montana", "nebraska", "nevada",
            "new hampshire", "new jersey", "new mexico", "new york",
            "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
            "pennsylvania", "rhode island", "south carolina", "south dakota",
            "tennessee", "texas", "utah", "vermont", "virginia", "washington",
            "west virginia", "wisconsin", "wyoming"
        ]
        
        
        def extract_two_states(q: str):
            """Extract exactly two state names from the query."""
            q_low = q.lower()
            found = []
            
            for state in US_STATES_LIST:
                if state in q_low:
                    found.append(state.title())
            
            found = list(set(found))
            if len(found) == 2:
                return found
            return None
        
        
        def is_state_comparison_query(q: str):
            """Check if query is comparing states."""
            q_low = q.lower()
            comparison_words = ["best", "better", "compare", "vs", "versus", "or", "comparison"]
            has_comparison = any(word in q_low for word in comparison_words)
            
            states_found = extract_two_states_fuzzy(q)
            return has_comparison and states_found is not None
        
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def get_state_detailed_stats(state_name: str):
            """Get comprehensive stats for a state."""
            engine = get_engine()
            
            sql = text("""
                SELECT 
                    state,
                    COUNT(*) as city_count,
                    SUM(population) as total_population,
                    AVG(population) as avg_population,
                    MIN(population) as min_population,
                    MAX(population) as max_population,
                    AVG(median_age) as avg_median_age,
                    MIN(median_age) as min_median_age,
                    MAX(median_age) as max_median_age,
                    AVG(avg_household_size) as avg_household_size
                FROM dbo.cities
                WHERE LOWER(state) = LOWER(:state)
                GROUP BY state
            """)
            
            with engine.connect() as conn:
                result = conn.execute(sql, {"state": state_name})
                rows = result.fetchall()
                cols = result.keys()
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows, columns=cols)
            return df.iloc[0].to_dict()
        
        
        @st.cache_data(ttl=3600, show_spinner=False)
        def get_top_cities_in_state(state_name: str, limit: int = 5):
            """Get top cities by population in a state."""
            engine = get_engine()
            
            sql = text("""
                SELECT TOP(:limit)
                    city,
                    state,
                    population,
                    median_age,
                    avg_household_size
                FROM dbo.cities
                WHERE LOWER(state) = LOWER(:state)
                ORDER BY population DESC
            """)
            
            with engine.connect() as conn:
                result = conn.execute(sql, {"state": state_name, "limit": limit})
                rows = result.fetchall()
                cols = result.keys()
            
            return pd.DataFrame(rows, columns=cols)
        
        
        def generate_state_comparison_insight(state1: str, stats1: dict, cities1: pd.DataFrame,
                                               state2: str, stats2: dict, cities2: pd.DataFrame):
            """Generate AI-powered comparison insight between two states."""
            client = get_openai_client()
            if client is None:
                return "AI comparison unavailable."
            
            prompt = f"""
        Compare these two U.S. states based on the data provided:
        
        **{state1}:**
        - Number of cities: {int(stats1['city_count'])}
        - Total population: {int(stats1['total_population']):,}
        - Average city population: {int(stats1['avg_population']):,}
        - Average median age: {stats1['avg_median_age']:.1f} years
        - Average household size: {stats1['avg_household_size']:.2f}
        - Top cities: {', '.join(cities1['city'].head(5).tolist())}
        
        **{state2}:**
        - Number of cities: {int(stats2['city_count'])}
        - Total population: {int(stats2['total_population']):,}
        - Average city population: {int(stats2['avg_population']):,}
        - Average median age: {stats2['avg_median_age']:.1f} years
        - Average household size: {stats2['avg_household_size']:.2f}
        - Top cities: {', '.join(cities2['city'].head(5).tolist())}
        
        TASK:
        1. Write a 2-3 sentence summary comparing both states
        2. List 3 key differences as bullet points
        3. Provide a recommendation:
           - Which state is better for families?
           - Which state is better for young professionals?
           - Which state is better for retirement?
        4. End with a balanced conclusion
        
        Keep it concise, friendly, and actionable.
        """
        
            try:
                rsp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=500,
                )
                return rsp.choices[0].message.content.strip()
            except Exception as e:
                return f"AI comparison unavailable: {str(e)}"

            
            if not rows:
                return None
            
            df = pd.DataFrame(rows, columns=cols)
            return df.iloc[0].to_dict()

        # -------------------------------------------------
        # STATE COMPARISON (e.g., "which is best florida or texas")
        # -------------------------------------------------
        if is_state_comparison_query_fuzzy(q):
            two_states = extract_two_states_fuzzy(q)
            
            if two_states:
                state1, state2 = two_states
                
                with st.spinner(f"Comparing {state1} and {state2}..."):
                    stats1 = get_state_detailed_stats(state1)
                    stats2 = get_state_detailed_stats(state2)
                    cities1 = get_top_cities_in_state(state1, limit=5)
                    cities2 = get_top_cities_in_state(state2, limit=5)

                if stats1 and stats2:
                    # Header
                    st.markdown(
                        f"<div class='section-header'>🏆 State Comparison: {state1} vs {state2}</div>",
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
                                <div class="insight-title">📍 {state2}</div>
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
                    st.markdown("<div class='section-header'>🏙️ Top Cities Comparison</div>", unsafe_allow_html=True)
                    
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
                    st.markdown("<div class='section-header'>📊 Visual Comparison</div>", unsafe_allow_html=True)
                    
                    comparison_data = pd.DataFrame({
                        "Metric": ["Total Population (M)", "Avg City Population (K)", "Avg Median Age", "Avg Household Size"],
                        state1: [
                            stats1['total_population'] / 1_000_000,
                            stats1['avg_population'] / 1_000,
                            stats1['avg_median_age'],
                            stats1['avg_household_size'] * 10  # Scale for visibility
                        ],
                        state2: [
                            stats2['total_population'] / 1_000_000,
                            stats2['avg_population'] / 1_000,
                            stats2['avg_median_age'],
                            stats2['avg_household_size'] * 10
                        ]
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name=state1,
                        x=comparison_data["Metric"],
                        y=comparison_data[state1],
                        marker_color='#667eea'
                    ))
                    fig.add_trace(go.Bar(
                        name=state2,
                        x=comparison_data["Metric"],
                        y=comparison_data[state2],
                        marker_color='#764ba2'
                    ))
                    fig.update_layout(
                        barmode='group',
                        title="State Metrics Comparison",
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        height=350,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # -------------------------------------------------
                    # AI Insight & Recommendation (with user's specific question)
                    # -------------------------------------------------
                    st.markdown("<div class='section-header'>🤖 AI Analysis & Recommendation</div>", unsafe_allow_html=True)

                    with st.spinner("Generating AI analysis..."):
                        
                        # Extract any specific question from the user's query
                        def extract_specific_question_states(query: str, state1: str, state2: str) -> str:
                            """Extract any specific question beyond the comparison request."""
                            q_lower = query.lower()
                            s1 = state1.lower()
                            s2 = state2.lower()
                            
                            # Remove common comparison phrases to find the specific question
                            remove_phrases = [
                                f"compare {s1} and {s2}",
                                f"compare {s2} and {s1}",
                                f"compare {s1} with {s2}",
                                f"compare {s2} with {s1}",
                                f"{s1} vs {s2}",
                                f"{s2} vs {s1}",
                                f"{s1} versus {s2}",
                                f"{s2} versus {s1}",
                                f"{s1} or {s2}",
                                f"{s2} or {s1}",
                                f"which is best {s1} or {s2}",
                                f"which is better {s1} or {s2}",
                                f"which is best {s2} or {s1}",
                                f"which is better {s2} or {s1}",
                                f"{s1} and {s2}",
                                f"{s2} and {s1}",
                            ]
                            
                            remaining = q_lower
                            for phrase in remove_phrases:
                                remaining = remaining.replace(phrase, "")
                            
                            # Clean up
                            remaining = remaining.strip().strip(".").strip(",").strip("?").strip()
                            
                            # If something meaningful remains, it's a specific question
                            if len(remaining) > 10:
                                return remaining
                            return None

                        specific_question = extract_specific_question_states(q, state1, state2)
                        
                        # Build the prompt
                        specific_question_section = ""
                        if specific_question:
                            specific_question_section = f"""

**USER'S SPECIFIC QUESTION:**
The user specifically asked: "{specific_question}"
Please address this question directly at the BEGINNING of your response before the general comparison.
"""

                        state_comparison_prompt = f"""
Compare these two U.S. states based on the data provided:

**{state1}:**
- Number of cities in database: {int(stats1['city_count'])}
- Total population: {int(stats1['total_population']):,}
- Average city population: {int(stats1['avg_population']):,}
- Average median age: {stats1['avg_median_age']:.1f} years
- Average household size: {stats1['avg_household_size']:.2f}
- Top cities: {', '.join(cities1['city'].head(5).tolist())}

**{state2}:**
- Number of cities in database: {int(stats2['city_count'])}
- Total population: {int(stats2['total_population']):,}
- Average city population: {int(stats2['avg_population']):,}
- Average median age: {stats2['avg_median_age']:.1f} years
- Average household size: {stats2['avg_household_size']:.2f}
- Top cities: {', '.join(cities2['city'].head(5).tolist())}
{specific_question_section}

TASK:
1. {"FIRST, directly answer the user's specific question: " + specific_question if specific_question else "Write a 2-3 sentence summary comparing both states"}
2. List 3 key differences as bullet points
3. Provide recommendations:
   - Which state is better for families?
   - Which state is better for young professionals?
   - Which state is better for retirement?
4. End with a balanced conclusion

IMPORTANT:
- Use your general knowledge about these states (weather, climate, cost of living, culture, economy, beaches, outdoor activities, etc.)
- Keep it concise, friendly, and actionable
- If asked about weather, provide specific climate information
"""

                        client = get_openai_client()
                        if client:
                            try:
                                rsp = client.chat.completions.create(
                                    model="gpt-4.1-mini",
                                    messages=[{"role": "user", "content": state_comparison_prompt}],
                                    temperature=0.4,
                                    max_tokens=600,
                                )
                                insight = rsp.choices[0].message.content.strip()
                            except Exception as e:
                                insight = f"AI analysis unavailable: {str(e)}"
                        else:
                            insight = "AI analysis unavailable - OpenAI client not configured."

                    # Show what question was detected
                    if specific_question:
                        st.markdown(
                            f"<div style='font-size: 0.85rem; color: #a0aec0; margin-bottom: 0.5rem;'>📝 Detected specific question: <em>{specific_question}</em></div>",
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <div class="insight-label">AI-Powered Comparison</div>
                            <div class="insight-text">{convert_markdown_to_html(insight)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


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

                    st.stop()
                else:
                    st.warning(f"Could not find data for one or both states: {state1}, {state2}")
                    st.stop()



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
        # CITY PROFILE QUERY
        # -------------------------------------------------
        q_lower = q.lower()
        if "city profile" in q_lower or "profile for" in q_lower or "tell me about" in q_lower:
            detected_city = extract_single_city_fuzzy(q)
            
            if detected_city:
                city_row = df_features[df_features["city"].str.lower() == detected_city.lower()]
                
                if not city_row.empty:
                    row = city_row.iloc[0]
                    show_city_profile_card(
                        city_name=row["city"],
                        state_name=row["state"],
                        row=row
                    )
                    st.stop()
        
        if mode_intent == "sql":
            with st.spinner("Running SQL..."):
                sql = build_sql_with_fallback(q, use_gpt=True)
                
                # Check if this is a special query - don't override these
                q_lower = q.lower()
                is_special_query = any(phrase in q_lower for phrase in [
                    "how many", "count", "total", "number of",
                    "average", "avg", "sum", "percentage", "percent",
                    ">", "<", ">=", "<=", "greater", "less", "more than",
                    "top ", "largest", "smallest", "biggest", "highest", "lowest"
                ])
                
                # Only try to detect single city if NOT a special query
                if not is_special_query:
                    detected_city = extract_single_city_fuzzy(q)
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
                
            #     # ML city insights
            #     if len(df) == 1 and "city" in [c.lower() for c in df.columns]:
            #         city_col = [c for c in df.columns if c.lower() == "city"][0]
            #         city_name = df.iloc[0][city_col]
            #         if str(city_name).lower() in df_features["city"].str.lower().values:
            #             show_city_insights(city_name)
            #             show_similar_cities(city_name)
            # st.stop()

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
        
