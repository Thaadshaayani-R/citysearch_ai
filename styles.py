"""
CitySearch AI - CSS Styles
==========================
All custom CSS styling for the application.
Uses consistent dark theme styling.
"""


def get_theme_colors(theme: str = "dark") -> dict:
    """Get color palette - returns dark theme colors for consistency."""
    
    # Always return dark theme colors for consistent design
    return {
        "bg_main": "#0f1419",
        "bg_card": "#1a202c",
        "bg_input": "#1a202c",
        "text_primary": "#ffffff",
        "text_secondary": "#e2e8f0",
        "text_muted": "#a0aec0",
        "border_color": "#2d3748",
        "gradient_start": "#3b4a6b",
        "gradient_end": "#4a5f7f",
        "accent_gradient_start": "#5a67d8",
        "accent_gradient_end": "#7c3aed",
    }


def get_custom_css(theme: str = "dark") -> str:
    """Generate complete CSS for the application."""
    
    colors = get_theme_colors(theme)
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
        
    .main {{
        background-color: {colors['bg_main']};
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
        font-size: 0.2rem !important;
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
    
    /* Target the clear button */
    div[data-testid="stButton"] button:has(p:contains("✕")) {{
        background: rgba(255, 255, 255, 0.04) !important;
        border: none !important;
        border-radius: 999px !important;

        width: 36px !important;
        height: 36px !important;
        min-height: 36px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        color: #9ca3af !important; /* soft gray */
        font-size: 18px !important;
        font-weight: 400 !important;

        transition: all 0.2s ease-in-out !important;
    }}

    /* Hover effect */
    div[data-testid="stButton"] button:has(p:contains("✕")):hover {{
        background: rgba(239, 68, 68, 0.12) !important; /* soft red */
        color: #ef4444 !important;
        transform: scale(1.05);
    }}

    /* Active (click) */
    div[data-testid="stButton"] button:has(p:contains("✕")):active {{
        transform: scale(0.95);
    }}

    /* Remove focus outline */
    div[data-testid="stButton"] button:has(p:contains("✕")):focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    .hero-section {{
        background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
        padding: 1rem 1.5rem;
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
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
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
        background-color: {colors['bg_card']};
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid {colors['border_color']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .sql-container {{
        background-color: {colors['bg_card']};
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.75rem;
        border: 1px solid {colors['border_color']};
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
        background: linear-gradient(135deg, {colors['accent_gradient_start']} 0%, {colors['accent_gradient_end']} 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #ffffff;
    }}

    /* Title (left side) */
    .result-card-title {{
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
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
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }}

    .insight-card {{
        background: linear-gradient(135deg, {colors['accent_gradient_start']} 0%, {colors['accent_gradient_end']} 100%);
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
    
    /* =============================================================================
       PREMIUM INSIGHT CARDS
       ============================================================================= */
    
    .insight-card-premium {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.08) 100%);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }}
    
    .insight-card-premium .insight-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }}
    
    .insight-card-premium .insight-icon {{
        font-size: 1rem;
    }}
    
    .insight-card-premium .insight-label {{
        font-size: 0.7rem;
        color: #a78bfa;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }}
    
    .insight-card-premium .insight-content {{
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 0.95rem;
    }}
    
    .insight-card-premium .insight-footer {{
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        font-size: 0.75rem;
        color: #a0aec0;
    }}
    
    .lifestyle-card {{
        background: linear-gradient(135deg, {colors['accent_gradient_start']} 0%, {colors['accent_gradient_end']} 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .section-header {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {colors['text_primary']};
        margin: 0.3rem 0 0.3rem 0 !important;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid {colors['border_color']};
        letter-spacing: -0.01em;
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, {colors['accent_gradient_start']} 0%, {colors['accent_gradient_end']} 100%);
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
        background: {colors['bg_card']};
        color: {colors['text_secondary']};
        border: 1px solid {colors['border_color']};
        font-weight: 500;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
    }}
    
    .sidebar .stButton>button:hover {{
        background: {colors['border_color']};
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {colors['text_primary']};
    }}
    
    div[data-testid="stMetricLabel"] {{
        font-size: 0.7rem;
        color: {colors['text_muted']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {colors['bg_card']};
        padding: 1rem;
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {colors['text_secondary']};
    }}
    
    .stTextInput input {{
        background-color: {colors['bg_input']};
        color: {colors['text_primary']};
        border: 1px solid {colors['border_color']};
        border-radius: 6px;
        font-size: 0.9rem;
        padding: 0.6rem 1rem;
    }}
    
    .stTextInput input:focus {{
        border-color: {colors['accent_gradient_start']};
        box-shadow: 0 0 0 1px {colors['accent_gradient_start']};
    }}
    
    .stCheckbox {{
        color: {colors['text_secondary']};
    }}
    
    .dataframe {{
        font-size: 0.85rem;
    }}
    
    .stAlert {{
        background-color: {colors['bg_card']};
        border: 1px solid {colors['border_color']};
        color: {colors['text_secondary']};
        border-radius: 6px;
    }}
    
    .download-btn {{
        background: {colors['bg_card']};
        border: 1px solid {colors['border_color']};
        padding: 0.5rem 1rem;
        border-radius: 6px;
        color: {colors['text_primary']};
        font-weight: 600;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }}
    
    .download-btn:hover {{
        background: {colors['border_color']};
    }}
    
    /* AI Response Card */
    .ai-response-card {{
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }}
    
    .ai-response-label {{
        font-size: 0.8rem;
        color: #667eea;
        margin-bottom: 0.5rem;
    }}
    
    /* Out of Scope Card */
    .out-of-scope-card {{
        background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: white;
    }}
    
    /* Success Card (Purple - consistent with brand) */
    .success-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }}
    
    /* Profile Card (Purple) */
    .profile-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }}
    
    /* Highlight Item */
    .highlight-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
    }}
    
    .highlight-check {{
        color: #667eea;
    }}
    
    /* Comparison Cards */
    .comparison-container {{
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    
    .comparison-card {{
        flex: 1;
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }}
    
    .comparison-card-left {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }}
    
    .comparison-card-right {{
        background: linear-gradient(135deg, #764ba2 0%, #9f7aea 100%);
    }}
    
    .vs-badge {{
        display: flex;
        align-items: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }}

/* =============================================================================
       PREMIUM DATA TABLES
       ============================================================================= */
    
    /* Table wrapper */
    [data-testid="stDataFrame"] {{
        border-radius: 16px !important;
        overflow: hidden !important;
    }}
    
    /* Table container */
    [data-testid="stDataFrame"] > div {{
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(102, 126, 234, 0.15) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }}
    
    /* Table header */
    [data-testid="stDataFrame"] th,
    .stDataFrame thead tr th {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 1rem 1.25rem !important;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3) !important;
    }}
    
    /* Table body cells */
    [data-testid="stDataFrame"] td,
    .stDataFrame tbody tr td {{
        padding: 0.875rem 1.25rem !important;
        font-size: 0.875rem !important;
        color: rgba(255, 255, 255, 0.9) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.2s ease !important;
    }}
    
    /* Alternating rows - odd */
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) td,
    .stDataFrame tbody tr:nth-child(odd) td {{
        background: rgba(102, 126, 234, 0.03) !important;
    }}
    
    /* Alternating rows - even */
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td,
    .stDataFrame tbody tr:nth-child(even) td {{
        background: rgba(102, 126, 234, 0.06) !important;
    }}
    
    /* Row hover effect */
    [data-testid="stDataFrame"] tbody tr:hover td,
    .stDataFrame tbody tr:hover td {{
        background: rgba(102, 126, 234, 0.15) !important;
        color: #ffffff !important;
    }}
    
    /* First column (City name) - make it stand out */
    [data-testid="stDataFrame"] td:first-child,
    .stDataFrame tbody tr td:first-child {{
        font-weight: 600 !important;
        color: #ffffff !important;
    }}
    
    /* Number columns - right align */
    [data-testid="stDataFrame"] td:nth-child(n+3),
    .stDataFrame tbody tr td:nth-child(n+3) {{
        text-align: right !important;
        font-variant-numeric: tabular-nums !important;
    }}
    
    /* Download button styling */
    .stDownloadButton > button {{
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        padding: 0.5rem 1rem !important;
        height: auto !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }}
    
    .stDownloadButton > button:hover {{
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }}

    </style>
    """
