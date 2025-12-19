# =============================================================================
# CITYSEARCH AI - PREMIUM UI STYLES
# =============================================================================
# Add this to your app.py or import it
# Usage: st.markdown(PREMIUM_STYLES, unsafe_allow_html=True)
# =============================================================================

PREMIUM_STYLES = """
<style>
/* =============================================================================
   GOOGLE FONTS - Premium Typography
   ============================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* =============================================================================
   ROOT VARIABLES - Design System
   ============================================================================= */
:root {
    --primary-gradient: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
    --hero-gradient: linear-gradient(135deg, #2B3A55 0%, #1E293B 100%);
    --glow-color: rgba(99, 102, 241, 0.5);
    --glow-purple: rgba(168, 85, 247, 0.4);
    --border-subtle: rgba(255, 255, 255, 0.1);
    --border-hover: rgba(255, 255, 255, 0.2);
    --bg-dark: #0f1117;
    --bg-card: #1a1d24;
    --bg-input: #1e2028;
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
    --text-muted: rgba(255, 255, 255, 0.5);
    --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.15);
    --shadow-elevated: 0 8px 30px rgba(0, 0, 0, 0.25);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.3);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --transition-fast: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* =============================================================================
   GLOBAL STYLES
   ============================================================================= */
.stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* =============================================================================
   HERO SECTION - Premium Header Box
   ============================================================================= */
.hero-section {
    background: var(--hero-gradient);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-elevated);
    border: 1px solid var(--border-subtle);
    position: relative;
    overflow: hidden;
}

/* Subtle gradient overlay for depth */
.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse at top right, rgba(99, 102, 241, 0.1) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
    position: relative;
    z-index: 1;
}

/* =============================================================================
   SEARCH CONTAINER - ChatGPT Style
   ============================================================================= */
.search-container {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-top: 1.5rem;
    position: relative;
    z-index: 1;
}

/* =============================================================================
   SEARCH INPUT - Modern Premium Style
   ============================================================================= */
/* Target Streamlit's text input */
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    height: 52px !important;
    padding: 0 1.25rem !important;
    font-size: 1rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-smooth) !important;
    box-shadow: var(--shadow-soft) !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
    font-weight: 400 !important;
}

/* Focus state with glow effect */
.stTextInput > div > div > input:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 
        var(--shadow-soft),
        0 0 0 3px rgba(99, 102, 241, 0.15),
        0 0 20px rgba(99, 102, 241, 0.2) !important;
    outline: none !important;
}

/* Hover state */
.stTextInput > div > div > input:hover:not(:focus) {
    border-color: var(--border-hover) !important;
}

/* Hide the label */
.stTextInput > label {
    display: none !important;
}

/* =============================================================================
   SEARCH BUTTON - Premium Gradient with Micro-interactions
   ============================================================================= */
/* Target the primary button */
.stButton > button {
    background: var(--primary-gradient) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    height: 52px !important;
    min-width: 120px !important;
    padding: 0 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: white !important;
    cursor: pointer !important;
    transition: all var(--transition-smooth) !important;
    box-shadow: 
        0 4px 15px rgba(99, 102, 241, 0.4),
        0 2px 4px rgba(0, 0, 0, 0.1) !important;
    position: relative;
    overflow: hidden;
}

/* Hover effect - scale up and intensify glow */
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 
        0 8px 25px rgba(99, 102, 241, 0.5),
        0 4px 10px rgba(168, 85, 247, 0.3) !important;
}

/* Active/pressed state */
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
    box-shadow: 
        0 2px 10px rgba(99, 102, 241, 0.4) !important;
}

/* Focus state */
.stButton > button:focus {
    outline: none !important;
    box-shadow: 
        0 4px 15px rgba(99, 102, 241, 0.4),
        0 0 0 3px rgba(99, 102, 241, 0.3) !important;
}

/* =============================================================================
   SIDEBAR STYLING - Quick Examples
   ============================================================================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #1a1d24 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 1rem !important;
}

/* Sidebar buttons - Quick Examples */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.75rem 1rem !important;
    height: auto !important;
    min-width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    margin-bottom: 0.5rem !important;
    transition: all var(--transition-fast) !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.2) 100%) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2) !important;
}

/* =============================================================================
   FEATURE CARDS - City Data, AI-Powered, Instant Results
   ============================================================================= */
.feature-cards {
    display: flex;
    gap: 1.5rem;
    margin-top: 2rem;
    justify-content: center;
}

.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    text-align: center;
    flex: 1;
    max-width: 250px;
    transition: all var(--transition-smooth);
}

.feature-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-4px);
    box-shadow: var(--shadow-elevated);
}

.feature-card-icon {
    font-size: 1.5rem;
    margin-bottom: 0.75rem;
}

.feature-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.feature-card-desc {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* =============================================================================
   RESULTS CARDS - Premium Gradient Cards
   ============================================================================= */
.result-card {
    background: var(--hero-gradient);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-elevated);
    border: 1px solid var(--border-subtle);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.result-card-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.result-card-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.result-card-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

/* =============================================================================
   DATA TABLES - Modern Styling
   ============================================================================= */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
}

.stDataFrame [data-testid="stDataFrameContainer"] {
    background: var(--bg-card) !important;
}

/* Table headers */
.stDataFrame th {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 1rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* Table rows */
.stDataFrame td {
    color: var(--text-secondary) !important;
    padding: 0.875rem 1rem !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    transition: background var(--transition-fast) !important;
}

/* Alternating row colors */
.stDataFrame tr:nth-child(even) td {
    background: rgba(255, 255, 255, 0.02) !important;
}

/* Row hover */
.stDataFrame tr:hover td {
    background: rgba(99, 102, 241, 0.1) !important;
}

/* =============================================================================
   EXPANDER - View Generated SQL
   ============================================================================= */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-fast) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--border-hover) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

.streamlit-expanderContent {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

/* =============================================================================
   CLASSIFICATION BADGE
   ============================================================================= */
.classification-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 0.375rem 1rem;
    font-size: 0.8rem;
    color: rgba(99, 102, 241, 1);
    font-weight: 500;
    margin: 0.75rem 0;
}

.classification-badge.pattern {
    background: rgba(34, 197, 94, 0.1);
    border-color: rgba(34, 197, 94, 0.3);
    color: rgba(34, 197, 94, 1);
}

.classification-badge.llm {
    background: rgba(168, 85, 247, 0.1);
    border-color: rgba(168, 85, 247, 0.3);
    color: rgba(168, 85, 247, 1);
}

/* =============================================================================
   FOOTER STYLING
   ============================================================================= */
.footer {
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    border-top: 1px solid var(--border-subtle);
}

.footer-text {
    font-size: 0.875rem;
    color: var(--text-muted);
}

.footer-brand {
    font-weight: 600;
    color: var(--text-secondary);
}

/* =============================================================================
   LOADING ANIMATION
   ============================================================================= */
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.loading-skeleton {
    background: linear-gradient(90deg, 
        var(--bg-card) 25%, 
        rgba(99, 102, 241, 0.1) 50%, 
        var(--bg-card) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: var(--radius-md);
}

/* =============================================================================
   DOWNLOAD BUTTON
   ============================================================================= */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all var(--transition-fast) !important;
    box-shadow: none !important;
}

.stDownloadButton > button:hover {
    background: rgba(99, 102, 241, 0.1) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    color: var(--text-primary) !important;
    transform: none !important;
}

/* =============================================================================
   ERROR/WARNING/SUCCESS MESSAGES
   ============================================================================= */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: none !important;
}

/* Error */
.stAlert[data-baseweb="notification"] {
    background: rgba(239, 68, 68, 0.1) !important;
    border-left: 4px solid #ef4444 !important;
}

/* =============================================================================
   METRIC CARDS (Streamlit native)
   ============================================================================= */
[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-secondary) !important;
}

/* =============================================================================
   SCROLLBAR STYLING
   ============================================================================= */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
}

/* =============================================================================
   RESPONSIVE ADJUSTMENTS
   ============================================================================= */
@media (max-width: 768px) {
    .hero-section {
        padding: 1.5rem;
    }
    
    .hero-title {
        font-size: 1.5rem;
    }
    
    .feature-cards {
        flex-direction: column;
        align-items: center;
    }
    
    .feature-card {
        max-width: 100%;
    }
}
</style>
"""

# =============================================================================
# HERO SECTION HTML
# =============================================================================
def get_hero_html(title="CitySearch AI", subtitle="Ask anything about US cities. CitySearch AI finds the right data and gives clear insights instantly."):
    return f"""
    <div class="hero-section">
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """

# =============================================================================
# FEATURE CARDS HTML
# =============================================================================
def get_feature_cards_html():
    return """
    <div class="feature-cards">
        <div class="feature-card">
            <div class="feature-card-icon">📊</div>
            <div class="feature-card-title">City Data</div>
            <div class="feature-card-desc">Population, demographics, and more</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-icon">🤖</div>
            <div class="feature-card-title">AI-Powered</div>
            <div class="feature-card-desc">Smart recommendations and insights</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-icon">⚡</div>
            <div class="feature-card-title">Instant Results</div>
            <div class="feature-card-desc">Fast answers to any city question</div>
        </div>
    </div>
    """

# =============================================================================
# CLASSIFICATION BADGE
# =============================================================================
def get_classification_badge(method: str, query_type: str):
    """
    Returns HTML for classification badge
    method: "pattern" or "llm"
    query_type: the classified query type
    """
    if method == "pattern":
        icon = "⚡"
        label = "Pattern match (free)"
        badge_class = "pattern"
    else:
        icon = "🧠"
        label = "LLM classified"
        badge_class = "llm"
    
    return f"""
    <div class="classification-badge {badge_class}">
        {icon} {label} → {query_type}
    </div>
    """

# =============================================================================
# RESULT CARD HTML
# =============================================================================
def get_result_card_html(label: str, value: str, subtitle: str = ""):
    subtitle_html = f'<div class="result-card-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="result-card">
        <div class="result-card-label">{label}</div>
        <div class="result-card-value">{value}</div>
        {subtitle_html}
    </div>
    """

# =============================================================================
# FOOTER HTML
# =============================================================================
def get_footer_html():
    return """
    <div class="footer">
        <p class="footer-text">
            <span class="footer-brand">CitySearch AI</span> — Powered by GPT-4 and ML Clustering
        </p>
        <p class="footer-text">Ask anything about US cities</p>
    </div>
    """
