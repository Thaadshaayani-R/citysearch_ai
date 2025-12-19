#config.py
"""
CitySearch AI - Configuration and Constants
============================================
Central configuration file for all app settings.
"""

# APP SETTINGS
APP_TITLE = "CitySearch AI"
APP_SUBTITLE = "Ask anything about US cities. CitySearch AI finds the right data and gives clear insights instantly."
APP_ICON = "🏙️"

# MODEL SETTINGS
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE_CLASSIFICATION = 0.0  # Deterministic for classification
LLM_TEMPERATURE_SUMMARY = 0.7  # Creative for summaries
LLM_MAX_TOKENS_CLASSIFICATION = 500
LLM_MAX_TOKENS_SUMMARY = 300
LLM_MAX_TOKENS_COMPARISON = 600


# US STATES LIST
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

US_STATES_LOWER = [s.lower() for s in US_STATES_FULL]

# State abbreviations mapping
STATE_ABBREVIATIONS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}

# QUERY CLASSIFICATION TYPES
RESPONSE_TYPES = [
    "single_city",      # "What is the population of Denver?"
    "single_state",     # "What is the population of Texas?"
    "city_list",        # "Top 10 cities in Florida"
    "state_list",       # "List all states"
    "comparison",       # "Compare Miami and Dallas"
    "city_profile",     # "Tell me about Denver"
    "state_profile",    # "Tell me about Texas"
    "aggregate",        # "How many cities in California?"
    "recommendation",   # "Best city for families"
    "cluster",          # "Which cluster is Miami in?"
    "similar_cities",   # "Cities similar to Chicago"
    "lifestyle",        # "Life in Austin"
    "general_question", # Non-city questions
]

SPECIFIC_INTENTS = [
    "families",
    "young_professionals", 
    "retirement",
    "general",
]

SUMMARY_STYLES = [
    "none",             # No AI summary
    "brief",            # 1-2 bullet points
    "detailed",         # 2-3 sentences
    "recommendation",   # Pros/cons + pick
    "comparison",       # Side-by-side analysis
    "highlights",       # Key features
]

# PROTECTED WORDS (for spelling correction)
PROTECTED_WORDS = {
    # Query keywords
    "best", "worst", "good", "bad", "top", "bottom", "most", "least",
    "compare", "comparison", "versus", "between", "which", "what", "where",
    "city", "cities", "state", "states", "country", "countries",
    "population", "median", "average", "household", "size", "age",
    "highest", "lowest", "largest", "smallest", "biggest", "youngest", "oldest",
    
    # Demographics
    "young", "old", "older", "younger", "family", "families", "senior", "seniors",
    "professional", "professionals", "student", "students", "retired", "retiree",
    "retirement", "children", "kids", "adults", "people", "person",
    "remote", "worker", "workers", "millennials", "gen-z", "boomers",
    
    # Descriptors
    "large", "small", "big", "little", "high", "low", "cheap", "expensive",
    "affordable", "safe", "dangerous", "friendly", "beautiful", "nice",
    "growing", "declining", "urban", "rural", "suburban", "coastal",
    "populated", "dense", "crowded", "spacious",
    
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
    "score", "rank", "ranking", "rated", "rating", "profile",
    
    # Common words
    "the", "and", "for", "with", "from", "into", "over", "under",
    "how", "many", "much", "more", "less", "than", "that", "this",
    "are", "is", "was", "were", "been", "being", "have", "has", "had",
    "all", "any", "some", "every", "each", "both", "other", "another",
    "does", "total", "number", "count",
}

# FORBIDDEN WORDS (world/non-US queries)
FORBIDDEN_WORLD_KEYWORDS = [
    "canada", "india", "europe", "uk", "england",
    "china", "japan", "australia", "germany", "mexico",
    "france", "spain", "africa", "dubai", "uae",
    "london", "paris", "tokyo", "beijing", "mumbai",
    "toronto", "vancouver", "sydney", "melbourne",
]

# QUICK EXAMPLES FOR SIDEBAR
QUICK_EXAMPLES = [
    "Total population of California",
    "How many cities are in Texas",
    "Top 10 largest cities in Florida",
    "Which is best city for families",
    "Best cities for young professionals",
    "City profile for Denver",
    "Cities similar to Chicago",
    "Compare Miami and Austin",
]

# DATABASE SETTINGS
DB_TABLE_NAME = "dbo.cities"
DB_COLUMNS = ["city", "state", "state_code", "population", "median_age", "avg_household_size"]

# CACHE SETTINGS
CACHE_TTL_SECONDS = 3600  # 1 hour
