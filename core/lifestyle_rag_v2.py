# core/lifestyle_rag.py


import pandas as pd
from sqlalchemy import text
import streamlit as st

# Use shared database config
from db_config import get_engine


 
# OPENAI CLIENT
 
def get_openai_client():
    """Get OpenAI client from Streamlit secrets."""
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)


 
# LIFESTYLE QUERY DETECTION
 
LIFESTYLE_TRIGGERS = [
    "tell me about",
    "what is it like in",
    "what is life like in",
    "what's it like in",
    "what's life like in",
    "life in",
    "living in",
    "live in",
    "lifestyle in",
    "describe",
    "about",
]


def _extract_city_from_query(query: str) -> str | None:
    """Extract city name from a lifestyle query."""
    q = query.lower()

    # Check for trigger phrases and extract what follows
    for phrase in LIFESTYLE_TRIGGERS:
        if phrase in q:
            part = q.split(phrase)[-1].strip()
            part = part.rstrip("?.!,")
            if part:
                return part.title()

    # Fallback: take the last word
    tokens = query.strip().rstrip("?.!,").split()
    if not tokens:
        return None
    return tokens[-1].title()


def looks_like_lifestyle_query(query: str) -> bool:
    """Check if query appears to be asking about lifestyle."""
    q = query.lower()
    for phrase in LIFESTYLE_TRIGGERS:
        if phrase in q:
            return True
    return False


 
# GET CITY PROFILE FROM DB
 
def get_city_profile(city_name: str) -> dict | None:
    """
    Get city data with profile description from database.
    
    Args:
        city_name: Name of the city
        
    Returns:
        Dict with city data and profile, or None if not found
    """
    engine = get_engine()

    sql = text("""
        SELECT TOP 1
            c.city,
            c.state,
            c.state_code,
            c.population,
            c.median_age,
            c.avg_household_size,
            p.description
        FROM dbo.cities AS c
        LEFT JOIN dbo.city_profiles AS p
          ON c.city = p.city AND c.state = p.state
        WHERE LOWER(c.city) = LOWER(:city)
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"city": city_name}).fetchone()
        
        if result:
            return dict(result._mapping)
        return None
        
    except Exception as e:
        print(f"Error getting city profile: {e}")
        return None


 
# MAIN LIFESTYLE CARD BUILDER
 
def try_build_lifestyle_card(user_query: str) -> dict | None:
    """
    Attempt to build a lifestyle card for a city query.
    
    Args:
        user_query: User's query
        
    Returns:
        Dict with city info and AI summary, or None if not applicable
    """
    if not looks_like_lifestyle_query(user_query):
        return None

    city_name = _extract_city_from_query(user_query)
    if not city_name:
        return None

    # Get city profile from database
    profile = get_city_profile(city_name)
    if not profile:
        return None

    description = profile.get("description") or ""

    # Generate AI summary
    client = get_openai_client()

    if client is None:
        ai_summary = (
            "AI lifestyle summary unavailable, "
            "but the city exists in our database."
        )
    else:
        pop = profile.get('population', 'N/A')
        pop_str = f"{pop:,}" if isinstance(pop, (int, float)) else str(pop)
        
        prompt = f"""
User question: "{user_query}"

You are a helpful city-lifestyle assistant.

Here is structured data about the city:
- City: {profile['city']}
- State: {profile['state']}
- Population: {pop_str}
- Median age: {profile.get('median_age', 'N/A')}
- Average household size: {profile.get('avg_household_size', 'N/A')}

Here is a lifestyle description from our database:
\"\"\"{description}\"\"\"

TASK:
1. In 2–3 sentences, describe what it is like to live in this city.
2. Then give 3 bullet-style key points separated by line breaks.

IMPORTANT:
- Use ONLY the information above.
- Do NOT invent crime rates, salaries, or costs.
- Keep the language simple and friendly.
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "You explain cities in simple, friendly language."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300
            )
            ai_summary = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating lifestyle summary: {e}")
            ai_summary = "Could not generate AI summary."

    # Build card output
    return {
        "city": profile["city"],
        "state": profile["state"],
        "state_code": profile.get("state_code", ""),
        "population": int(profile["population"]) if profile.get("population") else None,
        "median_age": float(profile["median_age"]) if profile.get("median_age") else None,
        "avg_household_size": float(profile["avg_household_size"]) if profile.get("avg_household_size") else None,
        "description": description,
        "ai_summary": ai_summary,
    }


 
# LIFESTYLE CARD DISPLAY
 
def display_lifestyle_card(card_data: dict):
    """
    Display a lifestyle card in Streamlit.
    
    Args:
        card_data: Dict from try_build_lifestyle_card()
    """
    city = card_data.get("city", "Unknown")
    state = card_data.get("state", "")
    pop = card_data.get("population")
    age = card_data.get("median_age")
    household = card_data.get("avg_household_size")
    ai_summary = card_data.get("ai_summary", "")
    description = card_data.get("description", "")

    # Format values
    pop_str = f"{pop:,}" if pop else "N/A"
    age_str = f"{age:.1f}" if age else "N/A"
    household_str = f"{household:.2f}" if household else "N/A"

    # Main card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
    ">
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
            {city}, {state}
        </div>
        <div style="display: flex; gap: 2rem; margin-top: 1rem;">
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Population</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{pop_str}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Median Age</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{age_str}</div>
            </div>
            <div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Household Size</div>
                <div style="font-size: 1.5rem; font-weight: 600;">{household_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI Summary
    if ai_summary:
        st.markdown("### What's Life Like?")
        st.markdown(ai_summary)

    # Original description (if different from AI summary)
    if description and description not in ai_summary:
        with st.expander("Database Description", expanded=False):
            st.markdown(description)
