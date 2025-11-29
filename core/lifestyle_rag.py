import pandas as pd
import streamlit as st
from sqlalchemy import text
from openai import OpenAI
from db_config import get_engine
import spacy
from spacy.cli import download


# -----------------------------------------
# OpenAI client
# -----------------------------------------
def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


# -----------------------------------------
# Load spaCy model for Named Entity Recognition (NER)
# -----------------------------------------
nlp = spacy.load("en_core_web_sm")


# -----------------------------------------
# Query pattern triggers (flexible and dynamic)
# -----------------------------------------
LIFESTYLE_TRIGGERS = [
    "tell me about",
    "what is it like in",
    "what is life like in",
    "life in",
    "living in",
    "live in",
    "how is life in",  # Added another phrase for flexibility
    "what's life like in",  # Added another common phrase
]


def _extract_city_from_query(query: str):
    """
    Use Named Entity Recognition (NER) with spaCy to extract city names dynamically
    from the query.
    """
    # Process the query using spaCy NLP model
    doc = nlp(query)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # GPE = Geopolitical Entity
    if cities:
        return cities[0]  # Return the first detected city
    
    # Fallback: Use a list of common phrases to extract the city if NER fails
    q = query.lower()
    for phrase in LIFESTYLE_TRIGGERS:
        if phrase in q:
            part = q.split(phrase)[-1].strip().rstrip("?.!,")
            return part.title()
    return None  # Return None if no city is detected



def _looks_like_lifestyle_query(query: str):
    """
    Checks if the query seems to be asking about lifestyle based on trigger phrases.
    """
    q = query.lower()
    return any(p in q for p in LIFESTYLE_TRIGGERS)


# -----------------------------------------
# Main RAG builder
# -----------------------------------------
def try_build_lifestyle_card(user_query: str):
    """
    If the query looks like a lifestyle question,
    return a dict with city profile data + AI summary.
    Otherwise, return None.
    """
    # Check if it's a lifestyle-related query
    if not _looks_like_lifestyle_query(user_query):
        return None

    # Extract the city name from the query (using NER or fallback)
    city = _extract_city_from_query(user_query)
    if not city:
        return None  # Return None if no city is found

    # Query the database for city data (can be changed based on your data source)
    city_data = get_city_data_from_db(city)

    if not city_data:
        return None  # Return None if no city data is found

    # Generate AI summary based on available city data
    ai_summary = generate_ai_summary(user_query, city_data)

    return {
        "city": city_data['city'],
        "state": city_data['state'],
        "population": city_data['population'],
        "median_age": city_data['median_age'],
        "avg_household_size": city_data['avg_household_size'],
        "description": city_data['description'],
        "ai_summary": ai_summary,
    }


def get_city_data_from_db(city_name: str):
    """
    Query the database to fetch city data based on the city name.
    """
    # --- SAFE: SQLAlchemy engine ---
    engine = get_engine()

    sql = text("""
        SELECT TOP 1
            c.city,
            c.state,
            c.population,
            c.median_age,
            c.avg_household_size,
            p.description
        FROM dbo.cities AS c
        LEFT JOIN dbo.city_profiles AS p
            ON c.city = p.city AND c.state = p.state
        WHERE LOWER(c.city) = LOWER(:city)
    """)

    # --- Must use engine.connect() ---
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"city": city_name})

    if df.empty:
        return None  # No data for the city

    return df.iloc[0].to_dict()  # Return data as a dictionary


def generate_ai_summary(user_query: str, city_data):
    """
    Generate a friendly AI summary based on the city's data and description.
    """
    prompt = f"""
    User question: "{user_query}"

    City Data:
    - City: {city_data['city']}
    - State: {city_data['state']}
    - Population: {city_data['population']}
    - Median Age: {city_data['median_age']}
    - Average Household Size: {city_data['avg_household_size']}

    Lifestyle Notes:
    {city_data['description']}

    Task: Generate a short 2–3 sentence summary of what it's like to live in this city.
    Include key points like the city's vibe, family-friendliness, opportunities for young professionals, etc.
    """
    
    # Get OpenAI client and generate the summary
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_summary = response.choices[0].message.content.strip()
    except Exception as e:
        ai_summary = "AI summary unavailable due to an error."

    return ai_summary

