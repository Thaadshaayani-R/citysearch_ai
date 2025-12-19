# core/ml_router.py

import os
import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import text
from db_config import get_engine


 
# CACHED MODEL LOADING
 
MODEL_PATH = "models"

@st.cache_resource
def load_model(name: str):
    """Load ML model from models folder, cached."""
    path = os.path.join(MODEL_PATH, f"{name}_model.pkl")
    return joblib.load(path)


 
# CACHED CITY DATA
 
@st.cache_data(ttl=3600)
def get_city_df():
    """Load cities dataframe, cached for 1 hour."""
    engine = get_engine()

    query_str = """
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE population > 0
          AND median_age > 0
          AND avg_household_size > 0
    """

    with engine.connect() as conn:
        result = conn.execute(text(query_str))
        rows = result.fetchall()
        cols = result.keys()

    return pd.DataFrame(rows, columns=cols)


FEATURE_COLS = [
    "population",
    "median_age",
    "avg_household_size",
    "state"
]


 
# STATE EXTRACTION
 
US_STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York",
    "North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
    "Pennsylvania","Rhode Island","South Carolina","South Dakota",
    "Tennessee","Texas","Utah","Vermont","Virginia","Washington",
    "West Virginia","Wisconsin","Wyoming"
]

def extract_state_from_query(query):
    if not query:
        return None
    q = query.lower()
    for s in US_STATES:
        if s.lower() in q:
            return s
    return None


 
# APPLY MODEL
 
def _apply_model(df, model):
    X = df[FEATURE_COLS]
    df_out = df.copy()
    df_out["score"] = model.predict(X)
    return df_out.sort_values("score", ascending=False)


 
# RANKING FUNCTIONS
 
def run_family_ranking(query=None):
    df = get_city_df()
    state = extract_state_from_query(query)
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    model = load_model("family")
    return _apply_model(df, model).head(10)


def run_young_ranking(query=None):
    df = get_city_df()
    state = extract_state_from_query(query)
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    model = load_model("young")
    return _apply_model(df, model).head(10)


def run_retirement_ranking(query=None):
    df = get_city_df()
    state = extract_state_from_query(query)
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    model = load_model("retirement")
    return _apply_model(df, model).head(10)


 
# SINGLE CITY PREDICTION
 
def run_single_city_prediction(city_name):
    df = get_city_df()
    row = df[df["city"].str.lower() == city_name.lower()]

    if row.empty:
        return None

    row = row.iloc[0]

    model_family = load_model("family")
    model_young = load_model("young")
    model_ret = load_model("retirement")

    X = pd.DataFrame([{
        "population": row["population"],
        "median_age": row["median_age"],
        "avg_household_size": row["avg_household_size"],
        "state": row["state"],
    }])

    return {
        "city": row["city"],
        "state": row["state"],
        "population": row["population"],
        "median_age": row["median_age"],
        "avg_household_size": row["avg_household_size"],
        "family_score": float(model_family.predict(X)[0]),
        "young_score": float(model_young.predict(X)[0]),
        "retirement_score": float(model_ret.predict(X)[0]),
    }
