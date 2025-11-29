# core/ml_utils.py

import os
import json
import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import text
from db_config import get_engine

REGISTRY_DIR = "mlops/registry"


@st.cache_resource
def load_trained_model(model_name="city_clusters"):
    """Loads model + metadata, cached."""
    model_path = os.path.join("models", f"{model_name}.pkl")
    meta_path = os.path.join("models", f"{model_name}_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    model = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata


@st.cache_data(ttl=3600)
def load_feature_data():
    """Load feature data, cached for 1 hour."""
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
