# core/cluster_router.py

import os
import pickle
import pandas as pd
import streamlit as st
from sqlalchemy import text
from .cluster_definitions import CLUSTER_LABELS
from db_config import get_engine
from core.cluster_definitions import CLUSTER_LABELS, get_cluster_name

 
# 1. CACHED Model Loading
 
@st.cache_resource
def load_cluster_model():
    """Load cluster model once and cache."""
    MODEL_PATH = os.path.join(os.getcwd(), "models", "city_clusters.pkl")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Cluster model not found at: {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        cluster_data = pickle.load(f)
    
    return cluster_data["model"], cluster_data["scaler"]


# Load once
kmeans_model, scaler = load_cluster_model()


 
# 2. CACHED City Data Loading
 
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_city_data():
    """Load city data once and cache."""
    engine = get_engine()

    query_str = """
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
    """

    with engine.connect() as conn:
        result = conn.execute(text(query_str))
        rows = result.fetchall()
        cols = result.keys()

    return pd.DataFrame(rows, columns=cols)


# Rest of functions remain the same...
def add_cluster_labels(df):
    features = df[["population", "median_age", "avg_household_size"]]
    X_scaled = scaler.transform(features)
    df = df.copy()
    df["cluster_id"] = kmeans_model.predict(X_scaled)
    return df


def attach_cluster_labels(df):
    df = df.copy()
    df["cluster_name"] = df["cluster_id"].apply(
        lambda cid: CLUSTER_LABELS.get(cid, {}).get("name", f"Cluster {cid}")
    )
    return df


def cluster_all():
    df = load_city_data()
    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)
    return df.sort_values("cluster_id")


def cluster_by_state(state: str):
    df = load_city_data()
    df = df[df["state"].str.lower() == state.lower()]

    if df.empty:
        return None

    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)
    return df.sort_values("cluster_id")


def cluster_single_city(city: str):
    df = load_city_data()
    row = df[df["city"].str.lower() == city.lower()]

    if row.empty:
        return None

    row = add_cluster_labels(row).iloc[0]
    cluster_id = int(row["cluster_id"])

    return {
        "city": row["city"],
        "state": row["state"],
        "cluster": cluster_id,
        "cluster_name": CLUSTER_LABELS.get(cluster_id, {}).get("name", f"Cluster {cluster_id}"),
        "cluster_summary": CLUSTER_LABELS.get(cluster_id, {}).get("summary", ""),
    }


def cluster_similar_to(city: str):
    df = load_city_data()
    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)

    row = df[df["city"].str.lower() == city.lower()]
    if row.empty:
        return None

    cluster_id = int(row["cluster_id"].iloc[0])
    similar = df[df["cluster_id"] == cluster_id]

    return similar.sort_values("population", ascending=False)
