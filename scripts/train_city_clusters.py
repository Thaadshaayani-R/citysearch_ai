# scripts/train_city_clusters.py

import os
import pickle
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from db_config import get_engine

MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# LOAD CITY DATA

def load_data():
    engine = get_engine()
    with engine.begin() as conn:
        df = pd.read_sql(
            text("""
                SELECT city, state, population, median_age, avg_household_size
                FROM dbo.cities
            """),
            conn
        )
    return df


# TRAIN CLUSTERS

def train_clusters():
    df = load_data()

    features = ["population", "median_age", "avg_household_size"]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Train model
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Bundle everything
    bundle = {
        "df": df,
        "model": kmeans,
        "scaler": scaler,
        "features": features
    }

    # Save model file
    out_path = os.path.join(MODEL_DIR, "city_clusters.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print("✔ Cluster training complete!")
    print("✔ Saved:", out_path)
    return df

if __name__ == "__main__":
    df = train_clusters()
    print(df.head())
