#mlops/monitoring.py

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp
from datetime import datetime
import streamlit as st
from sqlalchemy import create_engine, text


# -----------------------------------------------------
# Helper to load registry
# -----------------------------------------------------
def load_registry():
    with open("mlops/registry.json", "r") as f:
        return json.load(f)


# -----------------------------------------------------
# Database Connection  (SQLAlchemy + pytds)
# -----------------------------------------------------
def get_engine():
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    username = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]

    conn_str = (
        f"mssql+pytds://{username}:{password}@{server}:1433/{database}"
        "?charset=utf8&autocommit=True"
    )

    return create_engine(conn_str)


# -----------------------------------------------------
# Load fresh data from SQL
# -----------------------------------------------------
def load_latest_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM dbo.cities", engine)
    return df


# -----------------------------------------------------
# Feature preparation (same as training)
# -----------------------------------------------------
def prepare_features(df):
    df = df.copy()

    df["population_norm"] = df["population"] / df["population"].max()
    df["median_age_norm"] = df["median_age"] / df["median_age"].max()
    df["household_norm"] = df["avg_household_size"] / df["avg_household_size"].max()

    return df[["population_norm", "median_age_norm", "household_norm"]]


# -----------------------------------------------------
# Drift detection (Kolmogorov-Smirnov test)
# -----------------------------------------------------
def ks_test(old, new):
    stat, p = ks_2samp(old, new)
    return {"statistic": float(stat), "p_value": float(p)}


# -----------------------------------------------------
# Main Monitoring Function
# -----------------------------------------------------
def run_monitoring():
    print("\n🔍 RUNNING MODEL MONITORING...\n")

    registry = load_registry()
    old_silhouette = registry["silhouette_score"]
    print(f"📌 Old silhouette: {old_silhouette}")

    # Load latest fresh dataset
    df = load_latest_data()
    print(f"📊 Loaded {len(df)} latest records")

    # Prepare features
    X = prepare_features(df)

    # Fit new KMeans temporarily (not saved)
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)

    # Compute new silhouette
    new_silhouette = silhouette_score(X, labels)
    print(f"✨ New silhouette: {new_silhouette}")

    silhouette_drop = new_silhouette < (old_silhouette - 0.02)

    # DRIFT CHECKS
    drift_report = {
        "population_drift": ks_test(
            registry.get("population_norm", X["population_norm"]),
            X["population_norm"],
        ),
        "median_age_drift": ks_test(
            registry.get("median_age_norm", X["median_age_norm"]),
            X["median_age_norm"],
        ),
        "household_size_drift": ks_test(
            registry.get("household_norm", X["household_norm"]),
            X["household_norm"],
        ),
        "silhouette_drop": silhouette_drop,
    }

    retrain_needed = (
        drift_report["population_drift"]["p_value"] < 0.05
        or drift_report["median_age_drift"]["p_value"] < 0.05
        or drift_report["household_size_drift"]["p_value"] < 0.05
        or silhouette_drop
    )

    print("\n📘 DRIFT REPORT:")
    print(json.dumps(drift_report, indent=4))

    print("\n🚨 Retraining Required?", "YES" if retrain_needed else "NO")

    return {
        "drift_report": drift_report,
        "retrain_needed": retrain_needed,
        "new_silhouette": new_silhouette,
    }


if __name__ == "__main__":
    run_monitoring()
