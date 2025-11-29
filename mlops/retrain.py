import json
import os
import shutil
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------
# Helper: Load Registry
# -----------------------------------------------------
def load_registry():
    with open("mlops/registry.json", "r") as f:
        return json.load(f)

def save_registry(registry):
    with open("mlops/registry.json", "w") as f:
        json.dump(registry, f, indent=4)


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
# Feature Engineering (SAME as training)
# -----------------------------------------------------
def prepare_features(df):
    df = df.copy()
    df["population_norm"] = df["population"] / df["population"].max()
    df["median_age_norm"] = df["median_age"] / df["median_age"].max()
    df["household_norm"] = df["avg_household_size"] / df["avg_household_size"].max()
    return df[["population_norm", "median_age_norm", "household_norm"]]


# -----------------------------------------------------
# MAIN RETRAIN PIPELINE
# -----------------------------------------------------
def retrain():
    print("\n🚀 STARTING MODEL RETRAINING...\n")

    registry = load_registry()
    old_version = float(registry["version"])
    old_silhouette = registry["silhouette_score"]

    print(f"📌 CURRENT MODEL VERSION: {old_version}")
    print(f"📌 OLD SILHOUETTE SCORE: {old_silhouette}")

    # Load fresh data
    df = load_latest_data()
    print(f"📊 Loaded {len(df)} rows from SQL Server")

    # Prepare features
    X = prepare_features(df)

    # Train NEW model
    print("\n⚙️ Training new KMeans model...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)

    new_silhouette = silhouette_score(X, labels)
    print(f"✨ NEW SILHOUETTE SCORE: {new_silhouette}")

    improvement = new_silhouette - old_silhouette

    print("\n📈 COMPARISON")
    print(f"Old silhouette: {old_silhouette}")
    print(f"New silhouette: {new_silhouette}")
    print(f"Improvement: {improvement}")

    if new_silhouette < old_silhouette:
        print("\n❌ NEW MODEL REJECTED (worse performance). Keeping old model.")
        return {
            "status": "rejected",
            "old_silhouette": old_silhouette,
            "new_silhouette": new_silhouette
        }

    print("\n✅ NEW MODEL ACCEPTED!")

    # Save & archive models
    old_model_path = registry["model_path"]
    new_version = round(old_version + 0.1, 1)
    new_model_name = f"city_kmeans_v{new_version}.pkl"
    new_model_path = f"mlops/registry/{new_model_name}"

    archive_dir = "mlops/registry/archive"
    os.makedirs(archive_dir, exist_ok=True)
    shutil.copy(old_model_path, f"{archive_dir}/city_kmeans_v{old_version}.pkl")

    import joblib
    joblib.dump(kmeans, new_model_path)

    # Update registry
    registry["version"] = str(new_version)
    registry["model_path"] = new_model_path
    registry["trained_on"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registry["silhouette_score"] = float(new_silhouette)
    registry["num_cities"] = len(df)
    registry["drift_detected"] = False
    registry["notes"] = "Auto-retrained model accepted"

    save_registry(registry)

    print("\n📦 MODEL UPDATED:")
    print(f"Version: {new_version}")
    print(f"Saved to: {new_model_path}")

    return {
        "status": "accepted",
        "new_version": new_version,
        "new_silhouette": new_silhouette
    }


if __name__ == "__main__":
    retrain()
