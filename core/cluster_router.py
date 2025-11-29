# core/cluster_router.py

import os
import pickle
import pandas as pd
from sqlalchemy import text
from .cluster_labels import CLUSTER_LABELS
from db_config import get_engine  # ← Use shared db_config

# -----------------------------
# 1. Load Model + Scaler
# -----------------------------

MODEL_PATH = os.path.join(os.getcwd(), "models", "city_clusters.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Cluster model not found at: {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as f:
        cluster_data = pickle.load(f)

    print("Model loaded successfully.")
    print("Model keys:", cluster_data.keys())

    kmeans_model = cluster_data["model"]
    scaler = cluster_data["scaler"]

except Exception as e:
    print(f"Error loading model: {e}")
    raise


# -----------------------------
# 2. Load City Data (FIXED)
# -----------------------------
def load_city_data():
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


# -----------------------------
# 3. Compute cluster IDs
# -----------------------------
def add_cluster_labels(df):
    features = df[["population", "median_age", "avg_household_size"]]
    X_scaled = scaler.transform(features)
    df["cluster_id"] = kmeans_model.predict(X_scaled)
    return df


# -----------------------------
# 4. Add human-friendly names
# -----------------------------
def attach_cluster_labels(df):
    df = df.copy()
    df["cluster_name"] = df["cluster_id"].apply(
        lambda cid: CLUSTER_LABELS.get(cid, {}).get("name", f"Cluster {cid}")
    )
    return df


# -----------------------------
# 5. Cluster All Cities
# -----------------------------
def cluster_all():
    df = load_city_data()
    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)
    return df.sort_values("cluster_id")


# -----------------------------
# 6. Cluster by State
# -----------------------------
def cluster_by_state(state: str):
    df = load_city_data()
    df = df[df["state"].str.lower() == state.lower()]

    if df.empty:
        return None

    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)
    return df.sort_values("cluster_id")


# -----------------------------
# 7. Single City Cluster
# -----------------------------
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


# -----------------------------
# 8. Cities Similar to X
# -----------------------------
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
