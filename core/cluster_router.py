import os
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from .cluster_labels import CLUSTER_LABELS   # NEW
import streamlit as st

# -----------------------------
# 1. Load Model + Scaler
# -----------------------------

MODEL_PATH = os.path.join(os.getcwd(), "models", "city_clusters.pkl")

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Cluster model not found at: {MODEL_PATH}")

try:
    # Try loading the model
    with open(MODEL_PATH, "rb") as f:
        cluster_data = pickle.load(f)

    print("Model loaded successfully.")
    # Print model structure if loaded successfully
    print("Model keys:", cluster_data.keys())

    # Load the model and scaler
    kmeans_model = cluster_data["model"]
    scaler = cluster_data["scaler"]
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise  # Raise the error to stop further execution if model loading fails


# -----------------------------
# 2. Database Connection (SQLAlchemy + pytds)
# -----------------------------
def get_engine():
    """
    Streamlit Cloud reads DB credentials from secrets.toml
    """
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    username = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]

    connection_url = URL.create(
        "mssql+pytds",
        username=username,
        password=password,
        host=server,
        port=1433,
        database=database,
    )

    return create_engine(connection_url)


def load_city_data():
    engine = get_engine()
    query = """
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
    """
    return pd.read_sql(query, engine)


# -----------------------------
# 3. Compute cluster IDs
# -----------------------------
def add_cluster_labels(df):
    # Ensure features are available
    features = df[["population", "median_age", "avg_household_size"]]
    
    # Scale the features using the previously loaded scaler
    X_scaled = scaler.transform(features)
    
    # Predict cluster labels using the KMeans model
    df["cluster_id"] = kmeans_model.predict(X_scaled)
    return df


# -----------------------------
# 4. Add human-friendly names
# -----------------------------
def attach_cluster_labels(df):
    df = df.copy()
    # Add a human-readable cluster name based on the cluster ID
    df["cluster_name"] = df["cluster_id"].apply(
        lambda cid: CLUSTER_LABELS.get(cid, {}).get("name", f"Cluster {cid}")
    )
    return df


# -----------------------------
# 5. Cluster All Cities
# -----------------------------
def cluster_all():
    # Load city data
    df = load_city_data()
    
    # Add cluster labels
    df = add_cluster_labels(df)
    
    # Attach human-readable cluster names
    df = attach_cluster_labels(df)
    
    # Sort by cluster ID
    return df.sort_values("cluster_id")


# -----------------------------
# 6. Cluster by State
# -----------------------------
def cluster_by_state(state: str):
    # Load city data
    df = load_city_data()
    
    # Filter by state
    df = df[df["state"].str.lower() == state.lower()]
    
    if df.empty:
        return None
    
    # Add cluster labels and human-readable names
    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)
    
    # Return sorted by cluster ID
    return df.sort_values("cluster_id")


# -----------------------------
# 7. Single City Cluster
# -----------------------------
def cluster_single_city(city: str):
    # Load city data
    df = load_city_data()
    
    # Find the specific city
    row = df[df["city"].str.lower() == city.lower()]
    
    if row.empty:
        return None

    # Add cluster labels
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
    # Load city data
    df = load_city_data()
    
    # Add cluster labels
    df = add_cluster_labels(df)
    df = attach_cluster_labels(df)

    # Find the specific city
    row = df[df["city"].str.lower() == city.lower()]
    if row.empty:
        return None

    # Get the cluster ID for the city
    cluster_id = int(row["cluster_id"].iloc[0])
    
    # Get other cities in the same cluster
    similar = df[df["cluster_id"] == cluster_id]

    return similar.sort_values("population", ascending=False)
