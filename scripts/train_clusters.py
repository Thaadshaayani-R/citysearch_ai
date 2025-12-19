# scripts/train_clusters.py (FIXED VERSION)
"""
Train KMeans clustering model for cities.
Saves model bundle compatible with cluster_router.py
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sqlalchemy import text
import streamlit as st

# Import from unified db_config
from db_config import get_engine


def load_city_data():
    """Load city data from database."""
    engine = get_engine()
    query = text("""
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE population > 0 AND median_age > 0 AND avg_household_size > 0
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def train_clusters(n_clusters: int = 5):
    """
    Train KMeans clustering model.
    
    Args:
        n_clusters: Number of clusters (default 5 to match cluster_labels.py)
    
    Returns:
        dict: Training results including metrics
    """
    print("Loading cities...")
    df = load_city_data()
    print(f"   Loaded {len(df)} cities")
    
    FEATURES = ["population", "median_age", "avg_household_size"]
    X = df[FEATURES]
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Training KMeans with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate quality metric
    sil_score = silhouette_score(X_scaled, labels)
    print(f"   Silhouette Score: {sil_score:.4f}")
    
    df["cluster"] = labels
    

    # CRITICAL: Bundle keys must match cluster_router.py

    os.makedirs("models", exist_ok=True)
    MODEL_PATH = os.path.join("models", "city_clusters.pkl")
    
    bundle = {
        "model": kmeans,      
        "scaler": scaler,     
        "features": FEATURES,
        "n_clusters": n_clusters,
        "silhouette_score": sil_score,
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"   Bundle keys: {list(bundle.keys())}")
    
    return {
        "model_path": MODEL_PATH,
        "n_clusters": n_clusters,
        "silhouette_score": sil_score,
        "num_cities": len(df),
    }


if __name__ == "__main__":
    results = train_clusters(n_clusters=5)
    print("\n📊 Training Results:")
    for k, v in results.items():
        print(f"   {k}: {v}")
