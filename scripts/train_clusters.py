# scripts/train_clusters.py

import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, text
import streamlit as st

# -------------------------------------------------
# 1. SQLAlchemy Engine (Azure SQL + Streamlit Cloud)
# -------------------------------------------------
def get_engine():
    server = st.secrets["SQL_SERVER_HOST"]
    database = st.secrets["SQL_SERVER_DB"]
    user = st.secrets["SQL_SERVER_USER"]
    password = st.secrets["SQL_SERVER_PASSWORD"]

    conn_str = (
        f"mssql+pytds://{user}:{password}@{server}:1433/{database}"
        "?charset=utf8&autocommit=True"
    )
    return create_engine(conn_str)


# -------------------------------------------------
# 2. Load City Data from Azure SQL
# -------------------------------------------------
def load_city_data():
    engine = get_engine()
    query = text("""
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
    """)

    with engine.begin() as conn:
        df = pd.read_sql(query, conn)

    return df


# -------------------------------------------------
# 3. Train KMeans Clusters
# -------------------------------------------------
def main():
    print("🔵 Loading cities...")
    df = load_city_data()

    FEATURES = ["population", "median_age", "avg_household_size"]
    X = df[FEATURES]

    print("🔵 Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🔵 Training KMeans...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    df["cluster"] = kmeans.labels_

    # -------------------------------------------------
    # 4. Save the model bundle
    # -------------------------------------------------
    os.makedirs("models", exist_ok=True)
    MODEL_PATH = os.path.join("models", "city_clusters.pkl")

    bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "features": FEATURES,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ SAVED MODEL: {MODEL_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
