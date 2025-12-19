# scripts/train_city_models.py

import os
import pandas as pd
import joblib
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from db_config import get_engine


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../citysearch_ai/scripts
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../citysearch_ai
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("MODEL_DIR:", MODEL_DIR)




# 2. Load data from Azure SQL

def load_city_data():
    engine = get_engine()

    query = text("""
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE population > 0 AND median_age > 0 AND avg_household_size > 0
    """)

    with engine.begin() as conn:
        df = pd.read_sql(query, conn)

    print("Loaded rows:", len(df))
    return df



# 3. Create synthetic targets

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize population
    df["population_norm"] = (df["population"] - df["population"].min()) / \
                            (df["population"].max() - df["population"].min())

    # Family score
    df["family_score"] = (
        0.4 * df["avg_household_size"] +
        0.3 * df["population_norm"] +
        0.3 * (1 / df["median_age"])
    )

    # Young professionals score
    df["young_score"] = (
        0.5 * (1 / df["median_age"]) +
        0.3 * df["population_norm"] +
        0.2 * (1 / df["avg_household_size"])
    )

    # Retirement score
    df["retirement_score"] = (
        0.6 * df["median_age"] +
        0.2 * (1 / df["population"]) +
        0.2 * (1 / df["avg_household_size"])
    )

    return df



# 4. Train + save one model

def train_model(df: pd.DataFrame, target_col: str, filename: str):
    X = df[["population", "median_age", "avg_household_size", "state"]]
    y = df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["population", "median_age", "avg_household_size"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["state"]),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"Model '{target_col}' R²: {r2:.4f}")

    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"Saved → {path}")



# 5. Main

def main():
    print("Loading data...")
    df = load_city_data()

    print("Creating targets...")
    df = create_targets(df)

    print("Training family model...")
    train_model(df, "family_score", "family_model.pkl")

    print("Training young model...")
    train_model(df, "young_score", "young_model.pkl")

    print("Training retirement model...")
    train_model(df, "retirement_score", "retirement_model.pkl")

    print("DONE! All models trained and saved.")


if __name__ == "__main__":
    main()
