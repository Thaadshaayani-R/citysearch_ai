#ml_utils.py

import os
import json
import joblib
import pandas as pd
from db_config import get_engine

# Folder that stores your trained ML models
REGISTRY_DIR = "mlops/registry"


# ===============================================================
# LOAD TRAINED MODEL + METADATA
# ===============================================================
def load_trained_model(model_name="city_clusters"):
    """
    Loads model + metadata from the models directory.
    """
    model_path = os.path.join("models", f"{model_name}.pkl")  # Corrected path
    meta_path = os.path.join("models", f"{model_name}_metadata.json")  # Corrected path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Load metadata
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata




# ===============================================================
# LOAD FEATURE DATA FROM SQL SERVER (CLEAN VERSION)
# ===============================================================
def load_feature_data():
    engine = get_engine()

    query_str = """
        SELECT city, state, population, median_age, avg_household_size
        FROM dbo.cities
        WHERE population > 0
          AND median_age > 0
          AND avg_household_size > 0
    """

    # ✅ DO NOT use connection.cursor()
    # ✅ DO NOT use pd.read_sql_query()
    # ✅ Use engine.execute() and convert to DataFrame manually

    with engine.connect() as conn:
        result = conn.execute(query_str)   # SQLAlchemy executes the string safely
        rows = result.fetchall()
        cols = result.keys()

    # Convert result → DataFrame
    df = pd.DataFrame(rows, columns=cols)
    return df

