#mlops/pipeline/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureEngineer:
    """
    Creates ML-ready features from raw city data:
    - normalization
    - demographic indexes
    - lifestyle score
    """

    def __init__(self):
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()

    def engineer_features(self, df: pd.DataFrame):
        df = df.copy()

        # --- SAFETY: drop missing values (should be none after validation)
        df = df.dropna(subset=["population", "median_age", "avg_household_size"])

        # ------------------------------------------------------------
        # 1. NORMALIZATION for ML Models (clustering, ranking)
        # ------------------------------------------------------------
        df["population_norm"] = self.minmax_scale(df["population"])
        df["median_age_norm"] = self.minmax_scale(df["median_age"])
        df["household_norm"] = self.minmax_scale(df["avg_household_size"])

        # ------------------------------------------------------------
        # 2. DEMOGRAPHIC INDEXES (real world meaning)
        # ------------------------------------------------------------

        # Younger cities = more attractive for young professionals
        df["youth_index"] = 1 - df["median_age_norm"]

        # Larger families = higher family-friendly score
        df["family_index"] = df["household_norm"]

        # Population as indicator of opportunities vs. crowding
        df["opportunity_index"] = df["population_norm"]

        # ------------------------------------------------------------
        # 3. COMBINED LIFESTYLE SCORE (weighted)
        # ------------------------------------------------------------
        df["lifestyle_score"] = (
            0.4 * df["opportunity_index"] +
            0.3 * df["youth_index"] +
            0.3 * df["family_index"]
        )

        # ------------------------------------------------------------
        # 4. RANKING
        # ------------------------------------------------------------
        df["lifestyle_rank"] = df["lifestyle_score"].rank(ascending=False).astype(int)

        # ------------------------------------------------------------
        # 5. FINAL ML FEATURES (for clustering or future ML models)
        # ------------------------------------------------------------
        df["ml_vector_population"] = df["population_norm"]
        df["ml_vector_age"] = df["median_age_norm"]
        df["ml_vector_household"] = df["household_norm"]

        return df

    # --- helper ----------------------------------------------------
    def minmax_scale(self, series: pd.Series):
        values = series.values.reshape(-1, 1)
        scaled = MinMaxScaler().fit_transform(values)
        return scaled.flatten()
