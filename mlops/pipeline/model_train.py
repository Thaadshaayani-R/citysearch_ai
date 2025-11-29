#mlops/pipeline/model_train.py

import os
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


class ModelTrainer:
    """
    Trains ML models on engineered city features.
    - Currently supports KMeans clustering
    - Saves model to registry
    - Logs experiments to CSV
    """

    def __init__(
        self,
        registry_dir: str = "mlops/registry",
        experiments_path: str = "mlops/experiments/results.csv",
    ):
        self.registry_dir = registry_dir
        self.experiments_path = experiments_path

        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.experiments_path), exist_ok=True)

    # -------------------------------------------------------
    # 1) TRAINING FUNCTION – KMEANS
    # -------------------------------------------------------
    def train_kmeans(
        self,
        df: pd.DataFrame,
        n_clusters: int = 5,
        random_state: int = 42,
    ):
        """
        Trains a KMeans model using ML feature columns.
        Returns: model, metrics_dict, df_with_clusters
        """

        feature_cols = [
            "ml_vector_population",
            "ml_vector_age",
            "ml_vector_household",
        ]

        # Safety: ensure all features exist
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing feature column: {col}")

        X = df[feature_cols].values

        # Train model
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_labels = kmeans.fit_predict(X)

        # Attach cluster labels to dataframe
        df = df.copy()
        df["cluster_label"] = cluster_labels

        # Quality metric (how well clusters are separated)
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(X, cluster_labels)
        else:
            sil_score = None

        metrics = {
            "model_type": "kmeans",
            "n_clusters": n_clusters,
            "silhouette_score": float(sil_score) if sil_score is not None else None,
            "random_state": random_state,
        }

        return kmeans, metrics, df

    # -------------------------------------------------------
    # 2) SAVE MODEL TO REGISTRY
    # -------------------------------------------------------
    def save_model(
        self,
        model,
        model_name: str,
        metrics: dict,
        feature_cols: list,
    ):
        """
        Saves model object + metadata into registry folder.
        """

        model_path = os.path.join(self.registry_dir, f"{model_name}.pkl")
        metadata_path = os.path.join(self.registry_dir, f"{model_name}_metadata.json")

        joblib.dump(model, model_path)

        metadata = {
            "model_name": model_name,
            "saved_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "feature_columns": feature_cols,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_path, metadata_path

    # -------------------------------------------------------
    # 3) LOG EXPERIMENT TO CSV
    # -------------------------------------------------------
    def log_experiment(self, model_name: str, metrics: dict, params: dict):
        """
        Appends training run info into experiments CSV.
        """

        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": model_name,
            **params,
            **metrics,
        }

        df_row = pd.DataFrame([row])

        # If file exists -> append, else create with header
        if os.path.exists(self.experiments_path):
            df_row.to_csv(self.experiments_path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(self.experiments_path, mode="w", header=True, index=False)

        return row
