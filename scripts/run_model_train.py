#scripts/run_model_train.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_config import get_connection
from mlops.pipeline.data_validation import DataValidator
from mlops.pipeline.feature_engineering import FeatureEngineer
from mlops.pipeline.model_train import ModelTrainer
import pandas as pd


def main():
    # 1) Load raw data from SQL
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM [dbo].[cities]", conn)

    # 2) Validate data
    validator = DataValidator()
    report = validator.validate(df)
    print("VALIDATION REPORT:", report)

    if report["status"] != "PASSED":
        print("Data validation FAILED. Fix data before training.")
        return

    # 3) Feature engineering
    engineer = FeatureEngineer()
    df_fe = engineer.engineer_features(df)

    # 4) Train model
    trainer = ModelTrainer()
    kmeans_model, metrics, df_with_clusters = trainer.train_kmeans(df_fe, n_clusters=5)

    print("TRAINING METRICS:", metrics)

    # 5) Save model to registry
    feature_cols = [
        "ml_vector_population",
        "ml_vector_age",
        "ml_vector_household",
    ]
    model_name = "city_kmeans"

    model_path, metadata_path = trainer.save_model(
        kmeans_model,
        model_name=model_name,
        metrics=metrics,
        feature_cols=feature_cols,
    )

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    # 6) Log experiment
    params = {"n_clusters": 5, "random_state": 42}
    exp_row = trainer.log_experiment(model_name=model_name, metrics=metrics, params=params)
    print("Experiment logged:", exp_row)

    # (Optional) show a preview of clusters
    print(df_with_clusters[["city", "state", "cluster_label", "lifestyle_score"]].head())


if __name__ == "__main__":
    main()
