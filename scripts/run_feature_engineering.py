#scripts/run_feature_engineering

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_config import get_connection
from mlops.pipeline.feature_engineering import FeatureEngineer
import pandas as pd

conn = get_connection()

df = pd.read_sql("SELECT * FROM [dbo].[cities]", conn)

engineer = FeatureEngineer()
new_df = engineer.engineer_features(df)

print(new_df.head())
print(new_df.columns)
