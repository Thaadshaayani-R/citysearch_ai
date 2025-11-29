#scripts/run_validation.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops.pipeline.data_validation import DataValidator
from db_config import get_connection
import pandas as pd

# Step 1 — Connect to SQL Server
conn = get_connection()

# Step 2 — Load data directly from SQL
query = "SELECT * FROM [dbo].[cities]"
df = pd.read_sql(query, conn)

# Step 3 — Run validation
validator = DataValidator()
report = validator.validate(df)

print("VALIDATION REPORT:")
print(report)
