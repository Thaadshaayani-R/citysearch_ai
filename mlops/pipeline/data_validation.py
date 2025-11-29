#mlops/pipeline/data_validation.py

import pandas as pd

class DataValidator:
    """
    Validates city dataset for:
    - missing values
    - wrong types
    - outliers
    - invalid ranges
    - duplicates
    """

    def __init__(self):
        # Acceptable ranges for basic sanity checks
        self.population_range = (0, 50_000_000)      # City population realistic range
        self.age_range = (0, 120)                    # Human age
        self.household_size_range = (0, 20)          # Typical household sizes

    def validate(self, df: pd.DataFrame):
        """Runs all validation checks and returns a report."""
        report = {
            "missing_values": self.check_missing(df),
            "invalid_types": self.check_types(df),
            "out_of_range": self.check_ranges(df),
            "duplicates": self.check_duplicates(df),
            "status": "PASSED"
        }

        # If any part fails, mark status
        for key, value in report.items():
            if key != "status" and len(value) > 0:
                report["status"] = "FAILED"

        return report

    def check_missing(self, df):
        """Returns columns with missing values."""
        missing = df.isnull().sum()
        return missing[missing > 0].to_dict()

    def check_types(self, df):
        """Detects wrong data types."""
        errors = {}

        expected_types = {
            "city": str,
            "state": str,
            "state_code": str,
            "population": (int, float),
            "median_age": (int, float),
            "avg_household_size": (int, float)
        }

        for col, expected in expected_types.items():
            if col in df.columns:
                if not df[col].map(lambda x: isinstance(x, expected)).all():
                    errors[col] = "Contains invalid types"

        return errors

    def check_ranges(self, df):
        """Checks values outside realistic ranges."""
        errors = {}

        # Population check
        if "population" in df.columns:
            invalid = df[
                (df["population"] < self.population_range[0]) |
                (df["population"] > self.population_range[1])
            ]
            if not invalid.empty:
                errors["population"] = f"{len(invalid)} values out of range"

        # Age check
        if "median_age" in df.columns:
            invalid = df[
                (df["median_age"] < self.age_range[0]) |
                (df["median_age"] > self.age_range[1])
            ]
            if not invalid.empty:
                errors["median_age"] = f"{len(invalid)} values out of range"

        # Household size check
        if "avg_household_size" in df.columns:
            invalid = df[
                (df["avg_household_size"] < self.household_size_range[0]) |
                (df["avg_household_size"] > self.household_size_range[1])
            ]
            if not invalid.empty:
                errors["avg_household_size"] = f"{len(invalid)} values out of range"

        return errors

    def check_duplicates(self, df):
        """Returns duplicate rows."""
        duplicates = df[df.duplicated()]
        return duplicates.to_dict(orient="records")
