"""Load, clean, and preprocess the Pima diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
TARGET_COLUMN = "Outcome"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# A zero in these columns is physically impossible and really means "missing".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
EXPECTED_COLUMNS = FEATURES + [TARGET_COLUMN]


def load_clean_data(
    data_path: Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, int]:
    """Load the raw CSV, drop duplicates, and mark impossible zeros as missing."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find the dataset at {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    duplicate_count = int(data.duplicated().sum())
    clean_data = data.drop_duplicates().reset_index(drop=True)
    clean_data[ZERO_AS_MISSING] = clean_data[ZERO_AS_MISSING].replace(0, np.nan)
    return clean_data, duplicate_count


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the clinical features X and diabetes target y."""
    X = data[FEATURES].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> Pipeline:
    """Impute the missing values with the median, then standardize."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
