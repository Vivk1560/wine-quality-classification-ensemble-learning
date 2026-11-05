"""
Dataset loading, schema validation, splitting and summary statistics.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


class DataValidationError(ValueError):
    """Raised when the dataset does not satisfy the expected schema."""


############################################################
# Function name : read_data
# Description   : Read the wine CSV into a pandas DataFrame
# Input         : path to the CSV file
# Output        : raw DataFrame
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def read_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the dataset from the specified path."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


############################################################
# Function name : validate_dataset
# Description   : Validate columns, data types, values and labels
# Input         : DataFrame
# Output        : validated DataFrame
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataset schema and values."""
    if df is None or df.empty:
        raise DataValidationError("Dataset is empty.")

    if not df.columns.is_unique:
        raise DataValidationError("Dataset contains duplicate column names.")

    expected = [config.TARGET_COLUMN] + config.FEATURE_COLUMNS

    missing = [column for column in expected if column not in df.columns]
    extra = [column for column in df.columns if column not in expected]

    if missing or extra:
        raise DataValidationError(
            f"Column mismatch. Missing: {missing}; unexpected: {extra}"
        )
    
    if df.duplicated().any():
        raise DataValidationError("Dataset contains duplicate rows.")

    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(df[column])
        or pd.api.types.is_bool_dtype(df[column])
    ]

    if non_numeric:
        raise DataValidationError(
            f"Invalid data types in columns: {non_numeric}"
        )

    if df[expected].isna().any().any():
        raise DataValidationError("Dataset contains missing values.")

    if not np.isfinite(df[expected].to_numpy(dtype=float)).all():
        raise DataValidationError("Dataset contains non-finite values.")

    labels = set(df[config.TARGET_COLUMN].unique())

    if not labels.issubset(config.CLASS_LABELS):
        raise DataValidationError(
            f"Unexpected class labels: "
            f"{sorted(labels - set(config.CLASS_LABELS))}"
        )

    if labels != set(config.CLASS_LABELS):
        raise DataValidationError(
            f"Classes absent from data: "
            f"{sorted(set(config.CLASS_LABELS) - labels)}"
        )

    return df


############################################################
# Function name : split_features_target
# Description   : Separate features and target columns
# Input         : validated DataFrame
# Output        : X and y
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature columns and target column."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()

    return X, y


############################################################
# Function name : make_train_test_split
# Description   : Create a stratified train-test split
# Input         : X, y, test_size and random_state
# Output        : X_train, X_test, y_train, y_test
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


############################################################
# Function name : summarize_dataset
# Description   : Generate dataset quality and statistical summary
# Input         : validated DataFrame
# Output        : JSON-serialisable dictionary
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def summarize_dataset(df: pd.DataFrame) -> Dict:
    """Generate dataset statistics and quality checks."""
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    features = df[config.FEATURE_COLUMNS]

    q1 = features.quantile(0.25)
    q3 = features.quantile(0.75)
    iqr = q3 - q1

    outside = (
        (features < q1 - 1.5 * iqr)
        | (features > q3 + 1.5 * iqr)
    )

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target": config.TARGET_COLUMN,
        "features": config.FEATURE_COLUMNS,
        "class_counts": {
            str(int(key)): int(value)
            for key, value in counts.items()
        },
        "class_percent": {
            str(int(key)): round(float(value) / len(df) * 100, 2)
            for key, value in counts.items()
        },
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "iqr_outlier_counts": {
            column: int(outside[column].sum())
            for column in config.FEATURE_COLUMNS
        },
        "iqr_outlier_note": (
            "Descriptive IQR outlier counts. No observations removed."
        ),
        "feature_describe": features.describe().T.round(4).to_dict(
            orient="index"
        ),
    }