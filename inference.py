
"""
Inference utilities for loading the trained model and predicting new samples.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd

import config


############################################################
# Function name : calculate_sha256
# Description   : Calculate SHA-256 checksum of a saved model
# Input         : model file path
# Output        : hexadecimal checksum
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


############################################################
# Function name : load_artifacts
# Description   : Load and validate the saved model and metadata
# Input         : model and metadata file paths
# Output        : fitted pipeline and metadata dictionary
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def load_artifacts(
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> Tuple[object, Dict]:
    """Load the trained pipeline after verifying its metadata and checksum."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_metadata = {
        "model_sha256",
        "model_file",
        "feature_order",
        "selected_features",
        "class_labels",
        "class_names",
        "training_ranges",
    }

    missing = required_metadata.difference(metadata)

    if missing:
        raise ValueError(
            f"Model metadata is missing fields: {sorted(missing)}"
        )

    if metadata["model_file"] != model_path.name:
        raise ValueError(
            "Model filename does not match the metadata."
        )

    if metadata["feature_order"] != config.FEATURE_COLUMNS:
        raise ValueError(
            "Model feature order does not match the current configuration."
        )

    if metadata["class_labels"] != config.CLASS_LABELS:
        raise ValueError(
            "Model class labels do not match the current configuration."
        )

    actual_hash = calculate_sha256(model_path)

    if actual_hash != metadata["model_sha256"]:
        raise ValueError(
            "Model checksum verification failed. "
            "The model file may have changed."
        )

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "predict"):
        raise TypeError("Loaded model does not support prediction.")

    if not hasattr(pipeline, "classes_"):
        raise ValueError("Loaded pipeline has no fitted class information.")

    if list(pipeline.classes_) != config.CLASS_LABELS:
        raise ValueError(
            "Loaded model classes do not match the expected class labels."
        )

    selected_features = metadata["selected_features"]

    if (
        not isinstance(selected_features, list)
        or not selected_features
        or len(selected_features) != len(set(selected_features))
        or not set(selected_features).issubset(config.FEATURE_COLUMNS)
    ):
        raise ValueError("Invalid selected feature information in metadata.")

    if metadata["n_selected_features"] != len(selected_features):
        raise ValueError(
            "Selected feature count does not match metadata."
        )

    if set(metadata["training_ranges"]) != set(config.FEATURE_COLUMNS):
        raise ValueError(
            "Training feature ranges are incomplete or inconsistent."
        )

    for feature, limits in metadata["training_ranges"].items():
        if not isinstance(limits, dict) or not {"min", "max"}.issubset(limits):
            raise ValueError(
                f"Invalid training range metadata for {feature}."
            )

        minimum = limits["min"]
        maximum = limits["max"]

        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or minimum > maximum
        ):
            raise ValueError(
                f"Invalid training range values for {feature}."
            )

    return pipeline, metadata


############################################################
# Function name : validate_input
# Description   : Validate and arrange new wine sample features
# Input         : dictionary or pandas DataFrame
# Output        : validated DataFrame in configured feature order
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def validate_input(
    samples: Union[Dict, pd.DataFrame],
) -> pd.DataFrame:
    """Validate incoming samples and arrange features in training order."""
    if isinstance(samples, dict):
        samples = pd.DataFrame([samples])

    elif not isinstance(samples, pd.DataFrame):
        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    if samples.empty:
        raise ValueError("At least one sample is required.")

    if samples.columns.duplicated().any():
        raise ValueError("Input contains duplicate feature columns.")

    missing = set(config.FEATURE_COLUMNS).difference(samples.columns)
    extra = set(samples.columns).difference(config.FEATURE_COLUMNS)

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected input columns: {sorted(extra)}"
        )

    samples = samples[config.FEATURE_COLUMNS].copy()

    for feature in config.FEATURE_COLUMNS:
        samples[feature] = pd.to_numeric(
            samples[feature],
            errors="raise",
        )

    values = samples.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains missing or non-finite numerical values."
        )

    return samples.astype(float)


############################################################
# Function name : detect_out_of_range_features
# Description   : Identify values outside the model training ranges
# Input         : validated samples and model metadata
# Output        : DataFrame containing out-of-range feature information
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def detect_out_of_range_features(
    samples: pd.DataFrame,
    metadata: Dict,
) -> pd.DataFrame:
    """Flag feature values outside the observed training ranges."""
    records = []

    training_ranges = metadata["training_ranges"]

    for row_index, row in samples.iterrows():
        for feature in config.FEATURE_COLUMNS:
            value = float(row[feature])
            limits = training_ranges[feature]

            minimum = float(limits["min"])
            maximum = float(limits["max"])

            if value < minimum or value > maximum:
                records.append(
                    {
                        "sample_index": row_index,
                        "feature": feature,
                        "value": value,
                        "training_min": minimum,
                        "training_max": maximum,
                        "range_status": "Outside training range",
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "sample_index",
            "feature",
            "value",
            "training_min",
            "training_max",
            "range_status",
        ],
    )


############################################################
# Function name : predict_samples
# Description   : Generate class predictions and probabilities
# Input         : fitted pipeline, metadata and input samples
# Output        : prediction DataFrame and range warnings DataFrame
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def predict_samples(
    pipeline,
    metadata: Dict,
    samples: Union[Dict, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predict cultivar classes for one or more wine samples."""
    X = validate_input(samples)

    # Pass all original features to the fitted pipeline.
    # StandardScaler was fitted before SelectKBest, so feature
    # selection must happen inside the pipeline.
    model_input = X

    predictions = pipeline.predict(model_input)

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "The loaded model does not support probability predictions."
        )

    probabilities = pipeline.predict_proba(model_input)
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape != (
        len(X),
        len(config.CLASS_LABELS),
    ):
        raise ValueError(
            "Model probability output has an unexpected shape."
        )

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Model returned invalid class probabilities.")

    class_names = metadata["class_names"]

    results = pd.DataFrame(
        {
            "predicted_class": predictions.astype(int),
            "predicted_cultivar": [
                class_names[str(int(label))]
                for label in predictions
            ],
            "confidence_percent": (
                probabilities.max(axis=1) * 100
            ).round(2),
        },
        index=X.index,
    )

    for index, label in enumerate(config.CLASS_LABELS):
        results[f"probability_class_{label}_percent"] = (
            probabilities[:, index] * 100
        ).round(2)

    range_warnings = detect_out_of_range_features(
        X,
        metadata,
    )

    return results, range_warnings


############################################################
# Function name : predict_single
# Description   : Predict the cultivar of one wine sample
# Input         : fitted pipeline, metadata and sample dictionary
# Output        : prediction result and range warnings
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def predict_single(
    pipeline,
    metadata: Dict,
    sample: Dict,
) -> Tuple[Dict, pd.DataFrame]:
    """Generate a prediction for a single wine sample."""
    results, warnings = predict_samples(
        pipeline,
        metadata,
        sample,
    )

    return results.iloc[0].to_dict(), warnings


############################################################
# Function name : predict_batch
# Description   : Predict cultivar classes for multiple wine samples
# Input         : fitted pipeline, metadata and sample DataFrame
# Output        : prediction DataFrame and range warnings
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def predict_batch(
    pipeline,
    metadata: Dict,
    samples: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions for a batch of wine samples."""
    return predict_samples(
        pipeline,
        metadata,
        samples,
    )
