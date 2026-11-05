"""
Multiclass metrics, holdout evaluation and report writers.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


############################################################
# Function name : compute_metrics
# Description   : Calculate multiclass classification metrics
# Input         : y_true, y_pred, optional y_proba and class labels
# Output        : dictionary containing classification metrics
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: Optional[np.ndarray] = None,
    labels: Sequence[int] = tuple(config.CLASS_LABELS),
) -> Dict[str, Optional[float]]:
    """Calculate multiclass classification metrics."""
    labels = list(labels)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc_ovr": None,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        y_true_array = np.asarray(y_true)

        valid = (
            y_proba.ndim == 2
            and y_proba.shape == (len(y_true_array), len(labels))
            and np.isfinite(y_proba).all()
            and np.all(y_proba >= 0)
            and np.all(y_proba <= 1)
            and np.allclose(
                y_proba.sum(axis=1),
                1.0,
                atol=1e-6,
            )
            and set(np.unique(y_true_array)) == set(labels)
        )

        if valid:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true_array,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )

    return metrics


############################################################
# Function name : evaluate_on_holdout
# Description   : Evaluate the final pipeline on the test set
# Input         : fitted pipeline, X_test and y_test
# Output        : dictionary containing metrics and predictions
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def evaluate_on_holdout(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """Evaluate the final pipeline on the untouched holdout set."""
    if X_test.empty or len(y_test) == 0:
        raise ValueError("Holdout data cannot be empty.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have equal lengths.")

    labels = [int(label) for label in pipeline.classes_]

    y_pred = pipeline.predict(X_test)

    y_proba = (
        pipeline.predict_proba(X_test)
        if hasattr(pipeline, "predict_proba")
        else None
    )

    if y_proba is not None and y_proba.shape[1] != len(labels):
        raise ValueError("Prediction probability columns do not match classes.")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[
            config.CLASS_NAMES[label]
            for label in labels
        ],
        output_dict=True,
        zero_division=0,
    )

    return {
        "metrics": compute_metrics(
            y_test,
            y_pred,
            y_proba,
            labels,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=labels,
        ).tolist(),
        "labels": labels,
        "class_names": [
            config.CLASS_NAMES[label]
            for label in labels
        ],
        "classification_report": report,
        "n_test_rows": int(len(y_test)),
        "y_true": np.asarray(y_test).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_proba": (
            None
            if y_proba is None
            else np.asarray(y_proba).tolist()
        ),
    }


############################################################
# Function name : bootstrap_confidence_intervals
# Description   : Estimate confidence intervals for holdout metrics
# Input         : y_true, y_pred, number of bootstrap samples and seed
# Output        : dictionary containing metric confidence intervals
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def bootstrap_confidence_intervals(
    y_true: Sequence,
    y_pred: Sequence,
    n_bootstrap: int = 2000,
    random_state: int = config.RANDOM_STATE,
) -> Dict:
    """Estimate percentile confidence intervals using bootstrap sampling."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Valid, equally sized evaluation arrays are required.")

    if (
        not isinstance(n_bootstrap, int)
        or isinstance(n_bootstrap, bool)
        or n_bootstrap < 100
    ):
        raise ValueError("n_bootstrap must be an integer of at least 100.")

    rng = np.random.default_rng(random_state)

    labels = list(config.CLASS_LABELS)
    accuracy_values = []
    f1_values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), size=len(y_true))

        true_sample = y_true[indices]
        pred_sample = y_pred[indices]

        accuracy_values.append(
            accuracy_score(true_sample, pred_sample)
        )

        f1_values.append(
            f1_score(
                true_sample,
                pred_sample,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    def calculate_interval(values):
        lower, upper = np.percentile(values, [2.5, 97.5])

        return {
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": 0.95,
        }

    return {
        "method": "Non-stratified percentile bootstrap",
        "n_bootstrap": n_bootstrap,
        "accuracy": calculate_interval(accuracy_values),
        "f1_macro": calculate_interval(f1_values),
        "interpretation": (
            "Intervals describe sampling uncertainty in the holdout "
            "metrics and are not external validation estimates."
        ),
    }


############################################################
# Function name : save_json
# Description   : Write a dictionary to a JSON file
# Input         : dictionary and destination path
# Output        : destination path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def save_json(obj: Dict, path: Path) -> Path:
    """Write a dictionary as an indented JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)

    return path


############################################################
# Function name : save_csv
# Description   : Write a DataFrame to a CSV file
# Input         : DataFrame and destination path
# Output        : destination path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame as a CSV file without its index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    return path