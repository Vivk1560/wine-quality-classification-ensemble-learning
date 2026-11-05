
"""
Reproduction and sensitivity analysis of historical Assignment 41 results.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config


############################################################
# Function name : validate_historical_inputs
# Description   : Validate features and target for historical experiments
# Input         : feature DataFrame and target Series
# Output        : validated inputs
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def validate_historical_inputs(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Validate the dataset inputs used by historical experiments."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y must be a pandas Series.")

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must contain exactly one target column.")
        y = y.iloc[:, 0]

    if X.empty or len(y) == 0:
        raise ValueError("Historical experiment inputs cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if list(X.columns) != config.FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the expected schema.")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Historical experiment inputs contain missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain non-finite values.")

    if not set(y.unique()).issubset(config.CLASS_LABELS):
        raise ValueError("Target contains unexpected class labels.")


############################################################
# Function name : reproduce_historical_results
# Description   : Reproduce the four original Assignment 41 experiments
# Input         : complete feature DataFrame and target Series
# Output        : DataFrame containing reproduced historical accuracies
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def reproduce_historical_results(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Reproduce Assignment 41 using its documented experimental setup."""
    validate_historical_inputs(X, y)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=config.HISTORICAL_TEST_SIZE,
        random_state=config.HISTORICAL_RANDOM_STATE,
    )

    results = []

    for experiment, selected_features in config.HISTORICAL_FEATURE_SETS.items():
        missing = [
            feature
            for feature in selected_features
            if feature not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Unknown features in {experiment}: {missing}"
            )

        model = RandomForestClassifier(
            n_estimators=config.HISTORICAL_RF_TREES,
            random_state=config.HISTORICAL_RANDOM_STATE,
        )

        model.fit(
            X_train[selected_features],
            y_train,
        )

        predictions = model.predict(
            X_test[selected_features]
        )

        reproduced_accuracy = accuracy_score(
            y_test,
            predictions,
        ) * 100

        documented_accuracy = config.HISTORICAL_DOCUMENTED_ACCURACY[
            experiment
        ]

        results.append(
            {
                "experiment": experiment,
                "n_features": len(selected_features),
                "reproduced_accuracy_percent": round(
                    reproduced_accuracy,
                    2,
                ),
                "documented_accuracy_percent": documented_accuracy,
                "difference_percentage_points": round(
                    reproduced_accuracy - documented_accuracy,
                    2,
                ),
                "matches_documented": bool(
                    np.isclose(
                        reproduced_accuracy,
                        documented_accuracy,
                        atol=0.01,
                    )
                ),
            }
        )

    return pd.DataFrame(results)


############################################################
# Function name : run_seed_sensitivity
# Description   : Evaluate historical experiments across random seeds
# Input         : features, target and iterable of random seeds
# Output        : DataFrame containing accuracy for every seed
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_seed_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    seeds: Iterable[int] = range(50),
) -> pd.DataFrame:
    """Measure how historical test accuracy changes across random splits."""
    validate_historical_inputs(X, y)

    seeds = list(seeds)

    if not seeds:
        raise ValueError("At least one random seed is required.")

    if any(
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        for seed in seeds
    ):
        raise ValueError("Every seed must be an integer.")

    results = []

    for seed in seeds:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=config.HISTORICAL_TEST_SIZE,
            random_state=int(seed),
        )

        for experiment, selected_features in config.HISTORICAL_FEATURE_SETS.items():
            model = RandomForestClassifier(
                n_estimators=config.HISTORICAL_RF_TREES,
                random_state=config.HISTORICAL_RANDOM_STATE,
            )

            model.fit(
                X_train[selected_features],
                y_train,
            )

            predictions = model.predict(
                X_test[selected_features]
            )

            accuracy = accuracy_score(
                y_test,
                predictions,
            ) * 100

            results.append(
                {
                    "seed": int(seed),
                    "experiment": experiment,
                    "n_features": len(selected_features),
                    "accuracy_percent": round(accuracy, 2),
                }
            )

    return pd.DataFrame(results)


############################################################
# Function name : summarize_seed_sensitivity
# Description   : Summarise historical accuracy variation across seeds
# Input         : seed sensitivity results DataFrame
# Output        : summary DataFrame with descriptive statistics
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def summarize_seed_sensitivity(
    sensitivity_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate descriptive accuracy statistics across random seeds."""
    required_columns = {
        "seed",
        "experiment",
        "accuracy_percent",
    }

    if not required_columns.issubset(sensitivity_results.columns):
        raise ValueError(
            "Sensitivity results must contain: "
            f"{sorted(required_columns)}"
        )

    if sensitivity_results.empty:
        raise ValueError("Sensitivity results cannot be empty.")

    summary = (
        sensitivity_results
        .groupby("experiment")["accuracy_percent"]
        .agg(
            mean_accuracy="mean",
            std_accuracy="std",
            minimum_accuracy="min",
            median_accuracy="median",
            maximum_accuracy="max",
            number_of_seeds="count",
        )
        .reset_index()
    )

    numeric_columns = [
        "mean_accuracy",
        "std_accuracy",
        "minimum_accuracy",
        "median_accuracy",
        "maximum_accuracy",
    ]

    summary[numeric_columns] = summary[numeric_columns].round(2)

    summary["number_of_seeds"] = (
        summary["number_of_seeds"].astype(int)
    )

    return summary
