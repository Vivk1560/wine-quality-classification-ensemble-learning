"""
Pipelines, cross-validated comparison, model selection, tuning and persistence.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from feature_selection import build_selector, get_selected_features


############################################################
# Function name : build_estimator
# Description   : Create an unfitted estimator for a model family
# Input         : family key from config.FAMILY_LABELS
# Output        : scikit-learn classifier
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def build_estimator(family: str):
    """Return the estimator for the specified model family."""
    seed = config.RANDOM_STATE

    if family == "dummy_majority":
        return DummyClassifier(strategy="prior")

    if family == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed)

    if family == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)

    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=1,
        )

    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=1,
        )

    if family == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)

    if family == "voting_ensemble":
        return VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(max_iter=2000, random_state=seed),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ],
            voting="soft",
        )

    raise ValueError(f"Unknown model family: {family!r}")


############################################################
# Function name : build_pipeline
# Description   : Create scaler, selector and classifier pipeline
# Input         : model family and number of features k
# Output        : unfitted sklearn Pipeline
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def build_pipeline(family: str, k: int) -> Pipeline:
    """Build the complete pipeline for a model candidate."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", build_selector(k)),
            ("clf", build_estimator(family)),
        ]
    )


############################################################
# Function name : run_cv_comparison
# Description   : Compare model families and feature counts using CV
# Input         : X_train, y_train and number of repeats
# Output        : DataFrame containing CV metrics
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_cv_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_repeats: int = config.CV_N_REPEATS,
) -> pd.DataFrame:
    """Cross-validate all candidates using training data only."""
    if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer.")

    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_N_SPLITS,
        n_repeats=n_repeats,
        random_state=config.CV_RANDOM_STATE,
    )

    candidates = [
        (config.BASELINE_FAMILY, len(config.FEATURE_COLUMNS))
    ]

    for family in config.FAMILY_LABELS:
        if family == config.BASELINE_FAMILY:
            continue

        candidates.extend(
            (family, k) for k in config.FEATURE_COUNTS
        )

    rows = []

    for family, k in candidates:
        scores = cross_validate(
            build_pipeline(family, k),
            X_train,
            y_train,
            cv=cv,
            scoring=config.CV_SCORING,
            n_jobs=1,
            error_score="raise",
        )

        row = {
            "family": family,
            "model": config.FAMILY_LABELS[family],
            "n_features": k,
            "is_selection_candidate": (
                family in config.SELECTION_FAMILIES
            ),
            "n_folds": int(len(scores["fit_time"])),
        }

        for name in config.CV_SCORING:
            values = scores[f"test_{name}"]

            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(
                np.std(values, ddof=1)
            )

        rows.append(row)

    return pd.DataFrame(rows)


############################################################
# Function name : select_best_candidate
# Description   : Select the best ensemble candidate using CV
# Input         : DataFrame returned by run_cv_comparison
# Output        : dictionary containing the selected candidate
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def select_best_candidate(cv_results: pd.DataFrame) -> Dict:
    """Select the highest-scoring ensemble candidate using CV results."""
    metric_mean = f"{config.PRIMARY_METRIC}_mean"
    metric_std = f"{config.PRIMARY_METRIC}_std"

    required_columns = {
        "family",
        "n_features",
        metric_mean,
        metric_std,
    }

    missing = required_columns - set(cv_results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in CV results: {sorted(missing)}"
        )

    pool = cv_results[
        cv_results["family"].isin(config.SELECTION_FAMILIES)
    ].copy()

    if pool.empty:
        raise ValueError(
            "No selection candidates present in cv_results."
        )

    if pool[metric_mean].isna().any():
        raise ValueError("CV results contain missing primary metric values.")

    best = pool.sort_values(
        [metric_mean, "n_features"],
        ascending=[False, True],
    ).iloc[0]

    pool["complexity"] = (
        pool["family"].map(config.FAMILY_COMPLEXITY)
    )

    if pool["complexity"].isna().any():
        raise ValueError(
            "Model complexity is missing for a selection candidate."
        )

    ranked = pool.sort_values(
        ["f1_macro_mean", "n_features", "complexity"],
        ascending=[False, True, True],
    )

    chosen = ranked.iloc[0]

    return {
        "family": str(chosen["family"]),
        "n_features": int(chosen["n_features"]),
        "cv_mean": float(chosen[metric_mean]),
        "cv_std": float(chosen[metric_std]),
        "best_raw_family": str(best["family"]),
        "best_raw_n_features": int(best["n_features"]),
        "best_raw_cv_mean": float(best[metric_mean]),
        "candidate_count": int(len(pool)),
        "selection_method": (
            "Highest mean CV macro-F1, with fewer features and "
            "simpler model family used as tie-breakers."
        ),
    }


############################################################
# Function name : tune_pipeline
# Description   : Tune the selected pipeline using GridSearchCV
# Input         : X_train, y_train, family and k
# Output        : fitted GridSearchCV
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def tune_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family: str,
    k: int,
) -> GridSearchCV:
    """Tune hyperparameters and refit the best pipeline on training data."""
    if family not in config.PARAM_GRIDS:
        raise ValueError(
            f"No hyperparameter grid defined for {family!r}."
        )

    search = GridSearchCV(
        build_pipeline(family, k),
        param_grid=config.PARAM_GRIDS[family],
        scoring=config.PRIMARY_METRIC,
        cv=StratifiedKFold(
            n_splits=config.TUNING_CV_SPLITS,
            shuffle=True,
            random_state=config.TUNING_CV_RANDOM_STATE,
        ),
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    return search.fit(X_train, y_train)


############################################################
# Function name : nested_cv_estimate
# Description   : Evaluate hyperparameter tuning using outer CV
# Input         : X_train, y_train, family and k
# Output        : dictionary containing nested CV metrics
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def nested_cv_estimate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family: str,
    k: int,
) -> Dict:
    """Estimate the tuning procedure using nested cross-validation."""
    if family not in config.PARAM_GRIDS:
        raise ValueError(
            f"No hyperparameter grid defined for {family!r}."
        )

    search = GridSearchCV(
        build_pipeline(family, k),
        param_grid=config.PARAM_GRIDS[family],
        scoring=config.PRIMARY_METRIC,
        cv=StratifiedKFold(
            n_splits=config.TUNING_CV_SPLITS,
            shuffle=True,
            random_state=config.TUNING_CV_RANDOM_STATE,
        ),
        n_jobs=1,
        error_score="raise",
    )

    outer = StratifiedKFold(
        n_splits=config.NESTED_CV_SPLITS,
        shuffle=True,
        random_state=config.NESTED_CV_RANDOM_STATE,
    )

    scores = cross_validate(
        search,
        X_train,
        y_train,
        cv=outer,
        scoring={
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
        },
        n_jobs=1,
        error_score="raise",
    )

    return {
        "f1_macro_mean": float(
            np.mean(scores["test_f1_macro"])
        ),
        "f1_macro_std": float(
            np.std(scores["test_f1_macro"], ddof=1)
        ),
        "accuracy_mean": float(
            np.mean(scores["test_accuracy"])
        ),
        "accuracy_std": float(
            np.std(scores["test_accuracy"], ddof=1)
        ),
        "outer_folds": int(config.NESTED_CV_SPLITS),
        "scope": (
            "Conditional on the previously selected model family "
            "and feature count. Does not evaluate the complete "
            "model-selection procedure."
        ),
    }


############################################################
# Function name : calculate_sha256
# Description   : Calculate the SHA-256 checksum of a file
# Input         : file path
# Output        : hexadecimal SHA-256 digest
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


############################################################
# Function name : build_metadata
# Description   : Generate metadata for inference and dashboard
# Input         : fitted pipeline, training data and run details
# Output        : JSON-serialisable dictionary
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def build_metadata(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family: str,
    best_params: Dict,
    n_test_rows: int,
    model_filename: str,
) -> Dict:
    """Collect model details and training feature ranges."""
    selected = get_selected_features(pipeline)

    ranges = {
        column: {
            "min": float(X_train[column].min()),
            "max": float(X_train[column].max()),
            "mean": float(X_train[column].mean()),
            "std": float(X_train[column].std()),
        }
        for column in config.FEATURE_COLUMNS
    }

    return {
        "project": (
            "Wine Classification Using Ensemble Learning "
            "and Robust Model Evaluation"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "model_family": family,
        "model_label": config.FAMILY_LABELS[family],
        "best_params": best_params,
        "feature_order": list(config.FEATURE_COLUMNS),
        "selected_features": selected,
        "n_selected_features": len(selected),
        "class_labels": [
            int(label) for label in pipeline.classes_
        ],
        "class_names": {
            str(key): value
            for key, value in config.CLASS_NAMES.items()
        },
        "training_ranges": ranges,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(n_test_rows),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "model_file": model_filename,
        "model_sha256": None,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }


############################################################
# Function name : save_pipeline
# Description   : Save the fitted pipeline and metadata
# Input         : pipeline, metadata and output paths
# Output        : model path and metadata path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def save_pipeline(
    pipeline: Pipeline,
    metadata: Dict,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
):
    """Save the fitted pipeline and its metadata."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    metadata = dict(metadata)
    metadata["model_sha256"] = calculate_sha256(model_path)
    metadata["model_file"] = model_path.name

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return model_path, metadata_path