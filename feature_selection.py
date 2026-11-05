"""
Leakage-safe feature selection.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

import config


############################################################
# Function name : build_selector
# Description   : Create a top-k feature selector
# Input         : k (number of features to retain)
# Output        : unfitted SelectKBest selector
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def build_selector(k: int) -> SelectKBest:
    """Return an ANOVA selector that retains exactly k features."""
    n_features = len(config.FEATURE_COLUMNS)

    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n_features:
        raise ValueError(
            f"k must be an integer between 1 and {n_features}, got {k!r}"
        )

    return SelectKBest(score_func=f_classif, k=k)


############################################################
# Function name : get_selected_features
# Description   : Recover selected feature names from a fitted pipeline
# Input         : fitted Pipeline containing a selector step
# Output        : list of selected feature names
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def get_selected_features(pipeline: Pipeline) -> List[str]:
    """Return the feature names retained by the fitted selector."""
    if "selector" not in pipeline.named_steps:
        raise ValueError("Pipeline does not contain a selector step.")

    selector = pipeline.named_steps["selector"]

    if not hasattr(selector, "get_feature_names_out"):
        raise ValueError("Selector does not support feature name extraction.")

    names = selector.get_feature_names_out(config.FEATURE_COLUMNS)

    return [str(name) for name in names]


############################################################
# Function name : rank_features
# Description   : Calculate ANOVA and Random Forest feature rankings
# Input         : training features and target
# Output        : DataFrame sorted by Random Forest importance
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def rank_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Rank features using training data for interpretation."""
    if list(X_train.columns) != config.FEATURE_COLUMNS:
        raise ValueError("Training feature columns do not match the expected schema.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")

    f_values, p_values = f_classif(X_train, y_train)

    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)

    forest = RandomForestClassifier(
        n_estimators=config.RANKING_RF_TREES,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )

    forest.fit(X_train, y_train)

    ranking = pd.DataFrame(
        {
            "feature": config.FEATURE_COLUMNS,
            "anova_f": f_values,
            "anova_p_value": p_values,
            "rf_importance": forest.feature_importances_,
        }
    )

    ranking["anova_rank"] = (
        ranking["anova_f"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    ranking["rf_rank"] = (
        ranking["rf_importance"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return (
        ranking
        .sort_values("rf_rank")
        .reset_index(drop=True)
    )