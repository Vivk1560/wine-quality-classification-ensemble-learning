"""
Centralised configuration for the Wine Classification project.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "WinePredictor.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "wine_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = {
    1: "Cultivar 1",
    2: "Cultivar 2",
    3: "Cultivar 3",
}


# ---------------------------------------------------------------------------
# Data splitting and cross-validation
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25

CV_N_SPLITS = 5
CV_N_REPEATS = 3
CV_RANDOM_STATE = 42

TUNING_CV_SPLITS = 5
TUNING_CV_RANDOM_STATE = 43

NESTED_CV_SPLITS = 5
NESTED_CV_RANDOM_STATE = 44


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "f1_macro"
FEATURE_COUNTS = [13, 10, 7, 5]
BASELINE_FAMILY = "dummy_majority"

SELECTION_FAMILIES = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "voting_ensemble",
]

FAMILY_COMPLEXITY = {
    "dummy_majority": 0,
    "logistic_regression": 1,
    "decision_tree": 1,
    "random_forest": 2,
    "extra_trees": 2,
    "gradient_boosting": 3,
    "voting_ensemble": 4,
}

FAMILY_LABELS = {
    "dummy_majority": "Majority-class baseline",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "voting_ensemble": "Soft Voting (LR + RF + ET)",
}

CV_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "roc_auc_ovr": "roc_auc_ovr",
}


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "random_forest": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5],
        "clf__min_samples_leaf": [1, 2],
    },
    "extra_trees": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5],
        "clf__min_samples_leaf": [1, 2],
    },
    "gradient_boosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [1, 2, 3],
    },
    "voting_ensemble": {
        "clf__lr__C": [0.1, 1.0, 10.0],
    },
}


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
RANKING_RF_TREES = 500


# ---------------------------------------------------------------------------
# Historical Assignment 41
# ---------------------------------------------------------------------------
HISTORICAL_TEST_SIZE = 0.4
HISTORICAL_RANDOM_STATE = 42
HISTORICAL_RF_TREES = 7

HISTORICAL_FEATURE_SETS = {
    "All 13 features": FEATURE_COLUMNS,
    "Top 7 features": [
        "Flavanoids",
        "OD280/OD315 of diluted wines",
        "Alcohol",
        "Alcalinity of ash",
        "Proline",
        "Total phenols",
        "Color intensity",
    ],
    "Top 5 features": [
        "Flavanoids",
        "OD280/OD315 of diluted wines",
        "Alcohol",
        "Proline",
        "Alcalinity of ash",
    ],
    "Selected 10 features": [
        "Alcohol",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ],
}

HISTORICAL_DOCUMENTED_ACCURACY = {
    "All 13 features": 97.22,
    "Top 7 features": 94.44,
    "Top 5 features": 90.28,
    "Selected 10 features": 98.61,
}