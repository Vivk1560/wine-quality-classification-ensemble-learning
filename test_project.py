
"""
Unit and integration tests for the Wine Classification project.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_wine

import config

from data_loader import (
    DataValidationError,
    validate_dataset,
    split_features_target,
    make_train_test_split,
)

from feature_selection import (
    build_selector,
    get_selected_features,
    rank_features,
)

from model_training import (
    build_estimator,
    build_pipeline,
    select_best_candidate,
)

from evaluation import (
    compute_metrics,
    evaluate_on_holdout,
    bootstrap_confidence_intervals,
)

from historical import (
    reproduce_historical_results,
    run_seed_sensitivity,
    summarize_seed_sensitivity,
)

from inference import (
    validate_input,
    detect_out_of_range_features,
    predict_samples,
)


############################################################
# Fixtures
############################################################
@pytest.fixture(scope="module")
def wine_dataframe():
    """Load the Wine dataset using the sklearn reference dataset."""
    dataset = load_wine()

    df = pd.DataFrame(
        dataset.data,
        columns=config.FEATURE_COLUMNS,
    )

    df[config.TARGET_COLUMN] = dataset.target + 1

    return df


@pytest.fixture(scope="module")
def wine_features_target(wine_dataframe):
    """Return the feature matrix and target series."""
    return split_features_target(wine_dataframe)


@pytest.fixture(scope="module")
def wine_split(wine_features_target):
    """Return a stratified training and holdout split."""
    X, y = wine_features_target

    return make_train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )


@pytest.fixture(scope="module")
def fitted_pipeline(wine_split):
    """Fit a lightweight classification pipeline for testing."""
    X_train, X_test, y_train, y_test = wine_split

    pipeline = build_pipeline(
        family="random_forest",
        k=5,
    )

    pipeline.fit(X_train, y_train)

    return pipeline


############################################################
# Function name : test_dataset_validation
# Description   : Test dataset schema and data validation
# Input         : Wine dataset fixture
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_dataset_validation(wine_dataframe):
    validated = validate_dataset(wine_dataframe.copy())

    assert not validated.empty
    assert config.TARGET_COLUMN in validated.columns
    assert set(config.FEATURE_COLUMNS).issubset(validated.columns)


def test_dataset_validation_missing_column(wine_dataframe):
    invalid_df = wine_dataframe.drop(
        columns=[config.FEATURE_COLUMNS[0]]
    )

    with pytest.raises(DataValidationError):
        validate_dataset(invalid_df)


def test_dataset_validation_missing_values(wine_dataframe):
    invalid_df = wine_dataframe.copy()

    invalid_df.loc[0, config.FEATURE_COLUMNS[0]] = np.nan

    with pytest.raises(DataValidationError):
        validate_dataset(invalid_df)


def test_dataset_validation_duplicate_rows(wine_dataframe):
    invalid_df = pd.concat(
        [wine_dataframe, wine_dataframe.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DataValidationError):
        validate_dataset(invalid_df)


############################################################
# Function name : test_feature_target_split
# Description   : Test feature and target separation
# Input         : Wine dataset fixture
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_feature_target_split(wine_dataframe):
    X, y = split_features_target(wine_dataframe)

    assert list(X.columns) == config.FEATURE_COLUMNS
    assert y.name == config.TARGET_COLUMN
    assert len(X) == len(y)
    assert config.TARGET_COLUMN not in X.columns


############################################################
# Function name : test_stratified_split
# Description   : Test train-test split dimensions and class coverage
# Input         : Wine features and target
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_stratified_split(wine_split):
    X_train, X_test, y_train, y_test = wine_split

    assert len(X_train) + len(X_test) == 178
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    assert set(y_train.unique()) == set(config.CLASS_LABELS)
    assert set(y_test.unique()) == set(config.CLASS_LABELS)


############################################################
# Function name : test_feature_selector
# Description   : Test SelectKBest feature selection
# Input         : Wine training split
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_feature_selector(wine_split):
    X_train, _, y_train, _ = wine_split

    selector = build_selector(5)

    selector.fit(X_train, y_train)

    selected = selector.get_support()

    assert selected.sum() == 5


def test_feature_selector_invalid_count():
    with pytest.raises(ValueError):
        build_selector(0)

    with pytest.raises(ValueError):
        build_selector(len(config.FEATURE_COLUMNS) + 1)


def test_feature_ranking(wine_split):
    X_train, _, y_train, _ = wine_split

    ranking = rank_features(X_train, y_train)

    assert len(ranking) == len(config.FEATURE_COLUMNS)

    assert {
        "feature",
        "anova_f",
        "anova_p_value",
        "rf_importance",
        "anova_rank",
        "rf_rank",
    }.issubset(ranking.columns)

    assert ranking["feature"].is_unique


############################################################
# Function name : test_model_construction
# Description   : Test classifier and pipeline creation
# Input         : None
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
@pytest.mark.parametrize(
    "family",
    [
        "dummy_majority",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "voting_ensemble",
    ],
)
def test_estimator_construction(family):
    estimator = build_estimator(family)

    assert estimator is not None
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict")


def test_invalid_estimator_family():
    with pytest.raises(ValueError):
        build_estimator("unknown_model")


def test_pipeline_prediction(wine_split):
    X_train, X_test, y_train, y_test = wine_split

    pipeline = build_pipeline(
        family="random_forest",
        k=5,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset(
        set(config.CLASS_LABELS)
    )


############################################################
# Function name : test_candidate_selection
# Description   : Test CV candidate selection and tie-breaking
# Input         : synthetic CV results
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_candidate_selection():
    results = pd.DataFrame(
        [
            {
                "family": "random_forest",
                "n_features": 13,
                "f1_macro_mean": 0.95,
                "f1_macro_std": 0.02,
            },
            {
                "family": "extra_trees",
                "n_features": 7,
                "f1_macro_mean": 0.95,
                "f1_macro_std": 0.02,
            },
            {
                "family": "logistic_regression",
                "n_features": 5,
                "f1_macro_mean": 0.91,
                "f1_macro_std": 0.03,
            },
        ]
    )

    selected = select_best_candidate(results)

    assert selected["family"] == "extra_trees"
    assert selected["n_features"] == 7


############################################################
# Function name : test_evaluation_metrics
# Description   : Test classification metric calculations
# Input         : known target and prediction values
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_evaluation_metrics():
    y_true = [1, 2, 3, 1, 2, 3]
    y_pred = [1, 2, 3, 1, 2, 2]

    probabilities = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.70, 0.20],
        ]
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert 0 <= metrics["f1_macro"] <= 1
    assert metrics["roc_auc_ovr"] is not None


def test_holdout_evaluation(fitted_pipeline, wine_split):
    X_train, X_test, y_train, y_test = wine_split

    evaluation = evaluate_on_holdout(
        fitted_pipeline,
        X_test,
        y_test,
    )

    assert evaluation["n_test_rows"] == len(y_test)
    assert len(evaluation["y_pred"]) == len(y_test)
    assert len(evaluation["confusion_matrix"]) == 3


def test_bootstrap_confidence_intervals():
    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    y_pred = [1, 2, 3, 1, 2, 2, 1, 2, 3]

    result = bootstrap_confidence_intervals(
        y_true,
        y_pred,
        n_bootstrap=100,
        random_state=42,
    )

    assert "accuracy" in result
    assert "f1_macro" in result

    for metric in ["accuracy", "f1_macro"]:
        assert result[metric]["lower"] <= result[metric]["upper"]


############################################################
# Function name : test_historical_reproduction
# Description   : Test reproduction of historical experiments
# Input         : Wine feature matrix and target
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_historical_reproduction(wine_features_target):
    X, y = wine_features_target

    results = reproduce_historical_results(X, y)

    assert len(results) == 4

    assert {
        "experiment",
        "reproduced_accuracy_percent",
        "documented_accuracy_percent",
        "difference_percentage_points",
    }.issubset(results.columns)


def test_seed_sensitivity(wine_features_target):
    X, y = wine_features_target

    results = run_seed_sensitivity(
        X,
        y,
        seeds=[42, 43],
    )

    assert len(results) == 8
    assert results["seed"].nunique() == 2

    summary = summarize_seed_sensitivity(results)

    assert len(summary) == 4
    assert "mean_accuracy" in summary.columns


############################################################
# Function name : test_inference_input
# Description   : Test validation of new prediction samples
# Input         : Wine dataset fixture
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_inference_input(wine_dataframe):
    sample = wine_dataframe[config.FEATURE_COLUMNS].iloc[0].to_dict()

    validated = validate_input(sample)

    assert validated.shape == (1, len(config.FEATURE_COLUMNS))
    assert list(validated.columns) == config.FEATURE_COLUMNS


def test_inference_missing_feature(wine_dataframe):
    sample = wine_dataframe[config.FEATURE_COLUMNS].iloc[0].to_dict()

    sample.pop(config.FEATURE_COLUMNS[0])

    with pytest.raises(ValueError):
        validate_input(sample)


def test_inference_invalid_numeric_value(wine_dataframe):
    sample = wine_dataframe[config.FEATURE_COLUMNS].iloc[0].to_dict()

    sample[config.FEATURE_COLUMNS[0]] = "invalid"

    with pytest.raises((ValueError, TypeError)):
        validate_input(sample)


def test_out_of_range_detection(wine_dataframe):
    sample = wine_dataframe[config.FEATURE_COLUMNS].iloc[[0]].copy()

    training_ranges = {
        feature: {
            "min": float(wine_dataframe[feature].min()),
            "max": float(wine_dataframe[feature].max()),
        }
        for feature in config.FEATURE_COLUMNS
    }

    metadata = {
        "training_ranges": training_ranges,
    }

    sample.loc[sample.index[0], config.FEATURE_COLUMNS[0]] = (
        training_ranges[config.FEATURE_COLUMNS[0]]["max"] + 100
    )

    warnings = detect_out_of_range_features(
        sample,
        metadata,
    )

    assert not warnings.empty
    assert warnings.iloc[0]["feature"] == config.FEATURE_COLUMNS[0]


############################################################
# Function name : test_end_to_end_prediction
# Description   : Test single and batch inference using a fitted pipeline
# Input         : fitted pipeline and Wine dataset
# Output        : pytest validation results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def test_end_to_end_prediction(
    fitted_pipeline,
    wine_dataframe,
):
    samples = wine_dataframe[config.FEATURE_COLUMNS].iloc[:3].copy()

    training_ranges = {
        feature: {
            "min": float(wine_dataframe[feature].min()),
            "max": float(wine_dataframe[feature].max()),
        }
        for feature in config.FEATURE_COLUMNS
    }

    metadata = {
        "selected_features": config.FEATURE_COLUMNS[:5],
        "training_ranges": training_ranges,
        "class_names": {
            str(label): config.CLASS_NAMES[label]
            for label in config.CLASS_LABELS
        },
    }

    predictions, warnings = predict_samples(
        fitted_pipeline,
        metadata,
        samples,
    )

    assert len(predictions) == 3
    assert "predicted_class" in predictions.columns
    assert "confidence_percent" in predictions.columns
    assert warnings.empty
