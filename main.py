
"""
Main execution pipeline for Wine Classification using Ensemble Learning.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader import (
    read_data,
    validate_dataset,
    split_features_target,
    make_train_test_split,
    summarize_dataset,
)
from evaluation import (
    evaluate_on_holdout,
    bootstrap_confidence_intervals,
    save_json,
    save_csv,
)
from feature_selection import rank_features
from historical import (
    reproduce_historical_results,
    run_seed_sensitivity,
    summarize_seed_sensitivity,
)
from model_training import (
    run_cv_comparison,
    select_best_candidate,
    tune_pipeline,
    nested_cv_estimate,
    build_metadata,
    save_pipeline,
)
from visualization import (
    plot_class_distribution,
    plot_feature_histograms,
    plot_feature_boxplots,
    plot_class_mean_heatmap,
    plot_correlation_heatmap,
    plot_scatter_pair,
    plot_feature_ranking,
    plot_cv_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_historical_comparison,
)


############################################################
# Function name : make_json_serializable
# Description   : Convert objects into JSON-compatible values
# Input         : Python object
# Output        : JSON-serialisable object
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def make_json_serializable(value):
    """Convert NumPy and pandas objects into standard Python values."""
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_serializable(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if value is pd.NA:
        return None

    return value


############################################################
# Function name : prepare_directories
# Description   : Create model and output directories
# Input         : None
# Output        : None
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def prepare_directories() -> None:
    """Create all directories required by the project."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


############################################################
# Function name : run_data_analysis
# Description   : Load, validate and summarise the wine dataset
# Input         : None
# Output        : DataFrame, X, y and dataset summary
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_data_analysis():
    """Load the dataset and generate descriptive analysis."""
    print("\n[1/9] Loading and validating dataset...")

    df = read_data()
    df = validate_dataset(df)

    X, y = split_features_target(df)
    summary = summarize_dataset(df)

    save_json(
        make_json_serializable(summary),
        config.REPORTS_DIR / "dataset_summary.json",
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"Class distribution: {summary['class_counts']}")

    return df, X, y, summary


############################################################
# Function name : run_eda_visualizations
# Description   : Generate exploratory data analysis figures
# Input         : complete validated dataset
# Output        : dictionary containing generated figure paths
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_eda_visualizations(df: pd.DataFrame) -> dict:
    """Generate descriptive EDA figures."""
    print("\n[2/9] Generating EDA visualizations...")

    figures = {}

    figures["class_distribution"] = plot_class_distribution(
        df,
        config.FIGURES_DIR / "class_distribution.png",
    )

    figures["feature_histograms"] = plot_feature_histograms(
        df,
        config.FIGURES_DIR / "feature_histograms.png",
    )

    figures["feature_boxplots"] = plot_feature_boxplots(
        df,
        config.FIGURES_DIR / "feature_boxplots.png",
    )

    figures["class_mean_heatmap"] = plot_class_mean_heatmap(
        df,
        config.FIGURES_DIR / "class_mean_heatmap.png",
    )

    figures["correlation_heatmap"] = plot_correlation_heatmap(
        df,
        config.FIGURES_DIR / "correlation_heatmap.png",
    )

    print(f"Generated {len(figures)} EDA figures.")

    return figures


############################################################
# Function name : run_feature_analysis
# Description   : Rank features using training data only
# Input         : X_train and y_train
# Output        : feature ranking DataFrame and figure paths
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_feature_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Calculate feature importance using only training observations."""
    print("\n[3/9] Analysing feature importance...")

    ranking = rank_features(X_train, y_train)

    save_csv(
        ranking,
        config.REPORTS_DIR / "feature_ranking.csv",
    )

    ranking_figure = plot_feature_ranking(
        ranking,
        config.FIGURES_DIR / "feature_ranking.png",
    )

    top_features = ranking["feature"].head(5).tolist()

    print("Top five features:")
    for position, feature in enumerate(top_features, start=1):
        print(f"{position}. {feature}")

    scatter_figure = None

    if len(top_features) >= 2:
        scatter_figure = plot_scatter_pair(
            pd.concat(
                [X_train, y_train.rename(config.TARGET_COLUMN)],
                axis=1,
            ),
            top_features[0],
            top_features[1],
            config.FIGURES_DIR / "top_feature_scatter.png",
        )

    return ranking, {
        "feature_ranking": ranking_figure,
        "top_feature_scatter": scatter_figure,
    }


############################################################
# Function name : run_model_comparison
# Description   : Compare candidate models using repeated stratified CV
# Input         : X_train and y_train
# Output        : CV results and selected candidate
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_model_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Compare model families and feature counts using training data."""
    print("\n[4/9] Running repeated stratified cross-validation...")

    cv_results = run_cv_comparison(
        X_train,
        y_train,
    )

    save_csv(
        cv_results,
        config.REPORTS_DIR / "cv_comparison.csv",
    )

    cv_figure = plot_cv_comparison(
        cv_results,
        config.FIGURES_DIR / "cv_comparison.png",
    )

    selected_candidate = select_best_candidate(cv_results)

    print("\nSelected candidate:")
    print(f"Model family: {selected_candidate['family']}")
    print(f"Number of features: {selected_candidate['n_features']}")
    print(
        f"CV {config.PRIMARY_METRIC}: "
        f"{selected_candidate['cv_mean']:.4f}"
    )

    return cv_results, selected_candidate, cv_figure


############################################################
# Function name : run_nested_cv
# Description   : Estimate the selected tuning procedure using nested CV
# Input         : X_train, y_train and selected candidate
# Output        : nested CV result dictionary
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_nested_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_candidate: dict,
) -> dict:
    """Run nested CV conditional on the selected family and feature count."""
    print("\n[5/9] Running nested cross-validation...")

    result = nested_cv_estimate(
        X_train,
        y_train,
        selected_candidate["family"],
        selected_candidate["n_features"],
    )

    save_json(
        make_json_serializable(result),
        config.REPORTS_DIR / "nested_cv_estimate.json",
    )

    print(
        f"Nested CV macro-F1: "
        f"{result['f1_macro_mean']:.4f} "
        f"+/- {result['f1_macro_std']:.4f}"
    )

    print(f"Scope: {result['scope']}")

    return result


############################################################
# Function name : train_final_model
# Description   : Tune and refit the selected pipeline on training data
# Input         : X_train, y_train and selected candidate
# Output        : fitted pipeline and tuning details
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_candidate: dict,
):
    """Tune the selected model and refit it on the complete training split."""
    print("\n[6/9] Tuning the selected model...")

    search = tune_pipeline(
        X_train,
        y_train,
        selected_candidate["family"],
        selected_candidate["n_features"],
    )

    pipeline = search.best_estimator_

    print(f"Best parameters: {search.best_params_}")
    print(f"Best tuning CV score: {search.best_score_:.4f}")

    return pipeline, {
        "best_params": search.best_params_,
        "best_tuning_cv_score": float(search.best_score_),
        "best_family": selected_candidate["family"],
        "n_features": selected_candidate["n_features"],
    }


############################################################
# Function name : evaluate_final_model
# Description   : Evaluate the final pipeline on the holdout set
# Input         : fitted pipeline, holdout data and training data
# Output        : evaluation results and bootstrap intervals
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def evaluate_final_model(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Evaluate the final model once on the untouched holdout set."""
    print("\n[7/9] Evaluating final model on the holdout set...")

    evaluation = evaluate_on_holdout(
        pipeline,
        X_test,
        y_test,
    )

    bootstrap = bootstrap_confidence_intervals(
        evaluation["y_true"],
        evaluation["y_pred"],
    )

    save_json(
        make_json_serializable(evaluation),
        config.REPORTS_DIR / "holdout_evaluation.json",
    )

    save_json(
        make_json_serializable(bootstrap),
        config.REPORTS_DIR / "bootstrap_confidence_intervals.json",
    )

    classification_report = pd.DataFrame(
        evaluation["classification_report"]
    ).T

    save_csv(
        classification_report.reset_index(names="class"),
        config.REPORTS_DIR / "classification_report.csv",
    )

    save_json(
        make_json_serializable(
            {
                "accuracy": evaluation["metrics"]["accuracy"],
                "precision_macro": evaluation["metrics"]["precision_macro"],
                "recall_macro": evaluation["metrics"]["recall_macro"],
                "f1_macro": evaluation["metrics"]["f1_macro"],
                "f1_weighted": evaluation["metrics"]["f1_weighted"],
                "roc_auc_ovr": evaluation["metrics"]["roc_auc_ovr"],
                "bootstrap": bootstrap,
            }
        ),
        config.REPORTS_DIR / "final_metrics.json",
    )

    plot_confusion_matrix(
        evaluation["confusion_matrix"],
        evaluation["labels"],
        config.FIGURES_DIR / "holdout_confusion_matrix.png",
    )

    if evaluation["y_proba"] is not None:
        plot_roc_curves(
            evaluation["y_true"],
            np.asarray(evaluation["y_proba"]),
            evaluation["labels"],
            config.FIGURES_DIR / "holdout_roc_curves.png",
        )

    metrics = evaluation["metrics"]

    print("\nHoldout metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Not available")

    print("\nBootstrap confidence intervals:")
    for metric in ["accuracy", "f1_macro"]:
        interval = bootstrap[metric]
        print(
            f"{metric}: "
            f"{interval['lower']:.4f} - {interval['upper']:.4f}"
        )

    return evaluation, bootstrap


############################################################
# Function name : run_historical_analysis
# Description   : Reproduce and analyse historical Assignment 41 results
# Input         : X and y from the complete dataset
# Output        : historical results and optional seed sensitivity
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_historical_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    include_seed_sensitivity: bool = True,
):
    """Run historical reproduction separately from modern evaluation."""
    print("\n[8/9] Reproducing historical Assignment 41 results...")

    historical_results = reproduce_historical_results(X, y)

    save_csv(
        historical_results,
        config.REPORTS_DIR / "historical_reproduction.csv",
    )

    plot_historical_comparison(
        historical_results,
        config.FIGURES_DIR / "historical_comparison.png",
    )

    print(
        historical_results[
            [
                "experiment",
                "reproduced_accuracy_percent",
                "documented_accuracy_percent",
                "matches_documented",
            ]
        ].to_string(index=False)
    )

    sensitivity = None
    sensitivity_summary = None

    if include_seed_sensitivity:
        print("\nRunning historical seed-sensitivity analysis...")

        sensitivity = run_seed_sensitivity(X, y)

        sensitivity_summary = summarize_seed_sensitivity(
            sensitivity
        )

        save_csv(
            sensitivity,
            config.REPORTS_DIR / "seed_sensitivity.csv",
        )

        save_csv(
            sensitivity_summary,
            config.REPORTS_DIR / "seed_sensitivity_summary.csv",
        )

    return historical_results, sensitivity, sensitivity_summary


############################################################
# Function name : run_project
# Description   : Execute the complete wine classification workflow
# Input         : nested CV and seed sensitivity execution options
# Output        : dictionary containing all project results
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def run_project(
    include_nested_cv: bool = True,
    include_seed_sensitivity: bool = True,
) -> dict:
    """Execute the complete training, evaluation and reporting workflow."""
    prepare_directories()

    df, X, y, dataset_summary = run_data_analysis()

    figures = run_eda_visualizations(df)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
    )

    print("\nTrain-test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Holdout samples: {len(X_test)}")

    ranking, feature_figures = run_feature_analysis(
        X_train,
        y_train,
    )

    figures.update(feature_figures)

    cv_results, selected_candidate, cv_figure = run_model_comparison(
        X_train,
        y_train,
    )

    figures["cv_comparison"] = cv_figure

    nested_results = None

    if include_nested_cv:
        nested_results = run_nested_cv(
            X_train,
            y_train,
            selected_candidate,
        )

    pipeline, tuning_results = train_final_model(
        X_train,
        y_train,
        selected_candidate,
    )

    evaluation, bootstrap = evaluate_final_model(
        pipeline,
        X_train,
        X_test,
        y_test,
    )

    metadata = build_metadata(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        family=selected_candidate["family"],
        best_params=tuning_results["best_params"],
        n_test_rows=len(X_test),
        model_filename=config.MODEL_PATH.name,
    )

    model_path, metadata_path = save_pipeline(
        pipeline,
        metadata,
    )

    print("\nSaved trained model:")
    print(model_path)
    print(metadata_path)

    historical_results, sensitivity, sensitivity_summary = (
        run_historical_analysis(
            X,
            y,
            include_seed_sensitivity=include_seed_sensitivity,
        )
    )

    print("\n[9/9] Saving final run summary...")

    run_summary = {
        "project": (
            "Wine Classification Using Ensemble Learning "
            "and Robust Model Evaluation"
        ),
        "dataset": dataset_summary,
        "train_rows": len(X_train),
        "holdout_rows": len(X_test),
        "selected_candidate": selected_candidate,
        "tuning": tuning_results,
        "nested_cv": nested_results,
        "holdout_metrics": evaluation["metrics"],
        "bootstrap_intervals": bootstrap,
        "historical_reproduction": historical_results.to_dict(
            orient="records"
        ),
        "seed_sensitivity_summary": (
            None
            if sensitivity_summary is None
            else sensitivity_summary.to_dict(orient="records")
        ),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "figures": {
            key: (
                None if value is None else str(value)
            )
            for key, value in figures.items()
        },
    }

    save_json(
        make_json_serializable(run_summary),
        config.REPORTS_DIR / "run_summary.json",
    )

    print("\n" + "=" * 60)
    print("WINE CLASSIFICATION WORKFLOW COMPLETED")
    print("=" * 60)
    print(f"Model: {selected_candidate['family']}")
    print(f"Selected features: {selected_candidate['n_features']}")
    print(
        f"Holdout accuracy: "
        f"{evaluation['metrics']['accuracy']:.4f}"
    )
    print(
        f"Holdout macro-F1: "
        f"{evaluation['metrics']['f1_macro']:.4f}"
    )
    print(f"Reports directory: {config.REPORTS_DIR}")
    print(f"Figures directory: {config.FIGURES_DIR}")

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "dataset_summary": dataset_summary,
        "feature_ranking": ranking,
        "cv_results": cv_results,
        "selected_candidate": selected_candidate,
        "nested_cv": nested_results,
        "tuning": tuning_results,
        "evaluation": evaluation,
        "bootstrap": bootstrap,
        "historical_results": historical_results,
        "seed_sensitivity": sensitivity,
        "seed_sensitivity_summary": sensitivity_summary,
        "figures": figures,
    }


############################################################
# Function name : main
# Description   : Parse command-line arguments and start execution
# Input         : command-line arguments
# Output        : None
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def main() -> None:
    """Parse execution options and launch the project."""
    parser = argparse.ArgumentParser(
        description=(
            "Wine Classification Using Ensemble Learning "
            "and Robust Model Evaluation"
        )
    )

    parser.add_argument(
        "--skip-nested-cv",
        action="store_true",
        help="Skip the optional nested cross-validation estimate.",
    )

    parser.add_argument(
        "--skip-seed-sensitivity",
        action="store_true",
        help="Skip historical random-seed sensitivity analysis.",
    )

    args = parser.parse_args()

    run_project(
        include_nested_cv=not args.skip_nested_cv,
        include_seed_sensitivity=not args.skip_seed_sensitivity,
    )


if __name__ == "__main__":
    main()
