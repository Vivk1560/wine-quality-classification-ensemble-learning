
"""
Headless plotting helpers. Every function saves a PNG, closes the figure
and returns the saved file path.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.preprocessing import label_binarize

import config


CLASS_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
}


############################################################
# Function name : _finish
# Description   : Save and close a matplotlib figure
# Input         : figure object and destination path
# Output        : saved file path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def _finish(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
    finally:
        plt.close(fig)

    return path


############################################################
# Function name : plot_class_distribution
# Description   : Plot the number of observations in each class
# Input         : validated dataset and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_class_distribution(df: pd.DataFrame, path: Path) -> Path:
    """Plot the class distribution."""
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(
        [config.CLASS_NAMES[int(c)] for c in counts.index],
        counts.values,
        color=[CLASS_COLORS[int(c)] for c in counts.index],
    )

    ax.bar_label(bars)
    ax.set_ylabel("Number of samples")
    ax.set_title("Class Distribution")

    return _finish(fig, path)


############################################################
# Function name : plot_feature_histograms
# Description   : Plot histograms for all numerical features
# Input         : validated dataset and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_feature_histograms(df: pd.DataFrame, path: Path) -> Path:
    """Plot feature distributions using histograms."""
    fig, axes = plt.subplots(5, 3, figsize=(13, 15))

    for ax, feature in zip(axes.ravel(), config.FEATURE_COLUMNS):
        ax.hist(
            df[feature],
            bins=15,
            color="#4c72b0",
            edgecolor="white",
        )
        ax.set_title(feature, fontsize=10)

    for ax in axes.ravel()[len(config.FEATURE_COLUMNS):]:
        ax.axis("off")

    fig.suptitle("Feature Distributions", y=1.0)

    return _finish(fig, path)


############################################################
# Function name : plot_feature_boxplots
# Description   : Compare feature distributions across classes
# Input         : validated dataset and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_feature_boxplots(df: pd.DataFrame, path: Path) -> Path:
    """Plot class-wise feature distributions."""
    fig, axes = plt.subplots(5, 3, figsize=(13, 17))

    for ax, feature in zip(axes.ravel(), config.FEATURE_COLUMNS):
        data = [
            df.loc[df[config.TARGET_COLUMN] == label, feature]
            for label in config.CLASS_LABELS
        ]

        ax.boxplot(
                    data,
                    tick_labels=[str(label) for label in config.CLASS_LABELS],
                )

        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("Class")

    for ax in axes.ravel()[len(config.FEATURE_COLUMNS):]:
        ax.axis("off")

    fig.suptitle("Feature Distributions by Class", y=1.0)

    return _finish(fig, path)


############################################################
# Function name : plot_class_mean_heatmap
# Description   : Plot standardised feature means for each class
# Input         : validated dataset and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_class_mean_heatmap(df: pd.DataFrame, path: Path) -> Path:
    """Plot class-wise means after feature standardisation."""
    features = df[config.FEATURE_COLUMNS]

    standard_deviation = features.std().replace(0, np.nan)

    z_scores = (
        features - features.mean()
    ) / standard_deviation

    z_scores = z_scores.fillna(0)

    means = z_scores.groupby(
        df[config.TARGET_COLUMN]
    ).mean().T

    fig, ax = plt.subplots(figsize=(6.5, 7))

    image = ax.imshow(
        means.values,
        cmap="coolwarm",
        vmin=-1.6,
        vmax=1.6,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(means.columns)),
        [config.CLASS_NAMES[int(c)] for c in means.columns],
    )

    ax.set_yticks(range(len(means.index)), means.index)

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(
                j,
                i,
                f"{means.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, label="Mean z-score")
    ax.set_title("Class Means of Standardised Features")

    return _finish(fig, path)


############################################################
# Function name : plot_correlation_heatmap
# Description   : Plot Pearson correlations between features
# Input         : validated dataset and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    """Plot the feature correlation matrix."""
    corr = df[config.FEATURE_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(
        corr.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(
        range(len(corr)),
        corr.columns,
        rotation=90,
    )

    ax.set_yticks(range(len(corr)), corr.columns)

    fig.colorbar(image, ax=ax, label="Pearson r")
    ax.set_title("Feature Correlation Matrix")

    return _finish(fig, path)


############################################################
# Function name : plot_scatter_pair
# Description   : Plot two numerical features by target class
# Input         : dataset, feature names and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_scatter_pair(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
) -> Path:
    """Plot a two-feature scatter comparison."""
    if x not in config.FEATURE_COLUMNS:
        raise ValueError(f"Unknown feature: {x}")

    if y not in config.FEATURE_COLUMNS:
        raise ValueError(f"Unknown feature: {y}")

    fig, ax = plt.subplots(figsize=(7, 5))

    for label in config.CLASS_LABELS:
        part = df[df[config.TARGET_COLUMN] == label]

        ax.scatter(
            part[x],
            part[y],
            label=config.CLASS_NAMES[label],
            color=CLASS_COLORS[label],
            alpha=0.8,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    ax.legend()

    return _finish(fig, path)


############################################################
# Function name : plot_feature_ranking
# Description   : Compare training-split feature ranking scores
# Input         : feature ranking DataFrame and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_feature_ranking(
    ranking: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot Random Forest importance and ANOVA F-statistics."""
    required = {
        "feature",
        "rf_importance",
        "anova_f",
    }

    if not required.issubset(ranking.columns):
        raise ValueError(
            f"Feature ranking must contain columns: {sorted(required)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    by_rf = ranking.sort_values("rf_importance")

    axes[0].barh(
        by_rf["feature"],
        by_rf["rf_importance"],
        color="#4c72b0",
    )

    axes[0].set_title("Random Forest Importance (Training Split)")

    by_f = ranking.sort_values("anova_f")

    axes[1].barh(
        by_f["feature"],
        by_f["anova_f"],
        color="#dd8452",
    )

    axes[1].set_title("ANOVA F-statistic (Training Split)")

    return _finish(fig, path)


############################################################
# Function name : plot_cv_comparison
# Description   : Compare candidate models using cross-validation
# Input         : CV results DataFrame and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_cv_comparison(
    cv_results: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot mean CV macro-F1 and standard deviation."""
    metric = config.PRIMARY_METRIC

    required = {
        "family",
        "n_features",
        f"{metric}_mean",
        f"{metric}_std",
    }

    if not required.issubset(cv_results.columns):
        raise ValueError(
            f"CV results must contain columns: {sorted(required)}"
        )

    families = [
        family
        for family in config.FAMILY_LABELS
        if family != config.BASELINE_FAMILY
    ]

    feature_counts = config.FEATURE_COUNTS
    width = 0.8 / len(feature_counts)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for index, count in enumerate(feature_counts):
        subset = (
            cv_results[cv_results["n_features"] == count]
            .set_index("family")
            .reindex(families)
        )

        means = subset[f"{metric}_mean"].to_numpy(dtype=float)
        deviations = subset[f"{metric}_std"].to_numpy(dtype=float)

        positions = np.arange(len(families)) + index * width

        ax.bar(
            positions,
            means,
            width,
            yerr=deviations,
            capsize=2,
            label=f"{count} features",
        )

    baseline = cv_results.loc[
        cv_results["family"] == config.BASELINE_FAMILY,
        f"{metric}_mean",
    ]

    if not baseline.empty:
        ax.axhline(
            float(baseline.iloc[0]),
            color="grey",
            linestyle="--",
            label="Majority baseline",
        )

    ax.set_xticks(
        np.arange(len(families)) + 0.4 - width / 2,
        [config.FAMILY_LABELS[family] for family in families],
        rotation=15,
        ha="right",
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Cross-validated {metric}")
    ax.set_title("Cross-Validated Model Comparison")
    ax.legend(ncol=3, fontsize=8)

    return _finish(fig, path)


############################################################
# Function name : plot_confusion_matrix
# Description   : Plot the final holdout confusion matrix
# Input         : confusion matrix, class labels and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    labels: Sequence[int],
    path: Path,
) -> Path:
    """Plot the final model's holdout confusion matrix."""
    cm = np.asarray(cm)

    if cm.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix dimensions do not match labels.")

    display_labels = [
        config.CLASS_NAMES[label]
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels,
    ).plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )

    ax.set_title("Final Model - Holdout Confusion Matrix")

    return _finish(fig, path)


############################################################
# Function name : plot_roc_curves
# Description   : Plot one-vs-rest ROC curves for each class
# Input         : true labels, probabilities, classes and path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_roc_curves(
    y_true: Sequence[int],
    y_proba: np.ndarray,
    labels: Sequence[int],
    path: Path,
) -> Path:
    """Plot one-vs-rest ROC curves on the holdout set."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    labels = list(labels)

    if y_proba.shape != (len(y_true), len(labels)):
        raise ValueError("Probability dimensions do not match labels.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Probabilities must contain finite values.")

    binary = label_binarize(y_true, classes=labels)

    if binary.shape[1] != len(labels):
        raise ValueError("Could not construct one-vs-rest class indicators.")

    fig, ax = plt.subplots(figsize=(6, 5))

    for index, label in enumerate(labels):
        if len(np.unique(binary[:, index])) < 2:
            continue

        fpr, tpr, _ = roc_curve(
            binary[:, index],
            y_proba[:, index],
        )

        ax.plot(
            fpr,
            tpr,
            color=CLASS_COLORS[label],
            label=config.CLASS_NAMES[label],
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final Model - One-vs-Rest ROC (Holdout)")
    ax.legend(loc="lower right")

    return _finish(fig, path)


############################################################
# Function name : plot_historical_comparison
# Description   : Compare historical Assignment 41 accuracies
# Input         : historical results DataFrame and destination path
# Output        : saved plot path
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def plot_historical_comparison(
    historical: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot historical results with their methodological caveat."""
    required = {
        "experiment",
        "reproduced_accuracy_percent",
    }

    if not required.issubset(historical.columns):
        raise ValueError(
            f"Historical results must contain columns: {sorted(required)}"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        historical["experiment"],
        historical["reproduced_accuracy_percent"],
        color="#9a9a9a",
    )

    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylim(80, 100)
    ax.set_ylabel("Test Accuracy (%)")

    ax.set_title(
        "HISTORICAL Protocol (60/40 Split, Test-Set Subset Selection)\n"
        "Not an unbiased estimate"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=15,
        ha="right",
    )

    return _finish(fig, path)
