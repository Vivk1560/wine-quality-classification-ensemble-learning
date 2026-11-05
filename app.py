
"""
Streamlit dashboard for Wine Classification using Ensemble Learning.

Author : Vivaan Kukreja
Date   : 04/11/2026
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from data_loader import read_data, validate_dataset
from inference import (
    load_artifacts,
    predict_single,
    predict_batch,
    validate_input,
)


############################################################
# Dashboard styling
############################################################
def apply_dashboard_style():
    """Apply a restrained, consistent visual theme."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.07);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 16px 18px;
            border-radius: 12px;
        }
        [data-testid="stMetricLabel"] p {font-size: 0.88rem;}
        [data-testid="stMetricValue"] {font-weight: 700;}
        div[data-testid="stTabs"] button {font-weight: 600;}
        div[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
        .hero-panel {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 16px;
            background: linear-gradient(120deg, rgba(128,128,128,.10), rgba(128,128,128,.025));
            margin-bottom: 1.2rem;
        }
        .eyebrow {
            text-transform: uppercase; letter-spacing: .12em;
            font-size: .72rem; font-weight: 700; opacity: .72;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


############################################################
# Page configuration
############################################################
st.set_page_config(
    page_title="Wine Classification | ML Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)


############################################################
# Function name : load_dataset
# Description   : Load and cache the validated wine dataset
# Input         : None
# Output        : validated dataset
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
@st.cache_data
def load_dataset():
    """Load the project dataset."""
    df = read_data()
    return validate_dataset(df)


############################################################
# Function name : load_reports
# Description   : Load saved reports from the project outputs
# Input         : None
# Output        : dictionary containing available reports
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
@st.cache_data
def load_reports():
    """Load generated CSV and JSON reports."""
    reports = {}

    csv_files = {
        "feature_ranking": "feature_ranking.csv",
        "cv_comparison": "cv_comparison.csv",
        "classification_report": "classification_report.csv",
        "historical_reproduction": "historical_reproduction.csv",
        "seed_sensitivity": "seed_sensitivity.csv",
        "seed_sensitivity_summary": "seed_sensitivity_summary.csv",
    }

    json_files = {
        "dataset_summary": "dataset_summary.json",
        "nested_cv": "nested_cv_estimate.json",
        "holdout_evaluation": "holdout_evaluation.json",
        "bootstrap": "bootstrap_confidence_intervals.json",
        "final_metrics": "final_metrics.json",
        "run_summary": "run_summary.json",
    }

    for key, filename in csv_files.items():
        path = config.REPORTS_DIR / filename

        if path.is_file():
            reports[key] = pd.read_csv(path)

    import json

    for key, filename in json_files.items():
        path = config.REPORTS_DIR / filename

        if path.is_file():
            with path.open("r", encoding="utf-8") as file:
                reports[key] = json.load(file)

    return reports


############################################################
# Function name : load_model_artifacts
# Description   : Load the trained pipeline and its metadata
# Input         : None
# Output        : fitted pipeline and metadata
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
@st.cache_resource
def load_model_artifacts():
    """Load the saved model and metadata."""
    return load_artifacts()


############################################################
# Function name : show_metric_card
# Description   : Display an individual evaluation metric
# Input         : metric label and value
# Output        : Streamlit metric component
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_metric_card(label, value):
    """Display a formatted metric."""
    if value is None:
        st.metric(label, "N/A")
    else:
        st.metric(label, f"{value:.4f}")


############################################################
# Function name : show_image
# Description   : Display a saved figure if it exists
# Input         : figure filename and display caption
# Output        : Streamlit image or unavailable message
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_image(filename, caption=None):
    """Display a saved project figure."""
    path = config.FIGURES_DIR / filename

    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not available yet: {filename}")


############################################################
# Function name : show_overview
# Description   : Display project and dataset overview
# Input         : dataset and reports
# Output        : Streamlit overview section
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_overview(df, reports):
    """Display project overview and dataset information."""
    st.markdown(
        '<div class="hero-panel"><div class="eyebrow">Machine Learning Case Study</div>',
        unsafe_allow_html=True,
    )
    st.title("🍷 Wine Classification")
    st.caption(
        "Ensemble Learning | Feature Selection | "
        "Robust Model Evaluation"
    )

    st.markdown(
        """
        Explore a machine learning workflow that compares multiple
        classification algorithms, evaluates feature subsets, and
        predicts wine cultivar classes using a trained ensemble model.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    summary = reports.get("dataset_summary", {})
    final_evaluation = reports.get("holdout_evaluation", {})
    final_metrics = final_evaluation.get("metrics", {})
    nested = reports.get("nested_cv", {})
    model_metadata = None
    try:
        _, model_metadata = load_model_artifacts()
    except Exception:
        model_metadata = None

    st.subheader("Model Snapshot")
    model_col, cv_col, nested_col, holdout_col = st.columns(4)
    model_label = (
        model_metadata.get("model_label", model_metadata.get("model_family", "Unavailable"))
        if model_metadata else "Unavailable"
    )
    model_col.metric("Selected Model", model_label)
    cv_score = None
    run_summary = reports.get("run_summary", {})
    selection = run_summary.get("selected_candidate", {})
    if isinstance(selection, dict):
        cv_score = selection.get("mean_score", selection.get("cv_mean"))
    cv_col.metric("Selection CV Macro-F1", f"{cv_score:.4f}" if isinstance(cv_score, (int, float)) else "N/A")
    nested_score = nested.get("f1_macro_mean")
    nested_col.metric("Nested CV Macro-F1", f"{nested_score:.4f}" if isinstance(nested_score, (int, float)) else "N/A")
    holdout_score = final_metrics.get("f1_macro")
    holdout_col.metric("Holdout Macro-F1", f"{holdout_score:.4f}" if isinstance(holdout_score, (int, float)) else "N/A")
    st.caption(
        "The holdout score is a post-selection evaluation on this dataset's "
        "held-out split; nested CV provides a separate, conditional estimate."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", len(df))
    col2.metric("Input Features", len(config.FEATURE_COLUMNS))
    col3.metric("Target Classes", df[config.TARGET_COLUMN].nunique())
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Dataset Information")

    info_col, chart_col = st.columns([1, 1])

    with info_col:
        st.write("**Features**")
        st.write(", ".join(config.FEATURE_COLUMNS))

        st.write("**Target variable**")
        st.write(config.TARGET_COLUMN)

        st.write("**Class labels**")
        st.write(", ".join(str(label) for label in config.CLASS_LABELS))

        if summary:
            st.write("**Dataset dimensions**")
            st.write(summary.get("shape", "See dataset table"))

    with chart_col:
        show_image(
            "class_distribution.png",
            "Distribution of wine cultivar classes",
        )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")


############################################################
# Function name : show_exploratory_analysis
# Description   : Display exploratory data analysis figures
# Input         : validated dataset and reports
# Output        : Streamlit EDA section
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_exploratory_analysis(df, reports):
    """Display feature distributions and relationships."""
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(
        [
            "Distributions",
            "Feature Relationships",
            "Feature Ranking",
        ]
    )

    with tab1:
        st.subheader("Feature Histograms")
        show_image("feature_histograms.png")

        st.subheader("Class-wise Boxplots")
        show_image("feature_boxplots.png")

    with tab2:
        st.subheader("Feature Correlation")
        show_image("correlation_heatmap.png")

        st.subheader("Class-wise Standardised Feature Means")
        show_image("class_mean_heatmap.png")

        st.subheader("Feature Pair Comparison")

        feature_x = st.selectbox(
            "Select X-axis feature",
            config.FEATURE_COLUMNS,
            index=0,
            key="scatter_x",
        )

        feature_y = st.selectbox(
            "Select Y-axis feature",
            config.FEATURE_COLUMNS,
            index=1,
            key="scatter_y",
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        for label in config.CLASS_LABELS:
            subset = df[df[config.TARGET_COLUMN] == label]

            ax.scatter(
                subset[feature_x],
                subset[feature_y],
                label=config.CLASS_NAMES[label],
                alpha=0.75,
            )

        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(f"{feature_x} vs {feature_y}")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Training-split Feature Ranking")

        ranking = reports.get("feature_ranking")

        if ranking is not None:
            st.dataframe(
                ranking,
                width="stretch",
                hide_index=True,
            )

        show_image("feature_ranking.png")

        st.caption(
            "Feature ranking was calculated using training data only."
        )


############################################################
# Function name : show_model_evaluation
# Description   : Display cross-validation and holdout performance
# Input         : reports
# Output        : Streamlit evaluation section
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_model_evaluation(reports):
    """Display model comparison and final evaluation."""
    st.title("Model Evaluation")

    cv_results = reports.get("cv_comparison")
    evaluation = reports.get("holdout_evaluation", {})
    metrics = evaluation.get("metrics", {})
    bootstrap = reports.get("bootstrap", {})

    st.subheader("Final Holdout Performance")

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "precision_macro"),
        ("Macro Recall", "recall_macro"),
        ("Macro F1", "f1_macro"),
        ("Weighted F1", "f1_weighted"),
        ("ROC-AUC OVR", "roc_auc_ovr"),
    ]

    for start in range(0, len(metric_names), 3):
        columns = st.columns(3)

        for column, (label, key) in zip(
            columns,
            metric_names[start:start + 3],
        ):
            with column:
                show_metric_card(label, metrics.get(key))

    st.caption(
        "These metrics describe the final held-out split after model selection "
        "and tuning. They are not an independent external validation."
    )

    if bootstrap:
        st.subheader("Bootstrap Confidence Intervals")
        st.caption(
            bootstrap.get(
                "interpretation",
                "Bootstrap intervals quantify uncertainty in the observed holdout metrics; "
                "they do not replace external validation."
            )
        )

        bootstrap_columns = st.columns(2)

        for column, key in zip(
            bootstrap_columns,
            ["accuracy", "f1_macro"],
        ):
            interval = bootstrap.get(key)

            if interval:
                with column:
                    lower = interval.get("lower")
                    upper = interval.get("upper")

                    if lower is not None and upper is not None:
                        st.metric(
                            f"{key} 95% interval",
                            f"{lower:.4f} – {upper:.4f}",
                        )

    st.divider()

    st.subheader("Cross-Validated Model Comparison")

    if cv_results is not None:
        st.dataframe(
            cv_results,
            width="stretch",
            hide_index=True,
        )

        metric_column = f"{config.PRIMARY_METRIC}_mean"

        if metric_column in cv_results.columns:
            chart_data = cv_results.pivot(
                index="family",
                columns="n_features",
                values=metric_column,
            )

            st.bar_chart(chart_data)

    st.divider()

    left, right = st.columns(2)

    with left:
        show_image(
            "holdout_confusion_matrix.png",
            "Final model confusion matrix",
        )

    with right:
        show_image(
            "holdout_roc_curves.png",
            "One-vs-rest ROC curves",
        )

    classification_report = reports.get("classification_report")

    if classification_report is not None:
        st.subheader("Classification Report")

        st.dataframe(
            classification_report,
            width="stretch",
            hide_index=True,
        )

    nested = reports.get("nested_cv")

    if nested:
        st.subheader("Nested Cross-Validation")

        st.write(
            f"Mean macro-F1: "
            f"{nested.get('f1_macro_mean', float('nan')):.4f}"
        )

        st.write(
            f"Standard deviation: "
            f"{nested.get('f1_macro_std', float('nan')):.4f}"
        )

        st.caption(nested.get("scope", ""))


############################################################
# Function name : show_historical_analysis
# Description   : Display historical Assignment 41 results
# Input         : reports
# Output        : Streamlit historical analysis section
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_historical_analysis(reports):
    """Display historical reproduction and seed sensitivity."""
    st.title("Historical Assignment 41 Analysis")

    st.warning(
        """
        Historical results reproduce the original 60:40 split and
        feature-subset selection protocol. The original workflow used
        test-set accuracy during subset selection, so these results
        must not be interpreted as unbiased generalisation estimates.
        """
    )

    historical = reports.get("historical_reproduction")

    if historical is not None:
        st.subheader("Historical Result Reproduction")

        st.dataframe(
            historical,
            width="stretch",
            hide_index=True,
        )

        show_image("historical_comparison.png")

    sensitivity = reports.get("seed_sensitivity_summary")

    if sensitivity is not None:
        st.subheader("Random Seed Sensitivity")

        st.dataframe(
            sensitivity,
            width="stretch",
            hide_index=True,
        )

        if "mean_accuracy" in sensitivity.columns:
            st.bar_chart(
                sensitivity.set_index("experiment")[
                    [
                        "mean_accuracy",
                        "minimum_accuracy",
                        "maximum_accuracy",
                    ]
                ]
            )

        st.caption(
            "The split seed varies while the Random Forest seed remains fixed."
        )

    else:
        st.info(
            "Seed sensitivity report is not available. "
            "Run main.py without --skip-seed-sensitivity."
        )


############################################################
# Function name : show_single_prediction
# Description   : Predict the cultivar for a single wine sample
# Input         : fitted pipeline and metadata
# Output        : Streamlit single prediction interface
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_single_prediction(pipeline, metadata, df):
    """Display a form for single-sample prediction."""
    st.title("Single Wine Prediction")

    st.write(
        "Enter the chemical properties of a wine sample "
        "to predict its cultivar."
    )

    with st.form("single_prediction_form"):
        sample = {}

        columns = st.columns(3)

        for index, feature in enumerate(config.FEATURE_COLUMNS):
            default_value = float(df[feature].median())

            with columns[index % 3]:
                sample[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.4f",
                    key=f"single_{feature}",
                )

        submitted = st.form_submit_button(
            "Predict Wine Cultivar",
            width="stretch",
        )

    if submitted:
        try:
            prediction, warnings = predict_single(
                pipeline,
                metadata,
                sample,
            )

            st.divider()
            st.subheader("Prediction Result")

            result_col, confidence_col = st.columns(2)

            result_col.metric(
                "Predicted Cultivar",
                prediction["predicted_cultivar"],
            )

            confidence_col.metric(
                "Prediction Confidence",
                f"{prediction['confidence_percent']:.2f}%",
            )

            st.subheader("Class Probabilities")

            probability_data = pd.DataFrame(
                {
                    "Cultivar": [
                        config.CLASS_NAMES[label]
                        for label in config.CLASS_LABELS
                    ],
                    "Probability (%)": [
                        prediction[
                            f"probability_class_{label}_percent"
                        ]
                        for label in config.CLASS_LABELS
                    ],
                }
            )

            st.dataframe(
                probability_data,
                width="stretch",
                hide_index=True,
            )

            st.bar_chart(
                probability_data.set_index("Cultivar")
            )

            if not warnings.empty:
                st.warning(
                    "Some input values fall outside the observed "
                    "training ranges. Review these values before "
                    "interpreting the prediction."
                )

                st.dataframe(
                    warnings,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.success(
                    "All input values fall within the observed "
                    "training ranges."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


############################################################
# Function name : show_batch_prediction
# Description   : Predict cultivar classes for an uploaded CSV
# Input         : fitted pipeline and metadata
# Output        : Streamlit batch prediction interface
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def show_batch_prediction(pipeline, metadata):
    """Display CSV upload and batch prediction functionality."""
    st.title("Batch Prediction")

    st.write(
        "Upload a CSV file containing all 13 wine features. "
        "The application will generate predictions for every row."
    )

    st.caption(
        "The uploaded CSV must contain the exact feature columns "
        "used during model training. The target column is not required."
    )

    uploaded_file = st.file_uploader(
        "Upload wine samples CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            input_df.head(10),
            width="stretch",
        )

        validated_input = validate_input(input_df)

        st.success(
            f"Successfully validated {len(validated_input)} samples."
        )

        if st.button(
            "Generate Batch Predictions",
            width="stretch",
        ):
            predictions, warnings = predict_batch(
                pipeline,
                metadata,
                validated_input,
            )

            output = pd.concat(
                [
                    input_df.reset_index(drop=True),
                    predictions.reset_index(drop=True),
                ],
                axis=1,
            )

            st.subheader("Prediction Results")

            st.dataframe(
                output,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                label="Download Predictions CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="wine_predictions.csv",
                mime="text/csv",
                width="stretch",
            )

            if not warnings.empty:
                st.warning(
                    "Some input values are outside the training ranges."
                )

                st.dataframe(
                    warnings,
                    width="stretch",
                    hide_index=True,
                )

            else:
                st.success(
                    "All samples are within the observed training ranges."
                )

    except Exception as error:
        st.error(f"Batch prediction failed: {error}")

############################################################
# Function name : show_feature_analysis
# Description   : Display feature rankings and selected features
# Input         : reports and optional model metadata
# Output        : Streamlit feature analysis section
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################

def show_feature_analysis(reports, metadata=None):
    """Display training-split feature rankings and selected features."""

    st.title("Feature Analysis")

    st.markdown(
        "Explore how the wine features rank according to "
        "Random Forest importance and ANOVA F-statistics."
    )

    ranking = reports.get("feature_ranking")

    if ranking is None or ranking.empty:
        st.warning(
            "Feature ranking report is unavailable. "
            "Run main.py to generate the report."
        )
        return

    selected_features = (
        metadata.get("selected_features", [])
        if metadata
        else []
    )

    if selected_features:
        st.subheader("Model Feature Selection")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Selected Features",
                len(selected_features),
            )

        with col2:
            st.metric(
                "Available Dataset Features",
                len(config.FEATURE_COLUMNS),
            )

        st.caption(
            "The saved pipeline accepts all 13 original features. "
            "Any internal feature selection is handled by the pipeline."
        )

        feature_status = ranking.copy()
        feature_status["Model Feature"] = feature_status["feature"].apply(
            lambda feature: (
                "Selected"
                if feature in selected_features
                else "Not selected"
            )
        )

        st.dataframe(
            feature_status,
            width="stretch",
            hide_index=True,
        )

    else:
        st.info(
            "Model feature-selection metadata is unavailable. "
            "Showing the training-split feature ranking only."
        )

        st.dataframe(
            ranking,
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Random Forest Feature Importance")

    rf_ranking = ranking.sort_values(
        "rf_importance",
        ascending=True,
    )

    st.bar_chart(
        rf_ranking.set_index("feature")["rf_importance"],
        horizontal=True,
    )

    st.subheader("ANOVA F-Statistics")

    anova_ranking = ranking.sort_values(
        "anova_f",
        ascending=True,
    )

    st.bar_chart(
        anova_ranking.set_index("feature")["anova_f"],
        horizontal=True,
    )

    st.caption(
        "These rankings are calculated using the training split. "
        "They are for interpretation and do not independently "
        "establish causal importance."
    )

############################################################
# Function name : main
# Description   : Initialise the Streamlit dashboard
# Input         : None
# Output        : Streamlit application
# Author        : Vivaan Kukreja
# Date          : 04/11/2026
############################################################
def main():
    """Launch the complete Streamlit dashboard."""
    apply_dashboard_style()
    st.sidebar.title("🍷 Wine ML Dashboard")

    st.sidebar.caption(
        "Wine Classification Using Ensemble Learning"
    )

    try:
        df = load_dataset()
    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        st.info(
            "Check that data/WinePredictor.csv exists in the project."
        )
        st.stop()

    reports = load_reports()

    try:
        pipeline, metadata = load_model_artifacts()
        model_available = True
    except Exception as error:
        pipeline = None
        metadata = None
        model_available = False
        model_error = str(error)

    pages = [
        "Overview",
        "Exploratory Analysis",
        "Feature Analysis",
        "Model Evaluation",
        "Historical Analysis",
        "Single Prediction",
        "Batch Prediction",
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
    )

    st.sidebar.divider()

    if model_available:
        st.sidebar.success("Trained model loaded")
        st.sidebar.write(
            f"Model: {metadata.get('model_label', metadata.get('model_family', 'Saved pipeline'))}"
        )
        st.sidebar.write(
            f"Features: {len(metadata['selected_features'])}"
        )
    else:
        st.sidebar.warning("Trained model unavailable")
        st.sidebar.caption(model_error)

    if selected_page == "Overview":
        show_overview(df, reports)

    elif selected_page == "Exploratory Analysis":
        show_exploratory_analysis(df, reports)

    elif selected_page == "Feature Analysis":
        show_feature_analysis(reports, metadata if model_available else None)

    elif selected_page == "Model Evaluation":
        show_model_evaluation(reports)

    elif selected_page == "Historical Analysis":
        show_historical_analysis(reports)

    elif selected_page == "Single Prediction":
        if model_available:
            show_single_prediction(
                pipeline,
                metadata,
                df,
            )
        else:
            st.error(
                "Prediction is unavailable. Run python main.py "
                "to train and save the model."
            )

    elif selected_page == "Batch Prediction":
        if model_available:
            show_batch_prediction(
                pipeline,
                metadata,
            )
        else:
            st.error(
                "Prediction is unavailable. Run python main.py "
                "to train and save the model."
            )

    st.sidebar.divider()

    st.sidebar.caption(
        "Developed by Vivaan Kukreja | B.Tech Information Technology"
    )


if __name__ == "__main__":
    main()
