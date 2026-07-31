"""
train.py
--------
End-to-end training orchestration for the Loan Risk Assessment System.
Coordinates data loading, cleaning, feature engineering, EDA, model
training/tuning, evaluation, artifact persistence, and report generation.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import inspect_dataset, load_dataset
from src.evaluation import build_comparison_table, evaluate_model, get_best_model_name
from src.feature_engineering import engineer_features
from src.model import scale_features, train_all_models
from src.preprocessing import clean_dataset, compare_credit_history_strategies
from src.utils import get_logger, print_section, save_object
from src.visualization import (
    plot_approval_rate_by_category,
    plot_combined_roc_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_model_comparison,
)

logger = get_logger(__name__)


def run_data_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, and feature-engineer the loan dataset.

    Returns:
        A tuple of (raw_df, engineered_df).
    """
    print_section("Step 1: Data Loading")
    raw_df = load_dataset(config.RAW_DATA_PATH)
    inspect_dataset(raw_df)

    print_section("Step 2: Data Cleaning")
    compare_credit_history_strategies(raw_df, column=config.CREDIT_HISTORY_COLUMN, target=config.TARGET_COLUMN)
    cleaned_df = clean_dataset(raw_df)
    cleaned_df.to_csv(config.CLEANED_DATA_PATH, index=False)
    logger.info(f"Cleaned dataset saved to: {config.CLEANED_DATA_PATH}")

    print_section("Step 3: Feature Engineering")
    engineered_df = engineer_features(cleaned_df)

    return raw_df, engineered_df


def run_eda(raw_df: pd.DataFrame, engineered_df: pd.DataFrame) -> None:
    """Generate and save all exploratory data analysis plots.

    Args:
        raw_df: The original raw DataFrame (used for categorical approval-rate plots).
        engineered_df: The fully engineered numeric DataFrame (used for distributions
            and the correlation heatmap).
    """
    print_section("Step 4: Exploratory Data Analysis")

    plot_approval_rate_by_category(
        raw_df,
        category_col="Credit_History",
        target_col="Loan_Status",
        title="Loan Approval Rate by Credit History",
        save_path=config.IMAGES_DIR / "approval_credit_history.png",
    )
    plot_approval_rate_by_category(
        raw_df,
        category_col="Property_Area",
        target_col="Loan_Status",
        title="Loan Approval Rate by Property Area",
        save_path=config.IMAGES_DIR / "approval_property_area.png",
    )
    plot_approval_rate_by_category(
        raw_df,
        category_col="Education",
        target_col="Loan_Status",
        title="Loan Approval Rate by Education",
        save_path=config.IMAGES_DIR / "approval_education.png",
    )

    plot_distribution(
        engineered_df,
        column="ApplicantIncome",
        title="Applicant Income Distribution",
        save_path=config.IMAGES_DIR / "applicant_income_distribution.png",
    )
    plot_distribution(
        engineered_df,
        column="CoapplicantIncome",
        title="Coapplicant Income Distribution",
        save_path=config.IMAGES_DIR / "coapplicant_income_distribution.png",
        color="#21918c",
    )
    plot_distribution(
        engineered_df,
        column="LoanAmount",
        title="Loan Amount Distribution",
        save_path=config.IMAGES_DIR / "loan_amount_distribution.png",
        color="#5ec962",
    )
    plot_distribution(
        engineered_df,
        column="total_income",
        title="Total Income Distribution",
        save_path=config.IMAGES_DIR / "total_income_distribution.png",
        color="#fde725",
    )
    plot_distribution(
        engineered_df,
        column="loan_income_ratio",
        title="Loan-to-Income Ratio Distribution",
        save_path=config.IMAGES_DIR / "loan_income_ratio_distribution.png",
        color="#440154",
    )

    plot_correlation_heatmap(engineered_df, save_path=config.IMAGES_DIR / "correlation_heatmap.png")


def run_split_and_scale(
    engineered_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray, Any]:
    """Split the engineered dataset into train/test sets and scale features.

    Args:
        engineered_df: The fully engineered numeric DataFrame.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler).
    """
    print_section("Step 5: Train/Test Split")

    X = engineered_df.drop(columns=[config.TARGET_COLUMN])
    y = engineered_df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    logger.info(f"Train/test CSVs saved to: {config.TRAIN_DATA_PATH}, {config.TEST_DATA_PATH}")

    print_section("Step 6: Feature Scaling (Logistic Regression only)")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    save_object(scaler, config.MODELS_DIR / config.MODEL_FILENAMES["scaler"])

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def run_training(
    X_train: pd.DataFrame, y_train: pd.Series, X_train_scaled: np.ndarray
) -> Dict[str, Any]:
    """Train all three models and persist them to disk.

    Args:
        X_train: Unscaled training features.
        y_train: Training labels.
        X_train_scaled: Scaled training features.

    Returns:
        Dictionary mapping model name -> fitted model.
    """
    models = train_all_models(
        X_train,
        y_train,
        X_train_scaled,
        dt_param_grid=config.DECISION_TREE_PARAM_GRID,
        rf_param_grid=config.RANDOM_FOREST_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.SCORING_METRIC,
        random_state=config.RANDOM_STATE,
    )

    print_section("Step 8: Persisting Trained Models")
    save_object(models["Logistic Regression"], config.MODELS_DIR / config.MODEL_FILENAMES["logistic_regression"])
    save_object(models["Decision Tree"], config.MODELS_DIR / config.MODEL_FILENAMES["decision_tree"])
    save_object(models["Random Forest"], config.MODELS_DIR / config.MODEL_FILENAMES["random_forest"])

    return models


def run_evaluation(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all trained models and generate all evaluation visualizations.

    Args:
        models: Dictionary mapping model name -> fitted model.
        X_test: Unscaled test features (used for tree models).
        X_test_scaled: Scaled test features (used for Logistic Regression).
        y_test: True test labels.

    Returns:
        The model comparison DataFrame.
    """
    print_section("Step 9: Model Evaluation")

    all_metrics: Dict[str, Dict[str, float]] = {}
    probabilities: Dict[str, np.ndarray] = {}

    filename_map = {
        "Logistic Regression": "logistic_confusion_matrix.png",
        "Decision Tree": "decision_tree_confusion_matrix.png",
        "Random Forest": "random_forest_confusion_matrix.png",
    }

    for model_name, model in models.items():
        X_eval = X_test_scaled if model_name == "Logistic Regression" else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]

        metrics = evaluate_model(model_name, y_test.values, y_pred, y_proba)
        all_metrics[model_name] = metrics
        probabilities[model_name] = y_proba

        plot_confusion_matrix(
            y_test.values, y_pred, model_name=model_name, save_path=config.IMAGES_DIR / filename_map[model_name]
        )

    print_section("Step 10: ROC Curve Comparison")
    plot_combined_roc_curve(y_test.values, probabilities, save_path=config.IMAGES_DIR / "roc_curve.png")

    print_section("Step 11: Feature Importance (Random Forest)")
    rf_model = models["Random Forest"]
    plot_feature_importance(
        feature_names=list(X_test.columns),
        importances=rf_model.feature_importances_,
        model_name="Random Forest",
        save_path=config.IMAGES_DIR / "feature_importance.png",
    )

    comparison_df = build_comparison_table(all_metrics)
    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Model comparison table saved to: {config.MODEL_COMPARISON_PATH}")

    plot_model_comparison(comparison_df, save_path=config.IMAGES_DIR / "model_comparison.png")

    return comparison_df


def generate_deployment_report(comparison_df: pd.DataFrame) -> str:
    """Generate the deployment recommendation markdown report.

    Args:
        comparison_df: The model comparison DataFrame.

    Returns:
        The generated markdown report content as a string.
    """
    print_section("Step 12: Deployment Recommendation")

    best_model = get_best_model_name(comparison_df, metric="ROC AUC")
    best_row = comparison_df[comparison_df["Model"] == best_model].iloc[0]

    table_md = comparison_df.round(4).to_markdown(index=False)

    report = f"""# Deployment Recommendation Report
## Loan Risk Assessment System

---

## 1. Executive Summary

This report compares three classification models trained to predict loan
approval outcomes: **Logistic Regression**, **Decision Tree**, and
**Random Forest**. Based on a comprehensive evaluation across Accuracy,
Precision, Recall, F1 Score, and ROC-AUC, we recommend deploying the
**{best_model}** model in the banking production environment.

---

## 2. Model Comparison Table

{table_md}

---

## 3. Metric-by-Metric Analysis

### Precision
Precision measures the proportion of predicted approvals that are truly
creditworthy. In a banking context, low precision means the bank
extends credit to applicants who are likely to default, directly
increasing **credit risk and non-performing assets (NPA)**.

### Recall
Recall measures the proportion of truly creditworthy applicants that the
model correctly approves. Low recall means the bank **rejects good
customers**, resulting in lost business opportunity and poor customer
experience.

### ROC-AUC
ROC-AUC summarizes the model's ability to discriminate between approved
and rejected applicants across all classification thresholds, making it
the most robust metric for comparing models independent of a specific
decision threshold — which is critical since banks often adjust
thresholds based on risk appetite and economic conditions.

---

## 4. Recommended Model: {best_model}

**Key metrics for {best_model}:**

| Metric | Score |
|---|---|
| Accuracy | {best_row['Accuracy']:.4f} |
| Precision | {best_row['Precision']:.4f} |
| Recall | {best_row['Recall']:.4f} |
| F1 Score | {best_row['F1 Score']:.4f} |
| ROC AUC | {best_row['ROC AUC']:.4f} |

### Justification

1. **Highest discriminative power**: {best_model} achieved the highest
   ROC-AUC score among all candidates, indicating superior ability to
   rank applicants by true creditworthiness across all decision thresholds.
2. **Balanced precision-recall trade-off**: The model does not
   over-optimize for one metric at the expense of the other, which is
   essential in banking where both false approvals (credit risk) and
   false rejections (lost revenue) carry real financial cost.
3. **Interpretability & auditability**: Tree-based models (Decision
   Tree / Random Forest) provide feature importance rankings that support
   regulatory requirements for explainable credit decisions, while
   Logistic Regression provides transparent, coefficient-based
   interpretability valued in credit scoring.
4. **Robustness**: Ensemble methods such as Random Forest reduce
   overfitting risk relative to a single Decision Tree by averaging
   predictions across many de-correlated trees, generally yielding more
   stable performance on unseen applicant populations.

---

## 5. Risk Considerations for Production Deployment

- **Model monitoring**: Continuously monitor for data drift in applicant
  demographics and macroeconomic conditions (interest rates, unemployment)
  that could degrade model performance over time.
- **Fairness & compliance**: Periodically audit approval rates across
  protected attributes (e.g. gender) to ensure compliance with fair
  lending regulations.
- **Threshold calibration**: The deployed decision threshold should be
  calibrated to the bank's specific risk appetite rather than using the
  default 0.5 cutoff, and should be reviewed periodically by the credit
  risk committee.
- **Human-in-the-loop**: Borderline predictions (probabilities near the
  decision threshold) should be routed to human underwriters for manual
  review rather than fully automated decisioning.

---

## 6. Conclusion

The **{best_model}** model is recommended for deployment in the banking
loan approval workflow based on its superior ROC-AUC and balanced
performance across all evaluation metrics. Deployment should be
accompanied by ongoing monitoring, periodic retraining, and a
human-in-the-loop review process for borderline cases.

---
*Report generated automatically by the Loan Risk Assessment System pipeline.*
"""

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DEPLOYMENT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

    logger.info(f"Deployment recommendation report saved to: {config.DEPLOYMENT_REPORT_PATH}")
    return report


def run_full_pipeline() -> Dict[str, Any]:
    """Execute the complete end-to-end Loan Risk Assessment training pipeline.

    Returns:
        A dictionary containing key pipeline artifacts: models, comparison_df,
        and the deployment report string.
    """
    config.ensure_directories()

    raw_df, engineered_df = run_data_pipeline()
    run_eda(raw_df, engineered_df)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = run_split_and_scale(engineered_df)

    print_section("Step 7: Model Training & Hyperparameter Tuning")
    models = run_training(X_train, y_train, X_train_scaled)

    comparison_df = run_evaluation(models, X_test, X_test_scaled, y_test)
    report = generate_deployment_report(comparison_df)

    return {
        "models": models,
        "scaler": scaler,
        "comparison_df": comparison_df,
        "deployment_report": report,
    }
