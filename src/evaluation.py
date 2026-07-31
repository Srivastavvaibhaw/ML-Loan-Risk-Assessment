"""
evaluation.py
-------------
Model evaluation utilities: computing classification metrics, generating
classification reports, and building the final model comparison table.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger, print_section

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute the core classification metrics for a single model.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        A dictionary with keys: Accuracy, Precision, Recall, F1 Score, ROC AUC.

    Raises:
        ValueError: If input arrays have mismatched lengths.
    """
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("y_true, y_pred, and y_proba must have the same length.")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_proba),
    }


def evaluate_model(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Evaluate a single model and print a detailed classification report.

    Args:
        model_name: Display name of the model.
        y_true: Ground-truth labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        A dictionary of computed metrics for this model.
    """
    print_section(f"Evaluation Report - {model_name}")

    metrics = compute_metrics(y_true, y_pred, y_proba)
    for metric_name, value in metrics.items():
        print(f"{metric_name:<12}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Rejected (0)", "Approved (1)"], zero_division=0))

    return metrics


def build_comparison_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build a model comparison DataFrame from a dictionary of per-model metrics.

    Args:
        all_metrics: Mapping of model name -> metrics dictionary (as returned
            by `compute_metrics`).

    Returns:
        A DataFrame with columns: Model, Accuracy, Precision, Recall, F1 Score, ROC AUC,
        sorted by ROC AUC descending.
    """
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"Model": model_name, **metrics}
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("ROC AUC", ascending=False).reset_index(drop=True)

    print_section("Model Comparison Table")
    print(comparison_df.to_string(index=False))

    return comparison_df


def get_best_model_name(comparison_df: pd.DataFrame, metric: str = "ROC AUC") -> str:
    """Identify the best-performing model based on a given metric.

    Args:
        comparison_df: Model comparison DataFrame (as returned by `build_comparison_table`).
        metric: The metric column to use for ranking.

    Returns:
        The name of the best-performing model.

    Raises:
        KeyError: If the metric column does not exist in the DataFrame.
    """
    if metric not in comparison_df.columns:
        raise KeyError(f"Metric '{metric}' not found in comparison table columns.")

    best_row = comparison_df.sort_values(metric, ascending=False).iloc[0]
    return best_row["Model"]
