"""
visualization.py
------------------
EDA and model-evaluation plotting utilities. All figures are saved at
300 DPI in a professional, consistent style into the Images/ directory.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless script execution

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc

from src.utils import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

FIGURE_DPI = 300


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure to disk at 300 DPI and close it.

    Args:
        fig: The matplotlib Figure to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def plot_approval_rate_by_category(
    df: pd.DataFrame, category_col: str, target_col: str, title: str, save_path: Path
) -> None:
    """Plot loan approval rate broken down by a categorical column.

    Args:
        df: DataFrame containing the category and target columns.
        category_col: Name of the categorical column (e.g. 'Credit_History').
        target_col: Name of the target column ('Y'/'N' values expected).
        title: Plot title.
        save_path: Path to save the figure.
    """
    data = df.copy()
    data["_approved"] = (data[target_col] == "Y").astype(int)
    rates = data.groupby(category_col)["_approved"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rates.index.astype(str), rates.values, color=sns.color_palette("viridis", len(rates)))
    for bar, value in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.1%}", ha="center", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(category_col)
    ax.set_ylabel("Approval Rate")
    ax.set_ylim(0, 1.05)
    _save_figure(fig, save_path)


def plot_distribution(
    df: pd.DataFrame, column: str, title: str, save_path: Path, color: str = "#3b528b"
) -> None:
    """Plot a histogram + KDE distribution for a numeric column.

    Args:
        df: DataFrame containing the column.
        column: Name of the numeric column to visualize.
        title: Plot title.
        save_path: Path to save the figure.
        color: Fill color for the histogram.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df[column], kde=True, color=color, ax=ax, bins=30)
    ax.axvline(df[column].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {df[column].mean():.0f}")
    ax.axvline(
        df[column].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {df[column].median():.0f}"
    )
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.legend()
    _save_figure(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Plot a correlation heatmap for all numeric columns in the DataFrame.

    Args:
        df: DataFrame containing numeric columns.
        save_path: Path to save the figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap")
    _save_figure(fig, save_path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path
) -> None:
    """Plot and save a confusion matrix for a single model.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Display name of the model.
        save_path: Path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected (0)", "Approved (1)"])
    disp.plot(ax=ax, cmap="viridis", colorbar=True, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    _save_figure(fig, save_path)


def plot_combined_roc_curve(
    y_true: np.ndarray, model_probabilities: Dict[str, np.ndarray], save_path: Path
) -> None:
    """Plot a single combined ROC curve comparing multiple models.

    Args:
        y_true: Ground-truth labels (shared across all models).
        model_probabilities: Mapping of model name -> predicted probabilities
            for the positive class.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = sns.color_palette("viridis", len(model_probabilities))

    for (model_name, y_proba), color in zip(model_probabilities.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", color=color, linewidth=2.2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random Classifier (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison - All Models")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    _save_figure(fig, save_path)


def plot_feature_importance(
    feature_names: List[str], importances: np.ndarray, model_name: str, save_path: Path, top_n: int = 15
) -> None:
    """Plot the top-N most important features for a tree-based model.

    Args:
        feature_names: List of feature column names.
        importances: Array of feature importance scores (aligned with feature_names).
        model_name: Display name of the model.
        save_path: Path to save the figure.
        top_n: Number of top features to display.
    """
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", palette="viridis", ax=ax, legend=False)
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    _save_figure(fig, save_path)


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path) -> None:
    """Plot a grouped bar chart comparing all evaluation metrics across models.

    Args:
        comparison_df: DataFrame with columns ['Model', 'Accuracy', 'Precision',
            'Recall', 'F1 Score', 'ROC AUC'].
        save_path: Path to save the figure.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    melted = comparison_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="viridis", ax=ax)
    ax.set_title("Model Comparison Across Evaluation Metrics")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(title="Model", loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    _save_figure(fig, save_path)
