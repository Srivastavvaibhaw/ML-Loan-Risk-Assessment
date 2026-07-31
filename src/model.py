"""
model.py
--------
Model definitions, training, and hyperparameter tuning for the Loan Risk
Assessment System: Logistic Regression, Decision Tree, and Random Forest.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.utils import get_logger, print_section

logger = get_logger(__name__)


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on the training data and transform train/test sets.

    Scaling is used exclusively for Logistic Regression, which (unlike
    tree-based models) is sensitive to feature magnitude.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.

    Returns:
        A tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Fitted StandardScaler on training data and transformed train/test sets.")
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train: np.ndarray, y_train: pd.Series, random_state: int = 42
) -> LogisticRegression:
    """Train a Logistic Regression classifier on scaled features.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training target labels.
        random_state: Random seed for reproducibility.

    Returns:
        A fitted LogisticRegression model.

    Raises:
        ValueError: If training data is empty.
    """
    if len(X_train) == 0:
        raise ValueError("Training data is empty; cannot fit Logistic Regression.")

    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    model.fit(X_train, y_train)
    logger.info("Trained Logistic Regression model.")
    return model


def train_decision_tree_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, list],
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier with GridSearchCV hyperparameter tuning.

    Args:
        X_train: Training feature matrix (unscaled; tree models are scale-invariant).
        y_train: Training target labels.
        param_grid: Hyperparameter grid to search over.
        cv: Number of cross-validation folds.
        scoring: Scoring metric used to select the best estimator.
        random_state: Random seed for reproducibility.

    Returns:
        The best fitted DecisionTreeClassifier found via grid search.

    Raises:
        ValueError: If training data is empty.
    """
    if len(X_train) == 0:
        raise ValueError("Training data is empty; cannot fit Decision Tree.")

    base_model = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Decision Tree GridSearchCV best params: {grid_search.best_params_}")
    logger.info(f"Decision Tree GridSearchCV best CV {scoring}: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train_random_forest_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, list],
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with GridSearchCV hyperparameter tuning.

    Args:
        X_train: Training feature matrix (unscaled; tree models are scale-invariant).
        y_train: Training target labels.
        param_grid: Hyperparameter grid to search over (n_estimators, max_depth,
            min_samples_split, max_features).
        cv: Number of cross-validation folds.
        scoring: Scoring metric used to select the best estimator.
        random_state: Random seed for reproducibility.

    Returns:
        The best fitted RandomForestClassifier found via grid search.

    Raises:
        ValueError: If training data is empty.
    """
    if len(X_train) == 0:
        raise ValueError("Training data is empty; cannot fit Random Forest.")

    base_model = RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Random Forest GridSearchCV best params: {grid_search.best_params_}")
    logger.info(f"Random Forest GridSearchCV best CV {scoring}: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_scaled: np.ndarray,
    dt_param_grid: Dict[str, list],
    rf_param_grid: Dict[str, list],
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Logistic Regression, Decision Tree, and Random Forest models.

    Args:
        X_train: Unscaled training feature matrix (used for tree models).
        y_train: Training target labels.
        X_train_scaled: Scaled training feature matrix (used for Logistic Regression).
        dt_param_grid: Hyperparameter grid for the Decision Tree.
        rf_param_grid: Hyperparameter grid for the Random Forest.
        cv: Number of cross-validation folds.
        scoring: Scoring metric used for model selection.
        random_state: Random seed for reproducibility.

    Returns:
        A dictionary mapping model name -> fitted model instance.
    """
    print_section("Model Training")

    logger.info("Training Logistic Regression...")
    log_reg = train_logistic_regression(X_train_scaled, y_train, random_state=random_state)

    logger.info("Training Decision Tree (with GridSearchCV tuning)...")
    decision_tree = train_decision_tree_with_tuning(
        X_train, y_train, param_grid=dt_param_grid, cv=cv, scoring=scoring, random_state=random_state
    )

    logger.info("Training Random Forest (with GridSearchCV tuning)...")
    random_forest = train_random_forest_with_tuning(
        X_train, y_train, param_grid=rf_param_grid, cv=cv, scoring=scoring, random_state=random_state
    )

    return {
        "Logistic Regression": log_reg,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }
