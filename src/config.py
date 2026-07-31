"""
config.py
---------
Centralized configuration for the Loan Risk Assessment System.

Keeping all paths, constants, and hyperparameter grids in a single module
avoids "magic strings/numbers" scattered across the codebase and makes the
pipeline easy to reconfigure.
"""

from pathlib import Path
from typing import Dict, List

# ----------------------------------------------------------------------
# Project root & directory paths
# ----------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
IMAGES_DIR: Path = PROJECT_ROOT / "Images"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

RAW_DATA_PATH: Path = DATASET_DIR / "loan_prediction.csv"
CLEANED_DATA_PATH: Path = OUTPUTS_DIR / "cleaned_dataset.csv"
TRAIN_DATA_PATH: Path = OUTPUTS_DIR / "train.csv"
TEST_DATA_PATH: Path = OUTPUTS_DIR / "test.csv"
MODEL_COMPARISON_PATH: Path = OUTPUTS_DIR / "model_comparison.csv"
DEPLOYMENT_REPORT_PATH: Path = REPORTS_DIR / "deployment_recommendation.md"

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RANDOM_STATE: int = 42

# ----------------------------------------------------------------------
# Train / test split configuration
# ----------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ----------------------------------------------------------------------
# Target / identifier columns
# ----------------------------------------------------------------------
TARGET_COLUMN: str = "Loan_Status"
ID_COLUMN: str = "Loan_ID"

# ----------------------------------------------------------------------
# Column groups
# ----------------------------------------------------------------------
MODE_IMPUTE_COLUMNS: List[str] = ["Gender", "Married", "Dependents", "Self_Employed"]
MEDIAN_IMPUTE_COLUMNS: List[str] = ["LoanAmount", "Loan_Amount_Term"]
CREDIT_HISTORY_COLUMN: str = "Credit_History"

ONE_HOT_COLUMNS: List[str] = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# ----------------------------------------------------------------------
# Model persistence file names
# ----------------------------------------------------------------------
MODEL_FILENAMES: Dict[str, str] = {
    "logistic_regression": "logistic_regression.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
    "scaler": "scaler.pkl",
}

# ----------------------------------------------------------------------
# Hyperparameter search grids (GridSearchCV)
# ----------------------------------------------------------------------
DECISION_TREE_PARAM_GRID: Dict[str, List] = {
    "max_depth": [3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
}

RANDOM_FOREST_PARAM_GRID: Dict[str, List] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}

CV_FOLDS: int = 5
SCORING_METRIC: str = "roc_auc"

# ----------------------------------------------------------------------
# Plotting configuration
# ----------------------------------------------------------------------
FIGURE_DPI: int = 300
FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"
PALETTE: str = "viridis"

# ----------------------------------------------------------------------
# Ensure required directories always exist
# ----------------------------------------------------------------------
def ensure_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in (DATASET_DIR, OUTPUTS_DIR, MODELS_DIR, IMAGES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
