"""
data_loader.py
---------------
Utilities for loading the raw loan prediction dataset and displaying an
initial inspection summary (shape, info, missing values, statistics).
"""

from pathlib import Path

import pandas as pd

from src.utils import get_logger, print_section

logger = get_logger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the loan prediction dataset from a CSV file.

    Args:
        path: Path to the CSV file containing the raw dataset.

    Returns:
        A pandas DataFrame with the raw loan data.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the loaded dataset is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        logger.error(f"Failed to parse CSV file at {path}: {exc}")
        raise

    if df.empty:
        raise ValueError(f"Dataset loaded from {path} is empty.")

    logger.info(f"Dataset loaded successfully from {path} with shape {df.shape}")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a comprehensive inspection summary of the dataset.

    Displays shape, schema info, missing-value counts, and descriptive
    statistics for both numeric and categorical columns.

    Args:
        df: The DataFrame to inspect.
    """
    print_section("Dataset Shape")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print_section("Dataset Info")
    df.info()

    print_section("Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(missing_summary[missing_summary["missing_count"] > 0].sort_values("missing_count", ascending=False))

    print_section("Numeric Statistics")
    print(df.describe().T)

    print_section("Categorical Statistics")
    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe().T)
    else:
        print("No categorical columns found.")
