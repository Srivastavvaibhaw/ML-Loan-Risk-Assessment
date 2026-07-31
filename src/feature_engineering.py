"""
feature_engineering.py
------------------------
Feature creation and encoding routines: derived income/ratio features,
Dependents numeric conversion, target label encoding, and one-hot encoding
of categorical predictors.
"""

from typing import List

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def create_total_income(df: pd.DataFrame) -> pd.DataFrame:
    """Create a `total_income` feature as the sum of applicant and co-applicant income.

    Args:
        df: Input DataFrame containing 'ApplicantIncome' and 'CoapplicantIncome'.

    Returns:
        DataFrame with a new 'total_income' column.

    Raises:
        KeyError: If required income columns are missing.
    """
    required = {"ApplicantIncome", "CoapplicantIncome"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for total_income: {missing}")

    df = df.copy()
    df["total_income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    logger.info("Created feature: 'total_income'")
    return df


def create_loan_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create a `loan_income_ratio` feature as LoanAmount / total_income.

    A small epsilon is added to the denominator to avoid division by zero.

    Args:
        df: Input DataFrame containing 'LoanAmount' and 'total_income'.

    Returns:
        DataFrame with a new 'loan_income_ratio' column.

    Raises:
        KeyError: If required columns are missing.
    """
    required = {"LoanAmount", "total_income"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for loan_income_ratio: {missing}")

    df = df.copy()
    epsilon = 1e-6
    df["loan_income_ratio"] = df["LoanAmount"] / (df["total_income"] + epsilon)
    logger.info("Created feature: 'loan_income_ratio'")
    return df


def convert_dependents_to_numeric(df: pd.DataFrame, column: str = "Dependents") -> pd.DataFrame:
    """Convert the categorical `Dependents` column ('0', '1', '2', '3+') to numeric.

    The '3+' category is mapped to the integer 3.

    Args:
        df: Input DataFrame.
        column: Name of the Dependents column.

    Returns:
        DataFrame with the Dependents column converted to integer dtype.

    Raises:
        KeyError: If the column is not present in the DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    df = df.copy()
    df[column] = df[column].astype(str).str.replace("3+", "3", regex=False)
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    logger.info(f"Converted '{column}' to numeric (3+ mapped to 3)")
    return df


def encode_target(df: pd.DataFrame, column: str = "Loan_Status") -> pd.DataFrame:
    """Encode the target column: 'Y' -> 1, 'N' -> 0.

    Args:
        df: Input DataFrame.
        column: Name of the target column.

    Returns:
        DataFrame with the target column encoded as integers.

    Raises:
        KeyError: If the target column is not present.
        ValueError: If the target contains values other than 'Y'/'N'.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    df = df.copy()
    unique_values = set(df[column].dropna().unique())
    if not unique_values.issubset({"Y", "N"}):
        raise ValueError(f"Unexpected values in target column '{column}': {unique_values}")

    df[column] = df[column].map({"Y": 1, "N": 0}).astype(int)
    logger.info(f"Encoded target column '{column}': Y -> 1, N -> 0")
    return df


def one_hot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """One-hot encode a list of categorical columns.

    Args:
        df: Input DataFrame.
        columns: List of categorical column names to one-hot encode.

    Returns:
        DataFrame with the specified columns one-hot encoded (drop_first=True
        to avoid the dummy variable trap) and boolean dummies cast to int.

    Raises:
        KeyError: If any requested column is missing.
    """
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for one-hot encoding: {missing}")

    df = df.copy()
    encoded = pd.get_dummies(df, columns=columns, drop_first=True)

    # Ensure dummy columns are integer (not bool) for downstream model compatibility
    new_cols = [c for c in encoded.columns if c not in df.columns]
    encoded[new_cols] = encoded[new_cols].astype(int)

    logger.info(f"One-hot encoded columns: {columns}. New shape: {encoded.shape}")
    return encoded


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
        1. Create 'total_income'.
        2. Create 'loan_income_ratio'.
        3. Convert 'Dependents' to numeric.
        4. Encode the 'Loan_Status' target.
        5. One-hot encode categorical predictors.

    Args:
        df: Cleaned input DataFrame (post `preprocessing.clean_dataset`).

    Returns:
        A fully engineered, numeric-only DataFrame ready for modeling.
    """
    from src.config import ONE_HOT_COLUMNS, TARGET_COLUMN

    logger.info("Starting feature engineering pipeline...")

    engineered = create_total_income(df)
    engineered = create_loan_income_ratio(engineered)
    engineered = convert_dependents_to_numeric(engineered)
    engineered = encode_target(engineered, column=TARGET_COLUMN)
    engineered = one_hot_encode(engineered, columns=ONE_HOT_COLUMNS)

    logger.info(f"Feature engineering complete. Final shape: {engineered.shape}")
    return engineered
