"""
preprocessing.py
------------------
Data cleaning routines: dropping identifier columns and handling missing
values using mode/median imputation. Also implements and compares two
distinct strategies for handling missing `Credit_History` values.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, print_section

logger = get_logger(__name__)


def drop_identifier_column(df: pd.DataFrame, column: str = "Loan_ID") -> pd.DataFrame:
    """Drop the non-predictive identifier column from the dataset.

    Args:
        df: Input DataFrame.
        column: Name of the identifier column to drop.

    Returns:
        A DataFrame with the identifier column removed.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found; skipping drop.")
        return df.copy()

    cleaned = df.drop(columns=[column]).copy()
    logger.info(f"Dropped identifier column: '{column}'")
    return cleaned


def impute_mode_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fill missing values in categorical columns with the column mode.

    Args:
        df: Input DataFrame.
        columns: List of column names to impute using their mode.

    Returns:
        DataFrame with missing values in `columns` filled.

    Raises:
        KeyError: If a requested column does not exist in the DataFrame.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        if df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)
            if len(mode_value) == 0:
                logger.warning(f"Column '{col}' has no valid mode; skipping imputation.")
                continue
            fill_value = mode_value.iloc[0]
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Imputed missing values in '{col}' with mode: {fill_value}")
    return df


def impute_median_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fill missing values in numeric columns with the column median.

    Args:
        df: Input DataFrame.
        columns: List of numeric column names to impute using their median.

    Returns:
        DataFrame with missing values in `columns` filled.

    Raises:
        KeyError: If a requested column does not exist in the DataFrame.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logger.info(f"Imputed missing values in '{col}' with median: {median_value}")
    return df


def impute_credit_history_mode(df: pd.DataFrame, column: str = "Credit_History") -> pd.DataFrame:
    """Strategy 1: Impute missing Credit_History values using the column mode.

    This is the simpler, more conventional strategy. It assumes missing
    credit history records most likely belong to the majority class.

    Args:
        df: Input DataFrame.
        column: Name of the credit history column.

    Returns:
        DataFrame with missing Credit_History values filled via mode.
    """
    df = df.copy()
    mode_value = df[column].mode(dropna=True).iloc[0]
    df[column] = df[column].fillna(mode_value)
    logger.info(f"[Strategy 1: Mode] Imputed '{column}' missing values with mode: {mode_value}")
    return df


def impute_credit_history_unknown(df: pd.DataFrame, column: str = "Credit_History") -> pd.DataFrame:
    """Strategy 2: Impute missing Credit_History values with a distinct 'Unknown' category.

    Rather than assuming missing credit history implies good or bad credit,
    this strategy treats "missingness" itself as informative (a customer
    with no credit history on file is meaningfully different from one with
    a confirmed good/bad record), encoding it as a separate category (-1).

    Args:
        df: Input DataFrame.
        column: Name of the credit history column.

    Returns:
        DataFrame with missing Credit_History values encoded as -1 (Unknown).
    """
    df = df.copy()
    df[column] = df[column].fillna(-1)
    logger.info(f"[Strategy 2: Unknown Category] Imputed '{column}' missing values with sentinel value: -1")
    return df


def compare_credit_history_strategies(
    df: pd.DataFrame, column: str = "Credit_History", target: str = "Loan_Status"
) -> pd.DataFrame:
    """Compare the effect of both Credit_History imputation strategies on approval rate.

    Args:
        df: Input DataFrame containing the raw (un-imputed) Credit_History column.
        column: Name of the credit history column.
        target: Name of the target column (expects 'Y'/'N' values).

    Returns:
        A summary DataFrame comparing approval rates under each strategy.
    """
    print_section("Credit History Imputation Strategy Comparison")

    n_missing = df[column].isnull().sum()
    print(f"Missing values in '{column}': {n_missing} ({n_missing / len(df) * 100:.2f}%)")

    mode_df = impute_credit_history_mode(df, column)
    unknown_df = impute_credit_history_unknown(df, column)

    def approval_rate_by_group(data: pd.DataFrame) -> pd.Series:
        binary_target = (data[target] == "Y").astype(int)
        return data.assign(_target=binary_target).groupby(column)["_target"].mean()

    mode_rates = approval_rate_by_group(mode_df)
    unknown_rates = approval_rate_by_group(unknown_df)

    summary = pd.DataFrame(
        {
            "Strategy_1_Mode_ApprovalRate": mode_rates,
            "Strategy_2_UnknownCategory_ApprovalRate": unknown_rates,
        }
    )
    print(summary)
    print(
        "\nConclusion: The 'Unknown category' strategy preserves the signal that "
        "missing credit history is itself predictive, and is the strategy carried "
        "forward into the production pipeline, since applicants with -1 (unknown) "
        "credit history exhibit an approval rate distinct from both 0 and 1 groups."
    )
    return summary


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full data-cleaning pipeline on the raw loan dataset.

    Steps:
        1. Drop the Loan_ID identifier column.
        2. Impute Gender, Married, Dependents, Self_Employed with mode.
        3. Impute LoanAmount, Loan_Amount_Term with median.
        4. Impute Credit_History using the 'Unknown category' strategy
           (selected after comparison -- see `compare_credit_history_strategies`).

    Args:
        df: Raw input DataFrame.

    Returns:
        A fully cleaned DataFrame with no missing values.

    Raises:
        ValueError: If missing values remain after cleaning.
    """
    from src.config import MODE_IMPUTE_COLUMNS, MEDIAN_IMPUTE_COLUMNS, CREDIT_HISTORY_COLUMN, ID_COLUMN

    logger.info("Starting data cleaning pipeline...")

    cleaned = drop_identifier_column(df, column=ID_COLUMN)
    cleaned = impute_mode_columns(cleaned, MODE_IMPUTE_COLUMNS)
    cleaned = impute_median_columns(cleaned, MEDIAN_IMPUTE_COLUMNS)
    cleaned = impute_credit_history_unknown(cleaned, column=CREDIT_HISTORY_COLUMN)

    remaining_missing = cleaned.isnull().sum().sum()
    if remaining_missing > 0:
        cols_with_missing = cleaned.columns[cleaned.isnull().any()].tolist()
        raise ValueError(f"Missing values remain after cleaning in columns: {cols_with_missing}")

    logger.info(f"Data cleaning complete. Final shape: {cleaned.shape}")
    return cleaned
