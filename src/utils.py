"""
utils.py
--------
Shared helper utilities used across the Loan Risk Assessment pipeline
(logging, joblib persistence helpers, generic error-handling wrappers).
"""

import logging
import sys
from pathlib import Path
from typing import Any

import joblib


def get_logger(name: str = "loan_risk_assessment") -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Args:
        name: Name of the logger.

    Returns:
        A configured `logging.Logger` instance that writes to stdout.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger()


def save_object(obj: Any, path: Path) -> None:
    """Persist a Python object to disk using joblib.

    Args:
        obj: The object to serialize (e.g. a trained model or scaler).
        path: Destination file path.

    Raises:
        IOError: If the object cannot be written to disk.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        logger.info(f"Saved object to: {path}")
    except (IOError, OSError) as exc:
        logger.error(f"Failed to save object to {path}: {exc}")
        raise


def load_object(path: Path) -> Any:
    """Load a Python object previously persisted with joblib.

    Args:
        path: Path to the serialized object.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the object cannot be read from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    try:
        obj = joblib.load(path)
        logger.info(f"Loaded object from: {path}")
        return obj
    except (IOError, OSError) as exc:
        logger.error(f"Failed to load object from {path}: {exc}")
        raise


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header to stdout for readable console output.

    Args:
        title: The section title to display.
        width: Total width of the separator line.
    """
    print("\n" + "=" * width)
    print(title.upper())
    print("=" * width)
