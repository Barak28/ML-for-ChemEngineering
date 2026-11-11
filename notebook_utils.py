"""Shared utilities for teaching notebooks in ML for Chemical Engineering.

These helpers keep repeated notebook patterns consistent and reduce boilerplate.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across common ML libraries."""
    random.seed(seed)
    np.random.seed(seed)


def set_plot_style() -> None:
    """Apply a clean default style for plots used in teaching notebooks."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def ensure_2d_array(X: Iterable[float]) -> np.ndarray:
    """Convert an input array-like to a 2D NumPy array if needed."""
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def regression_summary(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Return a compact set of regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_summary(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Return a compact set of classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def show_table(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Display a quick preview of a dataframe."""
    return df.head(n_rows)


__all__ = [
    "seed_everything",
    "set_plot_style",
    "ensure_2d_array",
    "regression_summary",
    "classification_summary",
    "show_table",
]
