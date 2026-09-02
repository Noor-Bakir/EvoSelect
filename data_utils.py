"""Data loading and leakage-safe preprocessing utilities for EvoSelect."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class DatasetInfo:
    target: str
    n_samples: int
    n_features: int
    feature_names: list[str]
    classes: list


def generate_random_dataset(
    n_rows: int = 200,
    n_cols: int = 30,
    n_informative: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, str]:
    if n_rows < 20:
        raise ValueError("n_rows must be at least 20.")
    if n_cols < 2:
        raise ValueError("n_cols must be at least 2.")
    if not 1 <= n_informative < n_cols:
        raise ValueError("n_informative must be between 1 and n_cols - 1.")

    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_cols,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
        shuffle=True,
    )
    columns = [f"feature_{i + 1}" for i in range(n_cols)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df, "target"


def read_csv_text(raw_csv: str) -> pd.DataFrame:
    if not raw_csv or not raw_csv.strip():
        raise ValueError("Dataset is empty.")
    df = pd.read_csv(io.StringIO(raw_csv))
    if df.empty:
        raise ValueError("Dataset contains no rows.")
    return df


def validate_dataset(df: pd.DataFrame, target_name: str) -> DatasetInfo:
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' does not exist.")

    data = df.dropna(axis=1, how="all").copy()
    if data.shape[1] < 2:
        raise ValueError("Dataset must contain a target and at least one feature.")

    X = data.drop(columns=[target_name])
    if X.shape[1] == 0:
        raise ValueError("Dataset has no feature columns.")

    # EvoSelect is currently a numerical feature-selection benchmark.
    # Do not silently convert categorical strings to arbitrary integer codes.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "EvoSelect currently requires numeric features. "
            f"Non-numeric columns: {non_numeric}"
        )

    y = data[target_name]
    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    if y.nunique(dropna=False) < 2:
        raise ValueError("Target must contain at least two classes.")

    return DatasetInfo(
        target=target_name,
        n_samples=len(data),
        n_features=X.shape[1],
        feature_names=X.columns.tolist(),
        classes=pd.unique(y).tolist(),
    )


def prepare_xy(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw numeric X/y. Imputation/scaling belongs inside CV pipelines."""
    validate_dataset(df, target_name)
    data = df.dropna(axis=1, how="all").copy()
    X = data.drop(columns=[target_name]).apply(pd.to_numeric, errors="raise")
    y = data[target_name]

    # Encode target labels deterministically without touching X.
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(
            pd.Categorical(y).codes,
            index=y.index,
            name=target_name,
        )
    return X, y
