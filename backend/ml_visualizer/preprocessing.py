"""ML Visualizer - Preprocessing utilities."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from .datasets import TaskType


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    scaler: Optional[StandardScaler]
    feature_names: list[str]


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Scaler that preserves DataFrame structure with column names."""

    def __init__(self, scaler=None):
        self.scaler = scaler or StandardScaler()
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            self.scaler.fit(X.values)
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            scaled = self.scaler.transform(X.values)
            return pd.DataFrame(scaled, columns=X.columns, index=X.index)
        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            inversed = self.scaler.inverse_transform(X.values)
            return pd.DataFrame(inversed, columns=X.columns, index=X.index)
        return self.scaler.inverse_transform(X)


def split_and_scale(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    scaler_type: str = "standard",
    task_type: TaskType = TaskType.CLASSIFICATION,
) -> SplitResult:
    """
    Split data and optionally scale features.
    Fits scaler on training data only to prevent leakage.
    Stratification is only used for classification tasks.
    """
    feature_names = X.columns.tolist()

    stratify = y if (y is not None and task_type == TaskType.CLASSIFICATION) else None

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        y_train = y_test = None

    scaler = None
    if scale:
        if scaler_type == "standard":
            scaler = DataFrameScaler(StandardScaler())
        elif scaler_type == "minmax":
            scaler = DataFrameScaler(MinMaxScaler())
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=feature_names,
    )


def validate_data(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[bool, str]:
    """Validate data for training."""
    if X.empty:
        return False, "Feature matrix is empty"
    if X.isnull().any().any():
        return False, "Feature matrix contains NaN values"
    if np.isinf(X.values).any():
        return False, "Feature matrix contains infinite values"
    if y is not None:
        if y.isnull().any():
            return False, "Target contains NaN values"
        if len(X) != len(y):
            return False, f"Feature matrix length ({len(X)}) != target length ({len(y)})"
    return True, "OK"


def select_features(X: pd.DataFrame, feature_indices: list[int]) -> pd.DataFrame:
    """Select subset of features by column index."""
    if not feature_indices:
        return X
    max_idx = max(feature_indices)
    if max_idx >= len(X.columns):
        raise ValueError(f"Feature index {max_idx} out of range (max {len(X.columns)-1})")
    return X.iloc[:, feature_indices]