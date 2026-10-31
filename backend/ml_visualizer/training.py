"""ML Visualizer - Training and prediction utilities."""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from .model_registry import ModelID
from .preprocessing import SplitResult


@dataclass
class TrainingResult:
    model: Pipeline
    model_id: ModelID
    params: dict
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    train_time: float
    feature_names: list[str]
    target_name: Optional[str]
    target_names: Optional[list[str]] = None


def train_model(
    model_id: ModelID,
    params: dict,
    split_result: SplitResult,
    target_name: Optional[str] = None,
    target_names: Optional[list[str]] = None,
) -> TrainingResult:
    """Train a model on the provided split data."""
    import time

    from .model_registry import create_pipeline

    start_time = time.time()

    X_train = split_result.X_train
    X_test = split_result.X_test
    y_train = split_result.y_train
    y_test = split_result.y_test

    pipeline = create_pipeline(model_id, params, scale=(split_result.scaler is not None))

    if model_id == ModelID.KMEANS:
        pipeline.fit(X_train)
    else:
        if y_train is None:
            raise ValueError(f"Model {model_id} requires target labels")
        pipeline.fit(X_train, y_train)

    train_time = time.time() - start_time

    return TrainingResult(
        model=pipeline,
        model_id=model_id,
        params=params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_time=train_time,
        feature_names=split_result.feature_names,
        target_name=target_name,
        target_names=target_names,
    )


def predict(
    training_result: TrainingResult,
    X: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions using the trained model.
    Returns (predictions, probabilities_or_none).
    """
    model = training_result.model
    model_id = training_result.model_id

    if X is None:
        X = training_result.X_test

    if model_id == ModelID.KMEANS:
        predictions = model.predict(X)
        return predictions, None

    predictions = model.predict(X)

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
        except Exception:
            pass

    return predictions, probabilities


def get_feature_importance(training_result: TrainingResult) -> Optional[pd.Series]:
    """Extract feature importance from trained model if available."""
    model = training_result.model
    model_id = training_result.model_id

    try:
        if model_id in [ModelID.LOGISTIC_REGRESSION, ModelID.LINEAR_REGRESSION]:
            estimator = model.named_steps["model"]
            if hasattr(estimator, "coef_"):
                coef = estimator.coef_
                if coef.ndim > 1:
                    importance = np.abs(coef).mean(axis=0)
                else:
                    importance = np.abs(coef)
                return pd.Series(importance, index=training_result.feature_names).sort_values(ascending=False)

        elif model_id in [ModelID.DECISION_TREE, ModelID.RANDOM_FOREST]:
            estimator = model.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                return pd.Series(
                    estimator.feature_importances_,
                    index=training_result.feature_names
                ).sort_values(ascending=False)

        elif model_id == ModelID.KMEANS:
            return None

    except Exception:
        pass

    return None


def get_decision_boundary_data(
    training_result: TrainingResult,
    feature_indices: tuple[int, int],
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate meshgrid and predictions for decision boundary visualization.
    Returns (xx, yy, Z) where Z is the predicted class for each grid point.
    """
    X_train = training_result.X_train
    feature_names = training_result.feature_names

    f1_idx, f2_idx = feature_indices
    f1_name = feature_names[f1_idx]
    f2_name = feature_names[f2_idx]

    x_min, x_max = X_train[f1_name].min() - 0.5, X_train[f1_name].max() + 0.5
    y_min, y_max = X_train[f2_name].min() - 0.5, X_train[f2_name].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=[f1_name, f2_name]
    )

    for col in X_train.columns:
        if col not in [f1_name, f2_name]:
            grid_points[col] = X_train[col].mean()

    grid_points = grid_points[X_train.columns]

    predictions, _ = predict(training_result, grid_points)
    Z = predictions.reshape(xx.shape)

    return xx, yy, Z