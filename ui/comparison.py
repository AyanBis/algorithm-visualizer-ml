"""Model comparison utilities for ML Visualizer."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ml_visualizer import (
    ModelID,
    TaskType,
    list_models,
    get_model_info,
    train_model,
    predict,
    split_and_scale,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
)


@dataclass
class ComparisonResult:
    model_name: str
    model_id: ModelID
    params: Dict[str, Any]
    metrics: Any
    training_time: float
    train_samples: int
    test_samples: int


def run_comparison(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    task_type: TaskType,
    model_configs: List[Dict[str, Any]],
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> List[ComparisonResult]:
    """
    Run fair comparison of multiple models on the same data split.

    Args:
        X: Feature matrix
        y: Target vector (None for clustering)
        task_type: Type of ML task
        model_configs: List of dicts with keys 'model_id', 'params', 'name'
        test_size: Fraction for test split
        random_state: Random seed
        scale_features: Whether to scale features

    Returns:
        List of ComparisonResult objects
    """
    # Single split for all models
    split_result = split_and_scale(
        X, y, test_size=test_size, random_state=random_state, scale=scale_features,
        task_type=task_type
    )

    results = []
    for config in model_configs:
        model_id = config["model_id"]
        params = config.get("params", {})
        name = config.get("name", model_id.value)

        training_result = train_model(
            model_id, params, split_result,
            target_name=getattr(y, "name", None) if y is not None else None,
        )

        y_pred, _ = predict(training_result)

        if task_type == TaskType.CLASSIFICATION:
            metrics = compute_classification_metrics(split_result.y_test.values, y_pred)
        elif task_type == TaskType.REGRESSION:
            metrics = compute_regression_metrics(split_result.y_test.values, y_pred)
        elif task_type == TaskType.CLUSTERING:
            inertia = training_result.model.named_steps["model"].inertia_
            metrics = compute_clustering_metrics(split_result.X_test.values, y_pred, inertia)
        else:
            metrics = None

        results.append(ComparisonResult(
            model_name=name,
            model_id=model_id,
            params=params,
            metrics=metrics,
            training_time=training_result.train_time,
            train_samples=len(split_result.X_train),
            test_samples=len(split_result.X_test),
        ))

    return results


def create_comparison_dataframe(results: List[ComparisonResult], task_type: TaskType) -> pd.DataFrame:
    """Create a comparison DataFrame from results."""
    rows = []
    for r in results:
        row = {
            "Model": r.model_name,
            "Train Time (s)": round(r.training_time, 3),
            "Train Samples": r.train_samples,
            "Test Samples": r.test_samples,
        }
        if r.metrics:
            if task_type == TaskType.CLASSIFICATION:
                row.update({
                    "Accuracy": round(r.metrics.accuracy, 4),
                    "Precision": round(r.metrics.precision, 4),
                    "Recall": round(r.metrics.recall, 4),
                    "F1": round(r.metrics.f1, 4),
                })
            elif task_type == TaskType.REGRESSION:
                row.update({
                    "MAE": round(r.metrics.mae, 4),
                    "RMSE": round(r.metrics.rmse, 4),
                    "R²": round(r.metrics.r2, 4),
                })
            elif task_type == TaskType.CLUSTERING:
                row.update({
                    "Inertia": round(r.metrics.inertia, 2),
                    "Silhouette": round(r.metrics.silhouette, 4) if r.metrics.silhouette is not None else None,
                })
        rows.append(row)

    return pd.DataFrame(rows)


def get_best_model(results: List[ComparisonResult], metric: str, task_type: TaskType) -> Optional[ComparisonResult]:
    """Get the best model for a given metric."""
    if not results:
        return None

    valid_results = [r for r in results if r.metrics and hasattr(r.metrics, metric)]
    if not valid_results:
        return None

    if metric in ["accuracy", "precision", "recall", "f1", "r2", "silhouette"]:
        return max(valid_results, key=lambda r: getattr(r.metrics, metric))
    else:
        return min(valid_results, key=lambda r: getattr(r.metrics, metric))