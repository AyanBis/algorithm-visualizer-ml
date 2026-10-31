"""Model comparison API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from ml_visualizer import (
    get_dataset_loader,
    get_model_info,
    list_models,
    TaskType,
    ModelID,
    split_and_scale,
    train_model,
    predict,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
    get_feature_importance,
    get_decision_boundary_data,
)
from ml_visualizer.model_registry import MODEL_REGISTRY

from backend.schemas.experiments import ExperimentConfig
from backend.schemas.models import ModelInfo
from backend.core.config import settings

import numpy as np
from fastapi import HTTPException
from typing import List, Optional
from pydantic import BaseModel


router = APIRouter()


class ModelComparisonConfig(BaseModel):
    model_id: str
    hyperparameters: dict = {}


class ComparisonRequest(BaseModel):
    dataset_id: str
    models: List[ModelComparisonConfig]
    test_size: float = 0.2
    random_state: int = 42
    scale_features: bool = True
    scaler_type: str = "standard"
    viz_features: Optional[List[int]] = None


class ModelComparisonResult(BaseModel):
    model_id: str
    model_name: str
    metrics: dict
    train_time_ms: float
    train_samples: int
    test_samples: int
    visualizations: dict = {}


class ComparisonResponse(BaseModel):
    experiment_ids: List[str]
    results: List[ModelComparisonResult]
    best_model: Optional[str] = None
    primary_metric: str


@router.post("/compare", response_model=ComparisonResponse)
async def compare_models(request: ComparisonRequest):
    """Compare multiple models on the same dataset with identical train/test split."""
    # Validate dataset
    loader = get_dataset_loader(random_state=request.random_state)
    try:
        dataset_info = loader.get_info(request.dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Validate models and check compatibility
    model_configs = []
    for model_config in request.models:
        try:
            model_info = get_model_info(ModelID(model_config.model_id))
        except (ValueError, KeyError):
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_config.model_id}")

        if model_info.task_type != dataset_info.task_type:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_info.name} ({model_info.task_type.value}) not compatible with dataset {dataset_info.name} ({dataset_info.task_type.value})"
            )
        model_configs.append({
            "model_id": model_config.model_id,
            "params": model_config.hyperparameters,
            "name": model_info.name,
        })

    if len(model_configs) < 2:
        raise HTTPException(status_code=400, detail="At least 2 models required for comparison")

    # Load dataset
    loader = get_dataset_loader(random_state=request.random_state)
    X, y, _ = loader.load(request.dataset_id)

    # Create single split for all models
    split_result = split_and_scale(
        X, y,
        test_size=request.test_size,
        random_state=request.random_state,
        scale=request.scale_features,
        scaler_type=request.scaler_type,
        task_type=dataset_info.task_type,
    )

    # Train and evaluate each model
    results = []
    for model_config in model_configs:
        model_id = model_config["model_id"]
        params = model_config["params"]
        name = model_config["name"]

        training_result = train_model(
            ModelID(model_id), params, split_result,
            target_name=dataset_info.target_name,
            target_names=dataset_info.target_names,
        )

        y_pred, y_prob = predict(training_result)

        # Compute metrics based on task type
        if dataset_info.task_type == TaskType.CLASSIFICATION:
            metrics = compute_classification_metrics(
                split_result.y_test.values, y_pred, dataset_info.target_names
            )
            metrics_dict = {
                "task_type": "classification",
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "confusion_matrix": metrics.confusion_matrix.tolist(),
                "per_class": metrics.per_class,
            }
        elif dataset_info.task_type == TaskType.REGRESSION:
            metrics = compute_regression_metrics(split_result.y_test.values, y_pred)
            metrics_dict = {
                "task_type": "regression",
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
            }
        else:  # CLUSTERING
            train_labels = training_result.model.predict(split_result.X_train)
            inertia = training_result.model.named_steps["model"].inertia_
            metrics = compute_clustering_metrics(split_result.X_train.values, train_labels, inertia)
            metrics_dict = {
                "task_type": "clustering",
                "inertia": metrics.inertia,
                "silhouette": metrics.silhouette,
                "n_clusters": metrics.n_clusters,
                "n_samples": metrics.n_samples,
            }

        # Generate visualizations
        visualizations = {}
        viz_features = request.viz_features

        if dataset_info.task_type == TaskType.CLASSIFICATION and viz_features and len(X.columns) >= 2:
            from ml_visualizer import get_decision_boundary_data
            xx, yy, Z = get_decision_boundary_data(training_result, tuple(viz_features))
            X_test_2d = split_result.X_test.iloc[:, list(viz_features)].values
            visualizations["decision_boundary"] = {
                "xx": xx.tolist(),
                "yy": yy.tolist(),
                "Z": Z.tolist(),
                "X_test_2d": X_test_2d.tolist(),
                "y_test": split_result.y_test.values.tolist(),
                "feature_names": [X.columns[viz_features[0]], X.columns[viz_features[1]]],
                "target_names": dataset_info.target_names,
            }

            if MODEL_REGISTRY[ModelID(model_config["model_id"])].supports_feature_importance:
                importance = get_feature_importance(training_result)
                if importance is not None:
                    visualizations["feature_importance"] = [
                        {"feature": f, "importance": float(v)} for f, v in importance.items()
                    ]

            visualizations["confusion_matrix"] = metrics.confusion_matrix.tolist()

        elif dataset_info.task_type == TaskType.REGRESSION:
            visualizations["actual_vs_predicted"] = {
                "actual": split_result.y_test.values.tolist(),
                "predicted": y_pred.tolist(),
            }
            residuals = split_result.y_test.values - y_pred
            visualizations["residuals"] = {
                "predicted": y_pred.tolist(),
                "residuals": residuals.tolist(),
            }

            if viz_features:
                feat_idx = viz_features[0]
                feat_name = X.columns[feat_idx]
                X_test_feat = split_result.X_test.iloc[:, feat_idx].values
                sort_idx = np.argsort(X_test_feat)
                visualizations["regression_line"] = {
                    "X": X_test_feat[sort_idx].tolist(),
                    "y_true": split_result.y_test.values[sort_idx].tolist(),
                    "y_pred": y_pred[sort_idx].tolist(),
                    "feature_name": feat_name,
                    "target_name": dataset_info.target_name or "target",
                }

            if MODEL_REGISTRY[ModelID(model_config["model_id"])].supports_feature_importance:
                importance = get_feature_importance(training_result)
                if importance is not None:
                    visualizations["feature_importance"] = [
                        {"feature": f, "importance": float(v)} for f, v in importance.items()
                    ]

        elif dataset_info.task_type == TaskType.CLUSTERING:
            if viz_features and training_result.X_train.shape[1] >= 2:
                X_train_2d = training_result.X_train.iloc[:, list(viz_features)].values
                train_labels = training_result.model.predict(training_result.X_train)

                centroids = None
                if hasattr(training_result.model.named_steps["model"], "cluster_centers_"):
                    scaler = training_result.model.named_steps.get("scaler")
                    if scaler:
                        centroids = scaler.inverse_transform(
                            training_result.model.named_steps["model"].cluster_centers_
                        )
                    else:
                        centroids = training_result.model.named_steps["model"].cluster_centers_
                    centroids = centroids[:, list(viz_features)]

                visualizations["clusters"] = {
                    "X_2d": X_train_2d.tolist(),
                    "labels": train_labels.tolist(),
                    "centroids": centroids.tolist() if centroids is not None else None,
                    "feature_names": [
                        training_result.feature_names[viz_features[0]],
                        training_result.feature_names[viz_features[1]],
                    ],
                }

                # Silhouette in 2D space
                if len(np.unique(train_labels)) > 1 and len(np.unique(train_labels)) < len(train_labels):
                    try:
                        from sklearn.metrics import silhouette_score, silhouette_samples
                        sil_2d = silhouette_score(X_train_2d, train_labels)
                        sil_samples_2d = silhouette_samples(X_train_2d, train_labels)
                        visualizations["silhouette"] = {
                            "silhouette_score": float(sil_2d),
                            "silhouette_samples": sil_samples_2d.tolist(),
                            "labels": train_labels.tolist(),
                            "X_2d": X_train_2d.tolist(),
                        }
                    except Exception:
                        pass

        results.append({
            "model_id": model_config["model_id"],
            "model_name": name,
            "metrics": metrics_dict,
            "train_time_ms": training_result.train_time * 1000,
            "train_samples": len(training_result.X_train),
            "test_samples": len(training_result.X_test),
            "visualizations": visualizations,
        })

    # Determine best model
    primary_metric = _get_primary_metric(dataset_info.task_type)
    best_model = _get_best_model(results, primary_metric, dataset_info.task_type)

    return ComparisonResponse(
        experiment_ids=[],  # Comparison doesn't create persistent experiments
        results=[ModelComparisonResult(**r) for r in results],
        best_model=best_model,
        primary_metric=primary_metric,
    )


def _get_primary_metric(task_type: TaskType) -> str:
    """Get the primary metric for a task type."""
    if task_type == TaskType.CLASSIFICATION:
        return "accuracy"
    elif task_type == TaskType.REGRESSION:
        return "r2"
    else:
        return "silhouette"


def _get_best_model(results: List[dict], metric: str, task_type: TaskType) -> Optional[str]:
    """Determine the best model based on the primary metric."""
    valid_results = [r for r in results if r["metrics"].get(metric) is not None]
    if not valid_results:
        return None

    if metric in ["accuracy", "precision", "recall", "f1", "r2", "silhouette"]:
        return max(valid_results, key=lambda r: r["metrics"][metric])["model_name"]
    else:
        return min(valid_results, key=lambda r: r["metrics"][metric])["model_name"]