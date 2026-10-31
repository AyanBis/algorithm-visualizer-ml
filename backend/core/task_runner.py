"""Task runner for executing ML experiments with workflow events."""
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import traceback

import numpy as np
import pandas as pd

# Import ML visualizer modules
from ml_visualizer import (
    get_dataset_loader,
    get_model_info,
    list_models,
    TaskType,
    ModelID,
    split_and_scale,
    validate_data,
    train_model,
    predict,
    get_feature_importance,
    get_decision_boundary_data,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_clustering_metrics,
)
from ml_visualizer.model_registry import MODEL_REGISTRY

from ..schemas.experiments import (
    ExperimentConfig,
    ExperimentResultsResponse,
    ExperimentStatus,
    StageTiming,
    ClassificationMetrics,
    RegressionMetrics,
    ClusteringMetrics,
    VisualizationData,
    DecisionBoundaryData,
    FeatureImportanceItem,
    ClusterData,
    PredictionItem,
)
from ..schemas.workflow import (
    WORKFLOW_STAGES,
    StageStatus,
    create_stage_started_event,
    create_stage_completed_event,
    create_stage_failed_event,
    create_stage_skipped_event,
    create_experiment_completed_event,
    create_experiment_failed_event,
)
from ..core.experiment_manager import experiment_manager
from ..core.event_emitter import event_emitter
from ..core.config import settings


class ExperimentRunner:
    """Runs ML experiment and emits workflow events."""

    def __init__(self, experiment_id: str, config: ExperimentConfig):
        self.experiment_id = experiment_id
        self.config = config
        self.sequence = 0
        self.stage_start_times: Dict[str, str] = {}
        self.stage_timings: Dict[str, float] = {}
        self.warnings: List[str] = []

        # Data placeholders
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.dataset_info = None
        self.split_result = None
        self.training_result = None
        self.y_pred = None
        self.y_prob = None
        self.metrics = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    async def _emit(self, event) -> None:
        await event_emitter.emit(event)

    async def _run_stage(self, stage_id: str, coro) -> Any:
        """Run a stage with event emission and timing."""
        stage_info = next(s for s in WORKFLOW_STAGES if s.id == stage_id)
        start_time = time.monotonic()
        start_timestamp = self._now()
        self.stage_start_times[stage_id] = start_timestamp
        seq = self._next_sequence()

        await self._emit(create_stage_started_event(
            self.experiment_id, stage_id, seq, start_timestamp
        ))

        await experiment_manager.update_status(
            self.experiment_id,
            ExperimentStatus.RUNNING,
            current_stage=stage_id,
            progress=stage_info.order / len(WORKFLOW_STAGES),
        )

        try:
            result = await coro
            duration_ms = (time.monotonic() - start_time) * 1000
            self.stage_timings[stage_id] = duration_ms
            end_timestamp = self._now()

            await self._emit(create_stage_completed_event(
                self.experiment_id, stage_id, seq, end_timestamp,
                start_timestamp, duration_ms,
                f"{stage_info.name} completed", {}
            ))

            return result
        except asyncio.CancelledError:
            duration_ms = (time.monotonic() - start_time) * 1000
            end_timestamp = self._now()
            await self._emit(create_stage_failed_event(
                self.experiment_id, stage_id, seq, end_timestamp,
                start_timestamp, duration_ms, "Cancelled"
            ))
            raise
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            end_timestamp = self._now()
            await self._emit(create_stage_failed_event(
                self.experiment_id, stage_id, seq, end_timestamp,
                start_timestamp, duration_ms, str(e)
            ))
            raise

    async def _skip_stage(self, stage_id: str, reason: str) -> None:
        """Mark a stage as skipped."""
        seq = self._next_sequence()
        timestamp = self._now()
        await self._emit(create_stage_skipped_event(
            self.experiment_id, stage_id, seq, timestamp, reason
        ))
        self.stage_timings[stage_id] = 0.0

    async def run(self) -> ExperimentResultsResponse:
        """Execute the full experiment workflow."""
        try:
            # Stage 1: Experiment Configuration
            await self._run_stage("experiment_config", self._stage_config())

            # Stage 2: Dataset Loading
            await self._run_stage("dataset_loading", self._stage_dataset_loading())

            # Stage 3: Data Validation
            await self._run_stage("data_validation", self._stage_data_validation())

            # Stage 4: Train/Test Split
            await self._run_stage("train_test_split", self._stage_train_test_split())

            # Stage 5: Preprocessing
            if self.config.scale_features:
                await self._run_stage("preprocessing", self._stage_preprocessing())
            else:
                await self._skip_stage("preprocessing", "Feature scaling disabled")

            # Stage 6: Model Initialization
            await self._run_stage("model_initialization", self._stage_model_initialization())

            # Stage 7: Model Training
            await self._run_stage("model_training", self._stage_model_training())

            # Stage 8: Prediction
            await self._run_stage("prediction", self._stage_prediction())

            # Stage 9: Evaluation
            await self._run_stage("evaluation", self._stage_evaluation())

            # Stage 10: Visualization
            await self._run_stage("visualization", self._stage_visualization())

            # Stage 11: Results Ready
            await self._run_stage("results_ready", self._stage_results_ready())

            # Build final results
            results = self._build_results()
            await experiment_manager.set_results(self.experiment_id, results)

            # Emit completion event
            total_duration = sum(self.stage_timings.values())
            seq = self._next_sequence()
            await self._emit(create_experiment_completed_event(
                self.experiment_id, seq, self._now(), total_duration, self.stage_timings
            ))

            return results

        except asyncio.CancelledError:
            await experiment_manager.update_status(self.experiment_id, ExperimentStatus.CANCELLED)
            raise
        except Exception as e:
            await experiment_manager.set_error(self.experiment_id, str(e))
            seq = self._next_sequence()
            await self._emit(create_experiment_failed_event(
                self.experiment_id, seq, self._now(),
                self.config.model_id, str(e)
            ))
            raise

    async def _stage_config(self) -> Dict[str, Any]:
        """Validate experiment configuration."""
        # Validate dataset exists
        loader = get_dataset_loader(random_state=self.config.random_state)
        try:
            loader.get_info(self.config.dataset_id)
        except ValueError as e:
            raise ValueError(f"Invalid dataset: {e}")

        # Validate model exists and is compatible
        try:
            model_info = get_model_info(ModelID(self.config.model_id))
        except (ValueError, KeyError):
            raise ValueError(f"Invalid model: {self.config.model_id}")

        # Check task compatibility
        dataset_info = loader.get_info(self.config.dataset_id)
        if model_info.task_type != dataset_info.task_type:
            raise ValueError(
                f"Model {model_info.name} ({model_info.task_type.value}) "
                f"not compatible with dataset {dataset_info.name} ({dataset_info.task_type.value})"
            )

        # Validate hyperparameters against model schema
        for param_name, param_value in self.config.hyperparameters.items():
            hp = next((p for p in model_info.hyperparameters if p.name == param_name), None)
            if hp:
                if hp.param_type == "int" and not isinstance(param_value, int):
                    self.config.hyperparameters[param_name] = int(param_value)
                elif hp.param_type == "float" and not isinstance(param_value, float):
                    self.config.hyperparameters[param_value] = float(param_value)

        return {"validated": True}

    async def _stage_dataset_loading(self) -> Dict[str, Any]:
        """Load dataset."""
        loader = get_dataset_loader(random_state=self.config.random_state)
        self.X, self.y, self.dataset_info = loader.load(self.config.dataset_id)

        return {
            "n_samples": len(self.X),
            "n_features": len(self.X.columns),
            "task_type": self.dataset_info.task_type.value,
            "target_name": self.dataset_info.target_name,
        }

    async def _stage_data_validation(self) -> Dict[str, Any]:
        """Validate data quality."""
        valid, msg = validate_data(self.X, self.y)
        if not valid:
            raise ValueError(f"Data validation failed: {msg}")

        # Check model/dataset compatibility
        model_info = get_model_info(ModelID(self.config.model_id))
        if model_info.task_type != self.dataset_info.task_type:
            raise ValueError("Model/task type mismatch")

        return {"validation": "passed", "message": msg}

    async def _stage_train_test_split(self) -> Dict[str, Any]:
        """Create train/test split."""
        task_type = self.dataset_info.task_type
        self.split_result = split_and_scale(
            self.X, self.y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            scale=False,  # Scaling done in separate stage
            task_type=task_type,
        )

        stratify_used = (
            self.y is not None and
            task_type == TaskType.CLASSIFICATION
        )

        return {
            "train_samples": len(self.split_result.X_train),
            "test_samples": len(self.split_result.X_test),
            "stratified": stratify_used,
        }

    async def _stage_preprocessing(self) -> Dict[str, Any]:
        """Apply feature scaling."""
        # Re-split with scaling (the split_and_scale function handles this)
        task_type = self.dataset_info.task_type
        scaler_type = self.config.scaler_type
        self.split_result = split_and_scale(
            self.X, self.y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            scale=True,
            scaler_type=scaler_type,
            task_type=task_type,
        )

        return {
            "scaler_type": scaler_type,
            "feature_names": self.split_result.feature_names,
        }

    async def _stage_model_initialization(self) -> Dict[str, Any]:
        """Initialize model with hyperparameters."""
        model_info = get_model_info(ModelID(self.config.model_id))
        # Model is created inside train_model, just validate params here
        return {
            "model": model_info.name,
            "hyperparameters": self.config.hyperparameters,
        }

    async def _stage_model_training(self) -> Dict[str, Any]:
        """Train the model."""
        self.training_result = train_model(
            ModelID(self.config.model_id),
            self.config.hyperparameters,
            self.split_result,
            target_name=self.dataset_info.target_name,
            target_names=self.dataset_info.target_names,
        )

        return {
            "train_time_ms": self.training_result.train_time * 1000,
            "train_samples": len(self.training_result.X_train),
        }

    async def _stage_prediction(self) -> Dict[str, Any]:
        """Generate predictions."""
        self.y_pred, self.y_prob = predict(self.training_result)

        return {
            "n_predictions": len(self.y_pred),
            "has_probabilities": self.y_prob is not None,
        }

    async def _stage_evaluation(self) -> Dict[str, Any]:
        """Compute metrics."""
        task_type = self.dataset_info.task_type
        y_test = self.split_result.y_test

        if task_type == TaskType.CLASSIFICATION:
            self.metrics = compute_classification_metrics(
                y_test.values, self.y_pred, self.dataset_info.target_names
            )
            return {
                "accuracy": self.metrics.accuracy,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "f1": self.metrics.f1,
            }

        elif task_type == TaskType.REGRESSION:
            self.metrics = compute_regression_metrics(y_test.values, self.y_pred)
            return {
                "mae": self.metrics.mae,
                "rmse": self.metrics.rmse,
                "r2": self.metrics.r2,
            }

        elif task_type == TaskType.CLUSTERING:
            # For clustering, evaluate on training data where model was fit
            train_labels = self.training_result.model.predict(self.training_result.X_train)
            inertia = self.training_result.model.named_steps["model"].inertia_
            self.metrics = compute_clustering_metrics(
                self.training_result.X_train.values, train_labels, inertia
            )
            return {
                "inertia": self.metrics.inertia,
                "silhouette": self.metrics.silhouette,
                "n_clusters": self.metrics.n_clusters,
            }

        return {}

    async def _stage_visualization(self) -> Dict[str, Any]:
        """Generate visualization data."""
        task_type = self.dataset_info.task_type
        viz_features = self.config.viz_features

        if task_type == TaskType.CLASSIFICATION:
            # Decision boundary
            if viz_features and len(self.X.columns) >= 2:
                try:
                    xx, yy, Z = get_decision_boundary_data(
                        self.training_result, tuple(viz_features)
                    )
                    X_test_2d = self.split_result.X_test.iloc[:, list(viz_features)].values
                    # Store for results
                    self._decision_boundary_data = {
                        "xx": xx.tolist(),
                        "yy": yy.tolist(),
                        "Z": Z.tolist(),
                        "X_test_2d": X_test_2d.tolist(),
                        "y_test": self.split_result.y_test.values.tolist(),
                        "feature_names": [
                            self.X.columns[viz_features[0]],
                            self.X.columns[viz_features[1]],
                        ],
                        "target_names": self.dataset_info.target_names,
                    }
                except Exception as e:
                    self.warnings.append(f"Decision boundary generation failed: {e}")

            # Feature importance
            if MODEL_REGISTRY[ModelID(self.config.model_id)].supports_feature_importance:
                importance = get_feature_importance(self.training_result)
                if importance is not None:
                    self._feature_importance = [
                        {"feature": f, "importance": float(v)}
                        for f, v in importance.items()
                    ]

            # Confusion matrix
            self._confusion_matrix = self.metrics.confusion_matrix.tolist()

        elif task_type == TaskType.REGRESSION:
            # Actual vs Predicted
            self._actual_vs_predicted = {
                "actual": self.split_result.y_test.values.tolist(),
                "predicted": self.y_pred.tolist(),
            }

            # Residuals
            residuals = self.split_result.y_test.values - self.y_pred
            self._residuals = {
                "predicted": self.y_pred.tolist(),
                "residuals": residuals.tolist(),
            }

            # Regression line (if viz feature selected)
            if viz_features:
                feat_idx = viz_features[0]
                feat_name = self.X.columns[feat_idx]
                X_test_feat = self.split_result.X_test.iloc[:, feat_idx].values
                sort_idx = np.argsort(X_test_feat)
                self._regression_line = {
                    "X": X_test_feat[sort_idx].tolist(),
                    "y_true": self.split_result.y_test.values[sort_idx].tolist(),
                    "y_pred": self.y_pred[sort_idx].tolist(),
                    "feature_name": feat_name,
                    "target_name": self.dataset_info.target_name or "target",
                }

            # Feature importance
            if MODEL_REGISTRY[ModelID(self.config.model_id)].supports_feature_importance:
                importance = get_feature_importance(self.training_result)
                if importance is not None:
                    self._feature_importance = [
                        {"feature": f, "importance": float(v)}
                        for f, v in importance.items()
                    ]

        elif task_type == TaskType.CLUSTERING:
            # Cluster visualization
            if viz_features and self.training_result.X_train.shape[1] >= 2:
                X_train_2d = self.training_result.X_train.iloc[:, list(viz_features)].values
                train_labels = self.training_result.model.predict(self.training_result.X_train)

                centroids = None
                if hasattr(self.training_result.model.named_steps["model"], "cluster_centers_"):
                    scaler = self.training_result.model.named_steps.get("scaler")
                    if scaler:
                        centroids = scaler.inverse_transform(
                            self.training_result.model.named_steps["model"].cluster_centers_
                        )
                    else:
                        centroids = self.training_result.model.named_steps["model"].cluster_centers_
                    centroids = centroids[:, list(viz_features)]

                self._cluster_data = {
                    "X_2d": X_train_2d.tolist(),
                    "labels": train_labels.tolist(),
                    "centroids": centroids.tolist() if centroids is not None else None,
                    "feature_names": [
                        self.training_result.feature_names[viz_features[0]],
                        self.training_result.feature_names[viz_features[1]],
                    ],
                }

                # Silhouette in 2D space
                if len(np.unique(train_labels)) > 1 and len(np.unique(train_labels)) < len(train_labels):
                    try:
                        from sklearn.metrics import silhouette_score, silhouette_samples
                        sil_2d = silhouette_score(X_train_2d, train_labels)
                        sil_samples_2d = silhouette_samples(X_train_2d, train_labels)
                        self._silhouette_data = {
                            "silhouette_score": float(sil_2d),
                            "silhouette_samples": sil_samples_2d.tolist(),
                            "labels": train_labels.tolist(),
                            "X_2d": X_train_2d.tolist(),
                        }
                    except Exception as e:
                        self.warnings.append(f"2D silhouette computation failed: {e}")

        return {"visualizations_generated": True}

    async def _stage_results_ready(self) -> Dict[str, Any]:
        """Finalize results."""
        return {"ready": True}

    def _build_results(self) -> ExperimentResultsResponse:
        """Build final results response."""
        task_type = self.dataset_info.task_type

        # Metrics
        if task_type == TaskType.CLASSIFICATION:
            metrics_dict = {
                "task_type": "classification",
                "accuracy": self.metrics.accuracy,
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "f1": self.metrics.f1,
                "confusion_matrix": self.metrics.confusion_matrix.tolist(),
                "per_class": self.metrics.per_class,
            }
        elif task_type == TaskType.REGRESSION:
            metrics_dict = {
                "task_type": "regression",
                "mae": self.metrics.mae,
                "rmse": self.metrics.rmse,
                "r2": self.metrics.r2,
            }
        else:
            metrics_dict = {
                "task_type": "clustering",
                "inertia": self.metrics.inertia,
                "silhouette": self.metrics.silhouette,
                "n_clusters": self.metrics.n_clusters,
                "n_samples": self.metrics.n_samples,
            }

        # Visualizations
        viz = VisualizationData()
        if task_type == TaskType.CLASSIFICATION:
            if hasattr(self, "_decision_boundary_data"):
                viz.decision_boundary = DecisionBoundaryData(**self._decision_boundary_data)
            if hasattr(self, "_feature_importance"):
                viz.feature_importance = [FeatureImportanceItem(**f) for f in self._feature_importance]
            if hasattr(self, "_confusion_matrix"):
                viz.confusion_matrix = self._confusion_matrix
        elif task_type == TaskType.REGRESSION:
            if hasattr(self, "_actual_vs_predicted"):
                viz.actual_vs_predicted = self._actual_vs_predicted
            if hasattr(self, "_residuals"):
                viz.residuals = self._residuals
            if hasattr(self, "_regression_line"):
                viz.regression_line = self._regression_line
            if hasattr(self, "_feature_importance"):
                viz.feature_importance = [FeatureImportanceItem(**f) for f in self._feature_importance]
        elif task_type == TaskType.CLUSTERING:
            if hasattr(self, "_cluster_data"):
                viz.clusters = ClusterData(**self._cluster_data)
            if hasattr(self, "_silhouette_data"):
                viz.silhouette = self._silhouette_data

        # Predictions
        predictions = []
        y_test = self.split_result.y_test
        for i in range(min(50, len(self.y_pred))):  # Limit to 50 for response size
            actual = y_test.values[i] if y_test is not None else None
            pred = self.y_pred[i]
            prob = self.y_prob[i].tolist() if self.y_prob is not None else None

            if task_type == TaskType.CLASSIFICATION and self.dataset_info.target_names:
                actual_label = self.dataset_info.target_names[int(actual)] if actual is not None else None
                pred_label = self.dataset_info.target_names[int(pred)]
            else:
                actual_label = actual
                pred_label = pred

            pred_item = PredictionItem(
                actual=actual_label,
                predicted=pred_label,
                probabilities=prob,
                match=(actual == pred) if actual is not None else None,
            )
            predictions.append(pred_item)

        # Stage timings
        stage_timings = [
            StageTiming(
                stage_id=stage_id,
                stage_name=next(s.name for s in WORKFLOW_STAGES if s.id == stage_id),
                duration_ms=duration,
                status="completed" if duration > 0 else "skipped",
            )
            for stage_id, duration in self.stage_timings.items()
        ]

        total_time = sum(self.stage_timings.values())
        train_time = self.stage_timings.get("model_training", 0)

        return ExperimentResultsResponse(
            experiment_id=self.experiment_id,
            status=ExperimentStatus.COMPLETED,
            config=self.config,
            metrics=metrics_dict,
            visualizations=viz,
            predictions=predictions,
            train_time_ms=train_time,
            total_time_ms=total_time,
            stage_timings=stage_timings,
            warnings=self.warnings,
        )


async def run_experiment(experiment_id: str, config: ExperimentConfig) -> None:
    """Entry point for running experiment as background task."""
    runner = ExperimentRunner(experiment_id, config)
    try:
        await runner.run()
    except asyncio.CancelledError:
        # Cancellation handled in runner
        raise
    except Exception as e:
        # Error handled in runner
        raise