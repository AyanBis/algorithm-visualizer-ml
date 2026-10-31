"""Experiment schemas."""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from .models import HyperParameter
from .datasets import TaskType


class ExperimentConfig(BaseModel):
    dataset_id: str
    model_id: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = Field(42, ge=0)
    scale_features: bool = True
    scaler_type: str = Field("standard", pattern="^(standard|minmax)$")
    viz_features: Optional[List[int]] = Field(None, min_length=2, max_length=2)


class ExperimentCreateRequest(BaseModel):
    config: ExperimentConfig


class ExperimentCreateResponse(BaseModel):
    experiment_id: str
    status: str
    created_at: str


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageTiming(BaseModel):
    stage_id: str
    stage_name: str
    duration_ms: float
    status: str


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    per_class: Dict[str, Dict[str, float]]


class RegressionMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float


class ClusteringMetrics(BaseModel):
    inertia: float
    silhouette: Optional[float]
    n_clusters: int
    n_samples: int


class DecisionBoundaryData(BaseModel):
    xx: List[List[float]]
    yy: List[List[float]]
    Z: List[List[int]]
    X_test_2d: List[List[float]]
    y_test: List[int]
    feature_names: List[str]
    target_names: Optional[List[str]] = None


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ClusterData(BaseModel):
    X_2d: List[List[float]]
    labels: List[int]
    centroids: Optional[List[List[float]]] = None
    feature_names: List[str]


class VisualizationData(BaseModel):
    decision_boundary: Optional[DecisionBoundaryData] = None
    feature_importance: Optional[List[FeatureImportanceItem]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    actual_vs_predicted: Optional[Dict[str, List[float]]] = None
    residuals: Optional[Dict[str, List[float]]] = None
    regression_line: Optional[Dict[str, List[float]]] = None
    clusters: Optional[ClusterData] = None
    silhouette: Optional[Dict[str, Any]] = None


class PredictionItem(BaseModel):
    actual: Optional[Union[int, float, str]] = None
    predicted: Union[int, float, str]
    probabilities: Optional[List[float]] = None
    match: Optional[bool] = None


class ExperimentResultsResponse(BaseModel):
    experiment_id: str
    status: ExperimentStatus
    config: ExperimentConfig
    metrics: Dict[str, Any]
    visualizations: VisualizationData
    predictions: List[PredictionItem]
    train_time_ms: float
    total_time_ms: float
    stage_timings: List[StageTiming]
    warnings: List[str] = Field(default_factory=list)


class ExperimentCancelRequest(BaseModel):
    pass  # No body needed


class ExperimentStatusResponse(BaseModel):
    experiment_id: str
    status: ExperimentStatus
    config: ExperimentConfig
    current_stage: Optional[str] = None
    progress: float = Field(0.0, ge=0.0, le=1.0)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_ms: Optional[float] = None