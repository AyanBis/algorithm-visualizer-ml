"""Dataset schemas."""
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class DatasetInfo(BaseModel):
    id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Human-readable dataset name")
    task_type: TaskType = Field(..., description="ML task type")
    description: str = Field(..., description="Dataset description")
    n_samples: int = Field(..., description="Number of samples")
    n_features: int = Field(..., description="Number of features")
    feature_names: List[str] = Field(..., description="Feature column names")
    target_name: Optional[str] = Field(None, description="Target column name")
    target_names: Optional[List[str]] = Field(None, description="Class names for classification")


class DatasetPreviewRow(BaseModel):
    """Single row of dataset preview."""
    pass  # Dynamic columns


class DatasetStatistics(BaseModel):
    """Descriptive statistics for a feature."""
    feature: str
    type: str
    non_null: int
    mean: float
    std: float
    min: float
    max: float


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]


class DatasetDetailResponse(BaseModel):
    info: DatasetInfo
    preview: List[dict]
    statistics: List[DatasetStatistics]
    target_distribution: Optional[List[dict]] = None