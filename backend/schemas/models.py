"""Model schemas."""
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum
from .datasets import TaskType


class HyperParameterType(str, Enum):
    INT = "int"
    FLOAT = "float"
    SELECT = "select"


class HyperParameter(BaseModel):
    name: str = Field(..., description="Parameter name (matches sklearn)")
    display_name: str = Field(..., description="Human-readable display name")
    param_type: HyperParameterType = Field(..., description="Parameter type")
    default: Union[int, float, str, bool, None] = Field(..., description="Default value")
    min_value: Optional[float] = Field(None, description="Minimum value (for numeric)")
    max_value: Optional[float] = Field(None, description="Maximum value (for numeric)")
    step: Optional[float] = Field(None, description="Step size (for numeric)")
    options: Optional[List[Union[str, int, float, bool, None]]] = Field(None, description="Options (for select)")
    help_text: str = Field("", description="Help text for UI")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    task_type: TaskType = Field(..., description="ML task type")
    description: str = Field(..., description="Model description")
    how_it_works: str = Field(..., description="How the model works")
    when_to_use: str = Field(..., description="When to use this model")
    limitations: str = Field(..., description="Model limitations")
    hyperparameters: List[HyperParameter] = Field(default_factory=list)
    supports_probability: bool = Field(False, description="Supports predict_proba")
    supports_decision_boundary: bool = Field(False, description="Supports decision boundary viz")
    supports_feature_importance: bool = Field(False, description="Supports feature importance")
    default_params: dict = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class ModelDetailResponse(BaseModel):
    model: ModelInfo