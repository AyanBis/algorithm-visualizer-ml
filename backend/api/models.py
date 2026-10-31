"""Model API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ml_visualizer import (
    get_model_info,
    list_models,
    TaskType,
    ModelID,
    MODEL_REGISTRY,
)

from backend.schemas.models import (
    ModelListResponse,
    ModelDetailResponse,
    ModelInfo,
    HyperParameter,
    HyperParameterType,
)


router = APIRouter()


@router.get("", response_model=ModelListResponse)
async def list_models_endpoint(task_type: Optional[TaskType] = Query(None)):
    """List all models, optionally filtered by task type."""
    models = list_models(task_type)
    result = []
    for model_info in models:
        result.append(_convert_model_info(model_info))
    return ModelListResponse(models=result)


@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(model_id: str):
    """Get model details including hyperparameters."""
    try:
        model_enum = ModelID(model_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    model_info = get_model_info(model_enum)
    return ModelDetailResponse(model=_convert_model_info(model_info))


def _convert_model_info(model_info) -> ModelInfo:
    """Convert internal ModelInfo to API schema."""
    hyperparams = []
    for hp in model_info.hyperparameters:
        param_type = HyperParameterType.INT
        if hp.param_type == "float":
            param_type = HyperParameterType.FLOAT
        elif hp.param_type == "select":
            param_type = HyperParameterType.SELECT

        hyperparams.append(HyperParameter(
            name=hp.name,
            display_name=hp.display_name,
            param_type=param_type,
            default=hp.default,
            min_value=hp.min_value,
            max_value=hp.max_value,
            step=hp.step,
            options=hp.options,
            help_text=hp.help_text,
        ))

    return ModelInfo(
        id=model_info.id.value,
        name=model_info.name,
        task_type=model_info.task_type,
        description=model_info.description,
        how_it_works=model_info.how_it_works,
        when_to_use=model_info.when_to_use,
        limitations=model_info.limitations,
        hyperparameters=hyperparams,
        supports_probability=model_info.supports_probability,
        supports_decision_boundary=model_info.supports_decision_boundary,
        supports_feature_importance=model_info.supports_feature_importance,
        default_params=model_info.default_params,
    )