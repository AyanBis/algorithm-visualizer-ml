"""Dataset API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
import numpy as np

from ml_visualizer import get_dataset_loader, DatasetLoader

from backend.schemas.datasets import (
    DatasetListResponse,
    DatasetDetailResponse,
    DatasetInfo,
    DatasetStatistics,
)
from backend.core.config import settings


router = APIRouter()


@router.get("", response_model=DatasetListResponse)
async def list_datasets():
    """List all available datasets."""
    loader = get_dataset_loader(random_state=settings.DEFAULT_RANDOM_STATE)
    datasets = []
    for dataset_id in loader.list_datasets():
        info = loader.get_info(dataset_id)
        datasets.append(DatasetInfo(
            id=dataset_id,
            name=info.name,
            task_type=info.task_type,
            description=info.description,
            n_samples=info.n_samples,
            n_features=info.n_features,
            feature_names=info.feature_names,
            target_name=info.target_name,
            target_names=info.target_names,
        ))
    return DatasetListResponse(datasets=datasets)


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: str,
    preview_rows: int = Query(10, ge=1, le=100),
):
    """Get dataset details, preview, and statistics."""
    loader = get_dataset_loader(random_state=settings.DEFAULT_RANDOM_STATE)

    try:
        info = loader.get_info(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X, y, _ = loader.load(dataset_id)

    # Preview
    preview_df = loader.get_preview(dataset_id, n_rows=preview_rows)
    preview = preview_df.to_dict(orient="records")

    # Statistics
    statistics = []
    for col in X.columns:
        statistics.append(DatasetStatistics(
            feature=col,
            type="numeric",
            non_null=int(X[col].notna().sum()),
            mean=float(X[col].mean()),
            std=float(X[col].std()),
            min=float(X[col].min()),
            max=float(X[col].max()),
        ))

    # Target distribution
    target_distribution = None
    if y is not None and info.target_names:
        target_counts = y.value_counts().sort_index()
        target_distribution = [
            {
                "class": info.target_names[int(idx)],
                "count": int(count),
                "percentage": float(count / len(y) * 100),
            }
            for idx, count in target_counts.items()
        ]

    return DatasetDetailResponse(
        info=DatasetInfo(
            id=dataset_id,
            name=info.name,
            task_type=info.task_type,
            description=info.description,
            n_samples=info.n_samples,
            n_features=info.n_features,
            feature_names=info.feature_names,
            target_name=info.target_name,
            target_names=info.target_names,
        ),
        preview=preview,
        statistics=statistics,
        target_distribution=target_distribution,
    )