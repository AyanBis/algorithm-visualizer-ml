"""Experiment API endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from typing import Optional
from sse_starlette.sse import EventSourceResponse

from backend.schemas.experiments import (
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    ExperimentStatusResponse,
    ExperimentResultsResponse,
    ExperimentCancelRequest,
    ExperimentConfig,
)
from backend.schemas.workflow import WorkflowEvent
from backend.core.experiment_manager import experiment_manager
from backend.core.task_runner import run_experiment
from backend.core.event_emitter import event_emitter
from backend.core.config import settings


router = APIRouter()


@router.post("", response_model=ExperimentCreateResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Create and start a new experiment."""
    exp = await experiment_manager.create_experiment(request.config)

    # Start experiment as background task
    background_tasks.add_task(run_experiment, exp.experiment_id, request.config)

    return ExperimentCreateResponse(
        experiment_id=exp.experiment_id,
        status=exp.status.value,
        created_at=exp.created_at,
    )


@router.get("/{experiment_id}", response_model=ExperimentStatusResponse)
async def get_experiment_status(experiment_id: str):
    """Get experiment status."""
    exp = await experiment_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp.to_status_response()


@router.get("/{experiment_id}/events")
async def get_experiment_events(
    experiment_id: str,
    after_sequence: int = 0,
):
    """Get experiment events (polling fallback)."""
    exp = await experiment_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    events = await event_emitter.get_events(experiment_id, after_sequence)
    return {"events": [e.model_dump() for e in events]}


@router.get("/{experiment_id}/events/stream")
async def stream_experiment_events(
    experiment_id: str,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """SSE stream for real-time experiment events."""
    exp = await experiment_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    async def event_generator():
        async for sse_data in event_emitter.stream_events(experiment_id, last_event_id):
            yield sse_data

    return EventSourceResponse(event_generator())


@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(experiment_id: str):
    """Get experiment results."""
    exp = await experiment_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if exp.status not in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Experiment not yet completed")

    if not exp.results:
        raise HTTPException(status_code=404, detail="Results not available")

    return exp.results


@router.post("/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str, request: ExperimentCancelRequest):
    """Cancel a running experiment."""
    success = await experiment_manager.cancel_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel experiment")
    return {"success": True, "message": "Cancellation requested"}