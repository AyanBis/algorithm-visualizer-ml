"""Workflow event schemas."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import uuid


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_SKIPPED = "stage_skipped"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"


class StageInfo(BaseModel):
    id: str
    name: str
    description: str
    order: int


# Fixed workflow stages
WORKFLOW_STAGES: List[StageInfo] = [
    StageInfo(
        id="experiment_config",
        name="Experiment Configuration",
        description="Validate inputs and prepare experiment configuration",
        order=1,
    ),
    StageInfo(
        id="dataset_loading",
        name="Dataset Loading",
        description="Load dataset and extract metadata",
        order=2,
    ),
    StageInfo(
        id="data_validation",
        name="Data Validation",
        description="Check for NaN, infinite values, alignment, and model/dataset compatibility",
        order=3,
    ),
    StageInfo(
        id="train_test_split",
        name="Train/Test Split",
        description="Create reproducible train/test partitions with stratification for classification",
        order=4,
    ),
    StageInfo(
        id="preprocessing",
        name="Preprocessing",
        description="Fit scaler on training data only, transform both train and test sets",
        order=5,
    ),
    StageInfo(
        id="model_initialization",
        name="Model Initialization",
        description="Instantiate model with selected hyperparameters",
        order=6,
    ),
    StageInfo(
        id="model_training",
        name="Model Training",
        description="Fit model on training partition",
        order=7,
    ),
    StageInfo(
        id="prediction",
        name="Prediction / Assignment",
        description="Generate predictions on test set or cluster assignments",
        order=8,
    ),
    StageInfo(
        id="evaluation",
        name="Evaluation",
        description="Compute metrics (accuracy, precision, recall, F1, MAE, RMSE, R², inertia, silhouette)",
        order=9,
    ),
    StageInfo(
        id="visualization",
        name="Visualization",
        description="Generate visualization data (decision boundary, confusion matrix, clusters, etc.)",
        order=10,
    ),
    StageInfo(
        id="results_ready",
        name="Results Ready",
        description="Finalize experiment, compile results",
        order=11,
    ),
]

STAGE_ID_TO_INFO = {s.id: s for s in WORKFLOW_STAGES}


class WorkflowEvent(BaseModel):
    experiment_id: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    stage_id: str
    stage_name: str
    status: StageStatus
    sequence: int
    timestamp: str  # ISO-8601
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ExperimentEventStreamResponse(BaseModel):
    """Response for SSE stream - same as WorkflowEvent but for documentation."""
    pass


def create_stage_started_event(
    experiment_id: str,
    stage_id: str,
    sequence: int,
    timestamp: str,
) -> WorkflowEvent:
    stage = STAGE_ID_TO_INFO[stage_id]
    return WorkflowEvent(
        experiment_id=experiment_id,
        event_type=EventType.STAGE_STARTED,
        stage_id=stage_id,
        stage_name=stage.name,
        status=StageStatus.RUNNING,
        sequence=sequence,
        timestamp=timestamp,
        started_at=timestamp,
        message=f"Starting {stage.name}",
    )


def create_stage_completed_event(
    experiment_id: str,
    stage_id: str,
    sequence: int,
    timestamp: str,
    started_at: str,
    duration_ms: float,
    message: str,
    details: Dict[str, Any],
) -> WorkflowEvent:
    stage = STAGE_ID_TO_INFO[stage_id]
    return WorkflowEvent(
        experiment_id=experiment_id,
        event_type=EventType.STAGE_COMPLETED,
        stage_id=stage_id,
        stage_name=stage.name,
        status=StageStatus.COMPLETED,
        sequence=sequence,
        timestamp=timestamp,
        started_at=started_at,
        completed_at=timestamp,
        duration_ms=duration_ms,
        message=message,
        details=details,
    )


def create_stage_failed_event(
    experiment_id: str,
    stage_id: str,
    sequence: int,
    timestamp: str,
    started_at: str,
    duration_ms: float,
    error: str,
) -> WorkflowEvent:
    stage = STAGE_ID_TO_INFO[stage_id]
    return WorkflowEvent(
        experiment_id=experiment_id,
        event_type=EventType.STAGE_FAILED,
        stage_id=stage_id,
        stage_name=stage.name,
        status=StageStatus.FAILED,
        sequence=sequence,
        timestamp=timestamp,
        started_at=started_at,
        completed_at=timestamp,
        duration_ms=duration_ms,
        message=f"{stage.name} failed",
        error=error,
    )


def create_stage_skipped_event(
    experiment_id: str,
    stage_id: str,
    sequence: int,
    timestamp: str,
    reason: str,
) -> WorkflowEvent:
    stage = STAGE_ID_TO_INFO[stage_id]
    return WorkflowEvent(
        experiment_id=experiment_id,
        event_type=EventType.STAGE_SKIPPED,
        stage_id=stage_id,
        stage_name=stage.name,
        status=StageStatus.SKIPPED,
        sequence=sequence,
        timestamp=timestamp,
        message=f"{stage.name} skipped",
        details={"reason": reason},
    )


def create_experiment_completed_event(
    experiment_id: str,
    sequence: int,
    timestamp: str,
    total_duration_ms: float,
    stage_timings: Dict[str, float],
) -> WorkflowEvent:
    return WorkflowEvent(
        experiment_id=experiment_id,
        event_type=EventType.EXPERIMENT_COMPLETED,
        stage_id="results_ready",
        stage_name="Results Ready",
        status=StageStatus.COMPLETED,
        sequence=sequence,
        timestamp=timestamp,
        message="Experiment completed successfully",
        details={
            "total_duration_ms": total_duration_ms,
            "stage_timings": stage_timings,
        },
    )


def create_experiment_failed_event(
    experiment_id: str,
    sequence: int,
    timestamp: str,
    failed_stage: str,
    error: str,
) -> WorkflowEvent:
    return WorkflowEvent(
        experiment_id=experiment_id,
        event_type=EventType.EXPERIMENT_FAILED,
        stage_id=failed_stage,
        stage_name=STAGE_ID_TO_INFO.get(failed_stage, StageInfo(id=failed_stage, name=failed_stage, description="", order=0)).name,
        status=StageStatus.FAILED,
        sequence=sequence,
        timestamp=timestamp,
        message="Experiment failed",
        error=error,
    )