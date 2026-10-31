"""Experiment manager for in-memory experiment storage and lifecycle."""
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from uuid import uuid4

from ..schemas.experiments import (
    ExperimentConfig,
    ExperimentStatus,
    ExperimentStatusResponse,
    ExperimentResultsResponse,
    StageTiming,
)
from ..schemas.workflow import (
    WorkflowEvent,
    EventType,
    StageStatus,
    WORKFLOW_STAGES,
    create_stage_started_event,
    create_stage_completed_event,
    create_stage_failed_event,
    create_stage_skipped_event,
    create_experiment_completed_event,
    create_experiment_failed_event,
)
from ..core.event_emitter import event_emitter
from ..core.config import settings


class Experiment:
    """In-memory experiment state."""

    def __init__(self, config: ExperimentConfig):
        self.experiment_id = str(uuid4())
        self.config = config
        self.status = ExperimentStatus.PENDING
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.progress: float = 0.0
        self.task: Optional[asyncio.Task] = None
        self.stage_timings: Dict[str, float] = {}
        self.stage_statuses: Dict[str, StageStatus] = {s.id: StageStatus.PENDING for s in WORKFLOW_STAGES}
        self.results: Optional[ExperimentResultsResponse] = None
        self.error: Optional[str] = None
        self._cancelled = False

    def to_status_response(self) -> ExperimentStatusResponse:
        total_duration = None
        if self.started_at and self.completed_at:
            start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
            total_duration = (end - start).total_seconds() * 1000

        return ExperimentStatusResponse(
            experiment_id=self.experiment_id,
            status=self.status,
            config=self.config,
            current_stage=self.current_stage,
            progress=self.progress,
            started_at=self.started_at,
            completed_at=self.completed_at,
            total_duration_ms=total_duration,
        )


class ExperimentManager:
    """Manages experiment lifecycle and in-memory storage."""

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup(self) -> None:
        """Start periodic cleanup of old experiments."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            await self._cleanup_old_experiments()

    async def _cleanup_old_experiments(self) -> None:
        now = datetime.now(timezone.utc)
        to_remove = []
        async with self._lock:
            for exp_id, exp in self._experiments.items():
                if exp.completed_at:
                    completed = datetime.fromisoformat(exp.completed_at.replace("Z", "+00:00"))
                    if (now - completed).total_seconds() > settings.EXPERIMENT_TTL_SECONDS:
                        to_remove.append(exp_id)
            for exp_id in to_remove:
                del self._experiments[exp_id]

    async def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create a new experiment."""
        async with self._lock:
            if len(self._experiments) >= settings.MAX_CONCURRENT_EXPERIMENTS:
                raise RuntimeError("Maximum concurrent experiments reached")
            exp = Experiment(config)
            self._experiments[exp.experiment_id] = exp
            return exp

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        async with self._lock:
            return self._experiments.get(experiment_id)

    async def list_experiments(self) -> List[Experiment]:
        async with self._lock:
            return list(self._experiments.values())

    async def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        current_stage: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        async with self._lock:
            exp = self._experiments.get(experiment_id)
            if exp:
                exp.status = status
                if current_stage:
                    exp.current_stage = current_stage
                if progress is not None:
                    exp.progress = progress
                if status == ExperimentStatus.RUNNING and not exp.started_at:
                    exp.started_at = datetime.now(timezone.utc).isoformat()
                if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED):
                    exp.completed_at = datetime.now(timezone.utc).isoformat()

    async def set_results(self, experiment_id: str, results: ExperimentResultsResponse) -> None:
        async with self._lock:
            exp = self._experiments.get(experiment_id)
            if exp:
                exp.results = results
                exp.status = ExperimentStatus.COMPLETED
                exp.progress = 1.0
                exp.completed_at = datetime.now(timezone.utc).isoformat()

    async def set_error(self, experiment_id: str, error: str) -> None:
        async with self._lock:
            exp = self._experiments.get(experiment_id)
            if exp:
                exp.error = error
                exp.status = ExperimentStatus.FAILED
                exp.completed_at = datetime.now(timezone.utc).isoformat()

    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Request cancellation of an experiment."""
        async with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            if exp.status not in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING):
                return False
            exp._cancelled = True
            if exp.task and not exp.task.done():
                exp.task.cancel()
            return True

    async def is_cancelled(self, experiment_id: str) -> bool:
        async with self._lock:
            exp = self._experiments.get(experiment_id)
            return exp._cancelled if exp else False


# Global experiment manager instance
experiment_manager = ExperimentManager()