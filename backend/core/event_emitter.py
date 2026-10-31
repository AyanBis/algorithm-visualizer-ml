"""Event emitter for SSE workflow events."""
import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime
import uuid

from ..schemas.workflow import WorkflowEvent, EventType, StageStatus
from ..schemas.experiments import ExperimentStatus
from ..core.config import settings


class EventBuffer:
    """Thread-safe event buffer with size limit."""

    def __init__(self, max_size: int = 1000):
        self._events: List[WorkflowEvent] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._subscribers: Dict[str, asyncio.Queue] = {}

    async def add(self, event: WorkflowEvent) -> None:
        async with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_size:
                self._events = self._events[-self._max_size:]

            # Notify subscribers
            for exp_id, queue in self._subscribers.items():
                if event.experiment_id == exp_id:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass  # Subscriber too slow, drop event

    async def get_events(self, experiment_id: str, after_sequence: int = 0) -> List[WorkflowEvent]:
        async with self._lock:
            return [
                e for e in self._events
                if e.experiment_id == experiment_id and e.sequence > after_sequence
            ]

    async def get_latest_sequence(self, experiment_id: str) -> int:
        async with self._lock:
            exp_events = [e for e in self._events if e.experiment_id == experiment_id]
            if not exp_events:
                return 0
            return max(e.sequence for e in exp_events)

    async def subscribe(self, experiment_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers[experiment_id] = queue
        return queue

    async def unsubscribe(self, experiment_id: str) -> None:
        async with self._lock:
            self._subscribers.pop(experiment_id, None)


class EventEmitter:
    """Manages experiment events and SSE streaming."""

    def __init__(self):
        self.buffer = EventBuffer(settings.EVENT_BUFFER_SIZE)

    async def emit(self, event: WorkflowEvent) -> None:
        await self.buffer.add(event)

    async def get_events(self, experiment_id: str, after_sequence: int = 0) -> List[WorkflowEvent]:
        return await self.buffer.get_events(experiment_id, after_sequence)

    async def get_latest_sequence(self, experiment_id: str) -> int:
        return await self.buffer.get_latest_sequence(experiment_id)

    async def stream_events(
        self,
        experiment_id: str,
        last_event_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """SSE event stream generator."""
        # Parse last_event_id to get sequence
        after_sequence = 0
        if last_event_id:
            try:
                after_sequence = int(last_event_id)
            except ValueError:
                pass

        # Send any missed events first
        missed = await self.get_events(experiment_id, after_sequence)
        for event in missed:
            yield self._format_sse(event)

        # Subscribe for new events
        queue = await self.buffer.subscribe(experiment_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield self._format_sse(event)
                    # Stop streaming on terminal events
                    if event.event_type in (EventType.EXPERIMENT_COMPLETED, EventType.EXPERIMENT_FAILED):
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat comment
                    yield ": heartbeat\n\n"
        finally:
            await self.buffer.unsubscribe(experiment_id)

    def _format_sse(self, event: WorkflowEvent) -> str:
        data = event.model_dump_json()
        return f"id: {event.sequence}\ndata: {data}\n\n"


# Global event emitter instance
event_emitter = EventEmitter()