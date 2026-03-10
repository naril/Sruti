from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Lock, Semaphore, Thread
from typing import ContextManager, Generic, Iterator, Protocol, TypeVar

from sruti.domain.enums import LlmProvider, StageId
from sruti.infrastructure import json_codec
from sruti.util.io import atomic_write_json, ensure_dir
from sruti.util.manifest import utc_now_iso

T = TypeVar("T")


class ExecutionResource(str, Enum):
    NONE = "none"
    LOCAL_HEAVY = "local_heavy"
    EXTERNAL_API = "external_api"


@dataclass(slots=True)
class BatchSchedulerConfig:
    runs_root: Path
    max_active_runs: int
    local_slots: int
    external_api_slots: int
    external_api_slots_per_run: int

    def effective_max_active_runs(self) -> int:
        if self.max_active_runs > 0:
            return self.max_active_runs
        return self.local_slots + self.external_api_slots


@dataclass(slots=True)
class RunWorkItem:
    audio_path: Path
    run_dir: Path
    run_index: int
    total_runs: int


@dataclass(slots=True)
class BatchEvent:
    ts: str
    seq: int
    event: str
    audio_path: str
    run_dir: str
    run_index: int
    total_runs: int
    stage: str | None = None
    resource: str | None = None
    status: str | None = None
    message: str | None = None
    task_label: str | None = None

    def to_row(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(slots=True)
class BatchRunState:
    audio_path: str
    run_index: int
    run_dir: str
    current_stage: str | None = None
    resource: str | None = None
    status: str = "queued"
    last_message: str | None = None
    started_at: str | None = None
    updated_at: str = ""
    finished_at: str | None = None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ExternalApiTask(Generic[T]):
    index: int
    label: str
    run: Callable[[], T]


class ExecutionCoordinator(Protocol):
    def emit_progress(self, message: str) -> None:
        ...

    def stage_scope(
        self,
        stage_id: StageId,
        *,
        llm_provider: LlmProvider,
    ) -> ContextManager[None]:
        ...

    def submit_external_api_task(
        self,
        *,
        stage_id: StageId,
        task_label: str,
        fn: Callable[[], T],
    ) -> Future[T]:
        ...

    def max_external_api_parallelism(self) -> int:
        ...


def resolve_stage_execution_resource(
    stage_id: StageId,
    *,
    llm_provider: LlmProvider,
) -> ExecutionResource:
    if stage_id in {StageId.S01, StageId.S02, StageId.S03}:
        return ExecutionResource.LOCAL_HEAVY
    if stage_id is StageId.S04:
        return ExecutionResource.NONE
    if llm_provider is LlmProvider.OPENAI:
        return ExecutionResource.EXTERNAL_API
    return ExecutionResource.LOCAL_HEAVY


def execute_ordered_external_api_tasks(
    coordinator: ExecutionCoordinator,
    *,
    stage_id: StageId,
    tasks: list[ExternalApiTask[T]],
    max_parallel: int | None = None,
) -> list[T]:
    if not tasks:
        return []

    effective_limit = coordinator.max_external_api_parallelism()
    if max_parallel is not None:
        effective_limit = min(effective_limit, max_parallel)
    effective_limit = max(1, min(effective_limit, len(tasks)))

    pending: dict[Future[T], ExternalApiTask[T]] = {}
    results: dict[int, T] = {}
    task_iter = iter(tasks)

    def _submit(task: ExternalApiTask[T]) -> None:
        future = coordinator.submit_external_api_task(
            stage_id=stage_id,
            task_label=task.label,
            fn=task.run,
        )
        pending[future] = task

    for _ in range(effective_limit):
        task = next(task_iter, None)
        if task is None:
            break
        _submit(task)

    while pending:
        done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
        for future in done:
            task = pending.pop(future)
            try:
                results[task.index] = future.result()
            except Exception:
                for pending_future in pending:
                    pending_future.cancel()
                raise
            next_task = next(task_iter, None)
            if next_task is not None:
                _submit(next_task)

    return [results[task.index] for task in sorted(tasks, key=lambda item: item.index)]


class RunExecutionCoordinator:
    def __init__(self, scheduler: "BatchScheduler", work_item: RunWorkItem) -> None:
        self._scheduler = scheduler
        self._work_item = work_item
        self._state_lock = Lock()
        self._current_stage: StageId | None = None

    def emit_progress(self, message: str) -> None:
        with self._state_lock:
            stage = self._current_stage
        self._scheduler.publish_event(
            self._work_item,
            event="progress",
            stage=stage,
            status="running",
            message=message,
        )

    @contextmanager
    def stage_scope(
        self,
        stage_id: StageId,
        *,
        llm_provider: LlmProvider,
    ) -> Iterator[None]:
        resource = resolve_stage_execution_resource(stage_id, llm_provider=llm_provider)
        release_local_slot = False
        if resource is ExecutionResource.LOCAL_HEAVY:
            self._scheduler.publish_event(
                self._work_item,
                event="stage_waiting",
                stage=stage_id,
                resource=resource,
                status="waiting",
            )
            self._scheduler.local_slot_semaphore.acquire()
            release_local_slot = True
            self._scheduler.publish_event(
                self._work_item,
                event="local_slot_acquired",
                stage=stage_id,
                resource=resource,
            )

        with self._state_lock:
            self._current_stage = stage_id
        self._scheduler.publish_event(
            self._work_item,
            event="stage_started",
            stage=stage_id,
            resource=resource,
            status="running",
        )
        try:
            yield
        except Exception as exc:
            self._scheduler.publish_event(
                self._work_item,
                event="stage_failed",
                stage=stage_id,
                resource=resource,
                status="failed",
                message=str(exc),
            )
            raise
        else:
            self._scheduler.publish_event(
                self._work_item,
                event="stage_finished",
                stage=stage_id,
                resource=resource,
                status="success",
            )
        finally:
            with self._state_lock:
                self._current_stage = None
            if release_local_slot:
                self._scheduler.publish_event(
                    self._work_item,
                    event="local_slot_released",
                    stage=stage_id,
                    resource=resource,
                )
                self._scheduler.local_slot_semaphore.release()

    def submit_external_api_task(
        self,
        *,
        stage_id: StageId,
        task_label: str,
        fn: Callable[[], T],
    ) -> Future[T]:
        executor = self._scheduler.external_api_executor
        if executor is None:
            raise RuntimeError("external API executor is not initialized")
        self._scheduler.publish_event(
            self._work_item,
            event="external_task_submitted",
            stage=stage_id,
            resource=ExecutionResource.EXTERNAL_API,
            task_label=task_label,
        )

        def _wrapped() -> T:
            self._scheduler.publish_event(
                self._work_item,
                event="external_task_started",
                stage=stage_id,
                resource=ExecutionResource.EXTERNAL_API,
                task_label=task_label,
            )
            try:
                result = fn()
            except Exception as exc:
                self._scheduler.publish_event(
                    self._work_item,
                    event="external_task_failed",
                    stage=stage_id,
                    resource=ExecutionResource.EXTERNAL_API,
                    task_label=task_label,
                    status="failed",
                    message=str(exc),
                )
                raise
            self._scheduler.publish_event(
                self._work_item,
                event="external_task_finished",
                stage=stage_id,
                resource=ExecutionResource.EXTERNAL_API,
                task_label=task_label,
                status="success",
            )
            return result

        return executor.submit(_wrapped)

    def max_external_api_parallelism(self) -> int:
        return min(
            self._scheduler.config.external_api_slots,
            self._scheduler.config.external_api_slots_per_run,
        )


class BatchScheduler:
    def __init__(
        self,
        *,
        config: BatchSchedulerConfig,
        progress_emitter: Callable[[str], None],
    ) -> None:
        self.config = config
        self._progress_emitter = progress_emitter
        self._event_queue: Queue[BatchEvent | None] = Queue()
        self._event_seq = 0
        self._event_seq_lock = Lock()
        self._snapshot_lock = Lock()
        self._run_states: dict[str, BatchRunState] = {}
        self._local_slots_in_use = 0
        self._external_api_in_use = 0
        self.local_slot_semaphore = Semaphore(config.local_slots)
        self.external_api_executor: ThreadPoolExecutor | None = None
        self._events_path = config.runs_root / "batch_scheduler_events.jsonl"
        self._state_path = config.runs_root / "batch_scheduler_state.json"

    def make_run_coordinator(self, work_item: RunWorkItem) -> RunExecutionCoordinator:
        return RunExecutionCoordinator(self, work_item)

    def run(
        self,
        work_items: list[RunWorkItem],
        *,
        worker: Callable[[RunWorkItem, RunExecutionCoordinator], None],
    ) -> list[tuple[RunWorkItem, str]]:
        ensure_dir(self.config.runs_root)
        self._events_path.write_text("", encoding="utf-8")

        for work_item in work_items:
            self._run_states[work_item.run_dir.name] = BatchRunState(
                audio_path=str(work_item.audio_path),
                run_index=work_item.run_index,
                run_dir=work_item.run_dir.name,
                updated_at=utc_now_iso(),
            )

        consumer = Thread(target=self._consume_events, name="batch-scheduler-events", daemon=True)
        consumer.start()

        for work_item in work_items:
            self.publish_event(work_item, event="run_queued", status="queued")

        failures: list[tuple[RunWorkItem, str]] = []
        max_workers = max(1, self.config.effective_max_active_runs())
        with ThreadPoolExecutor(max_workers=self.config.external_api_slots) as external_api_executor:
            self.external_api_executor = external_api_executor
            with ThreadPoolExecutor(max_workers=max_workers) as run_executor:
                future_to_item = {
                    run_executor.submit(self._run_work_item, work_item, worker): work_item
                    for work_item in work_items
                }
                for future, work_item in future_to_item.items():
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append((work_item, str(exc)))

        self.publish_snapshot()
        self._event_queue.put(None)
        consumer.join()
        self.external_api_executor = None
        return failures

    def _run_work_item(
        self,
        work_item: RunWorkItem,
        worker: Callable[[RunWorkItem, RunExecutionCoordinator], None],
    ) -> None:
        self.publish_event(work_item, event="run_started", status="running")
        try:
            worker(work_item, self.make_run_coordinator(work_item))
        except Exception as exc:
            self.publish_event(
                work_item,
                event="run_failed",
                status="failed",
                message=str(exc),
            )
            raise
        self.publish_event(work_item, event="run_finished", status="completed")

    def publish_event(
        self,
        work_item: RunWorkItem,
        *,
        event: str,
        stage: StageId | None = None,
        resource: ExecutionResource | None = None,
        status: str | None = None,
        message: str | None = None,
        task_label: str | None = None,
    ) -> None:
        with self._event_seq_lock:
            self._event_seq += 1
            seq = self._event_seq
        self._event_queue.put(
            BatchEvent(
                ts=utc_now_iso(),
                seq=seq,
                event=event,
                audio_path=str(work_item.audio_path),
                run_dir=work_item.run_dir.name,
                run_index=work_item.run_index,
                total_runs=work_item.total_runs,
                stage=stage.value if stage is not None else None,
                resource=resource.value if resource is not None else None,
                status=status,
                message=message,
                task_label=task_label,
            )
        )

    def publish_snapshot(self) -> None:
        with self._snapshot_lock:
            atomic_write_json(self._state_path, self._snapshot_payload())

    def _consume_events(self) -> None:
        with self._events_path.open("a", encoding="utf-8") as handle:
            while True:
                event = self._event_queue.get()
                if event is None:
                    with self._snapshot_lock:
                        atomic_write_json(self._state_path, self._snapshot_payload())
                    return
                handle.write(json_codec.dumps(event.to_row()) + "\n")
                handle.flush()
                with self._snapshot_lock:
                    self._apply_event(event)
                    atomic_write_json(self._state_path, self._snapshot_payload())
                line = self._format_cli_line(event)
                if line:
                    self._progress_emitter(line)

    def _apply_event(self, event: BatchEvent) -> None:
        run_state = self._run_states[event.run_dir]
        run_state.updated_at = event.ts
        if event.message:
            run_state.last_message = event.message

        if event.event == "run_queued":
            run_state.status = "queued"
            return
        if event.event == "run_started":
            run_state.status = "running"
            run_state.started_at = event.ts
            return
        if event.event == "run_finished":
            run_state.status = "completed"
            run_state.finished_at = event.ts
            run_state.current_stage = None
            run_state.resource = None
            return
        if event.event == "run_failed":
            run_state.status = "failed"
            run_state.finished_at = event.ts
            return
        if event.event == "stage_waiting":
            run_state.status = "waiting"
            run_state.current_stage = event.stage
            run_state.resource = event.resource
            return
        if event.event == "stage_started":
            run_state.status = "running"
            run_state.current_stage = event.stage
            run_state.resource = event.resource
            return
        if event.event == "stage_finished":
            run_state.status = "running"
            run_state.current_stage = None
            run_state.resource = None
            return
        if event.event == "stage_failed":
            run_state.status = "failed"
            run_state.current_stage = event.stage
            run_state.resource = event.resource
            return
        if event.event == "local_slot_acquired":
            self._local_slots_in_use += 1
            return
        if event.event == "local_slot_released":
            self._local_slots_in_use = max(0, self._local_slots_in_use - 1)
            return
        if event.event == "external_task_started":
            self._external_api_in_use += 1
            return
        if event.event in {"external_task_failed", "external_task_finished"}:
            self._external_api_in_use = max(0, self._external_api_in_use - 1)

    def _snapshot_payload(self) -> dict[str, object]:
        runs = {name: state.to_row() for name, state in sorted(self._run_states.items())}
        queued_runs = [name for name, state in runs.items() if state["status"] == "queued"]
        completed_runs = [name for name, state in runs.items() if state["status"] == "completed"]
        failed_runs = [name for name, state in runs.items() if state["status"] == "failed"]
        running_runs = [
            name
            for name, state in runs.items()
            if state["status"] not in {"queued", "completed", "failed"}
        ]
        return {
            "summary": {
                "total_runs": len(runs),
                "queued_runs": len(queued_runs),
                "running_runs": len(running_runs),
                "completed_runs": len(completed_runs),
                "failed_runs": len(failed_runs),
            },
            "resources": {
                "max_active_runs": self.config.effective_max_active_runs(),
                "local_slots": self.config.local_slots,
                "local_slots_in_use": self._local_slots_in_use,
                "external_api_slots": self.config.external_api_slots,
                "external_api_slots_in_use": self._external_api_in_use,
                "external_api_slots_per_run": self.config.external_api_slots_per_run,
            },
            "queued_runs": queued_runs,
            "running_runs": running_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "runs": runs,
        }

    def _format_cli_line(self, event: BatchEvent) -> str | None:
        if event.event == "run_started":
            return f"[run-batch] file {event.run_index}/{event.total_runs}: {event.audio_path}"
        if event.event == "progress" and event.message is not None:
            return f"[run-batch][{event.run_dir}] {event.message}"
        if event.event == "run_failed":
            return f"[run-batch] failed for {event.audio_path}: {event.message}"
        return None
