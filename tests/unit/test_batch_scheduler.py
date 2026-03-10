from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from sruti.application.batch_scheduler import (
    BatchScheduler,
    BatchSchedulerConfig,
    ExecutionResource,
    RunWorkItem,
    execute_ordered_external_api_tasks,
    resolve_stage_execution_resource,
    ExternalApiTask,
)
from sruti.domain.enums import LlmProvider, StageId


class FakeCoordinator:
    def __init__(self, *, max_parallel: int) -> None:
        self._max_parallel = max_parallel
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)

    def emit_progress(self, message: str) -> None:
        _ = message

    def stage_scope(self, stage_id: StageId, *, llm_provider: LlmProvider):
        _ = (stage_id, llm_provider)
        return nullcontext()

    def submit_external_api_task(
        self,
        *,
        stage_id: StageId,
        task_label: str,
        fn,
    ) -> Future[int]:
        _ = (stage_id, task_label)
        return self._executor.submit(fn)

    def max_external_api_parallelism(self) -> int:
        return self._max_parallel

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def test_resolve_stage_execution_resource_uses_provider_and_stage() -> None:
    assert resolve_stage_execution_resource(StageId.S01, llm_provider=LlmProvider.LOCAL) is ExecutionResource.LOCAL_HEAVY
    assert resolve_stage_execution_resource(StageId.S04, llm_provider=LlmProvider.OPENAI) is ExecutionResource.NONE
    assert resolve_stage_execution_resource(StageId.S05, llm_provider=LlmProvider.OPENAI) is ExecutionResource.EXTERNAL_API
    assert resolve_stage_execution_resource(StageId.S05, llm_provider=LlmProvider.LOCAL) is ExecutionResource.LOCAL_HEAVY


def test_execute_ordered_external_api_tasks_preserves_input_order() -> None:
    coordinator = FakeCoordinator(max_parallel=2)
    try:
        tasks = [
            ExternalApiTask(index=1, label="first", run=lambda: (time.sleep(0.05), 1)[1]),
            ExternalApiTask(index=2, label="second", run=lambda: (time.sleep(0.01), 2)[1]),
            ExternalApiTask(index=3, label="third", run=lambda: 3),
        ]
        results = execute_ordered_external_api_tasks(
            coordinator,
            stage_id=StageId.S07,
            tasks=tasks,
        )
    finally:
        coordinator.shutdown()

    assert results == [1, 2, 3]


def test_batch_scheduler_writes_state_and_events(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    config = BatchSchedulerConfig(
        runs_root=runs_root,
        max_active_runs=2,
        local_slots=1,
        external_api_slots=2,
        external_api_slots_per_run=1,
    )
    progress_lines: list[str] = []
    scheduler = BatchScheduler(config=config, progress_emitter=progress_lines.append)
    work_items = [
        RunWorkItem(
            audio_path=tmp_path / "a.wav",
            run_dir=runs_root / "a",
            run_index=1,
            total_runs=2,
        ),
        RunWorkItem(
            audio_path=tmp_path / "b.wav",
            run_dir=runs_root / "b",
            run_index=2,
            total_runs=2,
        ),
    ]

    def _worker(work_item: RunWorkItem, coordinator) -> None:
        _ = work_item
        with coordinator.stage_scope(StageId.S02, llm_provider=LlmProvider.LOCAL):
            coordinator.emit_progress("[s02] chunking audio...")

    failures = scheduler.run(work_items, worker=_worker)

    assert failures == []
    state = json.loads((runs_root / "batch_scheduler_state.json").read_text(encoding="utf-8"))
    assert state["summary"] == {
        "total_runs": 2,
        "queued_runs": 0,
        "running_runs": 0,
        "completed_runs": 2,
        "failed_runs": 0,
    }
    assert sorted(state["completed_runs"]) == ["a", "b"]
    event_lines = (runs_root / "batch_scheduler_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert event_lines
    first_event = json.loads(event_lines[0])
    assert first_event["event"] == "run_queued"
    assert any("[run-batch][a] [s02] chunking audio..." == line for line in progress_lines)
