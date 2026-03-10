from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Callable

from sruti.domain.enums import RunStatus, StageId, StageStatus
from sruti.domain.models import RunEvent, RunState, StageResult
from sruti.infrastructure import json_codec
from sruti.util.io import atomic_write_json, atomic_write_text, ensure_dir
from sruti.util.manifest import utc_now_iso

RUN_STATE_FILENAME = "run_state.json"
RUN_EVENTS_FILENAME = "run_events.jsonl"


def state_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / RUN_STATE_FILENAME, run_dir / RUN_EVENTS_FILENAME


class RunStateTracker:
    def __init__(
        self,
        *,
        run_dir: Path,
        from_stage: StageId,
        to_stage: StageId,
        input_path: Path | None,
        progress_emitter: Callable[[str], None] | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._state_path, self._events_path = state_paths(run_dir)
        self._progress_emitter = progress_emitter
        self._lock = Lock()
        self._seq = 0
        self._state = RunState(
            run_dir=str(run_dir),
            input_path=str(input_path) if input_path is not None else None,
            from_stage=from_stage,
            to_stage=to_stage,
            status=RunStatus.QUEUED,
        )

    def context_with_progress(self, context) -> object:
        return replace(context, progress_emitter=self.emit_progress)

    @property
    def current_state(self) -> RunState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def start_run(self) -> None:
        ensure_dir(self._run_dir)
        atomic_write_text(self._events_path, "")
        with self._lock:
            self._state.status = RunStatus.RUNNING
            self._state.started_at = utc_now_iso()
            self._state.finished_at = None
            self._state.updated_at = utc_now_iso()
            self._write_state_locked()
        self._append_event(event="run_started", status=RunStatus.RUNNING.value)

    def stage_started(self, stage_id: StageId) -> None:
        with self._lock:
            self._state.current_stage = stage_id
            self._state.updated_at = utc_now_iso()
            self._write_state_locked()
        self._append_event(event="stage_started", stage=stage_id, status=RunStatus.RUNNING.value)

    def stage_finished(self, stage_id: StageId, status: StageStatus) -> None:
        with self._lock:
            self._state.stages[stage_id.value] = status
            self._state.current_stage = None
            self._state.updated_at = utc_now_iso()
            self._write_state_locked()
        self._append_event(event="stage_finished", stage=stage_id, status=status.value)

    def stage_failed(self, stage_id: StageId, message: str) -> None:
        with self._lock:
            self._state.stages[stage_id.value] = StageStatus.FAILED
            self._state.current_stage = stage_id
            self._state.status = RunStatus.FAILED
            self._state.last_message = message
            self._state.updated_at = utc_now_iso()
            self._write_state_locked()
        self._append_event(event="stage_failed", stage=stage_id, status=RunStatus.FAILED.value, message=message)

    def emit_progress(self, message: str) -> None:
        current_stage: StageId | None
        with self._lock:
            current_stage = self._state.current_stage
            self._state.last_message = message
            self._state.updated_at = utc_now_iso()
            self._write_state_locked()
        self._append_event(event="progress", stage=current_stage, status=RunStatus.RUNNING.value, message=message)
        if self._progress_emitter is not None:
            self._progress_emitter(message)

    def finish(self, results: list[StageResult]) -> None:
        if results and all(result.status is StageStatus.DRY_RUN for result in results):
            final_status = RunStatus.DRY_RUN
        else:
            final_status = RunStatus.COMPLETED
        with self._lock:
            self._state.status = final_status
            self._state.current_stage = None
            self._state.finished_at = utc_now_iso()
            self._state.updated_at = utc_now_iso()
            self._write_state_locked()
        self._append_event(event="run_finished", status=final_status.value)

    def fail(self, message: str) -> None:
        with self._lock:
            self._state.status = RunStatus.FAILED
            self._state.finished_at = utc_now_iso()
            self._state.updated_at = utc_now_iso()
            self._state.last_message = message
            self._write_state_locked()
        self._append_event(event="run_failed", status=RunStatus.FAILED.value, message=message)

    def _append_event(
        self,
        *,
        event: str,
        stage: StageId | None = None,
        status: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            self._seq += 1
            payload = RunEvent(
                ts=utc_now_iso(),
                seq=self._seq,
                event=event,
                run_dir=self._run_dir.name,
                stage=stage,
                status=status,
                message=message,
            ).model_dump(mode="json")
        ensure_dir(self._events_path.parent)
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json_codec.dumps(payload) + "\n")

    def _write_state_locked(self) -> None:
        atomic_write_json(self._state_path, self._state.model_dump(mode="json"))
