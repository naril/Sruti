from __future__ import annotations

import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sruti.application.batch_scheduler import BatchScheduler, BatchSchedulerConfig, RunWorkItem
from sruti.application.context import StageContext
from sruti.application.run_state import RunStateTracker
from sruti.config import load_settings
from sruti.domain.enums import LlmProvider, OnExistsMode, StageId
from sruti.domain.errors import SrutiError
from sruti.domain.models import StageResult
from sruti.domain.policies import stage_ids_in_range
from sruti.stages import (
    s01_normalize,
    s02_chunk,
    s03_asr_whispercli,
    s04_merge,
    s05_asr_cleanup,
    s06_remove_nonlecture,
    s07_editorial,
    s08_condense,
    s09_translate_faithful,
    s10_translate_edit,
)
from sruti.util.io import atomic_write_json

BATCH_MANIFEST_FILENAME = "batch_manifest.json"
AUDIO_EXTENSIONS: set[str] = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".m4b",
    ".mka",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}
RUN_DIR_NAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


def _noop_progress(_: str) -> None:
    return


@dataclass(slots=True)
class RunRequest:
    run_dir: Path
    in_path: Path | None
    source_stage: StageId
    target_stage: StageId
    seconds: int | None
    model_path: Path | None
    on_exists: OnExistsMode
    dry_run: bool
    force: bool
    verbose: bool
    llm_provider: LlmProvider | None
    cost_cap_usd: float | None
    token_cap_input: int | None
    token_cap_output: int | None
    command_label: str = "run"
    settings_dir: Path | None = None
    progress_emitter: Callable[[str], None] | None = None
    result_emitter: Callable[[StageResult], None] | None = None
    ask_user: Callable[[str], bool] | None = None
    execution_coordinator: object | None = None


@dataclass(slots=True)
class BatchRunRequest:
    runs_root: Path
    in_dir: Path
    source_stage: StageId
    target_stage: StageId
    seconds: int | None
    model_path: Path | None
    on_exists: OnExistsMode
    dry_run: bool
    force: bool
    verbose: bool
    llm_provider: LlmProvider | None
    cost_cap_usd: float | None
    token_cap_input: int | None
    token_cap_output: int | None
    max_active_runs: int | None
    local_slots: int | None
    external_api_slots: int | None
    external_api_slots_per_run: int | None
    progress_emitter: Callable[[str], None] | None = None
    result_emitter: Callable[[StageResult], None] | None = None
    ask_user: Callable[[str], bool] | None = None


def build_stage_context(
    *,
    run_dir: Path,
    settings_dir: Path | None,
    on_exists: OnExistsMode,
    dry_run: bool,
    force: bool,
    verbose: bool,
    llm_provider: LlmProvider | None,
    cost_cap_usd: float | None,
    token_cap_input: int | None,
    token_cap_output: int | None,
    progress_emitter: Callable[[str], None] | None,
    execution_coordinator=None,
) -> StageContext:
    effective_progress_emitter = progress_emitter
    if execution_coordinator is not None:
        effective_progress_emitter = _noop_progress
    return StageContext.build(
        run_dir=run_dir,
        settings_dir=settings_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider_override=llm_provider,
        cost_cap_usd_override=cost_cap_usd,
        token_cap_input_override=token_cap_input,
        token_cap_output_override=token_cap_output,
        progress_emitter=effective_progress_emitter,
        execution_coordinator=execution_coordinator,
    )


def run_single_stage(
    *,
    stage_id: StageId,
    context: StageContext,
    in_path: Path | None,
    seconds: int | None,
    model_path: Path | None,
    ask_user: Callable[[str], bool] | None,
) -> StageResult:
    if stage_id is StageId.S01:
        if in_path is None:
            raise ValueError("--in is required when running s01.")
        return s01_normalize.run_stage(context=context, input_audio=in_path, ask_user=ask_user)
    if stage_id is StageId.S02:
        effective_seconds = seconds if seconds is not None else context.settings.chunk_seconds
        return s02_chunk.run_stage(context=context, seconds=effective_seconds, ask_user=ask_user)
    if stage_id is StageId.S03:
        effective_model_path = (
            model_path
            if model_path is not None
            else context.settings.default_whisper_model_path
        )
        return s03_asr_whispercli.run_stage(
            context=context,
            model_path=effective_model_path,
            ask_user=ask_user,
        )
    if stage_id is StageId.S04:
        return s04_merge.run_stage(context=context, ask_user=ask_user)
    if stage_id is StageId.S05:
        return s05_asr_cleanup.run_stage(context=context, ask_user=ask_user)
    if stage_id is StageId.S06:
        return s06_remove_nonlecture.run_stage(context=context, ask_user=ask_user)
    if stage_id is StageId.S07:
        return s07_editorial.run_stage(context=context, ask_user=ask_user)
    if stage_id is StageId.S08:
        return s08_condense.run_stage(context=context, ask_user=ask_user)
    if stage_id is StageId.S09:
        return s09_translate_faithful.run_stage(context=context, ask_user=ask_user)
    if stage_id is StageId.S10:
        return s10_translate_edit.run_stage(context=context, ask_user=ask_user)
    raise RuntimeError("unreachable")


def execute_run(request: RunRequest) -> list[StageResult]:
    base_context = build_stage_context(
        run_dir=request.run_dir,
        settings_dir=request.settings_dir,
        on_exists=request.on_exists,
        dry_run=request.dry_run,
        force=request.force,
        verbose=request.verbose,
        llm_provider=request.llm_provider,
        cost_cap_usd=request.cost_cap_usd,
        token_cap_input=request.token_cap_input,
        token_cap_output=request.token_cap_output,
        progress_emitter=request.progress_emitter,
        execution_coordinator=request.execution_coordinator,
    )
    tracker = RunStateTracker(
        run_dir=request.run_dir,
        from_stage=request.source_stage,
        to_stage=request.target_stage,
        input_path=request.in_path,
        progress_emitter=request.progress_emitter,
    )
    context = tracker.context_with_progress(base_context)
    tracker.start_run()
    try:
        context.emit_progress(
            f"[{request.command_label}] starting: "
            f"{request.source_stage.value}->{request.target_stage.value} in {context.run_dir}"
        )
        results = _run_stage_range(
            context=context,
            source_stage=request.source_stage,
            target_stage=request.target_stage,
            in_path=request.in_path,
            seconds=request.seconds,
            model_path=request.model_path,
            ask_user=request.ask_user,
            result_emitter=request.result_emitter,
            tracker=tracker,
        )
    except Exception as exc:
        tracker.fail(str(exc))
        raise
    tracker.finish(results)
    return results


def _run_stage_range(
    *,
    context: StageContext,
    source_stage: StageId,
    target_stage: StageId,
    in_path: Path | None,
    seconds: int | None,
    model_path: Path | None,
    ask_user: Callable[[str], bool] | None,
    result_emitter: Callable[[StageResult], None] | None,
    tracker: RunStateTracker | None,
) -> list[StageResult]:
    results: list[StageResult] = []
    for stage_id in stage_ids_in_range(source_stage, target_stage):
        if tracker is not None:
            tracker.stage_started(stage_id)
        stage_scope = (
            context.execution_coordinator.stage_scope(
                stage_id,
                llm_provider=context.settings.llm_provider,
            )
            if context.execution_coordinator is not None
            else nullcontext()
        )
        try:
            with stage_scope:
                result = run_single_stage(
                    stage_id=stage_id,
                    context=context,
                    in_path=in_path,
                    seconds=seconds,
                    model_path=model_path,
                    ask_user=ask_user,
                )
        except Exception as exc:
            if tracker is not None:
                tracker.stage_failed(stage_id, str(exc))
            raise
        results.append(result)
        if tracker is not None:
            tracker.stage_finished(stage_id, result.status)
        if result_emitter is not None:
            result_emitter(result)
    return results


def build_batch_scheduler_config(
    *,
    runs_root: Path,
    max_active_runs: int | None,
    local_slots: int | None,
    external_api_slots: int | None,
    external_api_slots_per_run: int | None,
) -> BatchSchedulerConfig:
    settings = load_settings(runs_root)
    resolved_max_active_runs = (
        settings.batch_max_active_runs if max_active_runs is None else max_active_runs
    )
    resolved_local_slots = settings.batch_local_slots if local_slots is None else local_slots
    resolved_external_api_slots = (
        settings.batch_external_api_slots if external_api_slots is None else external_api_slots
    )
    resolved_external_api_slots_per_run = (
        settings.batch_external_api_slots_per_run
        if external_api_slots_per_run is None
        else external_api_slots_per_run
    )
    if resolved_max_active_runs < 0:
        raise ValueError("batch max active runs must be >= 0.")
    if resolved_local_slots < 1:
        raise ValueError("batch local slots must be >= 1.")
    if resolved_external_api_slots < 1:
        raise ValueError("batch external API slots must be >= 1.")
    if resolved_external_api_slots_per_run < 1:
        raise ValueError("batch external API slots per run must be >= 1.")
    return BatchSchedulerConfig(
        runs_root=runs_root,
        max_active_runs=resolved_max_active_runs,
        local_slots=resolved_local_slots,
        external_api_slots=resolved_external_api_slots,
        external_api_slots_per_run=resolved_external_api_slots_per_run,
    )


def discover_audio_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path.relative_to(input_dir)))


def load_batch_mapping(runs_root: Path) -> dict[str, str]:
    manifest_path = runs_root / BATCH_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    raw_mapping = payload.get("audio_to_run_dir")
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{manifest_path} must contain object key 'audio_to_run_dir'.")
    mapping: dict[str, str] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{manifest_path} contains non-string mapping entries.")
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{manifest_path} contains invalid run dir name '{value}'.")
        mapping[key] = value
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{manifest_path} contains duplicate run dir assignments.")
    return mapping


def save_batch_mapping(runs_root: Path, mapping: dict[str, str]) -> None:
    manifest_path = runs_root / BATCH_MANIFEST_FILENAME
    payload = {"audio_to_run_dir": dict(sorted(mapping.items()))}
    atomic_write_json(manifest_path, payload)


def sanitize_run_dir_name(stem: str) -> str:
    normalized = RUN_DIR_NAME_SANITIZER.sub("-", stem).strip("._-").lower()
    return normalized or "audio"


def next_run_dir_name(base_stem: str, used_names: set[str]) -> str:
    candidate = sanitize_run_dir_name(base_stem)
    if candidate not in used_names:
        return candidate
    suffix = 2
    while True:
        suffixed = f"{candidate}-{suffix}"
        if suffixed not in used_names:
            return suffixed
        suffix += 1


def assign_batch_run_dirs(
    *,
    runs_root: Path,
    audio_files: list[Path],
    mapping: dict[str, str],
) -> tuple[dict[Path, Path], bool]:
    used_names = {path.name for path in runs_root.iterdir() if path.is_dir()}
    used_names.update(mapping.values())
    assignments: dict[Path, Path] = {}
    changed = False
    for audio_path in audio_files:
        key = str(audio_path.resolve())
        run_dir_name = mapping.get(key)
        if run_dir_name is None:
            run_dir_name = next_run_dir_name(audio_path.stem, used_names)
            mapping[key] = run_dir_name
            changed = True
        used_names.add(run_dir_name)
        assignments[audio_path] = runs_root / run_dir_name
    return assignments, changed


def batch_error_message(exc: Exception) -> str:
    if isinstance(exc, (SrutiError, ValueError, FileNotFoundError)):
        return str(exc)
    return f"{exc.__class__.__name__}: {exc}"


def execute_batch_run(request: BatchRunRequest) -> tuple[list[tuple[RunWorkItem, str]], int]:
    if not request.runs_root.is_dir():
        raise ValueError(f"{request.runs_root} is not a directory.")
    pipeline_path = request.runs_root / "pipeline.toml"
    if not pipeline_path.is_file():
        raise ValueError(f"{pipeline_path} is required for run-batch.")
    if not request.in_dir.is_dir():
        raise ValueError(f"{request.in_dir} is not a directory.")

    audio_files = discover_audio_files(request.in_dir)
    if not audio_files:
        raise ValueError(f"No supported audio files found under {request.in_dir}.")

    scheduler_config = build_batch_scheduler_config(
        runs_root=request.runs_root,
        max_active_runs=request.max_active_runs,
        local_slots=request.local_slots,
        external_api_slots=request.external_api_slots,
        external_api_slots_per_run=request.external_api_slots_per_run,
    )
    if (
        request.on_exists is OnExistsMode.ASK
        and scheduler_config.effective_max_active_runs() > 1
    ):
        raise ValueError(
            "--on-exists ask is only supported with sequential batch execution. "
            "Use skip|overwrite|fail or set --max-active-runs 1."
        )

    mapping = load_batch_mapping(request.runs_root)
    assignments, mapping_changed = assign_batch_run_dirs(
        runs_root=request.runs_root,
        audio_files=audio_files,
        mapping=mapping,
    )
    if mapping_changed:
        save_batch_mapping(request.runs_root, mapping)

    total = len(audio_files)
    scheduler = BatchScheduler(
        config=scheduler_config,
        progress_emitter=request.progress_emitter or _noop_progress,
    )
    work_items = [
        RunWorkItem(
            audio_path=audio_path,
            run_dir=assignments[audio_path],
            run_index=index,
            total_runs=total,
        )
        for index, audio_path in enumerate(audio_files, start=1)
    ]

    def _worker(work_item: RunWorkItem, coordinator) -> None:
        execute_run(
            RunRequest(
                run_dir=work_item.run_dir,
                in_path=work_item.audio_path,
                source_stage=request.source_stage,
                target_stage=request.target_stage,
                seconds=request.seconds,
                model_path=request.model_path,
                on_exists=request.on_exists,
                dry_run=request.dry_run,
                force=request.force,
                verbose=request.verbose,
                llm_provider=request.llm_provider,
                cost_cap_usd=request.cost_cap_usd,
                token_cap_input=request.token_cap_input,
                token_cap_output=request.token_cap_output,
                command_label="run-batch",
                settings_dir=request.runs_root,
                progress_emitter=None,
                result_emitter=request.result_emitter,
                ask_user=request.ask_user,
                execution_coordinator=coordinator,
            )
        )

    return scheduler.run(work_items, worker=_worker), total
