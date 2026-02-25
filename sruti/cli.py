from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from sruti.application.context import StageContext
from sruti.config import render_default_pipeline_toml
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
    s08_translate_faithful,
    s09_translate_edit,
)
from sruti.util.io import atomic_write_json

app = typer.Typer(no_args_is_help=True, help="sruti: local lecture pipeline")

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


def _not_implemented(stage: str) -> None:
    typer.secho(f"{stage} is not implemented yet.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


def _ask_user(prompt: str) -> bool:
    return typer.confirm(prompt, default=False)


def _emit_progress(message: str) -> None:
    typer.echo(message)


def _stage_context(
    *,
    run_dir: Path,
    settings_dir: Path | None = None,
    on_exists: OnExistsMode,
    dry_run: bool,
    force: bool,
    verbose: bool,
    llm_provider: LlmProvider | None,
    cost_cap_usd: float | None,
    token_cap_input: int | None,
    token_cap_output: int | None,
) -> StageContext:
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
        progress_emitter=_emit_progress,
    )


def _print_result(result: StageResult, *, include_status: bool = True) -> None:
    if include_status:
        typer.secho(f"[{result.stage.value}] {result.status.value}", fg=typer.colors.GREEN)
    if result.outputs:
        for output in result.outputs:
            typer.echo(f"  - {output}")


def _handle_failure(exc: Exception) -> None:
    if isinstance(exc, (SrutiError, ValueError, FileNotFoundError)):
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    raise exc


def _run_stage_range(
    *,
    context: StageContext,
    source_stage: StageId,
    target_stage: StageId,
    in_path: Path | None,
    seconds: int | None,
    model_path: Path | None,
    command_label: str,
) -> None:
    context.emit_progress(
        f"[{command_label}] starting: {source_stage.value}->{target_stage.value} in {context.run_dir}"
    )
    for stage_id in stage_ids_in_range(source_stage, target_stage):
        result = _run_single_stage(
            stage_id=stage_id,
            context=context,
            in_path=in_path,
            seconds=seconds,
            model_path=model_path,
        )
        _print_result(result, include_status=False)


def _discover_audio_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path.relative_to(input_dir)))


def _load_batch_mapping(runs_root: Path) -> dict[str, str]:
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


def _save_batch_mapping(runs_root: Path, mapping: dict[str, str]) -> None:
    manifest_path = runs_root / BATCH_MANIFEST_FILENAME
    payload = {"audio_to_run_dir": dict(sorted(mapping.items()))}
    atomic_write_json(manifest_path, payload)


def _sanitize_run_dir_name(stem: str) -> str:
    normalized = RUN_DIR_NAME_SANITIZER.sub("-", stem).strip("._-").lower()
    return normalized or "audio"


def _next_run_dir_name(base_stem: str, used_names: set[str]) -> str:
    candidate = _sanitize_run_dir_name(base_stem)
    if candidate not in used_names:
        return candidate
    suffix = 2
    while True:
        suffixed = f"{candidate}-{suffix}"
        if suffixed not in used_names:
            return suffixed
        suffix += 1


def _assign_batch_run_dirs(
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
            run_dir_name = _next_run_dir_name(audio_path.stem, used_names)
            mapping[key] = run_dir_name
            changed = True
        used_names.add(run_dir_name)
        assignments[audio_path] = runs_root / run_dir_name
    return assignments, changed


def _batch_error_message(exc: Exception) -> str:
    if isinstance(exc, (SrutiError, ValueError, FileNotFoundError, typer.BadParameter)):
        return str(exc)
    return f"{exc.__class__.__name__}: {exc}"


def _run_single_stage(
    *,
    stage_id: StageId,
    context: StageContext,
    in_path: Path | None,
    seconds: int | None,
    model_path: Path | None,
) -> StageResult:
    if stage_id is StageId.S01:
        if in_path is None:
            raise typer.BadParameter("--in is required when running s01.")
        return s01_normalize.run_stage(context=context, input_audio=in_path, ask_user=_ask_user)
    if stage_id is StageId.S02:
        effective_seconds = seconds if seconds is not None else context.settings.chunk_seconds
        return s02_chunk.run_stage(context=context, seconds=effective_seconds, ask_user=_ask_user)
    if stage_id is StageId.S03:
        effective_model_path = (
            model_path
            if model_path is not None
            else context.settings.default_whisper_model_path
        )
        return s03_asr_whispercli.run_stage(
            context=context,
            model_path=effective_model_path,
            ask_user=_ask_user,
        )
    if stage_id is StageId.S04:
        return s04_merge.run_stage(context=context, ask_user=_ask_user)
    if stage_id is StageId.S05:
        return s05_asr_cleanup.run_stage(context=context, ask_user=_ask_user)
    if stage_id is StageId.S06:
        return s06_remove_nonlecture.run_stage(context=context, ask_user=_ask_user)
    if stage_id is StageId.S07:
        return s07_editorial.run_stage(context=context, ask_user=_ask_user)
    if stage_id is StageId.S08:
        return s08_translate_faithful.run_stage(context=context, ask_user=_ask_user)
    if stage_id is StageId.S09:
        return s09_translate_edit.run_stage(context=context, ask_user=_ask_user)
    raise RuntimeError("unreachable")


@app.command("init", help="Create RUN_DIR and prefill pipeline.toml with default settings.")
def init_run_dir(
    run_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "pipeline.toml"
    if config_path.exists():
        raise typer.BadParameter(f"{config_path} already exists.")
    config_path.write_text(render_default_pipeline_toml(), encoding="utf-8")
    typer.secho(f"initialized {run_dir}", fg=typer.colors.GREEN)
    typer.echo(f"  - {config_path}")


@app.command("run", help="Run a stage range (s01-s09) in order.")
def run_pipeline(
    run_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    in_path: Path | None = typer.Option(None, "--in"),
    source_stage: StageId = typer.Option(
        StageId.S01,
        "--from",
        help="Start stage (inclusive), e.g. s01 for full pipeline or s05 to resume mid-run.",
    ),
    target_stage: StageId = typer.Option(
        StageId.S09,
        "--to",
        help="End stage (inclusive), e.g. s09 for full pipeline or s07 to stop earlier.",
    ),
    seconds: int | None = typer.Option(None, "--seconds"),
    model_path: Path | None = typer.Option(None, "--model-path"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        _run_stage_range(
            context=context,
            source_stage=source_stage,
            target_stage=target_stage,
            in_path=in_path,
            seconds=seconds,
            model_path=model_path,
            command_label="run",
        )
    except Exception as exc:
        _handle_failure(exc)


@app.command(
    "run-batch",
    help="Run a stage range (s01-s09) over all audio files in INPUT_DIR recursively.",
)
def run_batch(
    runs_root: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    in_dir: Path = typer.Option(..., "--in-dir", file_okay=False, dir_okay=True),
    source_stage: StageId = typer.Option(
        StageId.S01,
        "--from",
        help="Start stage (inclusive), e.g. s01 for full pipeline or s05 to resume mid-run.",
    ),
    target_stage: StageId = typer.Option(
        StageId.S09,
        "--to",
        help="End stage (inclusive), e.g. s09 for full pipeline or s07 to stop earlier.",
    ),
    seconds: int | None = typer.Option(None, "--seconds"),
    model_path: Path | None = typer.Option(None, "--model-path"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    if not runs_root.is_dir():
        raise typer.BadParameter(f"{runs_root} is not a directory.")
    pipeline_path = runs_root / "pipeline.toml"
    if not pipeline_path.is_file():
        raise typer.BadParameter(f"{pipeline_path} is required for run-batch.")
    if not in_dir.is_dir():
        raise typer.BadParameter(f"{in_dir} is not a directory.")

    try:
        audio_files = _discover_audio_files(in_dir)
        if not audio_files:
            raise ValueError(f"No supported audio files found under {in_dir}.")

        mapping = _load_batch_mapping(runs_root)
        assignments, mapping_changed = _assign_batch_run_dirs(
            runs_root=runs_root,
            audio_files=audio_files,
            mapping=mapping,
        )
        if mapping_changed:
            _save_batch_mapping(runs_root, mapping)

        failures: list[tuple[Path, str]] = []
        total = len(audio_files)
        for index, audio_path in enumerate(audio_files, start=1):
            run_dir = assignments[audio_path]
            typer.secho(f"[run-batch] file {index}/{total}: {audio_path}", fg=typer.colors.BLUE)
            context = _stage_context(
                run_dir=run_dir,
                settings_dir=runs_root,
                on_exists=on_exists,
                dry_run=dry_run,
                force=force,
                verbose=verbose,
                llm_provider=llm_provider,
                cost_cap_usd=cost_cap_usd,
                token_cap_input=token_cap_input,
                token_cap_output=token_cap_output,
            )
            try:
                _run_stage_range(
                    context=context,
                    source_stage=source_stage,
                    target_stage=target_stage,
                    in_path=audio_path,
                    seconds=seconds,
                    model_path=model_path,
                    command_label="run-batch",
                )
            except Exception as exc:
                message = _batch_error_message(exc)
                failures.append((audio_path, message))
                typer.secho(
                    f"[run-batch] failed for {audio_path}: {message}",
                    fg=typer.colors.RED,
                    err=True,
                )

        succeeded = total - len(failures)
        color = typer.colors.GREEN if not failures else typer.colors.RED
        typer.secho(
            f"[run-batch] summary: {succeeded} succeeded, {len(failures)} failed (total {total}).",
            fg=color,
        )
        if failures:
            for failed_path, message in failures:
                typer.secho(
                    f"  - {failed_path}: {message}",
                    fg=typer.colors.RED,
                    err=True,
                )
            raise typer.Exit(code=1)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s01-normalize", help="s01: Normalize input audio to deterministic WAV.")
def run_s01_normalize(
    run_dir: Path = typer.Argument(...),
    in_path: Path = typer.Option(..., "--in"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s01_normalize.run_stage(context=context, input_audio=in_path, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s02-chunk", help="s02: Split normalized audio into fixed-length chunks.")
def run_s02_chunk(
    run_dir: Path = typer.Argument(...),
    seconds: int | None = typer.Option(None, "--seconds"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        effective_seconds = seconds if seconds is not None else context.settings.chunk_seconds
        result = s02_chunk.run_stage(context=context, seconds=effective_seconds, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s03-asr", help="s03: Transcribe audio chunks with whisper-cli.")
def run_s03_asr(
    run_dir: Path = typer.Argument(...),
    model_path: Path | None = typer.Option(None, "--model-path"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        effective_model_path = (
            model_path
            if model_path is not None
            else context.settings.default_whisper_model_path
        )
        result = s03_asr_whispercli.run_stage(
            context=context,
            model_path=effective_model_path,
            ask_user=_ask_user,
        )
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s04-merge", help="s04: Merge per-chunk transcripts into one text/SRT.")
def run_s04_merge(
    run_dir: Path = typer.Argument(...),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s04_merge.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s05-asr-cleanup", help="s05: LLM cleanup of ASR transcript errors.")
def run_s05_asr_cleanup(
    run_dir: Path = typer.Argument(...),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s05_asr_cleanup.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s06-remove-nonlecture", help="s06: Remove non-lecture content from cleaned text.")
def run_s06_remove_nonlecture(
    run_dir: Path = typer.Argument(...),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s06_remove_nonlecture.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s07-editorial", help="s07: Editorially refine English text for publishing.")
def run_s07_editorial(
    run_dir: Path = typer.Argument(...),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s07_editorial.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s08-translate", help="s08: Faithful English-to-Czech translation.")
def run_s08_translate(
    run_dir: Path = typer.Argument(...),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s08_translate_faithful.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s09-translate-edit", help="s09: Editorial polish of Czech translation.")
def run_s09_translate_edit(
    run_dir: Path = typer.Argument(...),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
    llm_provider: LlmProvider | None = typer.Option(None, "--llm-provider"),
    cost_cap_usd: float | None = typer.Option(None, "--cost-cap-usd"),
    token_cap_input: int | None = typer.Option(None, "--token-cap-input"),
    token_cap_output: int | None = typer.Option(None, "--token-cap-output"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
    )
    try:
        result = s09_translate_edit.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


if __name__ == "__main__":
    app()
