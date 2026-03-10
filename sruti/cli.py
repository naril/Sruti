from __future__ import annotations

from pathlib import Path

import typer

from sruti.application.project_service import ProjectInitializer
from sruti.application.run_service import (
    BatchRunRequest,
    RunRequest,
    build_stage_context,
    execute_batch_run,
    execute_run,
    run_single_stage,
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
from sruti.config import load_settings
from sruti.domain.enums import LlmProvider, OnExistsMode, ProjectType, StageId
from sruti.domain.errors import SrutiError
from sruti.domain.models import StageResult

app = typer.Typer(no_args_is_help=True, help="sruti: local lecture pipeline")


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
    execution_coordinator=None,
):
    return build_stage_context(
        run_dir=run_dir,
        settings_dir=settings_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
        llm_provider=llm_provider,
        cost_cap_usd=cost_cap_usd,
        token_cap_input=token_cap_input,
        token_cap_output=token_cap_output,
        progress_emitter=_emit_progress,
        execution_coordinator=execution_coordinator,
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


@app.command("init", help="Create RUN_DIR and prefill pipeline.toml with default settings.")
def init_run_dir(
    run_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
) -> None:
    try:
        ProjectInitializer().create_project(project_dir=run_dir, project_type=ProjectType.SINGLE)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.secho(f"initialized {run_dir}", fg=typer.colors.GREEN)
    typer.echo(f"  - {run_dir / 'pipeline.toml'}")


@app.command("run", help="Run a stage range (s01-s10) in order.")
def run_pipeline(
    run_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    in_path: Path | None = typer.Option(None, "--in"),
    source_stage: StageId = typer.Option(
        StageId.S01,
        "--from",
        help="Start stage (inclusive), e.g. s01 for full pipeline or s05 to resume mid-run.",
    ),
    target_stage: StageId = typer.Option(
        StageId.S10,
        "--to",
        help="End stage (inclusive), e.g. s10 for full pipeline or s07 to stop earlier.",
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
    if source_stage is StageId.S01 and in_path is None:
        raise typer.BadParameter("--in is required when running s01.")
    try:
        execute_run(
            RunRequest(
                run_dir=run_dir,
                in_path=in_path,
                source_stage=source_stage,
                target_stage=target_stage,
                seconds=seconds,
                model_path=model_path,
                on_exists=on_exists,
                dry_run=dry_run,
                force=force,
                verbose=verbose,
                llm_provider=llm_provider,
                cost_cap_usd=cost_cap_usd,
                token_cap_input=token_cap_input,
                token_cap_output=token_cap_output,
                command_label="run",
                progress_emitter=_emit_progress,
                result_emitter=lambda result: _print_result(result, include_status=False),
                ask_user=_ask_user,
            )
        )
    except Exception as exc:
        _handle_failure(exc)


@app.command(
    "run-batch",
    help="Run a stage range (s01-s10) over all audio files in INPUT_DIR recursively.",
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
        StageId.S10,
        "--to",
        help="End stage (inclusive), e.g. s10 for full pipeline or s07 to stop earlier.",
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
    max_active_runs: int | None = typer.Option(
        None,
        "--max-active-runs",
        help="Max concurrent per-file runs. 0 in config means auto (= local_slots + external_api_slots).",
    ),
    local_slots: int | None = typer.Option(
        None,
        "--local-slots",
        help="Max concurrent local-heavy stages (ffmpeg, whisper, local Ollama).",
    ),
    external_api_slots: int | None = typer.Option(
        None,
        "--external-api-slots",
        help="Global cap for concurrent external API calls across all runs.",
    ),
    external_api_slots_per_run: int | None = typer.Option(
        None,
        "--external-api-slots-per-run",
        help="Per-run cap for concurrent external API calls.",
    ),
) -> None:
    try:
        failure_rows, total = execute_batch_run(
            BatchRunRequest(
                runs_root=runs_root,
                in_dir=in_dir,
                source_stage=source_stage,
                target_stage=target_stage,
                seconds=seconds,
                model_path=model_path,
                on_exists=on_exists,
                dry_run=dry_run,
                force=force,
                verbose=verbose,
                llm_provider=llm_provider,
                cost_cap_usd=cost_cap_usd,
                token_cap_input=token_cap_input,
                token_cap_output=token_cap_output,
                max_active_runs=max_active_runs,
                local_slots=local_slots,
                external_api_slots=external_api_slots,
                external_api_slots_per_run=external_api_slots_per_run,
                progress_emitter=_emit_progress,
                result_emitter=lambda result: _print_result(result, include_status=False),
                ask_user=_ask_user,
            )
        )
        failures = [(work_item.audio_path, message) for work_item, message in failure_rows]
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
    except typer.BadParameter:
        raise
    except Exception as exc:
        if str(exc).startswith("--on-exists ask is only supported"):
            raise typer.BadParameter(str(exc)) from exc
        if str(exc).endswith("is not a directory.") or "is required for run-batch." in str(exc):
            raise typer.BadParameter(str(exc)) from exc
        _handle_failure(exc)


@app.command("gui", help="Start the local sruti web GUI.")
def run_gui(
    workspace: Path = typer.Option(Path("./runs"), "--workspace", file_okay=False, dir_okay=True),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8420, "--port"),
) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise typer.BadParameter("uvicorn is not installed. Install GUI dependencies first.") from exc

    from sruti.gui.app import create_app

    app_instance = create_app(workspace_root=workspace.resolve())
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


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


@app.command("s08-condense", help="s08: Lightly condense English text into structured blocks.")
def run_s08_condense(
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
        result = s08_condense.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s09-translate", help="s09: Faithful English-to-Czech translation.")
def run_s09_translate(
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
        result = s09_translate_faithful.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s10-translate-edit", help="s10: Editorial polish of Czech translation.")
def run_s10_translate_edit(
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
        result = s10_translate_edit.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s08-translate", help="DEPRECATED alias for s09-translate.")
def run_s08_translate_alias(
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
    typer.secho(
        "[deprecated] 's08-translate' was moved to 's09-translate'.",
        fg=typer.colors.YELLOW,
        err=True,
    )
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
        result = s09_translate_faithful.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s09-translate-edit", help="DEPRECATED alias for s10-translate-edit.")
def run_s09_translate_edit_alias(
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
    typer.secho(
        "[deprecated] 's09-translate-edit' was moved to 's10-translate-edit'.",
        fg=typer.colors.YELLOW,
        err=True,
    )
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
        result = s10_translate_edit.run_stage(context=context, ask_user=_ask_user)
        _print_result(result, include_status=False)
    except Exception as exc:
        _handle_failure(exc)


if __name__ == "__main__":
    app()
