from __future__ import annotations

from pathlib import Path

import typer

from sruti.application.context import StageContext
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


def _print_result(result: StageResult) -> None:
    if result.outputs:
        for output in result.outputs:
            typer.echo(f"  - {output}")


def _handle_failure(exc: Exception) -> None:
    if isinstance(exc, (SrutiError, ValueError, FileNotFoundError)):
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    raise exc


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


@app.command("run")
def run_pipeline(
    run_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    in_path: Path | None = typer.Option(None, "--in"),
    source_stage: StageId = typer.Option(StageId.S01, "--from"),
    target_stage: StageId = typer.Option(StageId.S09, "--to"),
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
    context.emit_progress(
        f"[run] starting: {source_stage.value}->{target_stage.value} in {run_dir}"
    )
    for stage_id in stage_ids_in_range(source_stage, target_stage):
        try:
            result = _run_single_stage(
                stage_id=stage_id,
                context=context,
                in_path=in_path,
                seconds=seconds,
                model_path=model_path,
            )
            _print_result(result)
        except Exception as exc:
            _handle_failure(exc)


@app.command("s01-normalize")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s02-chunk")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s03-asr")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s04-merge")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s05-asr-cleanup")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s06-remove-nonlecture")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s07-editorial")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s08-translate")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


@app.command("s09-translate-edit")
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
        _print_result(result)
    except Exception as exc:
        _handle_failure(exc)


if __name__ == "__main__":
    app()
