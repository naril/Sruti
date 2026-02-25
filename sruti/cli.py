from __future__ import annotations

from pathlib import Path

import typer

from sruti.application.context import StageContext
from sruti.domain.enums import OnExistsMode, StageId
from sruti.domain.models import StageResult
from sruti.domain.policies import stage_ids_in_range
from sruti.stages import s01_normalize, s02_chunk, s03_asr_whispercli

app = typer.Typer(no_args_is_help=True, help="sruti: local lecture pipeline")


def _not_implemented(stage: str) -> None:
    typer.secho(f"{stage} is not implemented yet.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


def _ask_user(prompt: str) -> bool:
    return typer.confirm(prompt, default=False)


def _stage_context(
    *,
    run_dir: Path,
    on_exists: OnExistsMode,
    dry_run: bool,
    force: bool,
    verbose: bool,
) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
    )


def _print_result(result: StageResult) -> None:
    typer.secho(f"[{result.stage.value}] {result.status.value}", fg=typer.colors.GREEN)
    if result.outputs:
        for output in result.outputs:
            typer.echo(f"  - {output}")


def _run_single_stage(
    *,
    stage_id: StageId,
    context: StageContext,
    in_path: Path | None,
    seconds: int,
    model_path: Path,
) -> StageResult:
    if stage_id is StageId.S01:
        if in_path is None:
            raise typer.BadParameter("--in is required when running s01.")
        return s01_normalize.run_stage(context=context, input_audio=in_path, ask_user=_ask_user)
    if stage_id is StageId.S02:
        return s02_chunk.run_stage(context=context, seconds=seconds, ask_user=_ask_user)
    if stage_id is StageId.S03:
        return s03_asr_whispercli.run_stage(
            context=context,
            model_path=model_path,
            ask_user=_ask_user,
        )
    _not_implemented(stage_id.value)
    raise RuntimeError("unreachable")


@app.command("run")
def run_pipeline(
    run_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True),
    in_path: Path | None = typer.Option(None, "--in"),
    source_stage: StageId = typer.Option(StageId.S01, "--from"),
    target_stage: StageId = typer.Option(StageId.S09, "--to"),
    seconds: int = typer.Option(30, "--seconds"),
    model_path: Path = typer.Option(Path("./models/ggml-large-v3.bin"), "--model-path"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
    )
    for stage_id in stage_ids_in_range(source_stage, target_stage):
        result = _run_single_stage(
            stage_id=stage_id,
            context=context,
            in_path=in_path,
            seconds=seconds,
            model_path=model_path,
        )
        _print_result(result)


@app.command("s01-normalize")
def run_s01_normalize(
    run_dir: Path = typer.Argument(...),
    in_path: Path = typer.Option(..., "--in"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
    )
    result = s01_normalize.run_stage(context=context, input_audio=in_path, ask_user=_ask_user)
    _print_result(result)


@app.command("s02-chunk")
def run_s02_chunk(
    run_dir: Path = typer.Argument(...),
    seconds: int = typer.Option(30, "--seconds"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
    )
    result = s02_chunk.run_stage(context=context, seconds=seconds, ask_user=_ask_user)
    _print_result(result)


@app.command("s03-asr")
def run_s03_asr(
    run_dir: Path = typer.Argument(...),
    model_path: Path = typer.Option(Path("./models/ggml-large-v3.bin"), "--model-path"),
    on_exists: OnExistsMode = typer.Option(OnExistsMode.ASK, "--on-exists"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    context = _stage_context(
        run_dir=run_dir,
        on_exists=on_exists,
        dry_run=dry_run,
        force=force,
        verbose=verbose,
    )
    result = s03_asr_whispercli.run_stage(
        context=context,
        model_path=model_path,
        ask_user=_ask_user,
    )
    _print_result(result)


@app.command("s04-merge")
def s04_merge(_: Path = typer.Argument(...)) -> None:
    _not_implemented("s04")


@app.command("s05-asr-cleanup")
def s05_asr_cleanup(_: Path = typer.Argument(...)) -> None:
    _not_implemented("s05")


@app.command("s06-remove-nonlecture")
def s06_remove_nonlecture(_: Path = typer.Argument(...)) -> None:
    _not_implemented("s06")


@app.command("s07-editorial")
def s07_editorial(_: Path = typer.Argument(...)) -> None:
    _not_implemented("s07")


@app.command("s08-translate")
def s08_translate(_: Path = typer.Argument(...)) -> None:
    _not_implemented("s08")


@app.command("s09-translate-edit")
def s09_translate_edit(_: Path = typer.Argument(...)) -> None:
    _not_implemented("s09")


if __name__ == "__main__":
    app()
