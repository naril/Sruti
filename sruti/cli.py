from __future__ import annotations

from pathlib import Path

import typer

from sruti.config import Settings

app = typer.Typer(no_args_is_help=True, help="sruti: local lecture pipeline")
settings = Settings()


def _not_implemented(stage: str) -> None:
    typer.secho(f"{stage} is scaffolded but not implemented yet.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


@app.command("run")
def run_pipeline(
    run_dir: Path = typer.Argument(..., exists=False, file_okay=False, dir_okay=True),
    source_stage: str = typer.Option("s01", "--from"),
    target_stage: str = typer.Option("s09", "--to"),
) -> None:
    _ = (run_dir, source_stage, target_stage)
    _not_implemented("run")


@app.command("s01-normalize")
def s01_normalize(
    run_dir: Path = typer.Argument(...),
    in_path: Path = typer.Option(..., "--in"),
) -> None:
    _ = (run_dir, in_path)
    _not_implemented("s01")


@app.command("s02-chunk")
def s02_chunk(
    run_dir: Path = typer.Argument(...),
    seconds: int = typer.Option(settings.chunk_seconds, "--seconds"),
) -> None:
    _ = (run_dir, seconds)
    _not_implemented("s02")


@app.command("s03-asr")
def s03_asr(
    run_dir: Path = typer.Argument(...),
    model_path: Path = typer.Option(settings.default_whisper_model_path, "--model-path"),
) -> None:
    _ = (run_dir, model_path)
    _not_implemented("s03")


@app.command("s04-merge")
def s04_merge(run_dir: Path = typer.Argument(...)) -> None:
    _ = run_dir
    _not_implemented("s04")


@app.command("s05-asr-cleanup")
def s05_asr_cleanup(run_dir: Path = typer.Argument(...)) -> None:
    _ = run_dir
    _not_implemented("s05")


@app.command("s06-remove-nonlecture")
def s06_remove_nonlecture(run_dir: Path = typer.Argument(...)) -> None:
    _ = run_dir
    _not_implemented("s06")


@app.command("s07-editorial")
def s07_editorial(run_dir: Path = typer.Argument(...)) -> None:
    _ = run_dir
    _not_implemented("s07")


@app.command("s08-translate")
def s08_translate(run_dir: Path = typer.Argument(...)) -> None:
    _ = run_dir
    _not_implemented("s08")


@app.command("s09-translate-edit")
def s09_translate_edit(run_dir: Path = typer.Argument(...)) -> None:
    _ = run_dir
    _not_implemented("s09")


if __name__ == "__main__":
    app()
