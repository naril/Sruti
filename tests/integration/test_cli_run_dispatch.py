from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sruti.cli import app
from sruti.domain.enums import StageId, StageStatus
from sruti.domain.models import StageResult


def test_cli_run_dispatches_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_stage(stage_id: StageId):
        def _run_stage(**kwargs):
            calls.append(stage_id.value)
            context = kwargs["context"]
            stage_dir = context.run_dir / f"{stage_id.value}_fake"
            stage_dir.mkdir(parents=True, exist_ok=True)
            return StageResult(stage=stage_id, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

        return _run_stage

    monkeypatch.setattr("sruti.cli.s01_normalize.run_stage", fake_stage(StageId.S01))
    monkeypatch.setattr("sruti.cli.s02_chunk.run_stage", fake_stage(StageId.S02))
    monkeypatch.setattr("sruti.cli.s03_asr_whispercli.run_stage", fake_stage(StageId.S03))

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    run_dir = tmp_path / "run1"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(run_dir),
            "--in",
            str(input_audio),
            "--from",
            "s01",
            "--to",
            "s03",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls == ["s01", "s02", "s03"]
