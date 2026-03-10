from __future__ import annotations

import json
from pathlib import Path

from sruti.application.run_service import RunRequest, execute_run
from sruti.domain.enums import OnExistsMode, StageId, StageStatus
from sruti.domain.models import StageResult


def test_execute_run_writes_run_state_and_events(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_s01_run_stage(**kwargs):
        calls.append("s01")
        context = kwargs["context"]
        stage_dir = context.run_dir / "s01_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        context.emit_progress("[s01] fake normalize")
        return StageResult(stage=StageId.S01, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    def fake_s02_run_stage(**kwargs):
        calls.append("s02")
        context = kwargs["context"]
        stage_dir = context.run_dir / "s02_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        context.emit_progress("[s02] fake chunk")
        return StageResult(stage=StageId.S02, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.application.run_service.s01_normalize.run_stage", fake_s01_run_stage)
    monkeypatch.setattr("sruti.application.run_service.s02_chunk.run_stage", fake_s02_run_stage)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    run_dir = tmp_path / "run"
    execute_run(
        RunRequest(
            run_dir=run_dir,
            in_path=input_audio,
            source_stage=StageId.S01,
            target_stage=StageId.S02,
            seconds=None,
            model_path=None,
            on_exists=OnExistsMode.OVERWRITE,
            dry_run=False,
            force=False,
            verbose=False,
            llm_provider=None,
            cost_cap_usd=None,
            token_cap_input=None,
            token_cap_output=None,
        )
    )

    assert calls == ["s01", "s02"]
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["stages"] == {"s01": "success", "s02": "success"}
    event_rows = [
        json.loads(line)
        for line in (run_dir / "run_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert event_rows[0]["event"] == "run_started"
    assert any(row["event"] == "progress" and row["message"] == "[s02] fake chunk" for row in event_rows)

