from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s03_asr_whisper_uc import S03AsrWhisperUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.json_codec import loads
from sruti.util import manifest as manifest_util
from sruti.util.io import atomic_write_json


class FakeWhisper:
    def transcribe_chunk(self, *, model_path: Path, chunk_path: Path, output_prefix: Path) -> list[str]:
        _ = (model_path, chunk_path)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_prefix.with_suffix(".txt").write_text("transcript", encoding="utf-8")
        output_prefix.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nline\n", encoding="utf-8")
        return ["whisper-cli", "-m", str(model_path), "-f", str(chunk_path)]


def _ctx(run_dir: Path, *, mode: OnExistsMode = OnExistsMode.OVERWRITE, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=mode,
        dry_run=dry_run,
        force=False,
        verbose=False,
    )


def test_s03_asr_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s03_asr_whisper_uc.require_executable", lambda _: None)
    monkeypatch.setattr(
        "sruti.application.stages.s03_asr_whisper_uc.executable_version",
        lambda _: "whisper test",
    )
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")
    s02_dir = manifest_util.stage_dir_for(tmp_path, "s02")
    chunks_dir = s02_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "0001.wav").write_bytes(b"chunk")
    atomic_write_json(
        s02_dir / "chunks.json",
        [{"id": 1, "filename": "0001.wav", "start_time": 0, "end_time": 30, "sha256": "x"}],
    )
    use_case = S03AsrWhisperUseCase(
        whisper_model_path=model_path,
        whisper=FakeWhisper(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    rows = loads((tmp_path / "s03_asr" / "transcripts_index.json").read_text(encoding="utf-8"))
    assert rows[0]["txt_filename"] == "0001.txt"
    assert rows[0]["start_time"] == 0


def test_s03_asr_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s03_asr_whisper_uc.require_executable", lambda _: None)
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")
    s02_dir = manifest_util.stage_dir_for(tmp_path, "s02")
    s02_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(s02_dir / "chunks.json", [])
    use_case = S03AsrWhisperUseCase(
        whisper_model_path=model_path,
        whisper=FakeWhisper(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path, dry_run=True))
    assert result.status == StageStatus.DRY_RUN
