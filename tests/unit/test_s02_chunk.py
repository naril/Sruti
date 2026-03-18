from __future__ import annotations

from pathlib import Path
import wave

import pytest

from sruti.application.context import StageContext
from sruti.application.stages.s02_chunk_uc import S02ChunkUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.json_codec import loads
from sruti.util import manifest as manifest_util


class FakeFfmpeg:
    def segment(self, input_path: Path, output_pattern: Path, *, seconds: int) -> list[str]:
        _ = (input_path, seconds)
        output_pattern.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(output_pattern.parent / "0001.wav", seconds=30.0)
        self._write_wav(output_pattern.parent / "0002.wav", seconds=12.5)
        return ["ffmpeg", "segment", str(output_pattern)]

    def _write_wav(self, path: Path, *, seconds: float) -> None:
        sample_rate = 16000
        total_frames = int(seconds * sample_rate)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * total_frames)


def _ctx(run_dir: Path, *, mode: OnExistsMode = OnExistsMode.OVERWRITE, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=mode,
        dry_run=dry_run,
        force=False,
        verbose=False,
    )


def test_s02_chunk_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s02_chunk_uc.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages.s02_chunk_uc.executable_version", lambda _: "ffmpeg test")
    s01_dir = manifest_util.stage_dir_for(tmp_path, "s01")
    s01_dir.mkdir(parents=True, exist_ok=True)
    (s01_dir / "normalized.wav").write_bytes(b"normalized")
    use_case = S02ChunkUseCase(
        seconds=30,
        ffmpeg=FakeFfmpeg(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    rows = loads((tmp_path / "s02_chunk" / "chunks.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["filename"] == "0001.wav"
    assert rows[0]["start_time"] == pytest.approx(0.0)
    assert rows[0]["end_time"] == pytest.approx(30.0)
    assert rows[1]["start_time"] == pytest.approx(30.0)
    assert rows[1]["end_time"] == pytest.approx(42.5)


def test_s02_chunk_skip_when_exists(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s02_chunk_uc.require_executable", lambda _: None)
    s01_dir = manifest_util.stage_dir_for(tmp_path, "s01")
    s01_dir.mkdir(parents=True, exist_ok=True)
    (s01_dir / "normalized.wav").write_bytes(b"normalized")
    s02_dir = manifest_util.stage_dir_for(tmp_path, "s02")
    s02_dir.mkdir(parents=True, exist_ok=True)
    (s02_dir / "chunks.json").write_text("[]", encoding="utf-8")
    use_case = S02ChunkUseCase(
        seconds=30,
        ffmpeg=FakeFfmpeg(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path, mode=OnExistsMode.SKIP))
    assert result.status == StageStatus.SKIPPED


def test_s02_chunk_skip_when_chunk_directory_exists(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s02_chunk_uc.require_executable", lambda _: None)
    s01_dir = manifest_util.stage_dir_for(tmp_path, "s01")
    s01_dir.mkdir(parents=True, exist_ok=True)
    (s01_dir / "normalized.wav").write_bytes(b"normalized")
    chunks_dir = manifest_util.stage_dir_for(tmp_path, "s02") / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "0001.wav").write_bytes(b"partial")
    use_case = S02ChunkUseCase(
        seconds=30,
        ffmpeg=FakeFfmpeg(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path, mode=OnExistsMode.SKIP))
    assert result.status == StageStatus.SKIPPED
