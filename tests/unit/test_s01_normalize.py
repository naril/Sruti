from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s01_normalize_uc import S01NormalizeUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.infrastructure.fs_repository import FileSystemManifestStore


class FakeFfmpeg:
    def normalize(self, input_path: Path, output_path: Path) -> list[str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        return ["ffmpeg", "-i", str(input_path), str(output_path)]


def _ctx(run_dir: Path, *, mode: OnExistsMode = OnExistsMode.OVERWRITE, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=mode,
        dry_run=dry_run,
        force=False,
        verbose=False,
    )


def test_s01_normalize_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s01_normalize_uc.require_executable", lambda _: None)
    monkeypatch.setattr(
        "sruti.application.stages.s01_normalize_uc.executable_version", lambda _: "ffmpeg test"
    )
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"wave")
    use_case = S01NormalizeUseCase(
        input_audio=input_audio,
        ffmpeg=FakeFfmpeg(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s01_normalize" / "normalized.wav").exists()


def test_s01_normalize_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s01_normalize_uc.require_executable", lambda _: None)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"wave")
    use_case = S01NormalizeUseCase(
        input_audio=input_audio,
        ffmpeg=FakeFfmpeg(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path, dry_run=True))
    assert result.status == StageStatus.DRY_RUN
