from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s05_asr_cleanup_uc import S05AsrCleanupUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllama:
    def ensure_model_available(self, model: str) -> None:
        _ = model

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> str:
        _ = (model, prompt, temperature, timeout_seconds)
        return "cleaned"


def _ctx(run_dir: Path, *, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=dry_run,
        force=False,
        verbose=False,
    )


def test_s05_cleanup_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s05_asr_cleanup_uc.require_executable", lambda _: None)
    s04_dir = manifest_util.stage_dir_for(tmp_path, "s04")
    s04_dir.mkdir(parents=True, exist_ok=True)
    (s04_dir / "merged_raw.txt").write_text("hello world", encoding="utf-8")
    use_case = S05AsrCleanupUseCase(
        ollama=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s05_asr_cleanup" / "cleaned_1.txt").read_text(encoding="utf-8").strip() == "cleaned"


def test_s05_cleanup_dry_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s05_asr_cleanup_uc.require_executable", lambda _: None)
    s04_dir = manifest_util.stage_dir_for(tmp_path, "s04")
    s04_dir.mkdir(parents=True, exist_ok=True)
    (s04_dir / "merged_raw.txt").write_text("hello world", encoding="utf-8")
    use_case = S05AsrCleanupUseCase(
        ollama=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path, dry_run=True))
    assert result.status == StageStatus.DRY_RUN
