from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s07_editorial_uc import S07EditorialUseCase
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
        return "polished english"


def _ctx(run_dir: Path, *, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=dry_run,
        force=False,
        verbose=False,
    )


def test_s07_editorial_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s06_dir = manifest_util.stage_dir_for(tmp_path, "s06")
    s06_dir.mkdir(parents=True, exist_ok=True)
    (s06_dir / "content_only.txt").write_text("raw text", encoding="utf-8")
    use_case = S07EditorialUseCase(
        ollama=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s07_editorial" / "final_publishable_en.txt").read_text(encoding="utf-8").strip() == "polished english"
