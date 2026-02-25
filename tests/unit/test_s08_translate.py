from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s08_translate_faithful_uc import S08TranslateFaithfulUseCase
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
        return "verny preklad"


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_s08_translate_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    (s07_dir / "final_publishable_en.txt").write_text("final en", encoding="utf-8")
    use_case = S08TranslateFaithfulUseCase(
        ollama=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s08_translate" / "translated_faithful_cs.txt").read_text(encoding="utf-8").strip() == "verny preklad"
