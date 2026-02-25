from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s09_translate_edit_uc import S09TranslateEditUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.domain.models import LlmGenerateResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllama:
    def ensure_model_available(self, model: str) -> None:
        _ = model

    def provider_name(self) -> str:
        return "local"

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> LlmGenerateResult:
        _ = (model, prompt, temperature, timeout_seconds)
        return LlmGenerateResult(text="finalni cs")


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_s09_translate_edit_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s08_dir = manifest_util.stage_dir_for(tmp_path, "s08")
    s08_dir.mkdir(parents=True, exist_ok=True)
    (s08_dir / "translated_faithful_cs.txt").write_text("preklad", encoding="utf-8")
    use_case = S09TranslateEditUseCase(
        llm_client=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s09_translate_edit" / "final_publishable_cs.txt").read_text(encoding="utf-8").strip() == "finalni cs"
