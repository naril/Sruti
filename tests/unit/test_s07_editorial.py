from __future__ import annotations

import json
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s07_editorial_uc import S07EditorialUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.domain.models import LlmGenerateResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllama:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.prompts: list[str] = []

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
        _ = (model, temperature, timeout_seconds)
        self.generate_calls += 1
        self.prompts.append(prompt)
        return LlmGenerateResult(text="polished english")


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
        llm_client=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s07_editorial" / "final_publishable_en.txt").read_text(encoding="utf-8").strip() == "polished english"


def test_s07_editorial_empty_input_produces_empty_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s06_dir = manifest_util.stage_dir_for(tmp_path, "s06")
    s06_dir.mkdir(parents=True, exist_ok=True)
    (s06_dir / "content_only.txt").write_text(" \n\n\t", encoding="utf-8")
    ollama = FakeOllama()
    use_case = S07EditorialUseCase(
        llm_client=ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert ollama.generate_calls == 0
    assert (tmp_path / "s07_editorial" / "final_publishable_en.txt").read_text(encoding="utf-8") == ""


def test_s07_editorial_uses_prompt_templates_dir_from_pipeline_config(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    (tmp_path / "pipeline.toml").write_text(
        """
[sruti]
prompt_templates_dir = "prompts"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "s07_editorial.txt").write_text("CUSTOM STAGE PROMPT\n{{text}}\n", encoding="utf-8")

    s06_dir = manifest_util.stage_dir_for(tmp_path, "s06")
    s06_dir.mkdir(parents=True, exist_ok=True)
    (s06_dir / "content_only.txt").write_text("raw text", encoding="utf-8")
    ollama = FakeOllama()
    use_case = S07EditorialUseCase(
        llm_client=ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert ollama.generate_calls == 1
    assert ollama.prompts
    assert "CUSTOM STAGE PROMPT" in ollama.prompts[0]
    manifest_data = json.loads((tmp_path / "s07_editorial" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["params"]["prompt_templates_dir"] == str(tmp_path / "prompts")
