from __future__ import annotations

import json
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s09_translate_faithful_uc import S09TranslateFaithfulUseCase
from sruti.domain.enums import OnExistsMode, StageId, StageStatus
from sruti.domain.models import LlmGenerateResult, StageManifest
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllama:
    def __init__(self) -> None:
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
        self.prompts.append(prompt)
        return LlmGenerateResult(text="verny preklad")


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_s09_translate_prefers_current_s08_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    s07_path = s07_dir / "final_publishable_en.txt"
    s07_path.write_text("final en", encoding="utf-8")
    s08_dir = manifest_util.stage_dir_for(tmp_path, "s08")
    s08_dir.mkdir(parents=True, exist_ok=True)
    (s08_dir / "condensed_blocks_en.txt").write_text("condensed en", encoding="utf-8")
    manifest_util.save_stage_manifest(
        s08_dir,
        StageManifest(
            stage=StageId.S08,
            status=StageStatus.SUCCESS,
            params={"_inputs_signature": manifest_util.inputs_signature([s07_path])},
        ),
    )
    ollama = FakeOllama()
    use_case = S09TranslateFaithfulUseCase(
        llm_client_factory=lambda: ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s09_translate" / "translated_faithful_cs.txt").read_text(encoding="utf-8").strip() == "verny preklad"
    assert any("condensed en" in prompt for prompt in ollama.prompts)
    manifest_data = json.loads((tmp_path / "s09_translate" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["params"]["input_source_stage"] == "s08"


def test_s09_translate_falls_back_to_s07_when_s08_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    (s07_dir / "final_publishable_en.txt").write_text("final en", encoding="utf-8")
    ollama = FakeOllama()
    use_case = S09TranslateFaithfulUseCase(
        llm_client_factory=lambda: ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert any("final en" in prompt for prompt in ollama.prompts)
    manifest_data = json.loads((tmp_path / "s09_translate" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["params"]["input_source_stage"] == "s07"


def test_s09_translate_falls_back_to_s07_when_s08_is_stale(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    (s07_dir / "final_publishable_en.txt").write_text("new final en", encoding="utf-8")
    s08_dir = manifest_util.stage_dir_for(tmp_path, "s08")
    s08_dir.mkdir(parents=True, exist_ok=True)
    (s08_dir / "condensed_blocks_en.txt").write_text("stale condensed", encoding="utf-8")
    manifest_util.save_stage_manifest(
        s08_dir,
        StageManifest(
            stage=StageId.S08,
            status=StageStatus.SUCCESS,
            params={"_inputs_signature": "different-signature"},
        ),
    )
    ollama = FakeOllama()
    use_case = S09TranslateFaithfulUseCase(
        llm_client_factory=lambda: ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert any("new final en" in prompt for prompt in ollama.prompts)
    manifest_data = json.loads((tmp_path / "s09_translate" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["params"]["input_source_stage"] == "s07"
