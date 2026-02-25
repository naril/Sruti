from __future__ import annotations

from pathlib import Path

import pytest

from sruti.application.context import StageContext
from sruti.application.stages.s06_remove_nonlecture_uc import S06RemoveNonLectureUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.domain.errors import InvalidLlmJsonError
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllamaRetrySuccess:
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return '[{"span_id": 1, "action": "REMOVE", "label": "AUDIENCE", "reason": "chatter"}]'


class FakeOllamaAlwaysBad:
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
        return "bad"


class FakeOllamaLowercaseAction:
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
        return '[{"span_id": 1, "action": "remove", "label": "AUDIENCE"}]'


class FakeOllamaMustNotCall:
    def ensure_model_available(self, model: str) -> None:
        raise AssertionError("ensure_model_available should not be called for empty input")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> str:
        _ = (model, prompt, temperature, timeout_seconds)
        raise AssertionError("generate should not be called for empty input")


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_s06_remove_nonlecture_with_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        ollama=FakeOllamaRetrySuccess(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert "Lecture starts" in (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(
        encoding="utf-8"
    )


def test_s06_remove_nonlecture_fails_after_retries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("only text", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        ollama=FakeOllamaAlwaysBad(),
        manifest_store=FileSystemManifestStore(),
    )
    with pytest.raises(InvalidLlmJsonError):
        use_case.run(_ctx(tmp_path))


def test_s06_remove_nonlecture_normalizes_action_case(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        ollama=FakeOllamaLowercaseAction(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    final_text = (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8")
    assert "Audience question" not in final_text


def test_s06_remove_nonlecture_empty_input_short_circuits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text(" \n\n\t", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        ollama=FakeOllamaMustNotCall(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8") == ""
