from __future__ import annotations

import json
from pathlib import Path

import pytest

from sruti.application.context import StageContext
from sruti.application.stages.s06_remove_nonlecture_uc import S06RemoveNonLectureUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.domain.errors import InvalidLlmJsonError
from sruti.domain.models import LlmGenerateResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllamaRetrySuccess:
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        if self.calls == 1:
            return LlmGenerateResult(text="not json")
        return LlmGenerateResult(
            text='[{"span_id": 1, "action": "REMOVE", "label": "AUDIENCE", "reason": "chatter"}]'
        )


class FakeOllamaAlwaysBad:
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
        return LlmGenerateResult(text="bad")


class FakeOllamaLowercaseAction:
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
        return LlmGenerateResult(text='[{"span_id": 1, "action": "remove", "label": "AUDIENCE"}]')


class FakeOllamaMustNotCall:
    def ensure_model_available(self, model: str) -> None:
        raise AssertionError("ensure_model_available should not be called for empty input")

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
        raise AssertionError("generate should not be called for empty input")


class FakeOllamaMarkdownFence:
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
        return LlmGenerateResult(
            text='```json\n[{"span_id": 1, "action": "REMOVE", "label": "AUDIENCE"}]\n```'
        )


class FakeOllamaWrappedJson:
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
        return LlmGenerateResult(
            text='{"decisions": [{"span_id": 1, "action": "REMOVE", "label": "AUDIENCE"}]}'
        )


class FakeOllamaBatching:
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        span_ids: list[int] = []
        for line in prompt.splitlines():
            line = line.strip()
            if line.startswith("[") and "] " in line:
                try:
                    span_id = int(line[1 : line.index("]")])
                except ValueError:
                    continue
                span_ids.append(span_id)
        return LlmGenerateResult(
            text=json.dumps([{"span_id": item, "action": "KEEP"} for item in span_ids])
        )


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
        llm_client=FakeOllamaRetrySuccess(),
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
        llm_client=FakeOllamaAlwaysBad(),
        manifest_store=FileSystemManifestStore(),
    )
    with pytest.raises(InvalidLlmJsonError):
        use_case.run(_ctx(tmp_path))
    llm_log_path = tmp_path / "s06_remove_nonlecture" / "logs" / "model_calls.jsonl"
    assert llm_log_path.exists()
    retry_rows = llm_log_path.read_text(encoding="utf-8").splitlines()
    assert len(retry_rows) == _ctx(tmp_path).settings.llm_json_max_retries + 1


def test_s06_remove_nonlecture_normalizes_action_case(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        llm_client=FakeOllamaLowercaseAction(),
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
        llm_client=FakeOllamaMustNotCall(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8") == ""


def test_s06_remove_nonlecture_accepts_markdown_fenced_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        llm_client=FakeOllamaMarkdownFence(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    final_text = (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8")
    assert "Audience question" not in final_text


def test_s06_remove_nonlecture_accepts_wrapped_json_array(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        llm_client=FakeOllamaWrappedJson(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    final_text = (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8")
    assert "Audience question" not in final_text


def test_s06_remove_nonlecture_splits_large_inputs_into_batches(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    paragraphs = [f"Lecture segment {index}" for index in range(1, 121)]
    (s05_dir / "cleaned_1.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")
    fake_ollama = FakeOllamaBatching()
    use_case = S06RemoveNonLectureUseCase(
        llm_client=fake_ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert fake_ollama.calls > 1
