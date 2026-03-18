from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pytest

from sruti.application.context import StageContext
from sruti.application.stages.s06_remove_nonlecture_uc import S06RemoveNonLectureUseCase
from sruti.domain.enums import LlmProvider, OnExistsMode, StageId, StageStatus
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


class ThreadedCoordinator:
    def __init__(self, *, max_parallel: int) -> None:
        self._max_parallel = max_parallel
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)

    def emit_progress(self, message: str) -> None:
        _ = message

    def stage_scope(self, stage_id: StageId, *, llm_provider: LlmProvider):
        _ = (stage_id, llm_provider)
        return nullcontext()

    def submit_external_api_task(self, *, stage_id: StageId, task_label: str, fn) -> Future[LlmGenerateResult]:
        _ = (stage_id, task_label)
        return self._executor.submit(fn)

    def max_external_api_parallelism(self) -> int:
        return self._max_parallel

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class FakeOpenAIParallelBatching:
    def ensure_model_available(self, model: str) -> None:
        _ = model

    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> LlmGenerateResult:
        _ = (model, temperature, timeout_seconds)
        if "[1]" in prompt:
            time.sleep(0.05)
            return LlmGenerateResult(text='[{"span_id": 1, "action": "KEEP"}]')
        return LlmGenerateResult(text='[{"span_id": 2, "action": "KEEP"}]')


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def _openai_ctx(run_dir: Path, coordinator: ThreadedCoordinator) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
        llm_provider_override=LlmProvider.OPENAI,
        execution_coordinator=coordinator,
    )


def test_s06_remove_nonlecture_with_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        llm_client_factory=lambda: FakeOllamaRetrySuccess(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert "Lecture starts" in (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(
        encoding="utf-8"
    )
    report_path = tmp_path / "s06_remove_nonlecture" / "removal_report.html"
    report_html = report_path.read_text(encoding="utf-8")
    assert "status-REMOVE" in report_html
    assert "status-KEEP" in report_html
    assert "Audience question" in report_html
    assert "Lecture starts" in report_html


def test_s06_remove_nonlecture_fails_after_retries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("only text", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        llm_client_factory=lambda: FakeOllamaAlwaysBad(),
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
        llm_client_factory=lambda: FakeOllamaLowercaseAction(),
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
        llm_client_factory=lambda: FakeOllamaMustNotCall(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8") == ""


def test_s06_remove_nonlecture_parallel_openai_preserves_batch_log_order(tmp_path: Path) -> None:
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("First span.\n\nSecond span.", encoding="utf-8")
    coordinator = ThreadedCoordinator(max_parallel=2)
    use_case = S06RemoveNonLectureUseCase(
        llm_client_factory=lambda: FakeOpenAIParallelBatching(),
        manifest_store=FileSystemManifestStore(),
    )
    use_case._split_span_batches = lambda spans: [[span] for span in spans]  # type: ignore[method-assign]
    try:
        result = use_case.run(_openai_ctx(tmp_path, coordinator))
    finally:
        coordinator.shutdown()

    assert result.status == StageStatus.SUCCESS
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "s06_remove_nonlecture" / "logs" / "model_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["batch_index"] for row in log_rows] == [1, 2]
    final_text = (tmp_path / "s06_remove_nonlecture" / "content_only.txt").read_text(encoding="utf-8")
    assert "First span." in final_text
    assert "Second span." in final_text


def test_s06_remove_nonlecture_accepts_markdown_fenced_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None
    )
    s05_dir = manifest_util.stage_dir_for(tmp_path, "s05")
    s05_dir.mkdir(parents=True, exist_ok=True)
    (s05_dir / "cleaned_1.txt").write_text("Audience question\n\nLecture starts", encoding="utf-8")
    use_case = S06RemoveNonLectureUseCase(
        llm_client_factory=lambda: FakeOllamaMarkdownFence(),
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
        llm_client_factory=lambda: FakeOllamaWrappedJson(),
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
        llm_client_factory=lambda: fake_ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert fake_ollama.calls > 1
