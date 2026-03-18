from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s05_asr_cleanup_uc import S05AsrCleanupUseCase
from sruti.domain.enums import LlmProvider, OnExistsMode, StageId, StageStatus
from sruti.domain.models import LlmGenerateResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllama:
    def __init__(self) -> None:
        self.generate_calls = 0

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
        self.generate_calls += 1
        return LlmGenerateResult(text="cleaned")


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


class FakeOpenAIParallel:
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
        if "First paragraph" in prompt:
            time.sleep(0.05)
            return LlmGenerateResult(text="FIRST CLEAN")
        return LlmGenerateResult(text="SECOND CLEAN")


def _ctx(run_dir: Path, *, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=dry_run,
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


def test_s05_cleanup_happy_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s05_asr_cleanup_uc.require_executable", lambda _: None)
    s04_dir = manifest_util.stage_dir_for(tmp_path, "s04")
    s04_dir.mkdir(parents=True, exist_ok=True)
    (s04_dir / "merged_raw.txt").write_text("hello world", encoding="utf-8")
    use_case = S05AsrCleanupUseCase(
        llm_client_factory=lambda: FakeOllama(),
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
        llm_client_factory=lambda: FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path, dry_run=True))
    assert result.status == StageStatus.DRY_RUN


def test_s05_cleanup_skips_llm_for_empty_input(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s05_asr_cleanup_uc.require_executable", lambda _: None)
    s04_dir = manifest_util.stage_dir_for(tmp_path, "s04")
    s04_dir.mkdir(parents=True, exist_ok=True)
    (s04_dir / "merged_raw.txt").write_text("  \n\n\t", encoding="utf-8")
    ollama = FakeOllama()
    use_case = S05AsrCleanupUseCase(
        llm_client_factory=lambda: ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert ollama.generate_calls == 0
    assert (tmp_path / "s05_asr_cleanup" / "cleaned_1.txt").read_text(encoding="utf-8") == ""


def test_s05_cleanup_parallel_openai_keeps_output_and_logs_order(tmp_path: Path) -> None:
    s04_dir = manifest_util.stage_dir_for(tmp_path, "s04")
    s04_dir.mkdir(parents=True, exist_ok=True)
    text = ("First paragraph " * 300).strip() + "\n\n" + ("Second paragraph " * 300).strip()
    (s04_dir / "merged_raw.txt").write_text(text, encoding="utf-8")
    coordinator = ThreadedCoordinator(max_parallel=2)
    use_case = S05AsrCleanupUseCase(
        llm_client_factory=lambda: FakeOpenAIParallel(),
        manifest_store=FileSystemManifestStore(),
    )
    try:
        result = use_case.run(_openai_ctx(tmp_path, coordinator))
    finally:
        coordinator.shutdown()

    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s05_asr_cleanup" / "cleaned_1.txt").read_text(encoding="utf-8") == "FIRST CLEAN\n\nSECOND CLEAN\n"
    edit_rows = [
        json.loads(line)
        for line in (tmp_path / "s05_asr_cleanup" / "edits.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["chunk_id"] for row in edit_rows] == [1, 2]
