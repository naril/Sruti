from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s01_normalize_uc import S01NormalizeUseCase
from sruti.application.stages.s02_chunk_uc import S02ChunkUseCase
from sruti.application.stages.s03_asr_whisper_uc import S03AsrWhisperUseCase
from sruti.application.stages.s04_merge_uc import S04MergeUseCase
from sruti.application.stages.s05_asr_cleanup_uc import S05AsrCleanupUseCase
from sruti.application.stages.s06_remove_nonlecture_uc import S06RemoveNonLectureUseCase
from sruti.application.stages.s07_editorial_uc import S07EditorialUseCase
from sruti.application.stages.s08_translate_faithful_uc import S08TranslateFaithfulUseCase
from sruti.application.stages.s09_translate_edit_uc import S09TranslateEditUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.domain.models import LlmGenerateResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore


class FakeFfmpeg:
    def normalize(self, input_path: Path, output_path: Path) -> list[str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        return ["ffmpeg", "normalize", str(input_path), str(output_path)]

    def segment(self, input_path: Path, output_pattern: Path, *, seconds: int) -> list[str]:
        _ = (input_path, seconds)
        output_pattern.parent.mkdir(parents=True, exist_ok=True)
        (output_pattern.parent / "0001.wav").write_bytes(b"chunk-1")
        return ["ffmpeg", "segment", str(output_pattern)]


class FakeWhisper:
    def transcribe_chunk(self, *, model_path: Path, chunk_path: Path, output_prefix: Path) -> list[str]:
        _ = (model_path, chunk_path)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_prefix.with_suffix(".txt").write_text("Audience: hello\n\nCore lecture text", encoding="utf-8")
        output_prefix.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")
        return ["whisper-cli", str(output_prefix)]


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
        _ = (model, temperature, timeout_seconds)
        if "Classify each span as KEEP or REMOVE" in prompt:
            return LlmGenerateResult(
                text='[{"span_id": 1, "action": "REMOVE", "label": "AUDIENCE", "reason": "non-lecture"}, {"span_id": 2, "action": "KEEP", "label": "LECTURE", "reason": "main content"}]'
            )
        if "You are cleaning ASR output from an English lecture" in prompt:
            return LlmGenerateResult(text="Audience: hello\n\nCore lecture text")
        if "Translate English to Czech faithfully" in prompt:
            return LlmGenerateResult(text="Hlavni text prednasky.")
        if "Improve Czech readability and style" in prompt:
            return LlmGenerateResult(text="Hlavni text prednasky.")
        if "Edit the following English lecture text" in prompt:
            return LlmGenerateResult(text="Core lecture text.")
        return LlmGenerateResult(text="Core lecture text.")


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_full_pipeline_simulated(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s01_normalize_uc.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages.s02_chunk_uc.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages.s03_asr_whisper_uc.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages.s05_asr_cleanup_uc.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages.s06_remove_nonlecture_uc.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages._llm_text_transform.require_executable", lambda _: None)
    monkeypatch.setattr("sruti.application.stages.s01_normalize_uc.executable_version", lambda _: "ffmpeg")
    monkeypatch.setattr("sruti.application.stages.s02_chunk_uc.executable_version", lambda _: "ffmpeg")
    monkeypatch.setattr("sruti.application.stages.s03_asr_whisper_uc.executable_version", lambda _: "whisper")

    run_dir = tmp_path / "run-e2e"
    input_audio = tmp_path / "lecture.wav"
    input_audio.write_bytes(b"audio")
    model_path = tmp_path / "ggml-large-v3.bin"
    model_path.write_bytes(b"model")
    ctx = _ctx(run_dir)
    store = FileSystemManifestStore()

    s01 = S01NormalizeUseCase(input_audio=input_audio, ffmpeg=FakeFfmpeg(), manifest_store=store)
    s02 = S02ChunkUseCase(seconds=30, ffmpeg=FakeFfmpeg(), manifest_store=store)
    s03 = S03AsrWhisperUseCase(whisper_model_path=model_path, whisper=FakeWhisper(), manifest_store=store)
    s04 = S04MergeUseCase(manifest_store=store)
    fake_ollama = FakeOllama()
    s05 = S05AsrCleanupUseCase(llm_client=fake_ollama, manifest_store=store)
    s06 = S06RemoveNonLectureUseCase(llm_client=fake_ollama, manifest_store=store)
    s07 = S07EditorialUseCase(llm_client=fake_ollama, manifest_store=store)
    s08 = S08TranslateFaithfulUseCase(llm_client=fake_ollama, manifest_store=store)
    s09 = S09TranslateEditUseCase(llm_client=fake_ollama, manifest_store=store)

    for use_case in [s01, s02, s03, s04, s05, s06, s07, s08, s09]:
        result = use_case.run(ctx)
        assert result.status == StageStatus.SUCCESS

    final_path = run_dir / "s09_translate_edit" / "final_publishable_cs.txt"
    assert final_path.exists()
    assert "Hlavni text prednasky." in final_path.read_text(encoding="utf-8")
