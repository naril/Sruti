from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import StageId
from sruti.domain.models import LlmCallRecord, StageResult
from sruti.domain.ports import ManifestStore
from sruti.infrastructure.llm_ollama import OllamaClient
from sruti.llm.chunking import chunk_text
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file


class LlmTextTransformUseCase:
    stage_id: StageId
    input_stage_id: StageId
    input_filename: str
    output_filename: str
    model_setting_attr: str
    temperature_setting_attr: str
    prompt_builder: Callable[[str], str]
    chunk_max_chars: int = 6000

    def __init__(
        self,
        *,
        ollama: OllamaClient,
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._ollama = ollama
        self._manifest_store = manifest_store
        self._ask_user = ask_user

    def run(self, context: StageContext) -> StageResult:
        require_executable(context.settings.ollama_bin)
        model = getattr(context.settings, self.model_setting_attr)
        temperature = getattr(context.settings, self.temperature_setting_attr)
        self._ollama.ensure_model_available(model)

        stage_dir = manifest_util.stage_dir_for(context.run_dir, self.stage_id.value)
        input_path = (
            manifest_util.stage_dir_for(context.run_dir, self.input_stage_id.value) / self.input_filename
        )
        require_file(input_path, label=f"Input for {self.stage_id.value}")

        output_path = stage_dir / self.output_filename
        llm_log_path = stage_dir / "logs" / "model_calls.jsonl"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "model": model,
            "temperature": temperature,
            "chunk_max_chars": self.chunk_max_chars,
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=self.stage_id,
            stage_dir=stage_dir,
            expected_outputs=[output_path, llm_log_path],
            manifest_store=self._manifest_store,
            ask_user=self._ask_user,
        )
        manifest = runtime.initialize_manifest(params=params)

        if runtime.should_skip(params=params, inputs_signature=inputs_signature):
            return runtime.mark_skipped(manifest)

        policy = runtime.apply_on_exists_policy()
        if policy == "skip":
            return runtime.mark_skipped(manifest)

        if context.dry_run:
            return runtime.mark_dry_run(manifest)

        try:
            runtime.start(manifest)
            manifest.tool_versions["ollama_model"] = model
            source_text = input_path.read_text(encoding="utf-8")
            chunks = chunk_text(source_text, max_chars=self.chunk_max_chars)
            output_chunks: list[str] = []
            call_rows: list[dict[str, Any]] = []
            for idx, chunk in enumerate(chunks, start=1):
                prompt = self.prompt_builder(chunk)
                prompt_hash = sha256_text(prompt)
                response = self._ollama.generate(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    timeout_seconds=context.settings.stage_timeout_seconds,
                ).strip()
                output_chunks.append(response)
                call_rows.append(
                    {
                        "chunk_id": idx,
                        "model": model,
                        "temperature": temperature,
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "response": response,
                        "input_chars": len(chunk),
                        "output_chars": len(response),
                    }
                )
                manifest.llm_calls.append(
                    LlmCallRecord(
                        model=model,
                        temperature=temperature,
                        prompt_hash=prompt_hash,
                        input_chars=len(chunk),
                        output_chars=len(response),
                    )
                )

            final_text = "\n\n".join(output_chunks).strip()
            if final_text:
                final_text += "\n"
            atomic_write_text(output_path, final_text)
            write_jsonl(llm_log_path, call_rows)
            manifest.inputs = [manifest_util.artifact_for(input_path)]
            return runtime.mark_success(manifest, output_paths=[output_path, llm_log_path])
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise
