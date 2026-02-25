from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.models import LlmCallRecord, StageResult
from sruti.domain.ports import LlmClient, ManifestStore
from sruti.llm.chunking import chunk_text
from sruti.llm.prompts import s05_cleanup_prompt
from sruti.llm.runtime import StageCostGuardrails, resolve_llm_model
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file


class S05AsrCleanupUseCase:
    stage_name = StageId.S05.value

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._manifest_store = manifest_store
        self._ask_user = ask_user

    def run(self, context: StageContext) -> StageResult:
        if context.settings.llm_provider is LlmProvider.LOCAL:
            require_executable(context.settings.ollama_bin)
        model = resolve_llm_model(
            context.settings,
            stage_id=StageId.S05,
            local_model_attr="s05_model",
        )

        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S05.value)
        s04_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S04.value)
        input_path = s04_dir / "merged_raw.txt"
        require_file(input_path, label="s04 merged_raw.txt")

        cleaned_path = stage_dir / "cleaned_1.txt"
        edits_path = stage_dir / "edits.jsonl"
        llm_log_path = stage_dir / "logs" / "model_calls.jsonl"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "llm_provider": context.settings.llm_provider.value,
            "model": model,
            "temperature": context.settings.s05_temperature,
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S05,
            stage_dir=stage_dir,
            expected_outputs=[cleaned_path, edits_path, llm_log_path],
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
            manifest.tool_versions["llm_provider"] = self._llm_client.provider_name()
            manifest.tool_versions["llm_model"] = model
            source_text = input_path.read_text(encoding="utf-8")
            chunks = chunk_text(source_text, max_chars=6000)
            prompts = [s05_cleanup_prompt(chunk) for chunk in chunks]
            guardrails = StageCostGuardrails(
                settings=context.settings,
                stage_id=StageId.S05,
                provider=context.settings.llm_provider,
                model=model,
            )
            preflight = guardrails.preflight(prompts)
            if prompts:
                context.emit_progress(
                    f"[s05] preflight tokens in/out: {preflight['estimated_input_tokens']}/"
                    f"{preflight['estimated_output_tokens']}, est. cost ${preflight['estimated_cost_usd']}",
                    verbose_only=True,
                )

            if chunks:
                self._llm_client.ensure_model_available(model)

            cleaned_chunks: list[str] = []
            edit_rows: list[dict[str, Any]] = []
            call_rows: list[dict[str, Any]] = []
            total_chunks = len(chunks)

            for idx, chunk in enumerate(chunks, start=1):
                prompt = prompts[idx - 1]
                prompt_hash = sha256_text(prompt)
                context.emit_progress(f"[s05] llm chunk {idx}/{total_chunks}", verbose_only=True)
                guardrails.before_call()
                response_obj = self._llm_client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=context.settings.s05_temperature,
                    timeout_seconds=context.settings.stage_timeout_seconds,
                )
                response = response_obj.text.strip()
                est_input_tokens, est_output_tokens = guardrails.estimated_tokens_for_prompt(prompt)
                metrics = guardrails.record_call(
                    estimated_input_tokens=est_input_tokens,
                    estimated_output_tokens=est_output_tokens,
                    usage_input_tokens=response_obj.usage_input_tokens,
                    usage_output_tokens=response_obj.usage_output_tokens,
                )
                cleaned_chunks.append(response)
                edit_rows.append(
                    {
                        "chunk_id": idx,
                        "before": chunk,
                        "after": response,
                        "diff": self._diff_lines(chunk, response),
                    }
                )
                call_rows.append(
                    {
                        "chunk_id": idx,
                        "provider": self._llm_client.provider_name(),
                        "model": model,
                        "temperature": context.settings.s05_temperature,
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "response": response,
                        "input_chars": len(chunk),
                        "output_chars": len(response),
                        "usage_input_tokens": metrics.input_tokens,
                        "usage_output_tokens": metrics.output_tokens,
                        "estimated_cost_usd": metrics.estimated_cost_usd,
                        "cumulative_cost_usd": metrics.cumulative_cost_usd,
                    }
                )
                manifest.llm_calls.append(
                    LlmCallRecord(
                        model=model,
                        temperature=context.settings.s05_temperature,
                        prompt_hash=prompt_hash,
                        input_chars=len(chunk),
                        output_chars=len(response),
                        retries=0,
                    )
                )

            final_text = "\n\n".join(cleaned_chunks).strip()
            if final_text:
                final_text += "\n"
            atomic_write_text(cleaned_path, final_text)
            write_jsonl(edits_path, edit_rows)
            write_jsonl(llm_log_path, call_rows)
            manifest.inputs = [manifest_util.artifact_for(input_path)]
            return runtime.mark_success(
                manifest,
                output_paths=[cleaned_path, edits_path, llm_log_path],
            )
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise

    def _diff_lines(self, before: str, after: str) -> list[str]:
        return list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
