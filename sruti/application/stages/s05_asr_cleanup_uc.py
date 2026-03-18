from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sruti.application.batch_scheduler import ExternalApiTask, execute_ordered_external_api_tasks
from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.models import LlmCallRecord, LlmGenerateResult, StageResult
from sruti.domain.ports import LlmClient, ManifestStore
from sruti.llm.chunking import chunk_text
from sruti.llm.prompts import s05_cleanup_prompt
from sruti.llm.runtime import StageCostGuardrails, resolve_llm_model
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file


@dataclass(slots=True)
class CleanupChunkResult:
    chunk_id: int
    before: str
    prompt: str
    prompt_hash: str
    after: str
    usage_input_tokens: int | None
    usage_output_tokens: int | None


class S05AsrCleanupUseCase:
    stage_name = StageId.S05.value

    def __init__(
        self,
        *,
        llm_client_factory: Callable[[], LlmClient],
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._llm_client_factory = llm_client_factory
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
            "prompt_templates_dir": (
                str(context.settings.prompt_templates_dir)
                if context.settings.prompt_templates_dir is not None
                else None
            ),
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
            llm_client = self._llm_client_factory()
            manifest.tool_versions["llm_provider"] = llm_client.provider_name()
            manifest.tool_versions["llm_model"] = model
            source_text = input_path.read_text(encoding="utf-8")
            chunks = chunk_text(source_text, max_chars=6000)
            prompts = [
                s05_cleanup_prompt(
                    chunk,
                    template_dir=context.settings.prompt_templates_dir,
                )
                for chunk in chunks
            ]
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
                llm_client.ensure_model_available(model)

            if self._should_parallelize(context=context, chunk_count=len(chunks)):
                results = self._run_parallel_chunks(
                    context=context,
                    model=model,
                    chunks=chunks,
                    prompts=prompts,
                    guardrails=guardrails,
                )
            else:
                results = self._run_sequential_chunks(
                    context=context,
                    llm_client=llm_client,
                    model=model,
                    chunks=chunks,
                    prompts=prompts,
                    guardrails=guardrails,
                )

            cleaned_chunks: list[str] = []
            edit_rows: list[dict[str, Any]] = []
            call_rows: list[dict[str, Any]] = []
            for result in results:
                est_input_tokens, est_output_tokens = guardrails.estimated_tokens_for_prompt(result.prompt)
                metrics = guardrails.record_call(
                    estimated_input_tokens=est_input_tokens,
                    estimated_output_tokens=est_output_tokens,
                    usage_input_tokens=result.usage_input_tokens,
                    usage_output_tokens=result.usage_output_tokens,
                )
                cleaned_chunks.append(result.after)
                edit_rows.append(
                    {
                        "chunk_id": result.chunk_id,
                        "before": result.before,
                        "after": result.after,
                        "diff": self._diff_lines(result.before, result.after),
                    }
                )
                call_rows.append(
                    {
                        "chunk_id": result.chunk_id,
                        "provider": llm_client.provider_name(),
                        "model": model,
                        "temperature": context.settings.s05_temperature,
                        "prompt_hash": result.prompt_hash,
                        "prompt": result.prompt,
                        "response": result.after,
                        "input_chars": len(result.before),
                        "output_chars": len(result.after),
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
                        prompt_hash=result.prompt_hash,
                        input_chars=len(result.before),
                        output_chars=len(result.after),
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

    def _should_parallelize(self, *, context: StageContext, chunk_count: int) -> bool:
        return (
            context.settings.llm_provider is LlmProvider.OPENAI
            and context.execution_coordinator is not None
            and context.execution_coordinator.max_external_api_parallelism() > 1
            and chunk_count > 1
        )

    def _run_sequential_chunks(
        self,
        *,
        context: StageContext,
        llm_client: LlmClient,
        model: str,
        chunks: list[str],
        prompts: list[str],
        guardrails: StageCostGuardrails,
    ) -> list[CleanupChunkResult]:
        total_chunks = len(chunks)
        results: list[CleanupChunkResult] = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = prompts[idx - 1]
            prompt_hash = sha256_text(prompt)
            context.emit_progress(f"[s05] llm chunk {idx}/{total_chunks}", verbose_only=True)
            guardrails.before_call()
            try:
                response_obj = llm_client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=context.settings.s05_temperature,
                    timeout_seconds=context.settings.stage_timeout_seconds,
                )
            except Exception:
                guardrails.record_failure()
                raise
            results.append(
                self._chunk_result(
                    chunk_id=idx,
                    before=chunk,
                    prompt=prompt,
                    prompt_hash=prompt_hash,
                    response_obj=response_obj,
                )
            )
        return results

    def _run_parallel_chunks(
        self,
        *,
        context: StageContext,
        model: str,
        chunks: list[str],
        prompts: list[str],
        guardrails: StageCostGuardrails,
    ) -> list[CleanupChunkResult]:
        coordinator = context.execution_coordinator
        if coordinator is None:
            raise RuntimeError("parallel execution requires execution coordinator")

        total_chunks = len(chunks)
        tasks: list[ExternalApiTask[CleanupChunkResult]] = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = prompts[idx - 1]
            prompt_hash = sha256_text(prompt)

            def _run_chunk(
                chunk_id: int = idx,
                before_value: str = chunk,
                prompt_value: str = prompt,
                prompt_hash_value: str = prompt_hash,
            ) -> CleanupChunkResult:
                context.emit_progress(f"[s05] llm chunk {chunk_id}/{total_chunks}", verbose_only=True)
                guardrails.before_call()
                client = self._llm_client_factory()
                try:
                    response_obj = client.generate(
                        model=model,
                        prompt=prompt_value,
                        temperature=context.settings.s05_temperature,
                        timeout_seconds=context.settings.stage_timeout_seconds,
                    )
                except Exception:
                    guardrails.record_failure()
                    raise
                return self._chunk_result(
                    chunk_id=chunk_id,
                    before=before_value,
                    prompt=prompt_value,
                    prompt_hash=prompt_hash_value,
                    response_obj=response_obj,
                )

            tasks.append(
                ExternalApiTask(
                    index=idx,
                    label=f"chunk {idx}/{total_chunks}",
                    run=_run_chunk,
                )
            )

        return execute_ordered_external_api_tasks(
            coordinator,
            stage_id=StageId.S05,
            tasks=tasks,
        )

    def _chunk_result(
        self,
        *,
        chunk_id: int,
        before: str,
        prompt: str,
        prompt_hash: str,
        response_obj: LlmGenerateResult,
    ) -> CleanupChunkResult:
        return CleanupChunkResult(
            chunk_id=chunk_id,
            before=before,
            prompt=prompt,
            prompt_hash=prompt_hash,
            after=response_obj.text.strip(),
            usage_input_tokens=response_obj.usage_input_tokens,
            usage_output_tokens=response_obj.usage_output_tokens,
        )

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
