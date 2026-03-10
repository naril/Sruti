from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sruti.application.batch_scheduler import ExternalApiTask, execute_ordered_external_api_tasks
from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.models import LlmCallRecord, LlmGenerateResult, StageResult
from sruti.domain.ports import LlmClient, ManifestStore
from sruti.llm.chunking import chunk_text
from sruti.llm.runtime import StageCostGuardrails, resolve_llm_model
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file


class PromptBuilder(Protocol):
    def __call__(self, text: str, *, template_dir: Path | None = None) -> str: ...


@dataclass(slots=True)
class ChunkCallResult:
    chunk_id: int
    chunk_text: str
    prompt: str
    prompt_hash: str
    response: str
    usage_input_tokens: int | None
    usage_output_tokens: int | None


class LlmTextTransformUseCase:
    stage_id: StageId
    input_stage_id: StageId
    input_filename: str
    output_filename: str
    model_setting_attr: str
    temperature_setting_attr: str
    prompt_builder: PromptBuilder
    chunk_max_chars: int = 6000

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

    def resolve_input(self, context: StageContext) -> tuple[Path, dict[str, object]]:
        input_path = (
            manifest_util.stage_dir_for(context.run_dir, self.input_stage_id.value) / self.input_filename
        )
        return input_path, {}

    def run(self, context: StageContext) -> StageResult:
        if context.settings.llm_provider is LlmProvider.LOCAL:
            require_executable(context.settings.ollama_bin)
        model = resolve_llm_model(
            context.settings,
            stage_id=self.stage_id,
            local_model_attr=self.model_setting_attr,
        )
        temperature = getattr(context.settings, self.temperature_setting_attr)

        stage_dir = manifest_util.stage_dir_for(context.run_dir, self.stage_id.value)
        input_path, input_params = self.resolve_input(context)
        require_file(input_path, label=f"Input for {self.stage_id.value}")

        output_path = stage_dir / self.output_filename
        llm_log_path = stage_dir / "logs" / "model_calls.jsonl"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "llm_provider": context.settings.llm_provider.value,
            "model": model,
            "temperature": temperature,
            "chunk_max_chars": self.chunk_max_chars,
            "prompt_templates_dir": (
                str(context.settings.prompt_templates_dir)
                if context.settings.prompt_templates_dir is not None
                else None
            ),
            "_inputs_signature": inputs_signature,
        }
        params.update(input_params)

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
            llm_client = self._llm_client_factory()
            manifest.tool_versions["llm_provider"] = llm_client.provider_name()
            manifest.tool_versions["llm_model"] = model
            source_text = input_path.read_text(encoding="utf-8")
            chunks = chunk_text(source_text, max_chars=self.chunk_max_chars)
            prompts = [
                self.prompt_builder(
                    chunk,
                    template_dir=context.settings.prompt_templates_dir,
                )
                for chunk in chunks
            ]
            guardrails = StageCostGuardrails(
                settings=context.settings,
                stage_id=self.stage_id,
                provider=context.settings.llm_provider,
                model=model,
            )
            preflight = guardrails.preflight(prompts)
            if prompts:
                context.emit_progress(
                    f"[{self.stage_id.value}] preflight tokens in/out: "
                    f"{preflight['estimated_input_tokens']}/{preflight['estimated_output_tokens']}, "
                    f"est. cost ${preflight['estimated_cost_usd']}",
                    verbose_only=True,
                )
                llm_client.ensure_model_available(model)

            if self._should_parallelize(context=context, chunk_count=len(chunks)):
                results = self._run_parallel_chunks(
                    context=context,
                    model=model,
                    temperature=temperature,
                    chunks=chunks,
                    prompts=prompts,
                    guardrails=guardrails,
                )
            else:
                results = self._run_sequential_chunks(
                    context=context,
                    llm_client=llm_client,
                    model=model,
                    temperature=temperature,
                    chunks=chunks,
                    prompts=prompts,
                    guardrails=guardrails,
                )

            output_chunks: list[str] = []
            call_rows: list[dict[str, Any]] = []
            for result in results:
                est_input_tokens, est_output_tokens = guardrails.estimated_tokens_for_prompt(result.prompt)
                metrics = guardrails.record_call(
                    estimated_input_tokens=est_input_tokens,
                    estimated_output_tokens=est_output_tokens,
                    usage_input_tokens=result.usage_input_tokens,
                    usage_output_tokens=result.usage_output_tokens,
                )
                output_chunks.append(result.response)
                call_rows.append(
                    {
                        "chunk_id": result.chunk_id,
                        "provider": llm_client.provider_name(),
                        "model": model,
                        "temperature": temperature,
                        "prompt_hash": result.prompt_hash,
                        "prompt": result.prompt,
                        "response": result.response,
                        "input_chars": len(result.chunk_text),
                        "output_chars": len(result.response),
                        "usage_input_tokens": metrics.input_tokens,
                        "usage_output_tokens": metrics.output_tokens,
                        "estimated_cost_usd": metrics.estimated_cost_usd,
                        "cumulative_cost_usd": metrics.cumulative_cost_usd,
                    }
                )
                manifest.llm_calls.append(
                    LlmCallRecord(
                        model=model,
                        temperature=temperature,
                        prompt_hash=result.prompt_hash,
                        input_chars=len(result.chunk_text),
                        output_chars=len(result.response),
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
        temperature: float,
        chunks: list[str],
        prompts: list[str],
        guardrails: StageCostGuardrails,
    ) -> list[ChunkCallResult]:
        results: list[ChunkCallResult] = []
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            prompt = prompts[idx - 1]
            prompt_hash = sha256_text(prompt)
            context.emit_progress(
                f"[{self.stage_id.value}] llm chunk {idx}/{total_chunks}",
                verbose_only=True,
            )
            guardrails.before_call()
            try:
                response_obj = llm_client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    timeout_seconds=context.settings.stage_timeout_seconds,
                )
            except Exception:
                guardrails.record_failure()
                raise
            results.append(
                self._chunk_result(
                    chunk_id=idx,
                    chunk=chunk,
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
        temperature: float,
        chunks: list[str],
        prompts: list[str],
        guardrails: StageCostGuardrails,
    ) -> list[ChunkCallResult]:
        coordinator = context.execution_coordinator
        if coordinator is None:
            raise RuntimeError("parallel execution requires execution coordinator")

        total_chunks = len(chunks)
        tasks: list[ExternalApiTask[ChunkCallResult]] = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = prompts[idx - 1]
            prompt_hash = sha256_text(prompt)

            def _run_chunk(
                chunk_id: int = idx,
                chunk_text_value: str = chunk,
                prompt_value: str = prompt,
                prompt_hash_value: str = prompt_hash,
            ) -> ChunkCallResult:
                context.emit_progress(
                    f"[{self.stage_id.value}] llm chunk {chunk_id}/{total_chunks}",
                    verbose_only=True,
                )
                guardrails.before_call()
                client = self._llm_client_factory()
                try:
                    response_obj = client.generate(
                        model=model,
                        prompt=prompt_value,
                        temperature=temperature,
                        timeout_seconds=context.settings.stage_timeout_seconds,
                    )
                except Exception:
                    guardrails.record_failure()
                    raise
                return self._chunk_result(
                    chunk_id=chunk_id,
                    chunk=chunk_text_value,
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
            stage_id=self.stage_id,
            tasks=tasks,
        )

    def _chunk_result(
        self,
        *,
        chunk_id: int,
        chunk: str,
        prompt: str,
        prompt_hash: str,
        response_obj: LlmGenerateResult,
    ) -> ChunkCallResult:
        return ChunkCallResult(
            chunk_id=chunk_id,
            chunk_text=chunk,
            prompt=prompt,
            prompt_hash=prompt_hash,
            response=response_obj.text.strip(),
            usage_input_tokens=response_obj.usage_input_tokens,
            usage_output_tokens=response_obj.usage_output_tokens,
        )
