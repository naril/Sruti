from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from sruti.application.batch_scheduler import ExternalApiTask, execute_ordered_external_api_tasks
from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.errors import InvalidLlmJsonError
from sruti.domain.models import LlmCallRecord, StageResult
from sruti.domain.ports import LlmClient, ManifestStore
from sruti.infrastructure import json_codec
from sruti.llm.prompts import s08_condense_map_prompt, s08_condense_reduce_prompt
from sruti.llm.runtime import StageCostGuardrails, resolve_llm_model
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_json, atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class ParagraphBatch:
    start_paragraph: int
    rows: list[tuple[int, str]]

    @property
    def end_paragraph(self) -> int:
        return self.rows[-1][0]

    def render_lines(self) -> str:
        return "\n".join(f"[{idx}] {text}" for idx, text in self.rows)


class CandidateBlock(BaseModel):
    from_paragraph: int
    to_paragraph: int
    title: str
    body: str

    model_config = {"extra": "forbid"}

    @field_validator("title", "body", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_range(self) -> "CandidateBlock":
        if self.from_paragraph <= 0 or self.to_paragraph <= 0:
            raise ValueError("paragraph ids must be positive")
        if self.from_paragraph > self.to_paragraph:
            raise ValueError("from_paragraph must be <= to_paragraph")
        return self


@dataclass(slots=True)
class MapBatchResult:
    batch_id: int
    batch: ParagraphBatch
    prompt: str
    prompt_hash: str
    response: str
    parsed_blocks: list[CandidateBlock]
    usage_input_tokens: int | None
    usage_output_tokens: int | None


class S08CondenseUseCase:
    stage_name = StageId.S08.value
    map_chunk_max_paragraphs = 8
    map_chunk_overlap = 1

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
            stage_id=StageId.S08,
            local_model_attr="s08_model",
            openai_model_attr="openai_model_s08",
        )

        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S08.value)
        input_path = manifest_util.stage_dir_for(context.run_dir, StageId.S07.value) / "final_publishable_en.txt"
        require_file(input_path, label="Input for s08")

        output_path = stage_dir / "condensed_blocks_en.txt"
        candidate_blocks_path = stage_dir / "candidate_blocks.json"
        llm_log_path = stage_dir / "logs" / "model_calls.jsonl"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "llm_provider": context.settings.llm_provider.value,
            "model": model,
            "temperature": context.settings.s08_temperature,
            "map_chunk_max_paragraphs": self.map_chunk_max_paragraphs,
            "map_chunk_overlap": self.map_chunk_overlap,
            "prompt_templates_dir": (
                str(context.settings.prompt_templates_dir)
                if context.settings.prompt_templates_dir is not None
                else None
            ),
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S08,
            stage_dir=stage_dir,
            expected_outputs=[output_path, candidate_blocks_path, llm_log_path],
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
            paragraphs = self._to_paragraphs(source_text)
            if not paragraphs:
                atomic_write_text(output_path, "")
                atomic_write_json(candidate_blocks_path, [])
                write_jsonl(llm_log_path, [])
                manifest.inputs = [manifest_util.artifact_for(input_path)]
                return runtime.mark_success(
                    manifest,
                    output_paths=[output_path, candidate_blocks_path, llm_log_path],
                )

            llm_client.ensure_model_available(model)
            guardrails = StageCostGuardrails(
                settings=context.settings,
                stage_id=StageId.S08,
                provider=context.settings.llm_provider,
                model=model,
            )
            batches = self._paragraph_batches(paragraphs)
            map_prompts = [
                s08_condense_map_prompt(
                    batch.render_lines(),
                    template_dir=context.settings.prompt_templates_dir,
                )
                for batch in batches
            ]
            preflight = guardrails.preflight(map_prompts)
            context.emit_progress(
                f"[s08] preflight tokens in/out: "
                f"{preflight['estimated_input_tokens']}/{preflight['estimated_output_tokens']}, "
                f"est. cost ${preflight['estimated_cost_usd']}",
                verbose_only=True,
            )

            map_results = self._run_map_batches(
                context=context,
                llm_client=llm_client,
                model=model,
                batches=batches,
                prompts=map_prompts,
                guardrails=guardrails,
            )

            call_rows: list[dict[str, Any]] = []
            candidate_blocks: list[CandidateBlock] = []
            for result in map_results:
                est_input_tokens, est_output_tokens = guardrails.estimated_tokens_for_prompt(result.prompt)
                metrics = guardrails.record_call(
                    estimated_input_tokens=est_input_tokens,
                    estimated_output_tokens=est_output_tokens,
                    usage_input_tokens=result.usage_input_tokens,
                    usage_output_tokens=result.usage_output_tokens,
                )
                candidate_blocks.extend(result.parsed_blocks)
                call_rows.append(
                    {
                        "phase": "map",
                        "batch_id": result.batch_id,
                        "batch_start": result.batch.start_paragraph,
                        "batch_end": result.batch.end_paragraph,
                        "provider": llm_client.provider_name(),
                        "model": model,
                        "temperature": context.settings.s08_temperature,
                        "prompt_hash": result.prompt_hash,
                        "prompt": result.prompt,
                        "response": result.response,
                        "parsed_block_count": len(result.parsed_blocks),
                        "usage_input_tokens": metrics.input_tokens,
                        "usage_output_tokens": metrics.output_tokens,
                        "estimated_cost_usd": metrics.estimated_cost_usd,
                        "cumulative_cost_usd": metrics.cumulative_cost_usd,
                    }
                )
                manifest.llm_calls.append(
                    LlmCallRecord(
                        model=model,
                        temperature=context.settings.s08_temperature,
                        prompt_hash=result.prompt_hash,
                        input_chars=len(result.prompt),
                        output_chars=len(result.response),
                    )
                )

            merged_blocks = self._merge_overlapping_blocks(
                candidate_blocks,
                max_paragraph=len(paragraphs),
            )
            if not merged_blocks:
                merged_blocks = self._fallback_blocks(paragraphs)

            candidate_rows = [block.model_dump() for block in merged_blocks]
            atomic_write_json(candidate_blocks_path, candidate_rows)

            reduce_payload = json_codec.dumps(candidate_rows, indent=2)
            reduce_prompt = s08_condense_reduce_prompt(
                reduce_payload,
                template_dir=context.settings.prompt_templates_dir,
            )
            reduce_preflight = guardrails.preflight([reduce_prompt])
            context.emit_progress(
                f"[s08] reduce preflight tokens in/out: "
                f"{reduce_preflight['estimated_input_tokens']}/{reduce_preflight['estimated_output_tokens']}, "
                f"est. cost ${reduce_preflight['estimated_cost_usd']}",
                verbose_only=True,
            )
            context.emit_progress("[s08] condense reduce 1/1", verbose_only=True)
            guardrails.before_call()
            try:
                reduce_response_obj = llm_client.generate(
                    model=model,
                    prompt=reduce_prompt,
                    temperature=context.settings.s08_temperature,
                    timeout_seconds=context.settings.stage_timeout_seconds,
                )
            except Exception:
                guardrails.record_failure()
                raise
            reduce_response = reduce_response_obj.text.strip()
            est_input_tokens, est_output_tokens = guardrails.estimated_tokens_for_prompt(reduce_prompt)
            reduce_metrics = guardrails.record_call(
                estimated_input_tokens=est_input_tokens,
                estimated_output_tokens=est_output_tokens,
                usage_input_tokens=reduce_response_obj.usage_input_tokens,
                usage_output_tokens=reduce_response_obj.usage_output_tokens,
            )
            reduce_prompt_hash = sha256_text(reduce_prompt)
            call_rows.append(
                {
                    "phase": "reduce",
                    "provider": llm_client.provider_name(),
                    "model": model,
                    "temperature": context.settings.s08_temperature,
                    "prompt_hash": reduce_prompt_hash,
                    "prompt": reduce_prompt,
                    "response": reduce_response,
                    "usage_input_tokens": reduce_metrics.input_tokens,
                    "usage_output_tokens": reduce_metrics.output_tokens,
                    "estimated_cost_usd": reduce_metrics.estimated_cost_usd,
                    "cumulative_cost_usd": reduce_metrics.cumulative_cost_usd,
                }
            )
            manifest.llm_calls.append(
                LlmCallRecord(
                    model=model,
                    temperature=context.settings.s08_temperature,
                    prompt_hash=reduce_prompt_hash,
                    input_chars=len(reduce_prompt),
                    output_chars=len(reduce_response),
                )
            )

            final_text = self._normalize_final_text(reduce_response)
            if not final_text:
                final_text = self._render_blocks_text(merged_blocks)
            final_text = final_text.strip()
            if final_text:
                final_text += "\n"
            atomic_write_text(output_path, final_text)
            write_jsonl(llm_log_path, call_rows)
            manifest.inputs = [manifest_util.artifact_for(input_path)]
            return runtime.mark_success(
                manifest,
                output_paths=[output_path, candidate_blocks_path, llm_log_path],
            )
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise

    def _run_map_batches(
        self,
        *,
        context: StageContext,
        llm_client: LlmClient,
        model: str,
        batches: list[ParagraphBatch],
        prompts: list[str],
        guardrails: StageCostGuardrails,
    ) -> list[MapBatchResult]:
        if (
            context.settings.llm_provider is not LlmProvider.OPENAI
            or context.execution_coordinator is None
            or context.execution_coordinator.max_external_api_parallelism() <= 1
            or len(batches) <= 1
        ):
            results: list[MapBatchResult] = []
            total_batches = len(batches)
            for index, batch in enumerate(batches, start=1):
                prompt = prompts[index - 1]
                prompt_hash = sha256_text(prompt)
                results.append(
                    self._map_batch_result(
                        context=context,
                        llm_client=llm_client,
                        model=model,
                        batch=batch,
                        batch_id=index,
                        total_batches=total_batches,
                        prompt=prompt,
                        prompt_hash=prompt_hash,
                        guardrails=guardrails,
                    )
                )
            return results

        coordinator = context.execution_coordinator
        total_batches = len(batches)
        tasks: list[ExternalApiTask[MapBatchResult]] = []
        for index, batch in enumerate(batches, start=1):
            prompt = prompts[index - 1]
            prompt_hash = sha256_text(prompt)

            def _run_batch(
                batch_id: int = index,
                batch_value: ParagraphBatch = batch,
                prompt_value: str = prompt,
                prompt_hash_value: str = prompt_hash,
            ) -> MapBatchResult:
                client = self._llm_client_factory()
                return self._map_batch_result(
                    context=context,
                    llm_client=client,
                    model=model,
                    batch=batch_value,
                    batch_id=batch_id,
                    total_batches=total_batches,
                    prompt=prompt_value,
                    prompt_hash=prompt_hash_value,
                    guardrails=guardrails,
                )

            tasks.append(
                ExternalApiTask(
                    index=index,
                    label=f"map batch {index}/{total_batches}",
                    run=_run_batch,
                )
            )

        return execute_ordered_external_api_tasks(
            coordinator,
            stage_id=StageId.S08,
            tasks=tasks,
        )

    def _map_batch_result(
        self,
        *,
        context: StageContext,
        llm_client: LlmClient,
        model: str,
        batch: ParagraphBatch,
        batch_id: int,
        total_batches: int,
        prompt: str,
        prompt_hash: str,
        guardrails: StageCostGuardrails,
    ) -> MapBatchResult:
        context.emit_progress(
            f"[s08] condense map batch {batch_id}/{total_batches}",
            verbose_only=True,
        )
        guardrails.before_call()
        try:
            response_obj = llm_client.generate(
                model=model,
                prompt=prompt,
                temperature=context.settings.s08_temperature,
                timeout_seconds=context.settings.stage_timeout_seconds,
            )
        except Exception:
            guardrails.record_failure()
            raise
        response = response_obj.text.strip()
        parsed_blocks = self._parse_map_response(
            response,
            min_paragraph=batch.start_paragraph,
            max_paragraph=batch.end_paragraph,
        )
        return MapBatchResult(
            batch_id=batch_id,
            batch=batch,
            prompt=prompt,
            prompt_hash=prompt_hash,
            response=response,
            parsed_blocks=parsed_blocks,
            usage_input_tokens=response_obj.usage_input_tokens,
            usage_output_tokens=response_obj.usage_output_tokens,
        )

    def _to_paragraphs(self, text: str) -> list[str]:
        return [part.strip() for part in text.split("\n\n") if part.strip()]

    def _paragraph_batches(self, paragraphs: list[str]) -> list[ParagraphBatch]:
        if not paragraphs:
            return []
        max_paragraphs = self.map_chunk_max_paragraphs
        overlap = min(self.map_chunk_overlap, max_paragraphs - 1)
        step = max_paragraphs - overlap
        batches: list[ParagraphBatch] = []
        start = 0
        while start < len(paragraphs):
            end = min(len(paragraphs), start + max_paragraphs)
            rows = [(index + 1, paragraphs[index]) for index in range(start, end)]
            batches.append(ParagraphBatch(start_paragraph=start + 1, rows=rows))
            if end >= len(paragraphs):
                break
            start += step
        return batches

    def _parse_map_response(
        self,
        response: str,
        *,
        min_paragraph: int,
        max_paragraph: int,
    ) -> list[CandidateBlock]:
        payload = self._load_json(response)
        raw_blocks: Any
        if isinstance(payload, dict):
            raw_blocks = payload.get("blocks", [])
        else:
            raw_blocks = payload
        if not isinstance(raw_blocks, list):
            raise InvalidLlmJsonError("s08 map response must be a JSON array or object with 'blocks'.")

        blocks: list[CandidateBlock] = []
        for item in raw_blocks:
            try:
                block = CandidateBlock.model_validate(item)
            except ValidationError:
                continue
            clamped = self._clamp_block(block, min_paragraph=min_paragraph, max_paragraph=max_paragraph)
            if clamped is not None:
                blocks.append(clamped)
        return blocks

    def _load_json(self, value: str) -> Any:
        stripped = value.strip()
        if not stripped:
            raise InvalidLlmJsonError("s08 map response is empty.")
        try:
            return json_codec.loads(stripped)
        except Exception:
            match = JSON_FENCE_PATTERN.search(stripped)
            if match is None:
                raise InvalidLlmJsonError("s08 map response is not valid JSON.") from None
            fenced = match.group(1).strip()
            try:
                return json_codec.loads(fenced)
            except Exception as exc:
                raise InvalidLlmJsonError("s08 map fenced JSON is invalid.") from exc

    def _clamp_block(
        self,
        block: CandidateBlock,
        *,
        min_paragraph: int,
        max_paragraph: int,
    ) -> CandidateBlock | None:
        start = max(min_paragraph, block.from_paragraph)
        end = min(max_paragraph, block.to_paragraph)
        if start > end:
            return None
        return block.model_copy(update={"from_paragraph": start, "to_paragraph": end})

    def _merge_overlapping_blocks(
        self,
        blocks: list[CandidateBlock],
        *,
        max_paragraph: int,
    ) -> list[CandidateBlock]:
        if not blocks:
            return []
        sorted_blocks = sorted(blocks, key=lambda item: (item.from_paragraph, item.to_paragraph))
        merged: list[CandidateBlock] = [sorted_blocks[0]]
        for block in sorted_blocks[1:]:
            prev = merged[-1]
            if block.from_paragraph <= prev.to_paragraph:
                merged[-1] = CandidateBlock(
                    from_paragraph=prev.from_paragraph,
                    to_paragraph=min(max_paragraph, max(prev.to_paragraph, block.to_paragraph)),
                    title=prev.title if len(prev.title) >= len(block.title) else block.title,
                    body=self._merge_body(prev.body, block.body),
                )
                continue
            merged.append(block)
        return merged

    def _merge_body(self, first: str, second: str) -> str:
        if second in first:
            return first
        if first in second:
            return second
        return f"{first}\n\n{second}"

    def _fallback_blocks(self, paragraphs: list[str]) -> list[CandidateBlock]:
        batches = self._paragraph_batches(paragraphs)
        fallback: list[CandidateBlock] = []
        for index, batch in enumerate(batches, start=1):
            text = " ".join(value for _, value in batch.rows).strip()
            fallback.append(
                CandidateBlock(
                    from_paragraph=batch.start_paragraph,
                    to_paragraph=batch.end_paragraph,
                    title=f"Section {index}",
                    body=text,
                )
            )
        return fallback

    def _normalize_final_text(self, text: str) -> str:
        lines = text.strip().splitlines()
        cleaned_lines = [
            line
            for line in lines
            if not re.match(r"^\s*##\s*Block\s+\d+\s*:\s*.*$", line)
        ]
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned

    def _render_blocks_text(self, blocks: list[CandidateBlock]) -> str:
        return "\n\n".join(block.body for block in blocks).strip()
