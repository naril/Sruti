from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, field_validator

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import StageId
from sruti.domain.errors import InvalidLlmJsonError
from sruti.domain.models import LlmCallRecord, StageResult
from sruti.domain.ports import ManifestStore
from sruti.infrastructure.json_codec import loads
from sruti.infrastructure.llm_ollama import OllamaClient
from sruti.llm.prompts import s06_classification_prompt, s06_repair_json_prompt
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file


class SpanDecision(BaseModel):
    span_id: int
    action: Literal["KEEP", "REMOVE"]
    label: str | None = None
    reason: str | None = None

    model_config = {"extra": "forbid"}

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("action must be a string")
        normalized = value.strip().upper()
        if normalized not in {"KEEP", "REMOVE"}:
            raise ValueError("action must be KEEP or REMOVE")
        return normalized


class S06RemoveNonLectureUseCase:
    stage_name = StageId.S06.value

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

        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S06.value)
        s05_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S05.value)
        input_path = s05_dir / "cleaned_1.txt"
        require_file(input_path, label="s05 cleaned_1.txt")

        content_only_path = stage_dir / "content_only.txt"
        removed_spans_path = stage_dir / "removed_spans.jsonl"
        llm_log_path = stage_dir / "logs" / "model_calls.jsonl"
        decisions_path = stage_dir / "decisions_raw.json"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "model": context.settings.s06_model,
            "temperature": context.settings.s06_temperature,
            "llm_json_max_retries": context.settings.llm_json_max_retries,
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S06,
            stage_dir=stage_dir,
            expected_outputs=[content_only_path, removed_spans_path, llm_log_path, decisions_path],
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
            manifest.tool_versions["ollama_model"] = context.settings.s06_model
            source_text = input_path.read_text(encoding="utf-8")
            spans = self._to_spans(source_text)
            if not spans:
                atomic_write_text(content_only_path, "")
                write_jsonl(removed_spans_path, [])
                write_jsonl(llm_log_path, [])
                atomic_write_text(decisions_path, self._decisions_json([]))
                manifest.inputs = [manifest_util.artifact_for(input_path)]
                return runtime.mark_success(
                    manifest,
                    output_paths=[
                        content_only_path,
                        removed_spans_path,
                        llm_log_path,
                        decisions_path,
                    ],
                )

            self._ollama.ensure_model_available(context.settings.s06_model)
            decisions, call_rows = self._classify_spans(spans, context)
            remove_ids = {item.span_id for item in decisions if item.action == "REMOVE"}

            kept_lines: list[str] = []
            removed_rows: list[dict[str, Any]] = []
            for span in spans:
                if span["span_id"] in remove_ids:
                    decision = next(
                        (item for item in decisions if item.span_id == span["span_id"]),
                        None,
                    )
                    removed_rows.append(
                        {
                            "span_id": span["span_id"],
                            "label": decision.label if decision else None,
                            "reason": decision.reason if decision else None,
                            "text": span["text"],
                        }
                    )
                else:
                    kept_lines.append(span["text"])

            final_text = "\n\n".join(kept_lines).strip()
            if final_text:
                final_text += "\n"
            atomic_write_text(content_only_path, final_text)
            write_jsonl(removed_spans_path, removed_rows)
            write_jsonl(llm_log_path, call_rows)
            atomic_write_text(decisions_path, self._decisions_json(decisions))

            for call in call_rows:
                manifest.llm_calls.append(
                    LlmCallRecord(
                        model=context.settings.s06_model,
                        temperature=context.settings.s06_temperature,
                        prompt_hash=call["prompt_hash"],
                        input_chars=call["input_chars"],
                        output_chars=call["output_chars"],
                        retries=call["retry_index"],
                    )
                )
            manifest.inputs = [manifest_util.artifact_for(input_path)]
            return runtime.mark_success(
                manifest,
                output_paths=[content_only_path, removed_spans_path, llm_log_path, decisions_path],
            )
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise

    def _to_spans(self, text: str) -> list[dict[str, Any]]:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        return [{"span_id": idx, "text": para} for idx, para in enumerate(paragraphs, start=1)]

    def _classify_spans(
        self, spans: list[dict[str, Any]], context: StageContext
    ) -> tuple[list[SpanDecision], list[dict[str, Any]]]:
        span_lines = "\n".join(f"[{item['span_id']}] {item['text']}" for item in spans)
        prompt = s06_classification_prompt(span_lines)
        max_attempts = context.settings.llm_json_max_retries + 1
        call_rows: list[dict[str, Any]] = []
        last_response = ""

        for attempt in range(max_attempts):
            active_prompt = (
                prompt if attempt == 0 else s06_repair_json_prompt(prompt, last_response)
            )
            prompt_hash = sha256_text(active_prompt)
            response = self._ollama.generate(
                model=context.settings.s06_model,
                prompt=active_prompt,
                temperature=context.settings.s06_temperature,
                timeout_seconds=context.settings.stage_timeout_seconds,
            ).strip()
            call_rows.append(
                {
                    "retry_index": attempt,
                    "model": context.settings.s06_model,
                    "temperature": context.settings.s06_temperature,
                    "prompt_hash": prompt_hash,
                    "prompt": active_prompt,
                    "response": response,
                    "input_chars": len(active_prompt),
                    "output_chars": len(response),
                }
            )
            try:
                decisions = self._parse_decisions(response)
                return decisions, call_rows
            except InvalidLlmJsonError:
                last_response = response
                continue

        raise InvalidLlmJsonError("s06 failed to produce valid JSON decisions after retries")

    def _parse_decisions(self, response: str) -> list[SpanDecision]:
        try:
            payload = loads(response)
        except Exception as exc:
            raise InvalidLlmJsonError("LLM response is not valid JSON") from exc
        if not isinstance(payload, list):
            raise InvalidLlmJsonError("LLM response is not a JSON array")

        out: list[SpanDecision] = []
        for row in payload:
            try:
                decision = SpanDecision.model_validate(row)
            except ValidationError as exc:
                raise InvalidLlmJsonError(f"Invalid decision object: {row}") from exc
            out.append(decision)
        return out

    def _decisions_json(self, decisions: list[SpanDecision]) -> str:
        rows = [decision.model_dump() for decision in decisions]
        from sruti.infrastructure.json_codec import dumps

        return dumps(rows, indent=2) + "\n"
