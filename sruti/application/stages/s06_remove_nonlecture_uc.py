from __future__ import annotations

from html import escape
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, field_validator

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.errors import InvalidLlmJsonError
from sruti.domain.models import LlmCallRecord, StageResult
from sruti.domain.ports import LlmClient, ManifestStore
from sruti.infrastructure.json_codec import loads
from sruti.llm.prompts import s06_classification_prompt, s06_repair_json_prompt
from sruti.llm.runtime import StageCostGuardrails, resolve_llm_model
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_text
from sruti.util.io import atomic_write_text, write_jsonl
from sruti.util.system import require_executable, require_file

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
JSON_CONTAINER_KEYS = ("decisions", "items", "results", "data")


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


class S06InvalidLlmJsonError(InvalidLlmJsonError):
    def __init__(self, message: str, *, call_rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.call_rows = call_rows


class S06RemoveNonLectureUseCase:
    stage_name = StageId.S06.value

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
            stage_id=StageId.S06,
            local_model_attr="s06_model",
        )

        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S06.value)
        s05_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S05.value)
        input_path = s05_dir / "cleaned_1.txt"
        require_file(input_path, label="s05 cleaned_1.txt")

        content_only_path = stage_dir / "content_only.txt"
        removed_spans_path = stage_dir / "removed_spans.jsonl"
        report_html_path = stage_dir / "removal_report.html"
        llm_log_path = stage_dir / "logs" / "model_calls.jsonl"
        decisions_path = stage_dir / "decisions_raw.json"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "llm_provider": context.settings.llm_provider.value,
            "model": model,
            "temperature": context.settings.s06_temperature,
            "llm_json_max_retries": context.settings.llm_json_max_retries,
            "prompt_templates_dir": (
                str(context.settings.prompt_templates_dir)
                if context.settings.prompt_templates_dir is not None
                else None
            ),
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S06,
            stage_dir=stage_dir,
            expected_outputs=[
                content_only_path,
                removed_spans_path,
                report_html_path,
                llm_log_path,
                decisions_path,
            ],
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
            spans = self._to_spans(source_text)
            if not spans:
                atomic_write_text(content_only_path, "")
                write_jsonl(removed_spans_path, [])
                atomic_write_text(report_html_path, self._removal_report_html([]))
                write_jsonl(llm_log_path, [])
                atomic_write_text(decisions_path, self._decisions_json([]))
                manifest.inputs = [manifest_util.artifact_for(input_path)]
                return runtime.mark_success(
                    manifest,
                    output_paths=[
                        content_only_path,
                        removed_spans_path,
                        report_html_path,
                        llm_log_path,
                        decisions_path,
                    ],
                )

            self._llm_client.ensure_model_available(model)
            guardrails = StageCostGuardrails(
                settings=context.settings,
                stage_id=StageId.S06,
                provider=context.settings.llm_provider,
                model=model,
            )
            try:
                decisions, call_rows = self._classify_spans(
                    spans,
                    context,
                    model=model,
                    guardrails=guardrails,
                )
            except InvalidLlmJsonError as exc:
                write_jsonl(llm_log_path, getattr(exc, "call_rows", []))
                raise
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
            sentence_rows = self._sentence_rows(spans, decisions)
            atomic_write_text(content_only_path, final_text)
            write_jsonl(removed_spans_path, removed_rows)
            atomic_write_text(report_html_path, self._removal_report_html(sentence_rows))
            write_jsonl(llm_log_path, call_rows)
            atomic_write_text(decisions_path, self._decisions_json(decisions))

            for call in call_rows:
                manifest.llm_calls.append(
                    LlmCallRecord(
                        model=call["model"],
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
                output_paths=[
                    content_only_path,
                    removed_spans_path,
                    report_html_path,
                    llm_log_path,
                    decisions_path,
                ],
            )
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise

    def _to_spans(self, text: str) -> list[dict[str, Any]]:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        return [{"span_id": idx, "text": para} for idx, para in enumerate(paragraphs, start=1)]

    def _classify_spans(
        self,
        spans: list[dict[str, Any]],
        context: StageContext,
        *,
        model: str,
        guardrails: StageCostGuardrails,
    ) -> tuple[list[SpanDecision], list[dict[str, Any]]]:
        all_decisions: list[SpanDecision] = []
        all_call_rows: list[dict[str, Any]] = []
        batches = self._split_span_batches(spans)
        prompts = [
            s06_classification_prompt(
                "\n".join(f"[{item['span_id']}] {item['text']}" for item in batch),
                template_dir=context.settings.prompt_templates_dir,
            )
            for batch in batches
        ]
        preflight = guardrails.preflight(prompts)
        context.emit_progress(
            f"[s06] preflight tokens in/out: {preflight['estimated_input_tokens']}/"
            f"{preflight['estimated_output_tokens']}, est. cost ${preflight['estimated_cost_usd']}",
            verbose_only=True,
        )
        for batch_index, batch in enumerate(batches, start=1):
            try:
                decisions, call_rows = self._classify_span_batch(
                    spans=batch,
                    context=context,
                    model=model,
                    prompt=prompts[batch_index - 1],
                    guardrails=guardrails,
                    batch_index=batch_index,
                    batch_count=len(batches),
                )
            except InvalidLlmJsonError as exc:
                failed_rows = getattr(exc, "call_rows", [])
                raise S06InvalidLlmJsonError(
                    str(exc),
                    call_rows=[*all_call_rows, *failed_rows],
                ) from exc
            all_call_rows.extend(call_rows)
            by_span_id = {item.span_id: item for item in decisions}
            for span in batch:
                span_id = int(span["span_id"])
                decision = by_span_id.get(span_id)
                if decision is None:
                    # Keep-by-default avoids accidental removals when the model omits a span.
                    decision = SpanDecision(span_id=span_id, action="KEEP")
                all_decisions.append(decision)
        return all_decisions, all_call_rows

    def _split_span_batches(
        self,
        spans: list[dict[str, Any]],
        *,
        max_batch_size: int = 40,
        max_batch_chars: int = 8_000,
    ) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        current_batch: list[dict[str, Any]] = []
        current_chars = 0

        for span in spans:
            span_line = f"[{span['span_id']}] {span['text']}"
            span_chars = len(span_line) + 1
            exceeds_size = len(current_batch) >= max_batch_size
            exceeds_chars = current_batch and (current_chars + span_chars > max_batch_chars)
            if exceeds_size or exceeds_chars:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(span)
            current_chars += span_chars

        if current_batch:
            batches.append(current_batch)
        return batches

    def _classify_span_batch(
        self,
        *,
        spans: list[dict[str, Any]],
        context: StageContext,
        model: str,
        prompt: str,
        guardrails: StageCostGuardrails,
        batch_index: int,
        batch_count: int,
    ) -> tuple[list[SpanDecision], list[dict[str, Any]]]:
        max_attempts = context.settings.llm_json_max_retries + 1
        call_rows: list[dict[str, Any]] = []
        last_response = ""
        context.emit_progress(
            f"[s06] classifying span batch ({len(spans)} spans) {batch_index}/{batch_count}",
            verbose_only=True,
        )

        for attempt in range(max_attempts):
            active_prompt = (
                prompt
                if attempt == 0
                else s06_repair_json_prompt(
                    prompt,
                    last_response,
                    template_dir=context.settings.prompt_templates_dir,
                )
            )
            if attempt > 0:
                context.emit_progress(
                    f"[s06] retry {attempt}/{max_attempts - 1} for batch {batch_index}/{batch_count}",
                    verbose_only=True,
                )
            prompt_hash = sha256_text(active_prompt)
            guardrails.before_call()
            response_obj = self._llm_client.generate(
                model=model,
                prompt=active_prompt,
                temperature=context.settings.s06_temperature,
                timeout_seconds=context.settings.stage_timeout_seconds,
            )
            response = response_obj.text.strip()
            est_input_tokens, est_output_tokens = guardrails.estimated_tokens_for_prompt(active_prompt)
            metrics = guardrails.record_call(
                estimated_input_tokens=est_input_tokens,
                estimated_output_tokens=est_output_tokens,
                usage_input_tokens=response_obj.usage_input_tokens,
                usage_output_tokens=response_obj.usage_output_tokens,
            )
            row = {
                "batch_index": batch_index,
                "batch_count": batch_count,
                "span_count": len(spans),
                "retry_index": attempt,
                "provider": self._llm_client.provider_name(),
                "model": model,
                "temperature": context.settings.s06_temperature,
                "prompt_hash": prompt_hash,
                "prompt": active_prompt,
                "response": response,
                "input_chars": len(active_prompt),
                "output_chars": len(response),
                "usage_input_tokens": metrics.input_tokens,
                "usage_output_tokens": metrics.output_tokens,
                "estimated_cost_usd": metrics.estimated_cost_usd,
                "cumulative_cost_usd": metrics.cumulative_cost_usd,
            }
            call_rows.append(row)
            try:
                decisions = self._parse_decisions(response)
                return decisions, call_rows
            except InvalidLlmJsonError as exc:
                row["parse_error"] = str(exc)
                last_response = response
                continue

        raise S06InvalidLlmJsonError(
            "s06 failed to produce valid JSON decisions after retries",
            call_rows=call_rows,
        )

    def _parse_decisions(self, response: str) -> list[SpanDecision]:
        payload = self._parse_json_payload(response)
        payload = self._unwrap_json_container(payload)
        if not isinstance(payload, list):
            raise InvalidLlmJsonError("LLM response is not a JSON array")

        out: list[SpanDecision] = []
        for row in payload:
            row = self._normalize_row(row)
            try:
                decision = SpanDecision.model_validate(row)
            except ValidationError as exc:
                raise InvalidLlmJsonError(f"Invalid decision object: {row}") from exc
            out.append(decision)
        return out

    def _parse_json_payload(self, response: str) -> Any:
        for candidate in self._json_candidates(response):
            try:
                return loads(candidate)
            except Exception:
                continue
        raise InvalidLlmJsonError("LLM response is not valid JSON")

    def _json_candidates(self, response: str) -> list[str]:
        text = response.strip()
        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(value: str | None) -> None:
            if value is None:
                return
            candidate = value.strip()
            if not candidate or candidate in seen:
                return
            seen.add(candidate)
            candidates.append(candidate)

        add_candidate(text)
        for block in JSON_FENCE_PATTERN.findall(text):
            add_candidate(block)
        add_candidate(self._extract_balanced_json(text, "[", "]"))
        add_candidate(self._extract_balanced_json(text, "{", "}"))
        return candidates

    def _unwrap_json_container(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            for key in JSON_CONTAINER_KEYS:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return payload

    def _normalize_row(self, row: Any) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise InvalidLlmJsonError(f"Decision row is not an object: {row}")

        normalized = dict(row)
        if "span_id" not in normalized:
            for key in ("id", "span", "spanId", "spanID"):
                if key in normalized:
                    normalized["span_id"] = normalized[key]
                    break
        if "action" not in normalized:
            for key in ("decision", "classification", "verdict"):
                if key in normalized:
                    normalized["action"] = normalized[key]
                    break

        return {
            key: normalized[key]
            for key in ("span_id", "action", "label", "reason")
            if key in normalized
        }

    def _extract_balanced_json(
        self, text: str, open_char: str, close_char: str
    ) -> str | None:
        start = text.find(open_char)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                    continue
                if char == open_char:
                    depth += 1
                    continue
                if char == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]

            start = text.find(open_char, start + 1)
        return None

    def _decisions_json(self, decisions: list[SpanDecision]) -> str:
        rows = [decision.model_dump() for decision in decisions]
        from sruti.infrastructure.json_codec import dumps

        return dumps(rows, indent=2) + "\n"

    def _sentence_rows(
        self, spans: list[dict[str, Any]], decisions: list[SpanDecision]
    ) -> list[dict[str, Any]]:
        decision_by_span = {decision.span_id: decision for decision in decisions}
        rows: list[dict[str, Any]] = []
        for span in spans:
            span_id = int(span["span_id"])
            decision = decision_by_span.get(span_id)
            action = decision.action if decision is not None else "KEEP"
            label = decision.label if decision is not None else None
            reason = decision.reason if decision is not None else None
            for sentence in self._split_sentences(str(span["text"])):
                rows.append(
                    {
                        "span_id": span_id,
                        "action": action,
                        "label": label,
                        "reason": reason,
                        "sentence": sentence,
                    }
                )
        return rows

    def _split_sentences(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        sentences = [part.strip() for part in parts if part.strip()]
        if sentences:
            return sentences
        return [stripped]

    def _removal_report_html(self, rows: list[dict[str, Any]]) -> str:
        keep_count = sum(1 for row in rows if row["action"] == "KEEP")
        remove_count = sum(1 for row in rows if row["action"] == "REMOVE")
        lines = [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            "  <title>s06 removal report</title>",
            "  <style>",
            "    body { font-family: sans-serif; margin: 24px; color: #111827; }",
            "    h1 { margin-bottom: 8px; }",
            "    .summary { margin-bottom: 16px; color: #374151; }",
            "    table { border-collapse: collapse; width: 100%; }",
            "    th, td { border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }",
            "    th { background: #f3f4f6; text-align: left; }",
            "    tr.status-KEEP td.status { background: #ecfdf5; color: #065f46; font-weight: 600; }",
            "    tr.status-REMOVE td.status { background: #fef2f2; color: #991b1b; font-weight: 600; }",
            "    .empty { color: #6b7280; font-style: italic; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>s06 removal report</h1>",
            f"  <p class=\"summary\">sentences: {len(rows)} | KEEP: {keep_count} | REMOVE: {remove_count}</p>",
        ]

        if not rows:
            lines.append("  <p class=\"empty\">No sentences available.</p>")
        else:
            lines.extend(
                [
                    "  <table>",
                    "    <thead>",
                    "      <tr>",
                    "        <th>#</th>",
                    "        <th>span_id</th>",
                    "        <th>status</th>",
                    "        <th>label</th>",
                    "        <th>reason</th>",
                    "        <th>sentence</th>",
                    "      </tr>",
                    "    </thead>",
                    "    <tbody>",
                ]
            )
            for idx, row in enumerate(rows, start=1):
                label = "" if row["label"] is None else str(row["label"])
                reason = "" if row["reason"] is None else str(row["reason"])
                lines.extend(
                    [
                        f"      <tr class=\"status-{row['action']}\">",
                        f"        <td>{idx}</td>",
                        f"        <td>{row['span_id']}</td>",
                        f"        <td class=\"status\">{escape(str(row['action']))}</td>",
                        f"        <td>{escape(label)}</td>",
                        f"        <td>{escape(reason)}</td>",
                        f"        <td>{escape(str(row['sentence']))}</td>",
                        "      </tr>",
                    ]
                )
            lines.extend(["    </tbody>", "  </table>"])

        lines.extend(["</body>", "</html>", ""])
        return "\n".join(lines)
