from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplateSpec:
    stage_label: str
    allowed_placeholders: tuple[str, ...]
    recommended_placeholders: tuple[str, ...]


PROMPT_TEMPLATE_CATALOG: dict[str, PromptTemplateSpec] = {
    "s05_cleanup.txt": PromptTemplateSpec(
        stage_label="s05 ASR Cleanup",
        allowed_placeholders=("text",),
        recommended_placeholders=("text",),
    ),
    "s06_classification.txt": PromptTemplateSpec(
        stage_label="s06 Remove Nonlecture",
        allowed_placeholders=("span_lines",),
        recommended_placeholders=("span_lines",),
    ),
    "s06_repair_json.txt": PromptTemplateSpec(
        stage_label="s06 Remove Nonlecture (JSON repair)",
        allowed_placeholders=("original_prompt", "bad_response"),
        recommended_placeholders=("original_prompt", "bad_response"),
    ),
    "s07_editorial.txt": PromptTemplateSpec(
        stage_label="s07 Editorial",
        allowed_placeholders=("text",),
        recommended_placeholders=("text",),
    ),
    "s08_condense_map.txt": PromptTemplateSpec(
        stage_label="s08 Condense Map",
        allowed_placeholders=("paragraph_lines",),
        recommended_placeholders=("paragraph_lines",),
    ),
    "s08_condense_reduce.txt": PromptTemplateSpec(
        stage_label="s08 Condense Reduce",
        allowed_placeholders=("candidate_blocks_json",),
        recommended_placeholders=("candidate_blocks_json",),
    ),
    "s09_translate.txt": PromptTemplateSpec(
        stage_label="s09 Translate",
        allowed_placeholders=("text",),
        recommended_placeholders=("text",),
    ),
    "s10_czech_editorial.txt": PromptTemplateSpec(
        stage_label="s10 Czech Editorial",
        allowed_placeholders=("text",),
        recommended_placeholders=("text",),
    ),
}
