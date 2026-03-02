from __future__ import annotations

import math
from dataclasses import dataclass

from sruti.config import Settings
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.errors import ConfigurationError, StageExecutionError

MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.2": (1.75, 14.00),
}

STAGE_OUTPUT_MULTIPLIER: dict[StageId, float] = {
    StageId.S05: 0.95,
    StageId.S06: 0.35,
    StageId.S07: 0.80,
    StageId.S08: 0.70,
    StageId.S09: 1.05,
    StageId.S10: 0.90,
}


def resolve_llm_model(
    settings: Settings,
    *,
    stage_id: StageId,
    local_model_attr: str,
    openai_model_attr: str | None = None,
) -> str:
    if settings.llm_provider is LlmProvider.OPENAI:
        openai_attr = openai_model_attr if openai_model_attr is not None else f"openai_model_{stage_id.value}"
        value = getattr(settings, openai_attr)
    else:
        value = getattr(settings, local_model_attr)
    model = str(value).strip()
    if not model:
        raise ConfigurationError(f"Resolved empty model name for stage {stage_id.value}.")
    return model


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    # Conservative ~4 chars/token heuristic for guardrails.
    return max(1, math.ceil(len(stripped) / 4))


@dataclass(slots=True)
class CostMetrics:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    cumulative_cost_usd: float


class StageCostGuardrails:
    def __init__(
        self,
        *,
        settings: Settings,
        stage_id: StageId,
        provider: LlmProvider,
        model: str,
    ) -> None:
        self._settings = settings
        self._stage_id = stage_id
        self._provider = provider
        self._model = model
        self._input_price_per_1m, self._output_price_per_1m = self._resolve_model_prices()
        self._calls = 0
        self._cum_input = 0
        self._cum_output = 0
        self._cum_cost = 0.0

    def preflight(self, prompts: list[str]) -> dict[str, float | int]:
        est_input = sum(estimate_tokens(prompt) for prompt in prompts)
        stage_multiplier = STAGE_OUTPUT_MULTIPLIER.get(self._stage_id, 0.8)
        est_output = sum(
            max(1, math.ceil(estimate_tokens(prompt) * stage_multiplier))
            for prompt in prompts
            if prompt.strip()
        )
        est_cost = self._cost_usd(est_input, est_output)
        self._enforce_caps(
            input_tokens=est_input,
            output_tokens=est_output,
            cost_usd=est_cost,
            phase="preflight",
        )
        return {
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "estimated_cost_usd": round(est_cost, 6),
        }

    def estimated_tokens_for_prompt(self, prompt: str) -> tuple[int, int]:
        input_tokens = estimate_tokens(prompt)
        stage_multiplier = STAGE_OUTPUT_MULTIPLIER.get(self._stage_id, 0.8)
        output_tokens = max(1, math.ceil(input_tokens * stage_multiplier)) if input_tokens else 0
        return input_tokens, output_tokens

    def before_call(self) -> None:
        if self._provider is not LlmProvider.OPENAI:
            return
        if self._calls >= self._settings.max_llm_calls_per_stage:
            raise StageExecutionError(
                f"{self._stage_id.value}: exceeded max_llm_calls_per_stage="
                f"{self._settings.max_llm_calls_per_stage}"
            )

    def record_call(
        self,
        *,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        usage_input_tokens: int | None,
        usage_output_tokens: int | None,
    ) -> CostMetrics:
        self._calls += 1
        input_tokens = usage_input_tokens if usage_input_tokens is not None else estimated_input_tokens
        output_tokens = usage_output_tokens if usage_output_tokens is not None else estimated_output_tokens
        call_cost = self._cost_usd(input_tokens, output_tokens)
        self._cum_input += input_tokens
        self._cum_output += output_tokens
        self._cum_cost += call_cost
        self._enforce_caps(
            input_tokens=self._cum_input,
            output_tokens=self._cum_output,
            cost_usd=self._cum_cost,
            phase="runtime",
        )
        return CostMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(call_cost, 6),
            cumulative_cost_usd=round(self._cum_cost, 6),
        )

    def _resolve_model_prices(self) -> tuple[float, float]:
        model_lower = self._model.lower()
        for model_prefix, prices in MODEL_PRICING_PER_1M.items():
            if model_lower.startswith(model_prefix):
                return prices
        return (self._settings.openai_price_input_per_1m, self._settings.openai_price_output_per_1m)

    def _cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        if self._provider is not LlmProvider.OPENAI:
            return 0.0
        return (
            (input_tokens / 1_000_000) * self._input_price_per_1m
            + (output_tokens / 1_000_000) * self._output_price_per_1m
        )

    def _enforce_caps(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        phase: str,
    ) -> None:
        if self._provider is not LlmProvider.OPENAI:
            return
        if input_tokens > self._settings.token_cap_input:
            raise StageExecutionError(
                f"{self._stage_id.value}: {phase} input tokens {input_tokens} exceed token_cap_input "
                f"{self._settings.token_cap_input}"
            )
        if output_tokens > self._settings.token_cap_output:
            raise StageExecutionError(
                f"{self._stage_id.value}: {phase} output tokens {output_tokens} exceed token_cap_output "
                f"{self._settings.token_cap_output}"
            )
        if cost_usd > self._settings.cost_cap_usd:
            raise StageExecutionError(
                f"{self._stage_id.value}: {phase} estimated cost ${cost_usd:.6f} exceeds cost_cap_usd "
                f"${self._settings.cost_cap_usd:.6f}"
            )
