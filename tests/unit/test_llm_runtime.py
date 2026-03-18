from __future__ import annotations

import pytest

from sruti.config import Settings
from sruti.domain.enums import LlmProvider, StageId
from sruti.domain.errors import StageExecutionError
from sruti.llm.runtime import StageCostGuardrails, resolve_llm_model


def test_resolve_llm_model_uses_local_setting() -> None:
    settings = Settings(llm_provider=LlmProvider.LOCAL, s07_model="mistral:local")
    model = resolve_llm_model(settings, stage_id=StageId.S07, local_model_attr="s07_model")
    assert model == "mistral:local"


def test_resolve_llm_model_uses_openai_stage_mapping() -> None:
    settings = Settings(llm_provider=LlmProvider.OPENAI, openai_model_s07="gpt-5-mini")
    model = resolve_llm_model(settings, stage_id=StageId.S07, local_model_attr="s07_model")
    assert model == "gpt-5-mini"


def test_resolve_llm_model_supports_explicit_openai_attr_override() -> None:
    settings = Settings(
        llm_provider=LlmProvider.OPENAI,
        openai_model_s09="gpt-5-mini",
        openai_model_s08="gpt-5-nano",
    )
    model = resolve_llm_model(
        settings,
        stage_id=StageId.S09,
        local_model_attr="s09_model",
        openai_model_attr="openai_model_s08",
    )
    assert model == "gpt-5-nano"


def test_cost_guardrails_preflight_enforces_cost_cap() -> None:
    settings = Settings(
        llm_provider=LlmProvider.OPENAI,
        cost_cap_usd=0.000001,
        token_cap_input=10_000_000,
        token_cap_output=10_000_000,
    )
    guardrails = StageCostGuardrails(
        settings=settings,
        stage_id=StageId.S08,
        provider=LlmProvider.OPENAI,
        model="gpt-5-mini",
    )
    with pytest.raises(StageExecutionError):
        guardrails.preflight(["very long prompt " * 5000])


def test_cost_guardrails_runtime_enforces_output_token_cap() -> None:
    settings = Settings(
        llm_provider=LlmProvider.OPENAI,
        cost_cap_usd=100.0,
        token_cap_input=100_000,
        token_cap_output=10,
    )
    guardrails = StageCostGuardrails(
        settings=settings,
        stage_id=StageId.S05,
        provider=LlmProvider.OPENAI,
        model="gpt-5-nano",
    )
    with pytest.raises(StageExecutionError):
        guardrails.record_call(
            estimated_input_tokens=5,
            estimated_output_tokens=5,
            usage_input_tokens=5,
            usage_output_tokens=50,
        )
