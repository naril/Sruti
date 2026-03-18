from __future__ import annotations

import types

import pytest

from sruti.domain.errors import ConfigurationError, StageExecutionError
from sruti.infrastructure.llm_openai import OpenAIClient


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, text: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.output_text = text
        self.usage = _Usage(input_tokens=input_tokens, output_tokens=output_tokens)


class _Responses:
    def __init__(self, calls: list[object]) -> None:
        self._calls = calls
        self.count = 0

    def create(self, **kwargs):
        _ = kwargs
        value = self._calls[self.count]
        self.count += 1
        if isinstance(value, Exception):
            raise value
        return value


class _SdkClient:
    def __init__(self, calls: list[object]) -> None:
        self.responses = _Responses(calls)


class _CaptureSdkClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = self
        self._attempt = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self._attempt += 1
        if self._attempt == 1:
            raise Exception(
                "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'temperature' is not "
                "supported with this model.\"}}"
            )
        return _Response("ok")


def test_openai_client_generate_parses_text_and_usage() -> None:
    sdk = _SdkClient([_Response("ok", input_tokens=12, output_tokens=8)])
    client = OpenAIClient(
        api_key_env="OPENAI_API_KEY",
        api_key="",
        base_url="",
        timeout_seconds=30,
        max_retries=1,
        client=sdk,
    )
    result = client.generate(model="gpt-5-mini", prompt="hello", temperature=0.2)
    assert result.text == "ok"
    assert result.usage_input_tokens == 12
    assert result.usage_output_tokens == 8


def test_openai_client_retries_on_retryable_error() -> None:
    retryable = Exception("temporary connection timeout")
    sdk = _SdkClient([retryable, _Response("ok")])
    client = OpenAIClient(
        api_key_env="OPENAI_API_KEY",
        api_key="",
        base_url="",
        timeout_seconds=30,
        max_retries=2,
        client=sdk,
    )
    result = client.generate(model="gpt-5-mini", prompt="hello", temperature=0.2)
    assert result.text == "ok"
    assert sdk.responses.count == 2


def test_openai_client_raises_after_retry_exhaustion() -> None:
    sdk = _SdkClient([Exception("timeout"), Exception("timeout")])
    client = OpenAIClient(
        api_key_env="OPENAI_API_KEY",
        api_key="",
        base_url="",
        timeout_seconds=30,
        max_retries=1,
        client=sdk,
    )
    with pytest.raises(StageExecutionError):
        client.generate(model="gpt-5-mini", prompt="hello", temperature=0.2)


def test_openai_client_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_OPENAI_KEY", raising=False)

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    with pytest.raises(ConfigurationError):
        OpenAIClient(
            api_key_env="MISSING_OPENAI_KEY",
            api_key="",
            base_url="",
            timeout_seconds=30,
            max_retries=0,
            client=None,
        )


def test_openai_client_uses_config_key_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_OPENAI_KEY", raising=False)
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    _ = OpenAIClient(
        api_key_env="MISSING_OPENAI_KEY",
        api_key="config-key",
        base_url="",
        timeout_seconds=30,
        max_retries=0,
        client=None,
    )
    assert captured["api_key"] == "config-key"


def test_openai_client_env_key_has_priority_over_config(monkeypatch) -> None:
    monkeypatch.setenv("PREFERRED_KEY", "env-key")
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    _ = OpenAIClient(
        api_key_env="PREFERRED_KEY",
        api_key="config-key",
        base_url="",
        timeout_seconds=30,
        max_retries=0,
        client=None,
    )
    assert captured["api_key"] == "env-key"


def test_openai_client_retries_without_temperature_when_model_rejects_it() -> None:
    sdk = _CaptureSdkClient()
    client = OpenAIClient(
        api_key_env="OPENAI_API_KEY",
        api_key="",
        base_url="",
        timeout_seconds=30,
        max_retries=1,
        client=sdk,
    )
    result = client.generate(model="gpt-5-nano", prompt="hello", temperature=0.1)
    assert result.text == "ok"
    assert len(sdk.calls) == 2
    assert "temperature" in sdk.calls[0]
    assert "temperature" not in sdk.calls[1]
