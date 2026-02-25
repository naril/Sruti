from __future__ import annotations

from sruti.config import Settings
from sruti.domain.enums import LlmProvider
from sruti.infrastructure.llm_factory import create_llm_client
from sruti.infrastructure.llm_ollama import OllamaClient


def test_llm_factory_returns_ollama_for_local_provider() -> None:
    client = create_llm_client(Settings(llm_provider=LlmProvider.LOCAL))
    assert isinstance(client, OllamaClient)


def test_llm_factory_builds_openai_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAIClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("sruti.infrastructure.llm_factory.OpenAIClient", FakeOpenAIClient)
    settings = Settings(
        llm_provider=LlmProvider.OPENAI,
        openai_api_key_env="OPENAI_API_KEY",
        openai_base_url="https://example.test",
        openai_timeout_seconds=30,
        openai_max_retries=4,
    )
    client = create_llm_client(settings)
    assert isinstance(client, FakeOpenAIClient)
    assert captured["api_key_env"] == "OPENAI_API_KEY"
    assert captured["base_url"] == "https://example.test"
    assert captured["timeout_seconds"] == 30
    assert captured["max_retries"] == 4
