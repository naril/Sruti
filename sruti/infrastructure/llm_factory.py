from __future__ import annotations

from sruti.config import Settings
from sruti.domain.enums import LlmProvider
from sruti.domain.errors import ConfigurationError
from sruti.domain.ports import LlmClient
from sruti.infrastructure.llm_ollama import OllamaClient
from sruti.infrastructure.llm_openai import OpenAIClient


def create_llm_client(settings: Settings) -> LlmClient:
    if settings.llm_provider is LlmProvider.LOCAL:
        return OllamaClient()
    if settings.llm_provider is LlmProvider.OPENAI:
        return OpenAIClient(
            api_key_env=settings.openai_api_key_env,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
    raise ConfigurationError(f"Unsupported llm_provider '{settings.llm_provider}'.")
