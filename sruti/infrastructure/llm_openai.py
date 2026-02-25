from __future__ import annotations

import os
import time
from typing import Any

from sruti.domain.errors import ConfigurationError, DependencyMissingError, StageExecutionError
from sruti.domain.models import LlmGenerateResult


class OpenAIClient:
    def __init__(
        self,
        *,
        api_key_env: str,
        base_url: str,
        timeout_seconds: int,
        max_retries: int,
        client: Any | None = None,
    ) -> None:
        self._api_key_env = api_key_env
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise DependencyMissingError(
                "OpenAI provider requires the 'openai' package. Install it with: pip install openai"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigurationError(
                f"Missing API key in environment variable '{api_key_env}' for OpenAI provider."
            )

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def provider_name(self) -> str:
        return "openai"

    def ensure_model_available(self, model: str) -> None:
        if not model.strip():
            raise ConfigurationError("OpenAI model name is empty.")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> LlmGenerateResult:
        self.ensure_model_available(model)
        effective_timeout = timeout_seconds if timeout_seconds is not None else self._timeout_seconds
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    timeout=effective_timeout,
                )
                text = self._response_text(response).strip()
                if not text:
                    raise StageExecutionError("OpenAI response did not include output text.")
                usage = getattr(response, "usage", None)
                return LlmGenerateResult(
                    text=text,
                    usage_input_tokens=self._usage_tokens(usage, "input_tokens"),
                    usage_output_tokens=self._usage_tokens(usage, "output_tokens"),
                )
            except Exception as exc:  # pragma: no cover - exact SDK exception classes vary by version
                last_error = exc
                if not self._is_retryable(exc) or attempt == self._max_retries:
                    break
                time.sleep(min(2**attempt, 8))

        detail = str(last_error) if last_error else "unknown OpenAI error"
        raise StageExecutionError(f"OpenAI request failed for model '{model}': {detail}")

    def _response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text

        # Fallback for SDK versions that expose only nested output blocks.
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "\n".join(chunks)

    def _usage_tokens(self, usage: Any, key: str) -> int | None:
        value = getattr(usage, key, None) if usage is not None else None
        if isinstance(value, int):
            return value
        return None

    def _is_retryable(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if status_code == 429 or 500 <= status_code < 600:
                return True
        message = str(exc).lower()
        return "timeout" in message or "temporar" in message or "connection" in message
