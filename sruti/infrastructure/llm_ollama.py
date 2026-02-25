from __future__ import annotations

from json import JSONDecodeError
from typing import Any
from urllib import error, request

from sruti.domain.errors import DependencyMissingError, StageExecutionError
from sruti.infrastructure import json_codec

DEFAULT_HTTP_TIMEOUT_SECONDS = 30


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self._base_url = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        return [item["name"] for item in models if isinstance(item, dict) and "name" in item]

    def ensure_model_available(self, model: str) -> None:
        available = self.list_models()
        if model not in available:
            raise DependencyMissingError(
                f"Ollama model '{model}' not found. Available: {', '.join(available) or '(none)'}"
            )

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        payload = self._request_json(
            "POST",
            "/api/generate",
            body=body,
            timeout_seconds=timeout_seconds,
        )
        response = payload.get("response")
        if not isinstance(response, str):
            raise StageExecutionError("Ollama response missing 'response' text.")
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json_codec.dumps(body).encode("utf-8")
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else DEFAULT_HTTP_TIMEOUT_SECONDS
        )
        req = request.Request(
            url=f"{self._base_url}{path}",
            method=method,
            headers={"Content-Type": "application/json"},
            data=data,
        )
        try:
            with request.urlopen(req, timeout=effective_timeout) as resp:  # nosec B310
                text = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise DependencyMissingError(
                "Unable to reach local Ollama API at http://127.0.0.1:11434."
            ) from exc
        try:
            parsed = json_codec.loads(text)
        except JSONDecodeError as exc:
            raise StageExecutionError(f"Invalid JSON from Ollama API: {text[:200]}") from exc
        if not isinstance(parsed, dict):
            raise StageExecutionError("Unexpected non-object JSON from Ollama API.")
        return parsed
