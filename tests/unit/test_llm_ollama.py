from __future__ import annotations

from sruti.infrastructure.llm_ollama import DEFAULT_HTTP_TIMEOUT_SECONDS, OllamaClient


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)

    def read(self) -> bytes:
        return self._body.encode("utf-8")


def test_list_models_uses_default_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        _ = req
        captured["timeout"] = timeout
        return _FakeResponse('{"models": []}')

    monkeypatch.setattr("sruti.infrastructure.llm_ollama.request.urlopen", fake_urlopen)
    client = OllamaClient()
    assert client.list_models() == []
    assert captured["timeout"] == DEFAULT_HTTP_TIMEOUT_SECONDS


def test_generate_respects_explicit_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        _ = req
        captured["timeout"] = timeout
        return _FakeResponse('{"response": "ok"}')

    monkeypatch.setattr("sruti.infrastructure.llm_ollama.request.urlopen", fake_urlopen)
    client = OllamaClient()
    value = client.generate(model="m", prompt="p", temperature=0.1, timeout_seconds=5)
    assert value == "ok"
    assert captured["timeout"] == 5
