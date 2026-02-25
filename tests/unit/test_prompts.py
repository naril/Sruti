from __future__ import annotations

from pathlib import Path

import pytest

from sruti.llm import prompts


def test_s05_prompt_renders_from_template() -> None:
    rendered = prompts.s05_cleanup_prompt("example text")
    assert "You are cleaning ASR output from an English lecture." in rendered
    assert rendered.endswith("Input:\nexample text\n")


def test_prompt_templates_can_be_overridden_with_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_dir = tmp_path / "templates"
    custom_dir.mkdir(parents=True, exist_ok=True)
    (custom_dir / "s05_cleanup.txt").write_text("CUSTOM: {{text}}", encoding="utf-8")
    monkeypatch.setenv("SRUTI_PROMPTS_DIR", str(custom_dir))

    rendered = prompts.s05_cleanup_prompt("abc")
    assert rendered == "CUSTOM: abc"


def test_prompt_missing_placeholder_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_dir = tmp_path / "templates"
    custom_dir.mkdir(parents=True, exist_ok=True)
    (custom_dir / "s05_cleanup.txt").write_text("Broken {{missing}}", encoding="utf-8")
    monkeypatch.setenv("SRUTI_PROMPTS_DIR", str(custom_dir))

    with pytest.raises(ValueError):
        prompts.s05_cleanup_prompt("abc")
