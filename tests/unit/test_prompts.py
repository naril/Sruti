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


def test_prompt_template_dir_has_priority_over_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_dir = tmp_path / "env_templates"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "s05_cleanup.txt").write_text("ENV: {{text}}", encoding="utf-8")
    monkeypatch.setenv("SRUTI_PROMPTS_DIR", str(env_dir))

    config_dir = tmp_path / "config_templates"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "s05_cleanup.txt").write_text("CFG: {{text}}", encoding="utf-8")

    rendered = prompts.s05_cleanup_prompt("abc", template_dir=config_dir)
    assert rendered == "CFG: abc"


def test_prompt_template_dir_falls_back_to_builtin_when_file_missing(tmp_path: Path) -> None:
    config_dir = tmp_path / "config_templates"
    config_dir.mkdir(parents=True, exist_ok=True)

    rendered = prompts.s05_cleanup_prompt("abc", template_dir=config_dir)
    assert "You are cleaning ASR output from an English lecture." in rendered
    assert rendered.endswith("Input:\nabc\n")


def test_prompt_template_dir_missing_directory_fails(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        prompts.s05_cleanup_prompt("abc", template_dir=missing_dir)
