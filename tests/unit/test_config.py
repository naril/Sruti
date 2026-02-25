from __future__ import annotations

import tomllib
from pathlib import Path

from sruti.config import Settings, load_settings, render_default_pipeline_toml


def test_load_settings_ignores_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "pipeline.toml").write_text(
        """
chunk_seconds = 45
unknown = "value"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    assert settings.chunk_seconds == 45


def test_load_settings_supports_sruti_table_with_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "pipeline.toml").write_text(
        """
[sruti]
chunk_seconds = 15
noise = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    assert settings.chunk_seconds == 15


def test_load_settings_reads_prompt_templates_dir(tmp_path: Path) -> None:
    (tmp_path / "pipeline.toml").write_text(
        """
[sruti]
prompt_templates_dir = "prompts"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    assert settings.prompt_templates_dir == Path("prompts")


def test_load_settings_empty_prompt_templates_dir_maps_to_none(tmp_path: Path) -> None:
    (tmp_path / "pipeline.toml").write_text(
        """
[sruti]
prompt_templates_dir = ""
""".strip()
        + "\n",
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    assert settings.prompt_templates_dir is None


def test_render_default_pipeline_toml_contains_all_fields(tmp_path: Path) -> None:
    rendered = render_default_pipeline_toml()
    parsed = tomllib.loads(rendered)
    assert set(parsed["sruti"]) == set(Settings.model_fields)

    (tmp_path / "pipeline.toml").write_text(rendered, encoding="utf-8")
    settings = load_settings(tmp_path)
    assert settings == Settings()
