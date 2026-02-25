from __future__ import annotations

from pathlib import Path

from sruti.config import load_settings


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
