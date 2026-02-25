from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global defaults used by CLI and stages."""

    chunk_seconds: int = 30
    source_language: str = "en"
    whisper_beam_size: int = 5
    s05_model: str = "llama3.1:8b"
    s05_temperature: float = 0.1
    s06_model: str = "llama3.1:8b"
    s06_temperature: float = 0.1
    s07_model: str = "mistral:7b-instruct"
    s07_temperature: float = 0.2
    s08_model: str = "llama3.1:8b"
    s08_temperature: float = 0.1
    s09_model: str = "mistral:7b-instruct"
    s09_temperature: float = 0.2
    llm_json_max_retries: int = 3
    ffmpeg_bin: str = "ffmpeg"
    whisper_cli_bin: str = "whisper-cli"
    ollama_bin: str = "ollama"
    default_whisper_model_path: Path = Field(
        default=Path("./models/ggml-large-v3.bin"),
        description="Path to whisper.cpp large-v3 model.",
    )
    stage_timeout_seconds: int = 3600

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }


def load_settings(run_dir: Path | None = None) -> Settings:
    """Load settings with optional run-local override from RUN_DIR/pipeline.toml."""

    if run_dir is None:
        return Settings()

    config_path = run_dir / "pipeline.toml"
    if not config_path.exists():
        return Settings()

    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    # Support either root keys or [sruti] table.
    values = raw.get("sruti", raw)
    if not isinstance(values, dict):
        return Settings()

    # Ignore unrelated keys so run-local config can coexist with other tooling.
    known_keys = set(Settings.model_fields)
    values = {key: value for key, value in values.items() if key in known_keys}
    return Settings.model_validate(values)
