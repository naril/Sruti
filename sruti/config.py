from __future__ import annotations

import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sruti.domain.enums import LlmProvider


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
    llm_provider: LlmProvider = LlmProvider.LOCAL
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_timeout_seconds: int = 120
    openai_max_retries: int = 3
    openai_model_s05: str = "gpt-5-nano"
    openai_model_s06: str = "gpt-5-nano"
    openai_model_s07: str = "gpt-5-mini"
    openai_model_s08: str = "gpt-5-mini"
    openai_model_s09: str = "gpt-5-mini"
    cost_cap_usd: float = 2.0
    token_cap_input: int = 2_000_000
    token_cap_output: int = 1_000_000
    max_llm_calls_per_stage: int = 10_000
    openai_price_input_per_1m: float = 0.25
    openai_price_output_per_1m: float = 2.0
    llm_json_max_retries: int = 3
    prompt_templates_dir: Path | None = None
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
    if values.get("prompt_templates_dir") == "":
        values["prompt_templates_dir"] = None
    return Settings.model_validate(values)


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return _toml_literal(value.value)
    if isinstance(value, Path):
        return _toml_literal(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def render_default_pipeline_toml() -> str:
    defaults = Settings().model_dump(mode="python")
    lines = ["[sruti]"]
    for key in Settings.model_fields:
        lines.append(f"{key} = {_toml_literal(defaults[key])}")
    return "\n".join(lines) + "\n"
