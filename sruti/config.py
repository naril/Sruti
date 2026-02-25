from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
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
    default_whisper_model_path: Path = Path("./models/ggml-large-v3.bin")
