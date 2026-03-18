from __future__ import annotations

from pathlib import Path

from sruti.config import Settings
from sruti.domain.ports import ShellRunner


class WhisperCliAdapter:
    def __init__(self, runner: ShellRunner, settings: Settings) -> None:
        self._runner = runner
        self._settings = settings

    def transcribe_chunk(
        self,
        *,
        model_path: Path,
        chunk_path: Path,
        output_prefix: Path,
    ) -> list[str]:
        command = [
            self._settings.whisper_cli_bin,
            "-m",
            str(model_path),
            "-f",
            str(chunk_path),
            "-otxt",
            "-osrt",
            "-of",
            str(output_prefix),
            "-l",
            self._settings.source_language,
            "--beam-size",
            str(self._settings.whisper_beam_size),
        ]
        self._runner.run(command, timeout_seconds=self._settings.stage_timeout_seconds)
        return command
