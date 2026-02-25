from __future__ import annotations

from pathlib import Path

from sruti.config import Settings
from sruti.domain.ports import ShellRunner


class FfmpegAdapter:
    def __init__(self, runner: ShellRunner, settings: Settings) -> None:
        self._runner = runner
        self._settings = settings

    def normalize(self, input_path: Path, output_path: Path) -> list[str]:
        command = [
            self._settings.ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "loudnorm",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        self._runner.run(command, timeout_seconds=self._settings.stage_timeout_seconds)
        return command

    def segment(self, input_path: Path, output_pattern: Path, *, seconds: int) -> list[str]:
        command = [
            self._settings.ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-f",
            "segment",
            "-segment_time",
            str(seconds),
            "-c",
            "copy",
            str(output_pattern),
        ]
        self._runner.run(command, timeout_seconds=self._settings.stage_timeout_seconds)
        return command
