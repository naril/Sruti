from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s01_normalize_uc import S01NormalizeUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.audio_ffmpeg import FfmpegAdapter
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.subprocess_runner import SubprocessShellRunner


def run_stage(
    *,
    context: StageContext,
    input_audio: Path,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    runner = SubprocessShellRunner()
    ffmpeg = FfmpegAdapter(runner, context.settings)
    manifest_store = FileSystemManifestStore()
    use_case = S01NormalizeUseCase(
        input_audio=input_audio,
        ffmpeg=ffmpeg,
        manifest_store=manifest_store,
        ask_user=ask_user,
    )
    return use_case.run(context)
