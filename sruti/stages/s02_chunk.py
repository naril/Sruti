from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.application.stages.s02_chunk_uc import S02ChunkUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.audio_ffmpeg import FfmpegAdapter
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.subprocess_runner import SubprocessShellRunner


def run_stage(
    *,
    context: StageContext,
    seconds: int,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    runner = SubprocessShellRunner()
    ffmpeg = FfmpegAdapter(runner, context.settings)
    manifest_store = FileSystemManifestStore()
    use_case = S02ChunkUseCase(
        seconds=seconds,
        ffmpeg=ffmpeg,
        manifest_store=manifest_store,
        ask_user=ask_user,
    )
    return use_case.run(context)
