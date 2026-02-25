from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s03_asr_whisper_uc import S03AsrWhisperUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.asr_whisper_cli import WhisperCliAdapter
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.subprocess_runner import SubprocessShellRunner


def run_stage(
    *,
    context: StageContext,
    model_path: Path,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    runner = SubprocessShellRunner()
    whisper = WhisperCliAdapter(runner, context.settings)
    manifest_store = FileSystemManifestStore()
    use_case = S03AsrWhisperUseCase(
        whisper_model_path=model_path,
        whisper=whisper,
        manifest_store=manifest_store,
        ask_user=ask_user,
    )
    return use_case.run(context)
