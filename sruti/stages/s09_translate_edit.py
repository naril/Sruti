from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.application.stages.s09_translate_edit_uc import S09TranslateEditUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.llm_ollama import OllamaClient


def run_stage(
    *,
    context: StageContext,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    use_case = S09TranslateEditUseCase(
        ollama=OllamaClient(),
        manifest_store=FileSystemManifestStore(),
        ask_user=ask_user,
    )
    return use_case.run(context)
