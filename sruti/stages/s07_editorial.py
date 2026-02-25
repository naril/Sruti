from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.application.stages.s07_editorial_uc import S07EditorialUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.llm_ollama import OllamaClient


def run_stage(
    *,
    context: StageContext,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    use_case = S07EditorialUseCase(
        ollama=OllamaClient(),
        manifest_store=FileSystemManifestStore(),
        ask_user=ask_user,
    )
    return use_case.run(context)
