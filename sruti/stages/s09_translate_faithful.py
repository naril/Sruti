from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.application.stages.s09_translate_faithful_uc import S09TranslateFaithfulUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.llm_factory import create_llm_client


def run_stage(
    *,
    context: StageContext,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    use_case = S09TranslateFaithfulUseCase(
        llm_client_factory=lambda: create_llm_client(context.settings),
        manifest_store=FileSystemManifestStore(),
        ask_user=ask_user,
    )
    return use_case.run(context)
