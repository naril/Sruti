from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.application.stages.s08_translate_faithful_uc import S08TranslateFaithfulUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.infrastructure.llm_factory import create_llm_client


def run_stage(
    *,
    context: StageContext,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    use_case = S08TranslateFaithfulUseCase(
        llm_client=create_llm_client(context.settings),
        manifest_store=FileSystemManifestStore(),
        ask_user=ask_user,
    )
    return use_case.run(context)
