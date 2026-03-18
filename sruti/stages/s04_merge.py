from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.application.stages.s04_merge_uc import S04MergeUseCase
from sruti.domain.models import StageResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore


def run_stage(
    *,
    context: StageContext,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    use_case = S04MergeUseCase(
        manifest_store=FileSystemManifestStore(),
        ask_user=ask_user,
    )
    return use_case.run(context)
