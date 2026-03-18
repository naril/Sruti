from __future__ import annotations

from collections.abc import Callable

from sruti.application.context import StageContext
from sruti.domain.models import StageResult
from sruti.stages.s09_translate_faithful import run_stage as _run_stage_v2


def run_stage(
    *,
    context: StageContext,
    ask_user: Callable[[str], bool] | None = None,
) -> StageResult:
    """Backward-compatible alias for pre-v2 imports."""

    return _run_stage_v2(context=context, ask_user=ask_user)
