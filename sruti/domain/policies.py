from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sruti.domain.enums import OnExistsMode, StageId
from sruti.domain.errors import ExistingOutputError, NonInteractivePromptError

STAGE_ORDER: list[StageId] = [
    StageId.S01,
    StageId.S02,
    StageId.S03,
    StageId.S04,
    StageId.S05,
    StageId.S06,
    StageId.S07,
    StageId.S08,
    StageId.S09,
    StageId.S10,
]


def stage_ids_in_range(start: StageId, end: StageId) -> list[StageId]:
    i = STAGE_ORDER.index(start)
    j = STAGE_ORDER.index(end)
    if i > j:
        raise ValueError(f"Invalid stage range: {start.value} > {end.value}")
    return STAGE_ORDER[i : j + 1]


def resolve_existing_output_policy(
    *,
    mode: OnExistsMode,
    is_tty: bool,
    stage_label: str,
    outputs_exist: bool,
    ask_user: Callable[[str], bool] | None = None,
) -> str:
    """
    Return one of: proceed, skip.
    Raise ExistingOutputError when policy blocks execution.
    """
    if not outputs_exist:
        return "proceed"

    if mode is OnExistsMode.OVERWRITE:
        return "proceed"
    if mode is OnExistsMode.SKIP:
        return "skip"
    if mode is OnExistsMode.FAIL:
        raise ExistingOutputError(
            f"Outputs already exist for {stage_label}; use --on-exists overwrite/skip or --force."
        )

    if not is_tty:
        raise NonInteractivePromptError(
            f"{stage_label}: --on-exists ask is not allowed in non-interactive mode."
        )
    if ask_user is None:
        raise NonInteractivePromptError(
            f"{stage_label}: interactive prompt callback missing for ask mode."
        )

    should_overwrite = ask_user(f"{stage_label}: outputs exist. Overwrite?")
    return "proceed" if should_overwrite else "skip"


def any_paths_exist(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)
