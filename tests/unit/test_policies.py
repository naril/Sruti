from __future__ import annotations

import pytest

from sruti.domain.enums import OnExistsMode, StageId
from sruti.domain.errors import ExistingOutputError, NonInteractivePromptError
from sruti.domain.policies import resolve_existing_output_policy, stage_ids_in_range


def test_stage_ids_in_range_returns_inclusive_range() -> None:
    result = stage_ids_in_range(StageId.S03, StageId.S05)
    assert result == [StageId.S03, StageId.S04, StageId.S05]


def test_stage_ids_in_range_rejects_reversed_range() -> None:
    with pytest.raises(ValueError):
        stage_ids_in_range(StageId.S05, StageId.S03)


def test_existing_output_policy_overwrite_proceeds() -> None:
    decision = resolve_existing_output_policy(
        mode=OnExistsMode.OVERWRITE,
        is_tty=False,
        stage_label="s01",
        outputs_exist=True,
    )
    assert decision == "proceed"


def test_existing_output_policy_fail_raises() -> None:
    with pytest.raises(ExistingOutputError):
        resolve_existing_output_policy(
            mode=OnExistsMode.FAIL,
            is_tty=True,
            stage_label="s02",
            outputs_exist=True,
        )


def test_existing_output_policy_ask_noninteractive_raises() -> None:
    with pytest.raises(NonInteractivePromptError):
        resolve_existing_output_policy(
            mode=OnExistsMode.ASK,
            is_tty=False,
            stage_label="s03",
            outputs_exist=True,
        )
