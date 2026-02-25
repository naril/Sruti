from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sruti.domain.enums import StageId, StageStatus


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class FileArtifact(BaseModel):
    path: str
    sha256: str
    bytes: int

    model_config = {"extra": "forbid"}


class LlmCallRecord(BaseModel):
    model: str
    temperature: float
    prompt_hash: str
    input_chars: int
    output_chars: int
    retries: int = 0

    model_config = {"extra": "forbid"}


class StageManifest(BaseModel):
    stage: StageId
    status: StageStatus
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[FileArtifact] = Field(default_factory=list)
    outputs: list[FileArtifact] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    llm_calls: list[LlmCallRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False

    model_config = {"extra": "forbid"}


class RunManifest(BaseModel):
    run_id: str
    run_dir: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    settings: dict[str, Any] = Field(default_factory=dict)
    requested_range: dict[str, str] = Field(default_factory=dict)
    stages: dict[str, StageStatus] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class StageResult(BaseModel):
    stage: StageId
    status: StageStatus
    stage_dir: Path
    outputs: list[Path] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}
