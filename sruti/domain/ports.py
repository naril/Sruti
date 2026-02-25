from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sruti.application.context import StageContext
from sruti.domain.models import StageManifest, StageResult


class StageUseCase(Protocol):
    stage_name: str

    def run(self, context: StageContext) -> StageResult:
        ...


class ManifestStore(Protocol):
    def load_stage_manifest(self, stage_dir: Path) -> StageManifest | None:
        ...

    def save_stage_manifest(self, stage_dir: Path, manifest: StageManifest) -> None:
        ...


class ShellRunner(Protocol):
    def run(self, command: list[str], *, cwd: Path | None = None, timeout_seconds: int | None = None) -> str:
        ...


class LlmClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> str:
        ...
