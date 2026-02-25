from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sruti.application.context import StageContext
from sruti.domain.enums import StageId, StageStatus
from sruti.domain.models import StageManifest, StageResult
from sruti.domain.policies import any_paths_exist, resolve_existing_output_policy
from sruti.domain.ports import ManifestStore
from sruti.util import manifest as manifest_util
from sruti.util.io import ensure_dir


@dataclass(slots=True)
class StageRuntime:
    context: StageContext
    stage_id: StageId
    stage_dir: Path
    expected_outputs: list[Path]
    manifest_store: ManifestStore
    ask_user: Callable[[str], bool] | None = None

    def initialize_manifest(self, *, params: dict[str, object]) -> StageManifest:
        return StageManifest(
            stage=self.stage_id,
            status=StageStatus.PENDING,
            params=dict(params),
            dry_run=self.context.dry_run,
        )

    def should_skip(self, *, params: dict[str, object], inputs_signature: str) -> bool:
        if self.context.force:
            return False
        existing = self.manifest_store.load_stage_manifest(self.stage_dir)
        if existing is None:
            return False
        if existing.status is not StageStatus.SUCCESS:
            return False
        same_params = manifest_util.params_signature(existing.params) == manifest_util.params_signature(
            params
        )
        previous_signature = existing.params.get("_inputs_signature")
        return same_params and previous_signature == inputs_signature

    def apply_on_exists_policy(self) -> str:
        decision = resolve_existing_output_policy(
            mode=self.context.on_exists,
            is_tty=self.context.is_tty,
            stage_label=self.stage_id.value,
            outputs_exist=any_paths_exist(self.expected_outputs),
            ask_user=self.ask_user,
        )
        return decision

    def start(self, manifest: StageManifest) -> None:
        ensure_dir(self.stage_dir)
        manifest.status = StageStatus.RUNNING
        self.manifest_store.save_stage_manifest(self.stage_dir, manifest)

    def mark_dry_run(self, manifest: StageManifest) -> StageResult:
        manifest.status = StageStatus.DRY_RUN
        self.manifest_store.save_stage_manifest(self.stage_dir, manifest)
        return StageResult(stage=self.stage_id, status=StageStatus.DRY_RUN, stage_dir=self.stage_dir)

    def mark_skipped(self, manifest: StageManifest) -> StageResult:
        manifest.status = StageStatus.SKIPPED
        self.manifest_store.save_stage_manifest(self.stage_dir, manifest)
        return StageResult(stage=self.stage_id, status=StageStatus.SKIPPED, stage_dir=self.stage_dir)

    def mark_success(
        self,
        manifest: StageManifest,
        *,
        output_paths: list[Path],
    ) -> StageResult:
        manifest.status = StageStatus.SUCCESS
        manifest.outputs = manifest_util.artifacts_for_existing(output_paths)
        self.manifest_store.save_stage_manifest(self.stage_dir, manifest)
        return StageResult(
            stage=self.stage_id,
            status=StageStatus.SUCCESS,
            stage_dir=self.stage_dir,
            outputs=output_paths,
        )

    def mark_failure(self, manifest: StageManifest, error_message: str) -> None:
        manifest.status = StageStatus.FAILED
        manifest.errors.append(error_message)
        self.manifest_store.save_stage_manifest(self.stage_dir, manifest)
