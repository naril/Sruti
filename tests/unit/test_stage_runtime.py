from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import OnExistsMode, StageId, StageStatus
from sruti.domain.models import StageManifest
from sruti.domain.ports import ManifestStore


class MemoryManifestStore(ManifestStore):
    def __init__(self) -> None:
        self.saved: list[StageManifest] = []

    def load_stage_manifest(self, stage_dir: Path) -> StageManifest | None:
        _ = stage_dir
        return None

    def save_stage_manifest(self, stage_dir: Path, manifest: StageManifest) -> None:
        _ = stage_dir
        self.saved.append(manifest.model_copy(deep=True))


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_stage_runtime_sets_finished_at_only_on_terminal_status(tmp_path: Path) -> None:
    store = MemoryManifestStore()
    runtime = StageRuntime(
        context=_ctx(tmp_path),
        stage_id=StageId.S01,
        stage_dir=tmp_path / "s01_normalize",
        expected_outputs=[],
        manifest_store=store,
    )
    manifest = runtime.initialize_manifest(params={})
    runtime.start(manifest)
    assert store.saved[-1].status == StageStatus.RUNNING
    assert store.saved[-1].finished_at is None

    runtime.mark_success(manifest, output_paths=[])
    assert store.saved[-1].status == StageStatus.SUCCESS
    assert store.saved[-1].finished_at is not None
