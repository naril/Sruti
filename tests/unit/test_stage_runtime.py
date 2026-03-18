from __future__ import annotations

from collections.abc import Callable
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


class ResumeManifestStore(ManifestStore):
    def __init__(self, manifest: StageManifest | None) -> None:
        self._manifest = manifest

    def load_stage_manifest(self, stage_dir: Path) -> StageManifest | None:
        _ = stage_dir
        return self._manifest

    def save_stage_manifest(self, stage_dir: Path, manifest: StageManifest) -> None:
        _ = stage_dir
        self._manifest = manifest


def _ctx(run_dir: Path, *, emitter: Callable[[str], None] | None = None) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
        progress_emitter=emitter,
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


def test_stage_runtime_should_skip_only_for_matching_success_manifest(tmp_path: Path) -> None:
    existing = StageManifest(
        stage=StageId.S01,
        status=StageStatus.SUCCESS,
        params={"flag": 1, "_inputs_signature": "abc"},
    )
    runtime = StageRuntime(
        context=_ctx(tmp_path),
        stage_id=StageId.S01,
        stage_dir=tmp_path / "s01_normalize",
        expected_outputs=[],
        manifest_store=ResumeManifestStore(existing),
    )
    assert runtime.should_skip(params={"flag": 1, "_inputs_signature": "abc"}, inputs_signature="abc")
    assert not runtime.should_skip(
        params={"flag": 2, "_inputs_signature": "abc"}, inputs_signature="abc"
    )
    assert not runtime.should_skip(
        params={"flag": 1, "_inputs_signature": "abc"}, inputs_signature="changed"
    )


def test_stage_runtime_emits_start_and_terminal_progress(tmp_path: Path) -> None:
    events: list[str] = []
    store = MemoryManifestStore()
    runtime = StageRuntime(
        context=_ctx(tmp_path, emitter=events.append),
        stage_id=StageId.S02,
        stage_dir=tmp_path / "s02_chunk",
        expected_outputs=[],
        manifest_store=store,
    )
    manifest = runtime.initialize_manifest(params={})
    runtime.start(manifest)
    runtime.mark_success(manifest, output_paths=[])
    assert events[0] == "[s02] started"
    assert events[1].startswith("[s02] success (duration:")


def test_stage_runtime_skip_and_dry_run_do_not_emit_spurious_started(tmp_path: Path) -> None:
    events: list[str] = []
    runtime = StageRuntime(
        context=_ctx(tmp_path, emitter=events.append),
        stage_id=StageId.S03,
        stage_dir=tmp_path / "s03_asr",
        expected_outputs=[],
        manifest_store=MemoryManifestStore(),
    )
    manifest = runtime.initialize_manifest(params={})
    runtime.mark_skipped(manifest)
    runtime.mark_dry_run(manifest)
    assert events[0].startswith("[s03] skipped (duration:")
    assert events[1].startswith("[s03] dry_run (duration:")
    assert all(event != "[s03] started" for event in events)
