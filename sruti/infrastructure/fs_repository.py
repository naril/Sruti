from __future__ import annotations

from pathlib import Path

from sruti.domain.models import StageManifest
from sruti.domain.ports import ManifestStore
from sruti.util import manifest as manifest_util


class FileSystemManifestStore(ManifestStore):
    def load_stage_manifest(self, stage_dir: Path) -> StageManifest | None:
        return manifest_util.load_stage_manifest(stage_dir)

    def save_stage_manifest(self, stage_dir: Path, manifest: StageManifest) -> None:
        manifest_util.save_stage_manifest(stage_dir, manifest)
