from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sruti.domain.models import FileArtifact, RunManifest, StageManifest
from sruti.util.hashes import sha256_file
from sruti.util.io import atomic_write_json


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def stage_dir_for(run_dir: Path, stage_name: str) -> Path:
    return run_dir / f"{stage_name}_{stage_label(stage_name)}"


def stage_label(stage_name: str) -> str:
    mapping = {
        "s01": "normalize",
        "s02": "chunk",
        "s03": "asr",
        "s04": "merge",
        "s05": "asr_cleanup",
        "s06": "remove_nonlecture",
        "s07": "editorial",
        "s08": "translate",
        "s09": "translate_edit",
    }
    return mapping[stage_name]


def artifact_for(path: Path) -> FileArtifact:
    return FileArtifact(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def artifacts_for_existing(paths: list[Path]) -> list[FileArtifact]:
    artifacts: list[FileArtifact] = []
    for path in paths:
        if path.exists():
            artifacts.append(artifact_for(path))
    return artifacts


def save_stage_manifest(stage_dir: Path, manifest: StageManifest) -> None:
    manifest.finished_at = manifest.finished_at or utc_now_iso()
    atomic_write_json(stage_dir / "manifest.json", manifest.model_dump())


def load_stage_manifest(stage_dir: Path) -> StageManifest | None:
    path = stage_dir / "manifest.json"
    if not path.exists():
        return None
    data = path.read_text(encoding="utf-8")
    return StageManifest.model_validate_json(data)


def load_run_manifest(run_dir: Path) -> RunManifest | None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_run_manifest(run_dir: Path, manifest: RunManifest) -> None:
    manifest.updated_at = utc_now_iso()
    atomic_write_json(run_dir / "run_manifest.json", manifest.model_dump())


def params_signature(params: dict[str, Any]) -> str:
    import json

    # Keep deterministic signature for resume checks.
    return json.dumps(params, sort_keys=True, ensure_ascii=False)
