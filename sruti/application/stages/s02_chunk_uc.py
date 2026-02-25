from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from shutil import rmtree
from typing import Any

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import StageId
from sruti.domain.models import StageResult
from sruti.domain.ports import ManifestStore
from sruti.infrastructure.audio_ffmpeg import FfmpegAdapter
from sruti.util import manifest as manifest_util
from sruti.util.hashes import sha256_file
from sruti.util.io import atomic_write_json, ensure_dir
from sruti.util.system import executable_version, require_executable, require_file


class S02ChunkUseCase:
    stage_name = StageId.S02.value

    def __init__(
        self,
        *,
        seconds: int,
        ffmpeg: FfmpegAdapter,
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._seconds = seconds
        self._ffmpeg = ffmpeg
        self._manifest_store = manifest_store
        self._ask_user = ask_user

    def run(self, context: StageContext) -> StageResult:
        require_executable(context.settings.ffmpeg_bin)
        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S02.value)
        input_path = (
            manifest_util.stage_dir_for(context.run_dir, StageId.S01.value) / "normalized.wav"
        )
        require_file(input_path, label="Normalized audio for s02")

        chunks_dir = stage_dir / "chunks"
        chunks_json_path = stage_dir / "chunks.json"
        inputs_signature = manifest_util.inputs_signature([input_path])
        params: dict[str, object] = {
            "seconds": self._seconds,
            "ffmpeg_bin": context.settings.ffmpeg_bin,
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S02,
            stage_dir=stage_dir,
            expected_outputs=[chunks_json_path],
            manifest_store=self._manifest_store,
            ask_user=self._ask_user,
        )
        manifest = runtime.initialize_manifest(params=params)

        if runtime.should_skip(params=params, inputs_signature=inputs_signature):
            return runtime.mark_skipped(manifest)

        policy = runtime.apply_on_exists_policy()
        if policy == "skip":
            return runtime.mark_skipped(manifest)

        if context.dry_run:
            return runtime.mark_dry_run(manifest)

        try:
            runtime.start(manifest)
            if chunks_dir.exists():
                rmtree(chunks_dir)
            ensure_dir(chunks_dir)
            output_pattern = chunks_dir / "%04d.wav"
            command = self._ffmpeg.segment(input_path, output_pattern, seconds=self._seconds)
            chunk_files = sorted(chunks_dir.glob("*.wav"))
            chunk_rows = self._build_chunk_rows(chunk_files)
            atomic_write_json(chunks_json_path, chunk_rows)
            manifest.commands.append(" ".join(command))
            manifest.inputs = [manifest_util.artifact_for(input_path)]
            manifest.tool_versions["ffmpeg"] = executable_version([context.settings.ffmpeg_bin, "-version"])
            return runtime.mark_success(manifest, output_paths=[chunks_json_path, *chunk_files])
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise

    def _build_chunk_rows(self, chunk_files: list[Path]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, chunk_path in enumerate(chunk_files, start=1):
            start_time = (idx - 1) * self._seconds
            end_time = idx * self._seconds
            rows.append(
                {
                    "id": idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "filename": chunk_path.name,
                    "sha256": sha256_file(chunk_path),
                }
            )
        return rows
