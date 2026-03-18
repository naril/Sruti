from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import StageId
from sruti.domain.models import StageResult
from sruti.domain.ports import ManifestStore
from sruti.infrastructure.audio_ffmpeg import FfmpegAdapter
from sruti.util import manifest as manifest_util
from sruti.util.io import ensure_dir
from sruti.util.system import executable_version, require_executable, require_file


class S01NormalizeUseCase:
    stage_name = StageId.S01.value

    def __init__(
        self,
        *,
        input_audio: Path,
        ffmpeg: FfmpegAdapter,
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._input_audio = input_audio
        self._ffmpeg = ffmpeg
        self._manifest_store = manifest_store
        self._ask_user = ask_user

    def run(self, context: StageContext) -> StageResult:
        require_executable(context.settings.ffmpeg_bin)
        require_file(self._input_audio, label="Input audio")

        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S01.value)
        output_path = stage_dir / "normalized.wav"
        inputs_signature = manifest_util.inputs_signature([self._input_audio])
        params: dict[str, object] = {
            "input_audio": str(self._input_audio),
            "ffmpeg_bin": context.settings.ffmpeg_bin,
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S01,
            stage_dir=stage_dir,
            expected_outputs=[output_path],
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
            ensure_dir(stage_dir)
            command = self._ffmpeg.normalize(self._input_audio, output_path)
            manifest.commands.append(" ".join(command))
            manifest.inputs = [manifest_util.artifact_for(self._input_audio)]
            manifest.tool_versions["ffmpeg"] = executable_version([context.settings.ffmpeg_bin, "-version"])
            return runtime.mark_success(manifest, output_paths=[output_path])
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise
