from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import StageId
from sruti.domain.models import StageResult
from sruti.domain.ports import ManifestStore
from sruti.infrastructure.asr_whisper_cli import WhisperCliAdapter
from sruti.infrastructure.json_codec import loads
from sruti.util import manifest as manifest_util
from sruti.util.io import atomic_write_json
from sruti.util.system import executable_version, require_executable, require_file


class S03AsrWhisperUseCase:
    stage_name = StageId.S03.value

    def __init__(
        self,
        *,
        whisper_model_path: Path,
        whisper: WhisperCliAdapter,
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._whisper_model_path = whisper_model_path
        self._whisper = whisper
        self._manifest_store = manifest_store
        self._ask_user = ask_user

    def run(self, context: StageContext) -> StageResult:
        require_executable(context.settings.whisper_cli_bin)
        require_file(self._whisper_model_path, label="Whisper model")

        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S03.value)
        s02_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S02.value)
        chunks_json_path = s02_dir / "chunks.json"
        require_file(chunks_json_path, label="chunks.json from s02")

        transcripts_dir = stage_dir / "transcripts"
        index_json_path = stage_dir / "transcripts_index.json"
        inputs_signature = manifest_util.inputs_signature([chunks_json_path, self._whisper_model_path])
        params: dict[str, object] = {
            "model_path": str(self._whisper_model_path),
            "beam_size": context.settings.whisper_beam_size,
            "_inputs_signature": inputs_signature,
        }

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S03,
            stage_dir=stage_dir,
            expected_outputs=[index_json_path],
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
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            rows = loads(chunks_json_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Invalid chunks.json format: expected list, got {type(rows)}")

            index_rows: list[dict[str, Any]] = []
            output_paths: list[Path] = [index_json_path]
            for row in rows:
                chunk_file = s02_dir / "chunks" / row["filename"]
                chunk_id = int(row["id"])
                prefix = transcripts_dir / f"{chunk_id:04d}"
                command = self._whisper.transcribe_chunk(
                    model_path=self._whisper_model_path,
                    chunk_path=chunk_file,
                    output_prefix=prefix,
                )
                manifest.commands.append(" ".join(command))
                txt_path = prefix.with_suffix(".txt")
                srt_path = prefix.with_suffix(".srt")
                output_paths.extend([txt_path, srt_path])
                index_rows.append(
                    {
                        "id": chunk_id,
                        "chunk_filename": row["filename"],
                        "start_time": row.get("start_time"),
                        "end_time": row.get("end_time"),
                        "txt_filename": txt_path.name,
                        "srt_filename": srt_path.name,
                    }
                )

            atomic_write_json(index_json_path, index_rows)
            manifest.inputs = [manifest_util.artifact_for(chunks_json_path)]
            manifest.tool_versions["whisper-cli"] = executable_version(
                [context.settings.whisper_cli_bin, "--help"]
            )
            return runtime.mark_success(manifest, output_paths=output_paths)
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise
