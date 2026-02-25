from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stage_runner import StageRuntime
from sruti.domain.enums import StageId
from sruti.domain.models import StageResult
from sruti.domain.ports import ManifestStore
from sruti.infrastructure.json_codec import loads
from sruti.util import manifest as manifest_util
from sruti.util.io import atomic_write_text
from sruti.util.system import require_file


class S04MergeUseCase:
    stage_name = StageId.S04.value

    def __init__(
        self,
        *,
        manifest_store: ManifestStore,
        ask_user: Callable[[str], bool] | None = None,
    ) -> None:
        self._manifest_store = manifest_store
        self._ask_user = ask_user

    def run(self, context: StageContext) -> StageResult:
        stage_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S04.value)
        s03_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S03.value)
        index_json_path = s03_dir / "transcripts_index.json"
        require_file(index_json_path, label="s03 transcripts index")

        merged_txt_path = stage_dir / "merged_raw.txt"
        merged_srt_path = stage_dir / "merged_raw.srt"
        inputs_signature = manifest_util.inputs_signature([index_json_path])
        params: dict[str, object] = {"_inputs_signature": inputs_signature}

        runtime = StageRuntime(
            context=context,
            stage_id=StageId.S04,
            stage_dir=stage_dir,
            expected_outputs=[merged_txt_path, merged_srt_path],
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
            rows = loads(index_json_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("transcripts_index.json must be a list")
            txt_parts: list[str] = []
            srt_parts: list[str] = []
            for row in rows:
                txt_path = s03_dir / "transcripts" / row["txt_filename"]
                srt_path = s03_dir / "transcripts" / row["srt_filename"]
                require_file(txt_path, label=f"Transcript chunk {txt_path.name}")
                require_file(srt_path, label=f"SRT chunk {srt_path.name}")
                txt_parts.append(txt_path.read_text(encoding="utf-8").strip())
                srt_parts.append(srt_path.read_text(encoding="utf-8").strip())

            merged_txt = "\n\n".join(part for part in txt_parts if part).strip() + "\n"
            merged_srt = "\n\n".join(part for part in srt_parts if part).strip() + "\n"
            atomic_write_text(merged_txt_path, merged_txt)
            atomic_write_text(merged_srt_path, merged_srt)
            manifest.inputs = [manifest_util.artifact_for(index_json_path)]
            return runtime.mark_success(manifest, output_paths=[merged_txt_path, merged_srt_path])
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise
