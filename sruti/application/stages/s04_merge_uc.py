from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
            ordered_rows = sorted(rows, key=lambda item: int(item.get("id", 0)))
            txt_parts: list[str] = []
            merged_srt_blocks: list[str] = []
            global_srt_index = 1
            for row in ordered_rows:
                txt_path = s03_dir / "transcripts" / row["txt_filename"]
                srt_path = s03_dir / "transcripts" / row["srt_filename"]
                require_file(txt_path, label=f"Transcript chunk {txt_path.name}")
                require_file(srt_path, label=f"SRT chunk {srt_path.name}")
                txt_parts.append(txt_path.read_text(encoding="utf-8").strip())
                srt_text = srt_path.read_text(encoding="utf-8")
                offset_ms = int(float(row.get("start_time") or 0) * 1000)
                for block in self._parse_srt_blocks(srt_text):
                    shifted_start = block["start_ms"] + offset_ms
                    shifted_end = block["end_ms"] + offset_ms
                    merged_srt_blocks.append(
                        "\n".join(
                            [
                                str(global_srt_index),
                                f"{self._format_srt_timestamp(shifted_start)} --> "
                                f"{self._format_srt_timestamp(shifted_end)}",
                                *block["text_lines"],
                            ]
                        ).strip()
                    )
                    global_srt_index += 1

            merged_txt = "\n\n".join(part for part in txt_parts if part).strip()
            merged_srt = "\n\n".join(block for block in merged_srt_blocks if block).strip()
            if merged_txt:
                merged_txt += "\n"
            if merged_srt:
                merged_srt += "\n"
            atomic_write_text(merged_txt_path, merged_txt)
            atomic_write_text(merged_srt_path, merged_srt)
            manifest.inputs = [manifest_util.artifact_for(index_json_path)]
            return runtime.mark_success(manifest, output_paths=[merged_txt_path, merged_srt_path])
        except Exception as exc:
            runtime.mark_failure(manifest, str(exc))
            raise

    def _parse_srt_blocks(self, value: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        raw_blocks = [block for block in value.strip().split("\n\n") if block.strip()]
        for raw in raw_blocks:
            lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            if lines[0].isdigit():
                timing_line = lines[1]
                text_lines = lines[2:]
            else:
                timing_line = lines[0]
                text_lines = lines[1:]
            if "-->" not in timing_line:
                raise ValueError(f"Invalid SRT timing line: {timing_line}")
            start_raw, end_raw = [part.strip() for part in timing_line.split("-->", maxsplit=1)]
            out.append(
                {
                    "start_ms": self._parse_srt_timestamp(start_raw),
                    "end_ms": self._parse_srt_timestamp(end_raw),
                    "text_lines": text_lines,
                }
            )
        return out

    def _parse_srt_timestamp(self, value: str) -> int:
        # Format: HH:MM:SS,mmm
        hhmmss, millis = value.split(",")
        hh_raw, mm_raw, ss_raw = hhmmss.split(":")
        total_seconds = int(hh_raw) * 3600 + int(mm_raw) * 60 + int(ss_raw)
        return total_seconds * 1000 + int(millis)

    def _format_srt_timestamp(self, total_ms: int) -> str:
        if total_ms < 0:
            total_ms = 0
        total_seconds, millis = divmod(total_ms, 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
