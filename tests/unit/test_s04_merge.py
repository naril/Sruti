from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s04_merge_uc import S04MergeUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util
from sruti.util.io import atomic_write_json


def _ctx(run_dir: Path) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )


def test_s04_merge_happy_path(tmp_path: Path) -> None:
    s03_dir = manifest_util.stage_dir_for(tmp_path, "s03")
    transcripts_dir = s03_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "0001.txt").write_text("a", encoding="utf-8")
    (transcripts_dir / "0002.txt").write_text("b", encoding="utf-8")
    (transcripts_dir / "0001.srt").write_text(
        "1\n00:00:00,100 --> 00:00:00,900\nchunk1\n",
        encoding="utf-8",
    )
    (transcripts_dir / "0002.srt").write_text(
        "1\n00:00:00,200 --> 00:00:01,000\nchunk2\n",
        encoding="utf-8",
    )
    atomic_write_json(
        s03_dir / "transcripts_index.json",
        [
            {
                "id": 1,
                "start_time": 0,
                "end_time": 30,
                "txt_filename": "0001.txt",
                "srt_filename": "0001.srt",
            },
            {
                "id": 2,
                "start_time": 30,
                "end_time": 60,
                "txt_filename": "0002.txt",
                "srt_filename": "0002.srt",
            },
        ],
    )

    use_case = S04MergeUseCase(manifest_store=FileSystemManifestStore())
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert (tmp_path / "s04_merge" / "merged_raw.txt").read_text(encoding="utf-8").strip() == "a\n\nb"
    merged_srt = (tmp_path / "s04_merge" / "merged_raw.srt").read_text(encoding="utf-8")
    assert "1\n00:00:00,100 --> 00:00:00,900" in merged_srt
    assert "2\n00:00:30,200 --> 00:00:31,000" in merged_srt


def test_s04_merge_accepts_dot_millisecond_separator(tmp_path: Path) -> None:
    s03_dir = manifest_util.stage_dir_for(tmp_path, "s03")
    transcripts_dir = s03_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "0001.txt").write_text("chunk", encoding="utf-8")
    (transcripts_dir / "0001.srt").write_text(
        "1\n00:00:00.500 --> 00:00:01.000\nline\n",
        encoding="utf-8",
    )
    atomic_write_json(
        s03_dir / "transcripts_index.json",
        [
            {
                "id": 1,
                "start_time": 10,
                "end_time": 20,
                "txt_filename": "0001.txt",
                "srt_filename": "0001.srt",
            }
        ],
    )

    use_case = S04MergeUseCase(manifest_store=FileSystemManifestStore())
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    merged_srt = (tmp_path / "s04_merge" / "merged_raw.srt").read_text(encoding="utf-8")
    assert "1\n00:00:10,500 --> 00:00:11,000" in merged_srt
