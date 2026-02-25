from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.domain.enums import OnExistsMode


def test_stage_context_resolves_relative_prompt_templates_dir_against_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline.toml").write_text(
        """
[sruti]
prompt_templates_dir = "prompts"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    context = StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )
    assert context.settings.prompt_templates_dir == run_dir / "prompts"


def test_stage_context_uses_settings_dir_for_shared_pipeline(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "pipeline.toml").write_text(
        """
[sruti]
chunk_seconds = 17
prompt_templates_dir = "prompts"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_dir = runs_root / "audio-001"

    context = StageContext.build(
        run_dir=run_dir,
        settings_dir=runs_root,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=False,
        force=False,
        verbose=False,
    )
    assert context.settings.chunk_seconds == 17
    assert context.settings.prompt_templates_dir == runs_root / "prompts"
