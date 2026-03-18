from __future__ import annotations

from pathlib import Path

from sruti.application.project_service import PIPELINE_FILENAME, ProjectInitializer
from sruti.config import load_gui_settings, load_settings
from sruti.domain.enums import ProjectType


def test_project_initializer_can_create_gui_project_with_local_prompts(tmp_path: Path) -> None:
    project_dir = tmp_path / "gui-project"
    ProjectInitializer().create_project(
        project_dir=project_dir,
        project_type=ProjectType.BATCH,
        input_dir="/tmp/input-folder",
        copy_prompts=True,
        include_gui_metadata=True,
    )

    settings = load_settings(project_dir)
    gui_settings = load_gui_settings(project_dir)
    assert (project_dir / PIPELINE_FILENAME).exists()
    assert (project_dir / "prompts" / "s05_cleanup.txt").exists()
    assert settings.prompt_templates_dir == Path("prompts")
    assert gui_settings.project_type is ProjectType.BATCH
    assert gui_settings.input_dir == "/tmp/input-folder"

