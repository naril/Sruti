from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.toml_document import TOMLDocument

from sruti.config import GuiSettings, Settings, load_gui_settings, load_settings, render_default_pipeline_toml
from sruti.domain.enums import ProjectType
from sruti.llm.prompts import DEFAULT_TEMPLATE_DIR
from sruti.util.io import atomic_write_text, ensure_dir

PIPELINE_FILENAME = "pipeline.toml"


@dataclass(slots=True)
class ProjectRecord:
    path: Path
    name: str
    project_type: ProjectType
    settings: Settings
    gui_settings: GuiSettings
    run_state: dict[str, Any] | None
    batch_state: dict[str, Any] | None
    final_output: Path | None


class ProjectInitializer:
    def create_project(
        self,
        *,
        project_dir: Path,
        project_type: ProjectType,
        input_path: str = "",
        input_dir: str = "",
        copy_prompts: bool = False,
        include_gui_metadata: bool = False,
    ) -> Path:
        ensure_dir(project_dir)
        pipeline_path = project_dir / PIPELINE_FILENAME
        if pipeline_path.exists():
            raise ValueError(f"{pipeline_path} already exists.")
        doc = tomlkit.parse(render_default_pipeline_toml())
        if copy_prompts:
            set_toml_value(doc, ["sruti", "prompt_templates_dir"], "prompts")
        if include_gui_metadata:
            set_toml_value(doc, ["gui", "project_type"], project_type.value)
            set_toml_value(doc, ["gui", "created_by"], "gui")
            set_toml_value(doc, ["gui", "input_path"], input_path)
            set_toml_value(doc, ["gui", "input_dir"], input_dir)
        write_pipeline_document(project_dir, doc)
        if copy_prompts:
            copy_prompt_templates(project_dir, overwrite=False)
        return project_dir


def load_pipeline_document(project_dir: Path) -> TOMLDocument:
    path = project_dir / PIPELINE_FILENAME
    if path.exists():
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    return tomlkit.parse(render_default_pipeline_toml())


def write_pipeline_document(project_dir: Path, document: TOMLDocument) -> None:
    atomic_write_text(project_dir / PIPELINE_FILENAME, tomlkit.dumps(document))


def set_toml_value(document: TOMLDocument, path: list[str], value: Any) -> None:
    cursor: Any = document
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = tomlkit.table()
        cursor = cursor[key]
    cursor[path[-1]] = value


def update_sruti_settings(project_dir: Path, values: dict[str, Any]) -> None:
    doc = load_pipeline_document(project_dir)
    if "sruti" not in doc or not isinstance(doc["sruti"], dict):
        doc["sruti"] = tomlkit.table()
    sruti_table = doc["sruti"]
    settings = Settings.model_validate(values)
    for key in Settings.model_fields:
        value = getattr(settings, key)
        if isinstance(value, Path):
            sruti_table[key] = str(value)
        elif hasattr(value, "value"):
            sruti_table[key] = value.value
        else:
            sruti_table[key] = value
    write_pipeline_document(project_dir, doc)


def update_gui_settings(project_dir: Path, gui_settings: GuiSettings) -> None:
    doc = load_pipeline_document(project_dir)
    if "gui" not in doc or not isinstance(doc["gui"], dict):
        doc["gui"] = tomlkit.table()
    gui_table = doc["gui"]
    for key, value in gui_settings.model_dump(mode="json").items():
        gui_table[key] = value
    write_pipeline_document(project_dir, doc)


def copy_prompt_templates(project_dir: Path, *, overwrite: bool) -> Path:
    prompts_dir = project_dir / "prompts"
    ensure_dir(prompts_dir)
    for source in sorted(DEFAULT_TEMPLATE_DIR.iterdir()):
        if not source.is_file():
            continue
        target = prompts_dir / source.name
        if target.exists() and not overwrite:
            continue
        shutil.copy2(source, target)
    return prompts_dir


def ensure_project_prompt_overrides(project_dir: Path) -> Path:
    prompts_dir = copy_prompt_templates(project_dir, overwrite=False)
    doc = load_pipeline_document(project_dir)
    set_toml_value(doc, ["sruti", "prompt_templates_dir"], "prompts")
    write_pipeline_document(project_dir, doc)
    return prompts_dir


def discover_projects(workspace_root: Path) -> list[ProjectRecord]:
    if not workspace_root.exists():
        return []
    projects: list[ProjectRecord] = []
    for path in sorted(workspace_root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        pipeline_path = path / PIPELINE_FILENAME
        if not pipeline_path.is_file():
            continue
        gui_settings = load_gui_settings(path)
        project_type = _detect_project_type(path, gui_settings)
        run_state = _load_json(path / "run_state.json")
        batch_state = _load_json(path / "batch_scheduler_state.json")
        projects.append(
            ProjectRecord(
                path=path,
                name=path.name,
                project_type=project_type,
                settings=load_settings(path),
                gui_settings=gui_settings,
                run_state=run_state,
                batch_state=batch_state,
                final_output=resolve_final_output(path),
            )
        )
    return projects


def resolve_final_output(project_dir: Path) -> Path | None:
    candidates = [
        project_dir / "s10_translate_edit" / "final_publishable_cs.txt",
        project_dir / "s09_translate" / "translated_faithful_cs.txt",
        project_dir / "s08_condense" / "condensed_blocks_en.txt",
        project_dir / "s07_editorial" / "final_publishable_en.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _detect_project_type(project_dir: Path, gui_settings: GuiSettings) -> ProjectType:
    if gui_settings.project_type is not None:
        return gui_settings.project_type
    if (project_dir / "batch_manifest.json").exists():
        return ProjectType.BATCH
    return ProjectType.SINGLE


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
