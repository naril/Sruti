from __future__ import annotations

import mimetypes
import json
import re
from urllib.parse import parse_qs
from enum import Enum
from pathlib import Path
from typing import Any

import tomlkit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sruti.application.project_service import (
    PIPELINE_FILENAME,
    ProjectInitializer,
    ProjectRecord,
    discover_projects,
    ensure_project_prompt_overrides,
    load_pipeline_document,
    update_gui_settings,
    update_sruti_settings,
    write_pipeline_document,
)
from sruti.application.run_service import BatchRunRequest, RunRequest, execute_batch_run, execute_run
from sruti.config import GuiSettings, Settings, load_gui_settings, load_settings
from sruti.domain.enums import LlmProvider, OnExistsMode, ProjectType, RunStatus, StageId
from sruti.gui.job_manager import GuiJobManager
from sruti.gui.prompt_catalog import PROMPT_TEMPLATE_CATALOG
from sruti.llm.prompts import DEFAULT_TEMPLATE_DIR, PLACEHOLDER_PATTERN
from sruti.util.manifest import load_stage_manifest, stage_dir_for

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEXT_EXTENSIONS = {".txt", ".srt", ".toml", ".json", ".jsonl", ".log"}
INLINE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".opus"}
FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "Runtime": (
        "chunk_seconds",
        "source_language",
        "whisper_beam_size",
        "ffmpeg_bin",
        "whisper_cli_bin",
        "ollama_bin",
        "default_whisper_model_path",
        "stage_timeout_seconds",
        "prompt_templates_dir",
    ),
    "Models": (
        "llm_provider",
        "s05_model",
        "s05_temperature",
        "s06_model",
        "s06_temperature",
        "s07_model",
        "s07_temperature",
        "s08_model",
        "s08_temperature",
        "s09_model",
        "s09_temperature",
        "s10_model",
        "s10_temperature",
        "max_llm_calls_per_stage",
        "llm_json_max_retries",
    ),
    "OpenAI": (
        "openai_api_key_env",
        "openai_api_key",
        "openai_base_url",
        "openai_timeout_seconds",
        "openai_max_retries",
        "openai_model_s05",
        "openai_model_s06",
        "openai_model_s07",
        "openai_model_s08",
        "openai_model_s09",
        "openai_model_s10",
        "openai_price_input_per_1m",
        "openai_price_output_per_1m",
    ),
    "Scheduler": (
        "batch_max_active_runs",
        "batch_local_slots",
        "batch_external_api_slots",
        "batch_external_api_slots_per_run",
    ),
    "Advanced": (
        "cost_cap_usd",
        "token_cap_input",
        "token_cap_output",
    ),
}


def create_app(*, workspace_root: Path) -> FastAPI:
    app = FastAPI(title="sruti GUI")
    app.state.workspace_root = workspace_root
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.job_manager = GuiJobManager()
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        projects = discover_projects(app.state.workspace_root)
        return _render(
            request,
            "dashboard.html",
            {
                "projects": projects,
                "active_jobs": app.state.job_manager.active_jobs(),
            },
        )

    @app.get("/dashboard/partial", response_class=HTMLResponse)
    async def dashboard_partial(request: Request) -> HTMLResponse:
        projects = discover_projects(app.state.workspace_root)
        return _render(
            request,
            "partials/dashboard_panels.html",
            {
                "projects": projects,
                "active_jobs": app.state.job_manager.active_jobs(),
            },
        )

    @app.get("/projects/new", response_class=HTMLResponse)
    async def new_project(request: Request) -> HTMLResponse:
        return _render(request, "new_project.html", {"project_types": ProjectType})

    @app.post("/projects", response_class=HTMLResponse)
    async def create_project_route(request: Request) -> RedirectResponse:
        form = await _read_simple_form(request)
        name = str(form.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="Project name is required.")
        project_type = ProjectType(str(form.get("project_type", ProjectType.SINGLE.value)))
        input_path = str(form.get("input_path", "")).strip()
        input_dir = str(form.get("input_dir", "")).strip()
        project_dir = app.state.workspace_root / name
        ProjectInitializer().create_project(
            project_dir=project_dir,
            project_type=project_type,
            input_path=input_path,
            input_dir=input_dir,
            copy_prompts=True,
            include_gui_metadata=True,
        )
        return RedirectResponse(url=f"/projects/{name}", status_code=303)

    @app.get("/projects/{name}", response_class=HTMLResponse)
    async def project_detail(request: Request, name: str) -> HTMLResponse:
        project = _get_project(app.state.workspace_root, name)
        return _render(
            request,
            "project_detail.html",
            {
                "project": project,
                "status_fragment": _project_status_context(project, app.state.job_manager.is_running(project.path)),
                "stages": _project_stages(project),
                "StageId": StageId,
                "batch_runs": _batch_runs(project),
            },
        )

    @app.get("/projects/{name}/runs/{run_name}", response_class=HTMLResponse)
    async def batch_run_detail(request: Request, name: str, run_name: str) -> HTMLResponse:
        batch_project = _get_project(app.state.workspace_root, name)
        run_dir = _resolve_project_path(batch_project.path, run_name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail="Run not found.")
        return _render(
            request,
            "batch_run_detail.html",
            {
                "batch_project": batch_project,
                "run_name": run_name,
                "run_dir": run_dir,
                "run_state": _load_json(run_dir / "run_state.json"),
                "final_output": _resolve_final_output_for(run_dir),
                "stages": _project_stages_for_path(run_dir),
            },
        )

    @app.get("/projects/{name}/status", response_class=HTMLResponse)
    async def project_status(request: Request, name: str) -> HTMLResponse:
        project = _get_project(app.state.workspace_root, name)
        context = _project_status_context(project, app.state.job_manager.is_running(project.path))
        return _render(request, "partials/project_status.html", context)

    @app.get("/projects/{name}/settings", response_class=HTMLResponse)
    async def project_settings(request: Request, name: str) -> HTMLResponse:
        project = _get_project(app.state.workspace_root, name)
        doc = load_pipeline_document(project.path)
        return _render(
            request,
            "settings.html",
            {
                "project": project,
                "settings_groups": _settings_groups(project.settings),
                "raw_toml": tomlkit.dumps(doc),
            },
        )

    @app.post("/projects/{name}/settings", response_class=HTMLResponse)
    async def save_project_settings(request: Request, name: str) -> RedirectResponse:
        project = _get_project(app.state.workspace_root, name)
        form = await _read_simple_form(request)
        mode = str(form.get("mode", "structured"))
        if mode == "raw":
            raw_toml = str(form.get("raw_toml", ""))
            document = tomlkit.parse(raw_toml)
            write_pipeline_document(project.path, document)
            return RedirectResponse(url=f"/projects/{name}/settings?saved=1", status_code=303)

        sruti_values = _parse_structured_settings(form, project.settings)
        update_sruti_settings(project.path, sruti_values)
        gui_settings = GuiSettings(
            project_type=project.project_type,
            input_path=str(form.get("input_path", "")).strip(),
            input_dir=str(form.get("input_dir", "")).strip(),
            created_by=project.gui_settings.created_by or "gui",
        )
        update_gui_settings(project.path, gui_settings)
        return RedirectResponse(url=f"/projects/{name}/settings?saved=1", status_code=303)

    @app.get("/projects/{name}/prompts", response_class=HTMLResponse)
    async def project_prompts(request: Request, name: str) -> HTMLResponse:
        project = _get_project(app.state.workspace_root, name)
        prompts_dir = _project_prompts_dir(project.path)
        prompt_rows = []
        for template_name, spec in PROMPT_TEMPLATE_CATALOG.items():
            source_path = prompts_dir / template_name if prompts_dir is not None else DEFAULT_TEMPLATE_DIR / template_name
            if not source_path.exists():
                source_path = DEFAULT_TEMPLATE_DIR / template_name
            prompt_rows.append(
                {
                    "name": template_name,
                    "spec": spec,
                    "content": source_path.read_text(encoding="utf-8"),
                    "is_local": prompts_dir is not None and (prompts_dir / template_name).exists(),
                }
            )
        return _render(
            request,
            "prompts.html",
            {
                "project": project,
                "prompts_dir": prompts_dir,
                "prompt_rows": prompt_rows,
            },
        )

    @app.post("/projects/{name}/prompts/{template_name}", response_class=HTMLResponse)
    async def save_prompt(request: Request, name: str, template_name: str) -> RedirectResponse:
        if template_name not in PROMPT_TEMPLATE_CATALOG:
            raise HTTPException(status_code=404, detail="Unknown prompt template.")
        project = _get_project(app.state.workspace_root, name)
        form = await _read_simple_form(request)
        action = str(form.get("action", "save"))
        if action == "create-local":
            ensure_project_prompt_overrides(project.path)
            return RedirectResponse(url=f"/projects/{name}/prompts?localized=1", status_code=303)

        content = str(form.get("content", ""))
        spec = PROMPT_TEMPLATE_CATALOG[template_name]
        placeholders = set(PLACEHOLDER_PATTERN.findall(content))
        unknown = sorted(placeholders - set(spec.allowed_placeholders))
        if unknown:
            joined = ", ".join(unknown)
            return RedirectResponse(
                url=f"/projects/{name}/prompts?error=Unknown+placeholders+for+{template_name}%3A+{joined}",
                status_code=303,
            )
        prompts_dir = ensure_project_prompt_overrides(project.path)
        (prompts_dir / template_name).write_text(content, encoding="utf-8")
        missing = sorted(set(spec.recommended_placeholders) - placeholders)
        suffix = ""
        if missing:
            suffix = f"&warning=Missing+recommended+placeholders%3A+{'+'.join(missing)}"
        return RedirectResponse(url=f"/projects/{name}/prompts?saved=1{suffix}", status_code=303)

    @app.post("/projects/{name}/execute", response_class=HTMLResponse)
    async def execute_project(request: Request, name: str) -> RedirectResponse:
        project = _get_project(app.state.workspace_root, name)
        form = await _read_simple_form(request)
        source_stage = StageId(str(form.get("from_stage", StageId.S01.value)))
        target_stage = StageId(str(form.get("to_stage", StageId.S10.value)))
        on_exists = OnExistsMode(str(form.get("on_exists", OnExistsMode.OVERWRITE.value)))
        dry_run = bool(form.get("dry_run"))
        force = bool(form.get("force"))
        verbose = bool(form.get("verbose"))
        llm_provider_raw = str(form.get("llm_provider", "")).strip()
        llm_provider = LlmProvider(llm_provider_raw) if llm_provider_raw else None
        cost_cap_usd = _maybe_float(form.get("cost_cap_usd"))
        token_cap_input = _maybe_int(form.get("token_cap_input"))
        token_cap_output = _maybe_int(form.get("token_cap_output"))

        if project.project_type is ProjectType.SINGLE:
            input_path = (project.gui_settings.input_path or "").strip()
            if source_stage is StageId.S01 and not input_path:
                return RedirectResponse(url=f"/projects/{name}?error=Missing+input_path", status_code=303)
            request_payload = RunRequest(
                run_dir=project.path,
                in_path=Path(input_path) if input_path else None,
                source_stage=source_stage,
                target_stage=target_stage,
                seconds=None,
                model_path=None,
                on_exists=on_exists,
                dry_run=dry_run,
                force=force,
                verbose=verbose,
                llm_provider=llm_provider,
                cost_cap_usd=cost_cap_usd,
                token_cap_input=token_cap_input,
                token_cap_output=token_cap_output,
                command_label="gui-run",
            )
            app.state.job_manager.submit(project.path, lambda: execute_run(request_payload))
        else:
            input_dir = (project.gui_settings.input_dir or "").strip()
            if not input_dir:
                return RedirectResponse(url=f"/projects/{name}?error=Missing+input_dir", status_code=303)
            batch_request = BatchRunRequest(
                runs_root=project.path,
                in_dir=Path(input_dir),
                source_stage=source_stage,
                target_stage=target_stage,
                seconds=None,
                model_path=None,
                on_exists=on_exists,
                dry_run=dry_run,
                force=force,
                verbose=verbose,
                llm_provider=llm_provider,
                cost_cap_usd=cost_cap_usd,
                token_cap_input=token_cap_input,
                token_cap_output=token_cap_output,
                max_active_runs=None,
                local_slots=None,
                external_api_slots=None,
                external_api_slots_per_run=None,
            )
            app.state.job_manager.submit(project.path, lambda: execute_batch_run(batch_request))
        return RedirectResponse(url=f"/projects/{name}?started=1", status_code=303)

    @app.get("/projects/{name}/stages/{stage_id}", response_class=HTMLResponse)
    async def stage_detail(request: Request, name: str, stage_id: StageId) -> HTMLResponse:
        project = _get_project(app.state.workspace_root, name)
        payload = _stage_context(project.path, stage_id)
        return _render(request, "stage_detail.html", {"project": project, "stage": payload})

    @app.get("/projects/{name}/artifact")
    async def project_artifact(name: str, path: str) -> FileResponse:
        project = _get_project(app.state.workspace_root, name)
        artifact_path = _resolve_project_path(project.path, path)
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        media_type, _ = mimetypes.guess_type(str(artifact_path))
        return FileResponse(artifact_path, media_type=media_type or "application/octet-stream")

    return app


def _render(request: Request, template_name: str, context: dict[str, Any]) -> HTMLResponse:
    templates: Jinja2Templates = request.app.state.templates
    merged = {"request": request, **context}
    return templates.TemplateResponse(request, template_name, merged)


def _get_project(workspace_root: Path, name: str) -> ProjectRecord:
    project_dir = workspace_root / name
    if not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="Project not found.")
    projects = {project.name: project for project in discover_projects(workspace_root)}
    project = projects.get(name)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _settings_groups(settings: Settings) -> list[dict[str, Any]]:
    values = settings.model_dump(mode="python")
    fields = Settings.model_fields
    groups: list[dict[str, Any]] = []
    for group_name, field_names in FIELD_GROUPS.items():
        groups.append(
            {
                "name": group_name,
                "fields": [
                    {
                        "name": field_name,
                        "value": values[field_name],
                        "input_type": _field_input_type(values[field_name]),
                        "options": _field_options(values[field_name]),
                    }
                    for field_name in field_names
                    if field_name in fields
                ],
            }
        )
    return groups


def _field_input_type(value: Any) -> str:
    if isinstance(value, Enum):
        return "select"
    if isinstance(value, bool):
        return "checkbox"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "float"
    return "text"


def _field_options(value: Any) -> list[str] | None:
    if isinstance(value, LlmProvider):
        return [item.value for item in LlmProvider]
    return None


def _parse_structured_settings(form: Any, current: Settings) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_values = current.model_dump(mode="python")
    for field_name in Settings.model_fields:
        raw_value = form.get(field_name)
        current_value = current_values[field_name]
        if isinstance(current_value, bool):
            values[field_name] = bool(raw_value)
        elif isinstance(current_value, int):
            values[field_name] = int(str(raw_value or current_value))
        elif isinstance(current_value, float):
            values[field_name] = float(str(raw_value or current_value))
        elif isinstance(current_value, Path):
            values[field_name] = Path(str(raw_value or current_value))
        elif isinstance(current_value, Enum):
            values[field_name] = current_value.__class__(str(raw_value or current_value.value))
        else:
            text = str(raw_value if raw_value is not None else current_value)
            if field_name == "prompt_templates_dir" and text == "":
                values[field_name] = None
            else:
                values[field_name] = text
    return values


def _project_prompts_dir(project_dir: Path) -> Path | None:
    settings = load_settings(project_dir)
    prompt_dir = settings.prompt_templates_dir
    if prompt_dir is None:
        return None
    return prompt_dir if prompt_dir.is_absolute() else project_dir / prompt_dir


def _project_status_context(project: ProjectRecord, is_running: bool) -> dict[str, Any]:
    if project.project_type is ProjectType.BATCH and project.batch_state is not None:
        summary = project.batch_state.get("summary", {})
        state_status = "running" if summary.get("running_runs") else "idle"
        last_updated = None
        if project.batch_state.get("runs"):
            last_updated = max(
                (payload.get("updated_at") for payload in project.batch_state["runs"].values() if payload.get("updated_at")),
                default=None,
            )
        return {
            "project": project,
            "summary": summary,
            "status": state_status,
            "last_updated": last_updated,
            "is_running": is_running or state_status == "running",
        }
    run_state = project.run_state or {}
    return {
        "project": project,
        "summary": run_state,
        "status": run_state.get("status", "idle"),
        "last_updated": run_state.get("updated_at"),
        "is_running": is_running or run_state.get("status") == RunStatus.RUNNING.value,
    }


def _project_stages(project: ProjectRecord) -> list[dict[str, Any]]:
    return _project_stages_for_path(project.path)


def _project_stages_for_path(project_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for stage_id in StageId:
        rows.append(_stage_context(project_dir, stage_id))
    return rows


def _batch_runs(project: ProjectRecord) -> list[dict[str, Any]]:
    if project.project_type is not ProjectType.BATCH or project.batch_state is None:
        return []
    runs = project.batch_state.get("runs", {})
    return [
        {
            "name": name,
            "status": payload.get("status"),
            "current_stage": payload.get("current_stage"),
            "last_message": payload.get("last_message"),
            "updated_at": payload.get("updated_at"),
        }
        for name, payload in sorted(runs.items())
    ]


def _stage_context(project_dir: Path, stage_id: StageId) -> dict[str, Any]:
    stage_dir = stage_dir_for(project_dir, stage_id.value)
    manifest = load_stage_manifest(stage_dir)
    artifacts = []
    if stage_dir.exists():
        for artifact in sorted(stage_dir.rglob("*")):
            if artifact.is_file() and artifact.name != "manifest.json":
                artifacts.append(
                    {
                        "path": artifact,
                        "relative_path": str(artifact.relative_to(project_dir)),
                        "preview": _artifact_preview(artifact),
                    }
                )
    return {
        "id": stage_id,
        "dir": stage_dir,
        "exists": stage_dir.exists(),
        "manifest": manifest,
        "artifacts": artifacts,
    }


def _artifact_preview(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return {"kind": "text", "content": path.read_text(encoding="utf-8")}
    if suffix == ".html":
        return {"kind": "html"}
    if suffix in INLINE_AUDIO_EXTENSIONS:
        return {"kind": "audio"}
    return {"kind": "binary"}


def _resolve_project_path(project_dir: Path, raw_path: str) -> Path:
    candidate = (project_dir / raw_path).resolve()
    try:
        candidate.relative_to(project_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path.") from exc
    return candidate


def _resolve_final_output_for(project_dir: Path) -> Path | None:
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_int(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None


def _maybe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    return float(text) if text else None


async def _read_simple_form(request: Request) -> dict[str, str]:
    body = await request.body()
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}
