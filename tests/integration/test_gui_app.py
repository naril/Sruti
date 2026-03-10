from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sruti.gui.app import create_app


def test_gui_can_create_project_and_render_pages(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/projects",
        data={
            "name": "demo",
            "project_type": "single",
            "input_path": "/tmp/input.wav",
            "input_dir": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert (tmp_path / "demo" / "pipeline.toml").exists()
    assert (tmp_path / "demo" / "prompts" / "s05_cleanup.txt").exists()

    detail = client.get("/projects/demo")
    assert detail.status_code == 200
    assert "demo" in detail.text

    prompts = client.get("/projects/demo/prompts")
    assert prompts.status_code == 200
    assert "s05_cleanup.txt" in prompts.text

    dashboard_partial = client.get("/dashboard/partial")
    assert dashboard_partial.status_code == 200
    assert "<html" not in dashboard_partial.text
    assert "Workspace Dashboard" not in dashboard_partial.text
    assert "Projects" in dashboard_partial.text


def test_gui_settings_structured_save_preserves_unknown_sections(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)
    project_dir = tmp_path / "demo"
    client.post(
        "/projects",
        data={
            "name": "demo",
            "project_type": "single",
            "input_path": "/tmp/input.wav",
            "input_dir": "",
        },
        follow_redirects=False,
    )
    pipeline_path = project_dir / "pipeline.toml"
    pipeline_path.write_text(
        pipeline_path.read_text(encoding="utf-8") + "\n[notes]\nowner = \"tester\"\n",
        encoding="utf-8",
    )

    response = client.post(
        "/projects/demo/settings",
        data={
            "mode": "structured",
            "input_path": "/tmp/changed.wav",
            "input_dir": "",
            "chunk_seconds": "17",
            "source_language": "en",
            "whisper_beam_size": "5",
            "ffmpeg_bin": "ffmpeg",
            "whisper_cli_bin": "whisper-cli",
            "ollama_bin": "ollama",
            "default_whisper_model_path": "models/ggml-large-v3.bin",
            "stage_timeout_seconds": "3600",
            "prompt_templates_dir": "prompts",
            "llm_provider": "local",
            "s05_model": "llama3.1:8b",
            "s05_temperature": "0.1",
            "s06_model": "llama3.1:8b",
            "s06_temperature": "0.1",
            "s07_model": "mistral:7b-instruct",
            "s07_temperature": "0.2",
            "s08_model": "mistral:7b-instruct",
            "s08_temperature": "0.2",
            "s09_model": "llama3.1:8b",
            "s09_temperature": "0.1",
            "s10_model": "mistral:7b-instruct",
            "s10_temperature": "0.2",
            "max_llm_calls_per_stage": "10000",
            "llm_json_max_retries": "3",
            "openai_api_key_env": "OPENAI_API_KEY",
            "openai_api_key": "",
            "openai_base_url": "",
            "openai_timeout_seconds": "120",
            "openai_max_retries": "3",
            "openai_model_s05": "gpt-5-nano",
            "openai_model_s06": "gpt-5-nano",
            "openai_model_s07": "gpt-5-mini",
            "openai_model_s08": "gpt-5-mini",
            "openai_model_s09": "gpt-5-mini",
            "openai_model_s10": "gpt-5-mini",
            "openai_price_input_per_1m": "0.25",
            "openai_price_output_per_1m": "2.0",
            "batch_max_active_runs": "0",
            "batch_local_slots": "1",
            "batch_external_api_slots": "4",
            "batch_external_api_slots_per_run": "2",
            "cost_cap_usd": "2.0",
            "token_cap_input": "2000000",
            "token_cap_output": "1000000",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    saved = pipeline_path.read_text(encoding="utf-8")
    assert "chunk_seconds = 17" in saved
    assert '[notes]\nowner = "tester"' in saved


def test_gui_prompt_save_rejects_unknown_placeholder(tmp_path: Path) -> None:
    app = create_app(workspace_root=tmp_path)
    client = TestClient(app)
    client.post(
        "/projects",
        data={
            "name": "demo",
            "project_type": "single",
            "input_path": "/tmp/input.wav",
            "input_dir": "",
        },
        follow_redirects=False,
    )

    response = client.post(
        "/projects/demo/prompts/s05_cleanup.txt",
        data={"content": "Bad {{unknown}}"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "error=Unknown+placeholders" in response.headers["location"]
