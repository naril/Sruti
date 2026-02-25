from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sruti.cli import app
from sruti.config import Settings, load_settings
from sruti.domain.enums import LlmProvider, StageId, StageStatus
from sruti.domain.models import StageResult


def test_cli_run_dispatches_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_stage(stage_id: StageId):
        def _run_stage(**kwargs):
            calls.append(stage_id.value)
            context = kwargs["context"]
            stage_dir = context.run_dir / f"{stage_id.value}_fake"
            stage_dir.mkdir(parents=True, exist_ok=True)
            return StageResult(stage=stage_id, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

        return _run_stage

    monkeypatch.setattr("sruti.cli.s01_normalize.run_stage", fake_stage(StageId.S01))
    monkeypatch.setattr("sruti.cli.s02_chunk.run_stage", fake_stage(StageId.S02))
    monkeypatch.setattr("sruti.cli.s03_asr_whispercli.run_stage", fake_stage(StageId.S03))

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    run_dir = tmp_path / "run1"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(run_dir),
            "--in",
            str(input_audio),
            "--from",
            "s01",
            "--to",
            "s03",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls == ["s01", "s02", "s03"]


def test_cli_run_uses_pipeline_defaults_for_optional_stage_params(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_s02_run_stage(**kwargs):
        observed["seconds"] = kwargs["seconds"]
        context = kwargs["context"]
        stage_dir = context.run_dir / "s02_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S02, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    def fake_s03_run_stage(**kwargs):
        observed["model_path"] = kwargs["model_path"]
        context = kwargs["context"]
        stage_dir = context.run_dir / "s03_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S03, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s02_chunk.run_stage", fake_s02_run_stage)
    monkeypatch.setattr("sruti.cli.s03_asr_whispercli.run_stage", fake_s03_run_stage)

    run_dir = tmp_path / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline.toml").write_text(
        """
[sruti]
chunk_seconds = 17
default_whisper_model_path = "models/custom.bin"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(run_dir),
            "--from",
            "s02",
            "--to",
            "s03",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert observed["seconds"] == 17
    assert observed["model_path"] == Path("models/custom.bin")


def test_cli_run_applies_llm_provider_and_cap_overrides(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_s05_run_stage(**kwargs):
        context = kwargs["context"]
        observed["llm_provider"] = context.settings.llm_provider
        observed["cost_cap_usd"] = context.settings.cost_cap_usd
        observed["token_cap_input"] = context.settings.token_cap_input
        observed["token_cap_output"] = context.settings.token_cap_output
        stage_dir = context.run_dir / "s05_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S05, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s05_asr_cleanup.run_stage", fake_s05_run_stage)

    run_dir = tmp_path / "run3"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(run_dir),
            "--from",
            "s05",
            "--to",
            "s05",
            "--llm-provider",
            "openai",
            "--cost-cap-usd",
            "0.5",
            "--token-cap-input",
            "1234",
            "--token-cap-output",
            "4321",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "[run] starting: s05->s05 in" in result.stdout
    assert observed["llm_provider"] is LlmProvider.OPENAI
    assert observed["cost_cap_usd"] == 0.5
    assert observed["token_cap_input"] == 1234
    assert observed["token_cap_output"] == 4321


def test_cli_single_stage_applies_llm_provider_and_caps(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_s07_run_stage(**kwargs):
        context = kwargs["context"]
        observed["llm_provider"] = context.settings.llm_provider
        observed["cost_cap_usd"] = context.settings.cost_cap_usd
        observed["token_cap_input"] = context.settings.token_cap_input
        observed["token_cap_output"] = context.settings.token_cap_output
        stage_dir = context.run_dir / "s07_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S07, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s07_editorial.run_stage", fake_s07_run_stage)

    run_dir = tmp_path / "run4"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "s07-editorial",
            str(run_dir),
            "--llm-provider",
            "openai",
            "--cost-cap-usd",
            "0.25",
            "--token-cap-input",
            "200",
            "--token-cap-output",
            "300",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert observed["llm_provider"] is LlmProvider.OPENAI
    assert observed["cost_cap_usd"] == 0.25
    assert observed["token_cap_input"] == 200
    assert observed["token_cap_output"] == 300


def test_cli_init_creates_run_dir_with_default_pipeline(tmp_path: Path) -> None:
    run_dir = tmp_path / "lecture-001"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    assert run_dir.is_dir()
    assert (run_dir / "pipeline.toml").exists()
    assert "initialized" in result.stdout
    assert load_settings(run_dir) == Settings()


def test_cli_init_fails_if_pipeline_already_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "lecture-002"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline.toml").write_text("[sruti]\nchunk_seconds = 12\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["init", str(run_dir)])

    assert result.exit_code == 2


def test_cli_help_lists_commands_with_short_descriptions() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.stdout
    expected: dict[str, str] = {
        "init": "Create RUN_DIR and prefill pipeline.toml",
        "run": "Run a stage range (s01-s09) in order.",
        "s01-normalize": "s01: Normalize input audio",
        "s02-chunk": "s02: Split normalized audio",
        "s03-asr": "s03: Transcribe audio chunks",
        "s04-merge": "s04: Merge per-chunk transcripts",
        "s05-asr-cleanup": "s05: LLM cleanup of ASR transcript errors.",
        "s06-remove-nonlecture": "s06: Remove non-lecture content",
        "s07-editorial": "s07: Editorially refine English text",
        "s08-translate": "s08: Faithful English-to-Czech translation.",
        "s09-translate-edit": "s09: Editorial polish of Czech translation.",
    }
    for command, description in expected.items():
        assert command in result.stdout
        assert description in result.stdout


def test_cli_run_help_describes_from_and_to_options() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "Start stage (inclusive)" in result.stdout
    assert "End stage (inclusive)" in result.stdout
