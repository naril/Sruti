from __future__ import annotations

import json
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


def test_cli_deprecated_alias_s08_translate_routes_to_s09_and_warns(monkeypatch, tmp_path: Path) -> None:
    observed: list[str] = []

    def fake_s09_run_stage(**kwargs):
        context = kwargs["context"]
        observed.append(context.run_dir.name)
        stage_dir = context.run_dir / "s09_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S09, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s09_translate_faithful.run_stage", fake_s09_run_stage)
    run_dir = tmp_path / "run-alias-1"
    runner = CliRunner()
    result = runner.invoke(app, ["s08-translate", str(run_dir), "--on-exists", "overwrite"])
    assert result.exit_code == 0, result.stdout
    assert observed == ["run-alias-1"]
    assert "[deprecated] 's08-translate' was moved to 's09-translate'." in result.stderr


def test_cli_deprecated_alias_s09_translate_edit_routes_to_s10_and_warns(
    monkeypatch, tmp_path: Path
) -> None:
    observed: list[str] = []

    def fake_s10_run_stage(**kwargs):
        context = kwargs["context"]
        observed.append(context.run_dir.name)
        stage_dir = context.run_dir / "s10_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S10, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s10_translate_edit.run_stage", fake_s10_run_stage)
    run_dir = tmp_path / "run-alias-2"
    runner = CliRunner()
    result = runner.invoke(app, ["s09-translate-edit", str(run_dir), "--on-exists", "overwrite"])
    assert result.exit_code == 0, result.stdout
    assert observed == ["run-alias-2"]
    assert "[deprecated] 's09-translate-edit' was moved to 's10-translate-edit'." in result.stderr


def test_cli_run_batch_processes_recursive_audio_files_into_unique_run_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "pipeline.toml").write_text("[sruti]\nchunk_seconds = 30\n", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    (input_dir / "nested").mkdir(parents=True, exist_ok=True)
    (input_dir / "lecture.wav").write_bytes(b"a")
    (input_dir / "nested" / "lecture.MP3").write_bytes(b"b")
    (input_dir / "nested" / "ignore.txt").write_text("x", encoding="utf-8")

    def fake_s01_run_stage(**kwargs):
        context = kwargs["context"]
        input_audio = kwargs["input_audio"]
        calls.append((str(input_audio.relative_to(input_dir)), context.run_dir.name))
        stage_dir = context.run_dir / "s01_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S01, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s01_normalize.run_stage", fake_s01_run_stage)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-batch",
            str(runs_root),
            "--in-dir",
            str(input_dir),
            "--from",
            "s01",
            "--to",
            "s01",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls == [("lecture.wav", "lecture"), ("nested/lecture.MP3", "lecture-2")]
    manifest_data = json.loads((runs_root / "batch_manifest.json").read_text(encoding="utf-8"))
    mapping = manifest_data["audio_to_run_dir"]
    assert mapping[str((input_dir / "lecture.wav").resolve())] == "lecture"
    assert mapping[str((input_dir / "nested" / "lecture.MP3").resolve())] == "lecture-2"


def test_cli_run_batch_uses_shared_pipeline_settings_for_each_run(
    monkeypatch, tmp_path: Path
) -> None:
    observed: list[tuple[int, Path | None]] = []
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

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "lecture.wav").write_bytes(b"a")

    def fake_s02_run_stage(**kwargs):
        context = kwargs["context"]
        observed.append((kwargs["seconds"], context.settings.prompt_templates_dir))
        stage_dir = context.run_dir / "s02_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S02, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s02_chunk.run_stage", fake_s02_run_stage)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-batch",
            str(runs_root),
            "--in-dir",
            str(input_dir),
            "--from",
            "s02",
            "--to",
            "s02",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert observed == [(17, runs_root / "prompts")]


def test_cli_run_batch_continues_after_failure_and_returns_nonzero(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "pipeline.toml").write_text("[sruti]\nchunk_seconds = 30\n", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "bad.wav").write_bytes(b"a")
    (input_dir / "good.wav").write_bytes(b"b")

    def fake_s01_run_stage(**kwargs):
        input_audio = kwargs["input_audio"]
        context = kwargs["context"]
        calls.append(input_audio.stem)
        if input_audio.stem == "bad":
            raise ValueError("boom")
        stage_dir = context.run_dir / "s01_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S01, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s01_normalize.run_stage", fake_s01_run_stage)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-batch",
            str(runs_root),
            "--in-dir",
            str(input_dir),
            "--from",
            "s01",
            "--to",
            "s01",
            "--on-exists",
            "overwrite",
        ],
    )
    assert result.exit_code == 1
    assert calls == ["bad", "good"]
    assert "[run-batch] summary: 1 succeeded, 1 failed (total 2)." in result.stdout
    assert (runs_root / "good" / "s01_fake").is_dir()


def test_cli_run_batch_preserves_existing_mapping_and_adds_new_files(
    monkeypatch, tmp_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "pipeline.toml").write_text("[sruti]\nchunk_seconds = 30\n", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    (input_dir / "nested").mkdir(parents=True, exist_ok=True)
    (input_dir / "alpha.wav").write_bytes(b"a")
    (input_dir / "nested" / "alpha.wav").write_bytes(b"b")

    def fake_s01_run_stage(**kwargs):
        context = kwargs["context"]
        stage_dir = context.run_dir / "s01_fake"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return StageResult(stage=StageId.S01, status=StageStatus.SUCCESS, stage_dir=stage_dir, outputs=[])

    monkeypatch.setattr("sruti.cli.s01_normalize.run_stage", fake_s01_run_stage)

    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "run-batch",
            str(runs_root),
            "--in-dir",
            str(input_dir),
            "--from",
            "s01",
            "--to",
            "s01",
            "--on-exists",
            "overwrite",
        ],
    )
    assert first.exit_code == 0, first.stdout
    first_mapping = json.loads((runs_root / "batch_manifest.json").read_text(encoding="utf-8"))[
        "audio_to_run_dir"
    ]

    (input_dir / "other").mkdir(parents=True, exist_ok=True)
    new_audio = input_dir / "other" / "alpha.wav"
    new_audio.write_bytes(b"c")
    second = runner.invoke(
        app,
        [
            "run-batch",
            str(runs_root),
            "--in-dir",
            str(input_dir),
            "--from",
            "s01",
            "--to",
            "s01",
            "--on-exists",
            "overwrite",
        ],
    )
    assert second.exit_code == 0, second.stdout
    second_mapping = json.loads((runs_root / "batch_manifest.json").read_text(encoding="utf-8"))[
        "audio_to_run_dir"
    ]

    for key, value in first_mapping.items():
        assert second_mapping[key] == value
    assert second_mapping[str(new_audio.resolve())] == "alpha-3"


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
        "run": "Run a stage range (s01-s10) in order.",
        "run-batch": "Run a stage range (s01-s10) over all audio files",
        "s01-normalize": "s01: Normalize input audio",
        "s02-chunk": "s02: Split normalized audio",
        "s03-asr": "s03: Transcribe audio chunks",
        "s04-merge": "s04: Merge per-chunk transcripts",
        "s05-asr-cleanup": "s05: LLM cleanup of ASR transcript errors.",
        "s06-remove-nonlecture": "s06: Remove non-lecture content",
        "s07-editorial": "s07: Editorially refine English text",
        "s08-condense": "s08: Lightly condense English text",
        "s09-translate": "s09: Faithful English-to-Czech translation.",
        "s10-translate-edit": "s10: Editorial polish of Czech translation.",
        "s08-translate": "DEPRECATED alias for s09-translate.",
        "s09-translate-edit": "DEPRECATED alias for s10-translate-edit.",
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
