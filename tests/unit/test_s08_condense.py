from __future__ import annotations

import json
from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages.s08_condense_uc import S08CondenseUseCase
from sruti.domain.enums import OnExistsMode, StageStatus
from sruti.domain.models import LlmGenerateResult
from sruti.infrastructure.fs_repository import FileSystemManifestStore
from sruti.util import manifest as manifest_util


class FakeOllama:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.map_calls = 0
        self.prompts: list[str] = []

    def ensure_model_available(self, model: str) -> None:
        _ = model

    def provider_name(self) -> str:
        return "local"

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
    ) -> LlmGenerateResult:
        _ = (model, temperature, timeout_seconds)
        self.generate_calls += 1
        self.prompts.append(prompt)
        if "Candidate blocks JSON:" not in prompt and "CUSTOM REDUCE" not in prompt:
            self.map_calls += 1
            if self.map_calls == 1:
                return LlmGenerateResult(
                    text=(
                        '{"blocks":[{"from_paragraph":1,"to_paragraph":4,"title":"Intro","body":"A."},'
                        '{"from_paragraph":5,"to_paragraph":8,"title":"Middle","body":"B."}]}'
                    )
                )
            return LlmGenerateResult(
                text='{"blocks":[{"from_paragraph":8,"to_paragraph":9,"title":"Tail","body":"C."}]}'
            )
        return LlmGenerateResult(text="## Block 01: Intro\nA.\n\n## Block 02: Middle\nB.\n\nC.")


def _ctx(run_dir: Path, *, dry_run: bool = False) -> StageContext:
    return StageContext.build(
        run_dir=run_dir,
        on_exists=OnExistsMode.OVERWRITE,
        dry_run=dry_run,
        force=False,
        verbose=False,
    )


def test_s08_condense_happy_path_merges_overlap_and_strips_block_headings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("sruti.application.stages.s08_condense_uc.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    paragraphs = [f"Paragraph {idx}." for idx in range(1, 10)]
    (s07_dir / "final_publishable_en.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")
    use_case = S08CondenseUseCase(
        llm_client=FakeOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS

    output_path = tmp_path / "s08_condense" / "condensed_blocks_en.txt"
    output = output_path.read_text(encoding="utf-8")
    assert "## Block 01: Intro" not in output
    assert "## Block 02: Middle" not in output
    assert "A." in output
    assert "B." in output
    assert "C." in output

    candidate_rows = json.loads((tmp_path / "s08_condense" / "candidate_blocks.json").read_text(encoding="utf-8"))
    assert candidate_rows == [
        {"from_paragraph": 1, "to_paragraph": 4, "title": "Intro", "body": "A."},
        {"from_paragraph": 5, "to_paragraph": 9, "title": "Middle", "body": "B.\n\nC."},
    ]


def test_s08_condense_empty_input_produces_empty_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sruti.application.stages.s08_condense_uc.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    (s07_dir / "final_publishable_en.txt").write_text(" \n\n\t", encoding="utf-8")
    ollama = FakeOllama()
    use_case = S08CondenseUseCase(
        llm_client=ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert ollama.generate_calls == 0
    assert (tmp_path / "s08_condense" / "condensed_blocks_en.txt").read_text(encoding="utf-8") == ""


def test_s08_condense_uses_prompt_templates_dir_from_pipeline_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("sruti.application.stages.s08_condense_uc.require_executable", lambda _: None)
    (tmp_path / "pipeline.toml").write_text(
        """
[sruti]
prompt_templates_dir = "prompts"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "s08_condense_map.txt").write_text("CUSTOM MAP\n{{paragraph_lines}}\n", encoding="utf-8")
    (prompts_dir / "s08_condense_reduce.txt").write_text(
        "CUSTOM REDUCE\n{{candidate_blocks_json}}\n",
        encoding="utf-8",
    )
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    (s07_dir / "final_publishable_en.txt").write_text("Paragraph one.", encoding="utf-8")

    ollama = FakeOllama()
    use_case = S08CondenseUseCase(
        llm_client=ollama,
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    assert ollama.generate_calls == 2
    assert any("CUSTOM MAP" in prompt for prompt in ollama.prompts)
    assert any("CUSTOM REDUCE" in prompt for prompt in ollama.prompts)
    manifest_data = json.loads((tmp_path / "s08_condense" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["params"]["prompt_templates_dir"] == str(tmp_path / "prompts")


def test_s08_condense_accepts_fenced_json_from_map_step(monkeypatch, tmp_path: Path) -> None:
    class FencedMapOllama(FakeOllama):
        def generate(
            self,
            *,
            model: str,
            prompt: str,
            temperature: float,
            timeout_seconds: int | None = None,
        ) -> LlmGenerateResult:
            _ = (model, temperature, timeout_seconds)
            self.generate_calls += 1
            self.prompts.append(prompt)
            if "Candidate blocks JSON:" not in prompt:
                return LlmGenerateResult(
                    text=(
                        "```json\n"
                        '{"blocks":[{"from_paragraph":1,"to_paragraph":1,"title":"Intro","body":"A."}]}\n'
                        "```"
                    )
                )
            return LlmGenerateResult(text="A.")

    monkeypatch.setattr("sruti.application.stages.s08_condense_uc.require_executable", lambda _: None)
    s07_dir = manifest_util.stage_dir_for(tmp_path, "s07")
    s07_dir.mkdir(parents=True, exist_ok=True)
    (s07_dir / "final_publishable_en.txt").write_text("Paragraph one.", encoding="utf-8")

    use_case = S08CondenseUseCase(
        llm_client=FencedMapOllama(),
        manifest_store=FileSystemManifestStore(),
    )
    result = use_case.run(_ctx(tmp_path))
    assert result.status == StageStatus.SUCCESS
    candidate_rows = json.loads((tmp_path / "s08_condense" / "candidate_blocks.json").read_text(encoding="utf-8"))
    assert candidate_rows == [
        {"from_paragraph": 1, "to_paragraph": 1, "title": "Intro", "body": "A."},
    ]
