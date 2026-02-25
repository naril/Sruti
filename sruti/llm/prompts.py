from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def _template_dir() -> Path:
    override = os.getenv("SRUTI_PROMPTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "prompt_templates"


def _load_template(name: str) -> str:
    path = _template_dir() / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_template(name: str, values: Mapping[str, str]) -> str:
    template = _load_template(name)
    placeholders = set(PLACEHOLDER_PATTERN.findall(template))
    missing = placeholders - set(values.keys())
    if missing:
        missing_values = ", ".join(sorted(missing))
        raise ValueError(f"Missing prompt template values for {name}: {missing_values}")

    rendered = template
    for placeholder in placeholders:
        rendered = rendered.replace(f"{{{{{placeholder}}}}}", values[placeholder])
    return rendered


def s05_cleanup_prompt(text: str) -> str:
    return _render_template("s05_cleanup.txt", {"text": text})


def s06_classification_prompt(span_lines: str) -> str:
    return _render_template("s06_classification.txt", {"span_lines": span_lines})


def s06_repair_json_prompt(original_prompt: str, bad_response: str) -> str:
    return _render_template(
        "s06_repair_json.txt",
        {"original_prompt": original_prompt, "bad_response": bad_response},
    )


def s07_editorial_prompt(text: str) -> str:
    return _render_template("s07_editorial.txt", {"text": text})


def s08_translate_prompt(text: str) -> str:
    return _render_template("s08_translate.txt", {"text": text})


def s09_czech_editorial_prompt(text: str) -> str:
    return _render_template("s09_czech_editorial.txt", {"text": text})
