from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"


def _env_template_dir() -> Path | None:
    override = os.getenv("SRUTI_PROMPTS_DIR")
    if override:
        return Path(override)
    return None


def _template_roots(*, template_dir: Path | None = None) -> list[Path]:
    if template_dir is not None:
        if not template_dir.exists():
            raise FileNotFoundError(f"Prompt template directory not found: {template_dir}")
        if not template_dir.is_dir():
            raise FileNotFoundError(f"Prompt template path is not a directory: {template_dir}")
        return [template_dir, DEFAULT_TEMPLATE_DIR]
    env_dir = _env_template_dir()
    if env_dir is not None:
        return [env_dir]
    return [DEFAULT_TEMPLATE_DIR]


def _load_template(name: str, *, template_dir: Path | None = None) -> str:
    roots = _template_roots(template_dir=template_dir)
    for root in roots:
        path = root / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    if template_dir is not None:
        raise FileNotFoundError(
            f"Prompt template not found: {template_dir / name} "
            f"(fallback: {DEFAULT_TEMPLATE_DIR / name})"
        )
    raise FileNotFoundError(f"Prompt template not found: {roots[0] / name}")


def _load_template_with_fallback(
    names: list[str],
    *,
    template_dir: Path | None = None,
) -> tuple[str, str]:
    roots = _template_roots(template_dir=template_dir)
    for root in roots:
        for name in names:
            path = root / name
            if path.exists():
                return path.read_text(encoding="utf-8"), name
    joined = ", ".join(names)
    if template_dir is not None:
        raise FileNotFoundError(
            f"Prompt template not found in {template_dir} or {DEFAULT_TEMPLATE_DIR}: {joined}"
        )
    raise FileNotFoundError(f"Prompt template not found: {joined}")


def _render_template(
    name: str,
    values: Mapping[str, str],
    *,
    template_dir: Path | None = None,
) -> str:
    template = _load_template(name, template_dir=template_dir)
    placeholders = set(PLACEHOLDER_PATTERN.findall(template))
    missing = placeholders - set(values.keys())
    if missing:
        missing_values = ", ".join(sorted(missing))
        raise ValueError(f"Missing prompt template values for {name}: {missing_values}")

    rendered = template
    for placeholder in placeholders:
        rendered = rendered.replace(f"{{{{{placeholder}}}}}", values[placeholder])
    return rendered


def _render_template_with_fallback(
    names: list[str],
    values: Mapping[str, str],
    *,
    template_dir: Path | None = None,
) -> str:
    template, resolved_name = _load_template_with_fallback(names, template_dir=template_dir)
    placeholders = set(PLACEHOLDER_PATTERN.findall(template))
    missing = placeholders - set(values.keys())
    if missing:
        missing_values = ", ".join(sorted(missing))
        raise ValueError(f"Missing prompt template values for {resolved_name}: {missing_values}")

    rendered = template
    for placeholder in placeholders:
        rendered = rendered.replace(f"{{{{{placeholder}}}}}", values[placeholder])
    return rendered


def s05_cleanup_prompt(text: str, *, template_dir: Path | None = None) -> str:
    return _render_template("s05_cleanup.txt", {"text": text}, template_dir=template_dir)


def s06_classification_prompt(span_lines: str, *, template_dir: Path | None = None) -> str:
    return _render_template(
        "s06_classification.txt",
        {"span_lines": span_lines},
        template_dir=template_dir,
    )


def s06_repair_json_prompt(
    original_prompt: str,
    bad_response: str,
    *,
    template_dir: Path | None = None,
) -> str:
    return _render_template(
        "s06_repair_json.txt",
        {"original_prompt": original_prompt, "bad_response": bad_response},
        template_dir=template_dir,
    )


def s07_editorial_prompt(text: str, *, template_dir: Path | None = None) -> str:
    return _render_template("s07_editorial.txt", {"text": text}, template_dir=template_dir)


def s08_condense_map_prompt(paragraph_lines: str, *, template_dir: Path | None = None) -> str:
    return _render_template(
        "s08_condense_map.txt",
        {"paragraph_lines": paragraph_lines},
        template_dir=template_dir,
    )


def s08_condense_reduce_prompt(
    candidate_blocks_json: str,
    *,
    template_dir: Path | None = None,
) -> str:
    return _render_template(
        "s08_condense_reduce.txt",
        {"candidate_blocks_json": candidate_blocks_json},
        template_dir=template_dir,
    )


def s09_translate_prompt(text: str, *, template_dir: Path | None = None) -> str:
    return _render_template_with_fallback(
        ["s09_translate.txt", "s08_translate.txt"],
        {"text": text},
        template_dir=template_dir,
    )


def s10_czech_editorial_prompt(text: str, *, template_dir: Path | None = None) -> str:
    return _render_template_with_fallback(
        ["s10_czech_editorial.txt", "s09_czech_editorial.txt"],
        {"text": text},
        template_dir=template_dir,
    )


def s08_translate_prompt(text: str, *, template_dir: Path | None = None) -> str:
    return s09_translate_prompt(text, template_dir=template_dir)


def s09_czech_editorial_prompt(text: str, *, template_dir: Path | None = None) -> str:
    return s10_czech_editorial_prompt(text, template_dir=template_dir)
