from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sruti.infrastructure import json_codec


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    text = json_codec.dumps(data, indent=indent) + "\n"
    atomic_write_text(path, text)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json_codec.dumps(row) for row in rows)
    if content:
        content += "\n"
    atomic_write_text(path, content)
