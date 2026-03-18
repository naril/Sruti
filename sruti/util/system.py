from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from sruti.domain.errors import DependencyMissingError


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise DependencyMissingError(f"Missing required executable: {name}")


def executable_version(command: list[str]) -> str:
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=True)
    except Exception:  # pragma: no cover - best effort only
        return "unknown"
    return (out.stdout.strip() or out.stderr.strip() or "unknown").splitlines()[0]


def require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise DependencyMissingError(f"{label} not found: {path}")
