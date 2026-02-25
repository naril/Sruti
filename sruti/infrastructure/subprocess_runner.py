from __future__ import annotations

import subprocess
from pathlib import Path

from sruti.domain.errors import StageExecutionError


class SubprocessShellRunner:
    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> str:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                check=True,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            details = stderr or stdout or "unknown subprocess error"
            raise StageExecutionError(
                f"Command failed ({' '.join(command)}): {details}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise StageExecutionError(f"Command timeout ({' '.join(command)}).") from exc

        return completed.stdout
