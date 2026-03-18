from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable


class GuiJobManager:
    def __init__(self, *, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sruti-gui")
        self._lock = Lock()
        self._jobs: dict[str, Future[object]] = {}

    def active_jobs(self) -> int:
        self._prune()
        with self._lock:
            return len(self._jobs)

    def is_running(self, project_dir: Path) -> bool:
        self._prune()
        with self._lock:
            future = self._jobs.get(str(project_dir))
            return future is not None and not future.done()

    def submit(self, project_dir: Path, fn: Callable[[], object]) -> Future[object]:
        self._prune()
        key = str(project_dir)
        with self._lock:
            current = self._jobs.get(key)
            if current is not None and not current.done():
                raise ValueError(f"Job for {project_dir.name} is already running.")
            future = self._executor.submit(fn)
            self._jobs[key] = future
            return future

    def _prune(self) -> None:
        with self._lock:
            done_keys = [key for key, future in self._jobs.items() if future.done()]
            for key in done_keys:
                self._jobs.pop(key, None)
