from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from sruti.config import Settings, load_settings
from sruti.domain.enums import OnExistsMode


@dataclass(frozen=True)
class StageContext:
    run_dir: Path
    settings: Settings
    on_exists: OnExistsMode
    dry_run: bool
    force: bool
    verbose: bool
    is_tty: bool

    @classmethod
    def build(
        cls,
        *,
        run_dir: Path,
        on_exists: OnExistsMode,
        dry_run: bool,
        force: bool,
        verbose: bool,
    ) -> "StageContext":
        settings = load_settings(run_dir=run_dir)
        return cls(
            run_dir=run_dir,
            settings=settings,
            on_exists=on_exists,
            dry_run=dry_run,
            force=force,
            verbose=verbose,
            is_tty=sys.stdin.isatty(),
        )
