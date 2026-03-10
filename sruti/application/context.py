from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sruti.application.batch_scheduler import ExecutionCoordinator
from sruti.config import Settings, load_settings
from sruti.domain.enums import LlmProvider, OnExistsMode


def _noop_progress(_: str) -> None:
    return


@dataclass(frozen=True)
class StageContext:
    run_dir: Path
    settings: Settings
    on_exists: OnExistsMode
    dry_run: bool
    force: bool
    verbose: bool
    is_tty: bool
    progress_emitter: Callable[[str], None] = _noop_progress
    execution_coordinator: ExecutionCoordinator | None = None

    @classmethod
    def build(
        cls,
        *,
        run_dir: Path,
        settings_dir: Path | None = None,
        on_exists: OnExistsMode,
        dry_run: bool,
        force: bool,
        verbose: bool,
        llm_provider_override: LlmProvider | None = None,
        cost_cap_usd_override: float | None = None,
        token_cap_input_override: int | None = None,
        token_cap_output_override: int | None = None,
        progress_emitter: Callable[[str], None] | None = None,
        execution_coordinator: ExecutionCoordinator | None = None,
    ) -> "StageContext":
        effective_settings_dir = settings_dir if settings_dir is not None else run_dir
        settings = load_settings(run_dir=effective_settings_dir)
        override_values: dict[str, object] = {}
        prompt_templates_dir = settings.prompt_templates_dir
        if prompt_templates_dir is not None and not prompt_templates_dir.is_absolute():
            override_values["prompt_templates_dir"] = effective_settings_dir / prompt_templates_dir
        if llm_provider_override is not None:
            override_values["llm_provider"] = llm_provider_override
        if cost_cap_usd_override is not None:
            override_values["cost_cap_usd"] = cost_cap_usd_override
        if token_cap_input_override is not None:
            override_values["token_cap_input"] = token_cap_input_override
        if token_cap_output_override is not None:
            override_values["token_cap_output"] = token_cap_output_override
        if override_values:
            settings = settings.model_copy(update=override_values)
        return cls(
            run_dir=run_dir,
            settings=settings,
            on_exists=on_exists,
            dry_run=dry_run,
            force=force,
            verbose=verbose,
            is_tty=sys.stdin.isatty(),
            progress_emitter=progress_emitter if progress_emitter is not None else _noop_progress,
            execution_coordinator=execution_coordinator,
        )

    def emit_progress(self, message: str, *, verbose_only: bool = False) -> None:
        if verbose_only and not self.verbose:
            return
        self.progress_emitter(message)
