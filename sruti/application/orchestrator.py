from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sruti.application.context import StageContext
from sruti.domain.enums import StageId, StageStatus
from sruti.domain.models import RunManifest, StageResult
from sruti.domain.policies import stage_ids_in_range
from sruti.domain.ports import StageUseCase
from sruti.util import manifest as manifest_util
from sruti.util.io import ensure_dir


@dataclass(slots=True)
class PipelineOrchestrator:
    stages: dict[StageId, StageUseCase]

    def run_range(
        self,
        *,
        context: StageContext,
        from_stage: StageId,
        to_stage: StageId,
    ) -> list[StageResult]:
        ensure_dir(context.run_dir)
        run_manifest = manifest_util.load_run_manifest(context.run_dir)
        if run_manifest is None:
            run_manifest = RunManifest(
                run_id=context.run_dir.name,
                run_dir=str(context.run_dir),
                settings=context.settings.model_dump(mode="json"),
                requested_range={"from": from_stage.value, "to": to_stage.value},
            )

        results: list[StageResult] = []
        for stage_id in stage_ids_in_range(from_stage, to_stage):
            use_case = self.stages[stage_id]
            result = use_case.run(context)
            run_manifest.stages[stage_id.value] = result.status
            results.append(result)
            if result.status is StageStatus.FAILED:
                break

        manifest_util.save_run_manifest(context.run_dir, run_manifest)
        return results


def parse_stage_id(value: str) -> StageId:
    try:
        return StageId(value)
    except ValueError as exc:  # pragma: no cover - defensive
        valid = ", ".join(item.value for item in StageId)
        raise ValueError(f"Unknown stage '{value}'. Expected one of: {valid}") from exc


def stage_dir(run_dir: Path, stage_id: StageId) -> Path:
    return manifest_util.stage_dir_for(run_dir, stage_id.value)
