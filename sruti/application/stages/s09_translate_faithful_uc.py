from __future__ import annotations

from pathlib import Path

from sruti.application.context import StageContext
from sruti.application.stages._llm_text_transform import LlmTextTransformUseCase
from sruti.domain.enums import StageId, StageStatus
from sruti.llm.prompts import s09_translate_prompt
from sruti.util import manifest as manifest_util


class S09TranslateFaithfulUseCase(LlmTextTransformUseCase):
    stage_name = StageId.S09.value
    stage_id = StageId.S09
    input_stage_id = StageId.S08
    input_filename = "condensed_blocks_en.txt"
    output_filename = "translated_faithful_cs.txt"
    model_setting_attr = "s09_model"
    temperature_setting_attr = "s09_temperature"
    prompt_builder = staticmethod(s09_translate_prompt)

    def resolve_input(self, context: StageContext) -> tuple[Path, dict[str, object]]:
        s07_path = manifest_util.stage_dir_for(context.run_dir, StageId.S07.value) / "final_publishable_en.txt"
        s08_dir = manifest_util.stage_dir_for(context.run_dir, StageId.S08.value)
        s08_path = s08_dir / "condensed_blocks_en.txt"
        s07_signature = manifest_util.inputs_signature([s07_path])
        s08_manifest = self._manifest_store.load_stage_manifest(s08_dir)
        s08_is_current = (
            s08_manifest is not None
            and s08_manifest.status is StageStatus.SUCCESS
            and s08_manifest.params.get("_inputs_signature") == s07_signature
            and s08_path.exists()
        )
        if s08_is_current:
            return s08_path, {"input_source_stage": "s08"}
        return s07_path, {"input_source_stage": "s07"}
