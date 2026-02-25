from __future__ import annotations

from sruti.application.stages._llm_text_transform import LlmTextTransformUseCase
from sruti.domain.enums import StageId
from sruti.llm.prompts import s08_translate_prompt


class S08TranslateFaithfulUseCase(LlmTextTransformUseCase):
    stage_name = StageId.S08.value
    stage_id = StageId.S08
    input_stage_id = StageId.S07
    input_filename = "final_publishable_en.txt"
    output_filename = "translated_faithful_cs.txt"
    model_setting_attr = "s08_model"
    temperature_setting_attr = "s08_temperature"
    prompt_builder = staticmethod(s08_translate_prompt)
