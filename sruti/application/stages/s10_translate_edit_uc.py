from __future__ import annotations

from sruti.application.stages._llm_text_transform import LlmTextTransformUseCase
from sruti.domain.enums import StageId
from sruti.llm.prompts import s10_czech_editorial_prompt


class S10TranslateEditUseCase(LlmTextTransformUseCase):
    stage_name = StageId.S10.value
    stage_id = StageId.S10
    input_stage_id = StageId.S09
    input_filename = "translated_faithful_cs.txt"
    output_filename = "final_publishable_cs.txt"
    model_setting_attr = "s10_model"
    temperature_setting_attr = "s10_temperature"
    prompt_builder = staticmethod(s10_czech_editorial_prompt)
