from __future__ import annotations

from sruti.application.stages._llm_text_transform import LlmTextTransformUseCase
from sruti.domain.enums import StageId
from sruti.llm.prompts import s07_editorial_prompt


class S07EditorialUseCase(LlmTextTransformUseCase):
    stage_name = StageId.S07.value
    stage_id = StageId.S07
    input_stage_id = StageId.S06
    input_filename = "content_only.txt"
    output_filename = "final_publishable_en.txt"
    model_setting_attr = "s07_model"
    temperature_setting_attr = "s07_temperature"
    prompt_builder = staticmethod(s07_editorial_prompt)
