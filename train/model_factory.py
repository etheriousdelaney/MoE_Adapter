from __future__ import annotations

from model.model.asr_model import LitMoEASR
from model.model.moe_classifier import LitMoEClassifier
from model.model.qwen_audio_model import LitQwenAudioModel
from model.model.qwen_dense_asr_model import LitQwenDenseASR
from model.model.qwen_frozen_adapter_instruction_model import (
    LitQwenFrozenAdapterASR,
    LitQwenFrozenAdapterInstructionModel,
)
from model.model.qwen_ntp_asr_model import LitQwenNTPASR
from model.model.qwen_qformer_asr_model import LitQwenQFormerASR
from model.model.testmodel import TestModel
from train.class_choice import ClassChoices
from train.config import TrainConfig


model_choices = ClassChoices(
    name="model",
    classes=dict(
        MoEClassifier=LitMoEClassifier,
        asr_model=LitMoEASR,
        qwen_dense_asr_model=LitQwenDenseASR,
        qwen_audio_model=LitQwenAudioModel,
        qwen_frozen_adapter_asr_model=LitQwenFrozenAdapterASR,
        qwen_frozen_adapter_instruction_model=LitQwenFrozenAdapterInstructionModel,
        qwen_ntp_asr_model=LitQwenNTPASR,
        qwen_qformer_asr_model=LitQwenQFormerASR,
        test_model=TestModel,
    ),
)


def build_model_from_config(
    config: TrainConfig,
    data_type: list[str] | None = None,
):
    model_class = model_choices.get_class(config.model_name)
    if model_class is None:
        raise ValueError("--model must be provided")

    effective_data_type = (
        list(config.dataset.data_type) if data_type is None else list(data_type)
    )
    use_frontend = "sound" in effective_data_type

    if config.model_name == "moeclassifier":
        return model_class(
            config=config.model,
            use_frontend=use_frontend,
        )
    if config.model_name == "asr_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    if config.model_name == "qwen_ntp_asr_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    if config.model_name == "qwen_dense_asr_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    if config.model_name == "qwen_audio_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    if config.model_name == "qwen_frozen_adapter_asr_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    if config.model_name == "qwen_frozen_adapter_instruction_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    if config.model_name == "qwen_qformer_asr_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    return model_class()
