from __future__ import annotations

from model.model.qwen_audio_model import LitQwenAudioModel
from train.class_choice import ClassChoices
from train.config import TrainConfig


model_choices = ClassChoices(
    name="model",
    classes=dict(
        qwen_audio_model=LitQwenAudioModel,
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

    if config.model_name == "qwen_audio_model":
        return model_class(
            config=config.model,
            token_list=config.token_list,
            use_frontend=use_frontend,
        )
    raise ValueError(f"Unsupported model: {config.model_name}")
