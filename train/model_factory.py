from __future__ import annotations

from model.adapter.Denseadapter import DenseAdapter
from model.adapter.MoEadapter import MoEAdapter
from model.adapter.Qformer import QFormerAdapter
from model.decoder.frozen_qwen import FrozenQwenDecoder
from model.decoder.qwen import QwenDecoder
from model.encoder.kimi_audio_encoder import KimiAudioFrontend
from model.model.qwen_audio_model import LitQwenAudioModel
from train.class_choice import ClassChoices
from train.config import TrainConfig


frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        kimi_audio=KimiAudioFrontend,
    ),
    default="kimi_audio",
)

adapter_choices = ClassChoices(
    name="adapter",
    classes=dict(
        moe=MoEAdapter,
        dense=DenseAdapter,
        qformer=QFormerAdapter,
    ),
    default="moe",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(
        qwen=QwenDecoder,
        frozen_qwen=FrozenQwenDecoder,
    ),
    default="qwen",
)

model_choices = ClassChoices(
    name="model",
    classes=dict(
        qwen_audio_model=LitQwenAudioModel,
    ),
    default="qwen_audio_model",
)


def build_model_from_config(
    config: TrainConfig,
    data_type: list[str] | None = None,
):
    effective_data_type = (
        list(config.dataset.data_type) if data_type is None else list(data_type)
    )
    use_frontend = "sound" in effective_data_type
    frontend = build_frontend(config, use_frontend=use_frontend)
    input_hidden_size = frontend.hidden_size if frontend is not None else 3584
    adapter = build_adapter(config, input_hidden_size=input_hidden_size)
    decoder_input_size = getattr(adapter, "output_hidden_size", input_hidden_size)
    decoder = build_decoder(config, input_hidden_size=decoder_input_size)
    model_class = model_choices.get_class(config.model_name)
    if model_class is None:
        raise ValueError("--model must be provided")

    if config.model_name == "qwen_audio_model":
        return model_class(
            frontend=frontend,
            adapter=adapter,
            decoder=decoder,
            token_list=config.token_list,
            aux_loss_weight=config.model.aux_loss_weight,
        )
    raise ValueError(f"Unsupported model: {config.model_name}")


def build_frontend(config: TrainConfig, use_frontend: bool):
    if not use_frontend:
        return None
    frontend_class = frontend_choices.get_class("kimi_audio")
    freeze_frontend = bool(config.model.freeze_frontend or config.model.decoder_type == "frozen_qwen")
    return frontend_class(
        model_repo=config.model.model_repo,
        tokenizer_repo=config.model.tokenizer_repo,
        sample_rate=config.model.sample_rate,
        freeze_frontend=freeze_frontend,
    )


def build_adapter(config: TrainConfig, input_hidden_size: int):
    adapter_type = str(config.model.adapter_type).strip().lower()
    adapter_class = adapter_choices.get_class(adapter_type)
    adapter_conf = config.model.adapter
    if adapter_type == "moe":
        return adapter_class(
            hidden_size=input_hidden_size,
            expert_ffn_dim=adapter_conf.expert_ffn_dim,
            num_experts=adapter_conf.num_experts,
            top_k=adapter_conf.top_k,
        )
    if adapter_type == "dense":
        return adapter_class(
            hidden_size=input_hidden_size,
            ffn_dim=adapter_conf.expert_ffn_dim,
        )
    if adapter_type == "qformer":
        return adapter_class(
            speech_width=input_hidden_size,
            qformer_model_name=adapter_conf.qformer_model_name,
            num_query_token=adapter_conf.num_query_token,
            num_hidden_layers=adapter_conf.num_hidden_layers,
            cross_attention_freq=adapter_conf.cross_attention_freq,
            hidden_dropout_prob=adapter_conf.hidden_dropout_prob,
            attention_probs_dropout_prob=adapter_conf.attention_probs_dropout_prob,
        )
    raise ValueError(f"Unsupported adapter_type: {config.model.adapter_type}")


def build_decoder(config: TrainConfig, input_hidden_size: int):
    decoder_type = str(config.model.decoder_type).strip().lower()
    decoder_class = decoder_choices.get_class(decoder_type)
    return decoder_class(
        input_hidden_size=input_hidden_size,
        **config.model.llm_decoder.__dict__,
    )
