from model.model.qwen_audio_model import LitQwenAudioModel


class LitQwenNTPASR(LitQwenAudioModel):
    def __init__(self, config, token_list: str, use_frontend: bool = True):
        super().__init__(
            config=config,
            token_list=token_list,
            use_frontend=use_frontend,
            adapter_type="moe",
            decoder_type="ntp",
        )
