import os
from dataclasses import dataclass
from typing import Sequence

import librosa
import lightning as L
import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from transformers.utils import logging as transformers_logging

from .tokenizer.gim4_tokenizer import Glm4Tokenizer
from .whisper.whisperencoder import WhisperEncoder


@dataclass
class FrontendBatch:
    fused_states: torch.Tensor
    padding_mask: torch.Tensor
    token_ids: torch.Tensor
    continuous_features: torch.Tensor
    lengths: torch.Tensor


class KimiAudioFrontend(nn.Module):
    def __init__(
        self,
        model_repo: str = "moonshotai/Kimi-Audio-7B-Instruct",
        tokenizer_repo: str = "THUDM/glm-4-voice-tokenizer",
        sample_rate: int = 16000,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        freeze_frontend: bool = True,
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        hf_disable_progress_bars()
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
        self.sample_rate = sample_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        logger.info("loading kimi-audio frontend")
        cache_path = snapshot_download(model_repo)
        alm = AutoModelForCausalLM.from_pretrained(
            cache_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        model_config = alm.config
        self.embed_tokens = alm.model.embed_tokens
        self.vq_adaptor = alm.model.vq_adaptor
        del alm
        self.audio_tokenizer = Glm4Tokenizer(tokenizer_repo)
        self.whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"),
            mel_batch_size=20,
        ).to(self.device)
        self.whisper_model = self.whisper_model.to(dtype)
        self.whisper_model.eval()

        self.kimia_token_offset = model_config.kimia_token_offset
        self.hidden_size = self.embed_tokens.embedding_dim

        if freeze_frontend:
            self.freeze()

    def freeze(self) -> None:
        self.eval()
        for module in (self.embed_tokens, self.vq_adaptor, self.whisper_model):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def _load_audio(self, audio_input: str | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio, _ = librosa.load(audio_input, sr=self.sample_rate)
            return torch.tensor(audio, dtype=torch.float32)
        if isinstance(audio_input, np.ndarray):
            return torch.from_numpy(audio_input).to(torch.float32).flatten()
        if isinstance(audio_input, torch.Tensor):
            return audio_input.detach().to(torch.float32).cpu().flatten()
        raise TypeError(f"Unsupported audio input type: {type(audio_input)!r}")

    @torch.inference_mode()
    def extract_audio_feat(
        self, audio_input: str | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self._load_audio(audio_input)
        audio_info = (audio.unsqueeze(0), self.sample_rate)

        wav_tokens = self.audio_tokenizer.tokenize(audio_info)
        wav_tokens = (wav_tokens + self.kimia_token_offset).squeeze(0).to(torch.long)

        wav_tensor = audio.unsqueeze(0).to(self.device)
        continuous_feature = self.whisper_model.tokenize_waveform(wav_tensor)
        continuous_feature = continuous_feature.reshape(
            continuous_feature.shape[0],
            int(continuous_feature.shape[1] // 4),
            continuous_feature.shape[2] * 4,
        ).squeeze(0)

        seq_len = min(wav_tokens.shape[0], continuous_feature.shape[0])
        wav_tokens = wav_tokens[:seq_len]
        continuous_feature = continuous_feature[:seq_len]
        return wav_tokens, continuous_feature

    @torch.inference_mode()
    def build_fused_states(
        self,
        input_ids: torch.Tensor,
        whisper_input_feature: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        whisper_input_feature = whisper_input_feature.to(self.device, dtype=self.dtype)

        if whisper_input_feature.ndim == 2:
            whisper_input_feature = whisper_input_feature.unsqueeze(0)

        audio_emb = self.embed_tokens(input_ids)
        whisper_emb = self.vq_adaptor(whisper_input_feature.transpose(0, 1)).transpose(0, 1)
        return (audio_emb + whisper_emb) * torch.sqrt(
            torch.tensor(2.0, dtype=whisper_emb.dtype, device=audio_emb.device)
        )

    @torch.inference_mode()
    def forward(
        self,
        audio_inputs: Sequence[str | np.ndarray | torch.Tensor] | str | np.ndarray | torch.Tensor,
    ) -> FrontendBatch:
        if isinstance(audio_inputs, (str, np.ndarray, torch.Tensor)):
            audio_inputs = [audio_inputs]

        token_ids_list: list[torch.Tensor] = []
        continuous_features_list: list[torch.Tensor] = []
        fused_states_list: list[torch.Tensor] = []
        lengths: list[int] = []

        for audio_input in audio_inputs:
            token_ids, continuous_feature = self.extract_audio_feat(audio_input)
            fused_states = self.build_fused_states(
                token_ids.unsqueeze(0),
                continuous_feature.unsqueeze(0),
            ).squeeze(0)
            token_ids_list.append(token_ids.cpu())
            continuous_features_list.append(continuous_feature.cpu())
            fused_states_list.append(fused_states.cpu())
            lengths.append(fused_states.shape[0])

        max_len = max(lengths)
        padded_token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=0)
        padded_continuous = pad_sequence(
            continuous_features_list,
            batch_first=True,
            padding_value=0.0,
        )
        padded_fused = pad_sequence(fused_states_list, batch_first=True, padding_value=0.0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        padding_mask = torch.arange(max_len).unsqueeze(0) < lengths_tensor.unsqueeze(1)

        return FrontendBatch(
            fused_states=padded_fused,
            padding_mask=padding_mask,
            token_ids=padded_token_ids,
            continuous_features=padded_continuous,
            lengths=lengths_tensor,
        )


class LitKimiAudioEncoder(L.LightningModule):
    def __init__(self, **frontend_kwargs):
        super().__init__()
        self.frontend = KimiAudioFrontend(**frontend_kwargs)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        audio_inputs = batch
        if isinstance(batch, dict):
            audio_inputs = batch.get("audio_path") or batch.get("path") or batch.get("audio")
        elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], str):
            audio_inputs = list(batch)
        return self.frontend(audio_inputs)
