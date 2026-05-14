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

from model.tokenizer.gim4_tokenizer import Glm4Tokenizer
from model.whisper.whisperencoder import WhisperEncoder


@dataclass
class FrontendBatch:
    fused_states: torch.Tensor
    padding_mask: torch.Tensor
    token_ids: torch.Tensor
    continuous_features: torch.Tensor
    lengths: torch.Tensor


def build_padding_mask_from_lengths(
    lengths: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, but got shape={tuple(lengths.shape)}")
    if max_len is None:
        max_len = int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


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

    def _load_audio(
        self,
        audio_input: str | np.ndarray | torch.Tensor,
        sound_length: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio, _ = librosa.load(audio_input, sr=self.sample_rate)
            loaded_audio = torch.tensor(audio, dtype=torch.float32)
            if sound_length is not None:
                loaded_audio = loaded_audio[: int(sound_length)]
            return loaded_audio
        if isinstance(audio_input, np.ndarray):
            audio = torch.from_numpy(audio_input).to(torch.float32).flatten()
            if sound_length is not None:
                audio = audio[: int(sound_length)]
            return audio
        if isinstance(audio_input, torch.Tensor):
            audio = audio_input.detach().to(torch.float32).cpu().flatten()
            if sound_length is not None:
                audio = audio[: int(sound_length)]
            return audio
        raise TypeError(f"Unsupported audio input type: {type(audio_input)!r}")

    def _normalize_audio_batch(
        self,
        audio_inputs: Sequence[str | np.ndarray | torch.Tensor]
        | str
        | np.ndarray
        | torch.Tensor
        | dict,
        sound_lengths: Sequence[int] | torch.Tensor | np.ndarray | int | None = None,
    ) -> tuple[list[str | np.ndarray | torch.Tensor], list[int | None]]:
        if isinstance(audio_inputs, dict):
            sound_lengths = audio_inputs.get("sound_lengths", sound_lengths)
            audio_value = None
            for key in ("audio", "sound", "audio_inputs", "audio_path", "path"):
                if key in audio_inputs and audio_inputs[key] is not None:
                    audio_value = audio_inputs[key]
                    break
            if audio_value is None:
                raise ValueError(
                    "audio_inputs dict must contain one of: "
                    "'audio', 'sound', 'audio_inputs', 'audio_path', or 'path'"
                )
            audio_inputs = audio_value

        normalized_lengths: list[int | None]
        if sound_lengths is None:
            normalized_lengths = []
        elif isinstance(sound_lengths, torch.Tensor):
            normalized_lengths = sound_lengths.detach().cpu().to(torch.long).flatten().tolist()
        elif isinstance(sound_lengths, np.ndarray):
            normalized_lengths = np.asarray(sound_lengths).reshape(-1).astype(np.int64).tolist()
        elif isinstance(sound_lengths, int):
            normalized_lengths = [sound_lengths]
        else:
            normalized_lengths = [int(length) for length in sound_lengths]

        if isinstance(audio_inputs, (str, np.ndarray)):
            normalized_audio_inputs = [audio_inputs]
        elif isinstance(audio_inputs, torch.Tensor):
            if audio_inputs.ndim <= 1:
                normalized_audio_inputs = [audio_inputs]
            else:
                normalized_audio_inputs = [audio_inputs[idx] for idx in range(audio_inputs.shape[0])]
        else:
            normalized_audio_inputs = list(audio_inputs)

        if not normalized_lengths:
            normalized_lengths = [None] * len(normalized_audio_inputs)
        elif len(normalized_lengths) != len(normalized_audio_inputs):
            raise ValueError(
                "sound_lengths must have the same batch size as audio_inputs: "
                f"{len(normalized_lengths)} != {len(normalized_audio_inputs)}"
            )

        return normalized_audio_inputs, normalized_lengths

    def _prepare_audio_batch(
        self,
        audio_inputs: Sequence[str | np.ndarray | torch.Tensor]
        | str
        | np.ndarray
        | torch.Tensor
        | dict,
        sound_lengths: Sequence[int] | torch.Tensor | np.ndarray | int | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_audio_inputs, normalized_lengths = self._normalize_audio_batch(
            audio_inputs,
            sound_lengths,
        )
        audio_tensors = [
            self._load_audio(audio_input, sound_length=sample_length)
            for audio_input, sample_length in zip(normalized_audio_inputs, normalized_lengths)
        ]
        if not audio_tensors:
            raise ValueError("audio_inputs must contain at least one sample")

        audio_lengths = torch.tensor(
            [audio_tensor.shape[0] for audio_tensor in audio_tensors],
            dtype=torch.long,
        )
        # Keep the padded raw batch available for future batched frontend work.
        # The current tokenize/whisper path still extracts features sample-by-sample.
        padded_audio = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
        audio_padding_mask = (
            torch.arange(padded_audio.shape[1]).unsqueeze(0) < audio_lengths.unsqueeze(1)
        )
        return audio_tensors, padded_audio, audio_padding_mask, audio_lengths

    @torch.inference_mode()
    def extract_audio_feat(
        self,
        audio_input: str | np.ndarray | torch.Tensor,
        sound_length: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = self._load_audio(audio_input, sound_length=sound_length)
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
    def extract_fused_states(
        self,
        audio_input: str | np.ndarray | torch.Tensor,
        sound_length: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        token_ids, continuous_feature = self.extract_audio_feat(
            audio_input,
            sound_length=sound_length,
        )
        fused_states = self.build_fused_states(
            token_ids.unsqueeze(0),
            continuous_feature.unsqueeze(0),
        ).squeeze(0)
        return fused_states.detach().cpu(), int(fused_states.shape[0])

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
        audio_inputs: Sequence[str | np.ndarray | torch.Tensor]
        | str
        | np.ndarray
        | torch.Tensor
        | dict,
        sound_lengths: Sequence[int] | torch.Tensor | np.ndarray | int | None = None,
    ) -> FrontendBatch:
        audio_tensors, _, _, _ = self._prepare_audio_batch(audio_inputs, sound_lengths)

        token_ids_list: list[torch.Tensor] = []
        continuous_features_list: list[torch.Tensor] = []
        fused_states_list: list[torch.Tensor] = []
        lengths: list[int] = []

        for audio_tensor in audio_tensors:
            token_ids, continuous_feature = self.extract_audio_feat(audio_tensor)
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
        padding_mask = build_padding_mask_from_lengths(lengths_tensor, max_len=max_len)

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
        if isinstance(batch, dict):
            return self.frontend(batch)
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], str):
            return self.frontend(list(batch))
        return self.frontend(batch)
