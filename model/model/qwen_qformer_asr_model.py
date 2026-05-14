from __future__ import annotations

import lightning as L
import torch

from model.adapter.Qformer import QFormerAdapter
from model.decoder.qwen_ntp import QwenNTPDecoder
from model.encoder.kimi_audio_encoder import KimiAudioFrontend, build_padding_mask_from_lengths
from train.config import ModelConfig


class LitQwenQFormerASR(L.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        token_list: str,
        use_frontend: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.frontend = (
            KimiAudioFrontend(
                model_repo=config.model_repo,
                tokenizer_repo=config.tokenizer_repo,
                sample_rate=config.sample_rate,
            )
            if use_frontend
            else None
        )
        self.hidden_size = self.frontend.hidden_size if self.frontend is not None else 3584
        self.supports_expert_heatmap = False
        self.supports_generation_inference = True

        
        self.adapter = QFormerAdapter(
            speech_width=self.hidden_size,
            **config.qformer.__dict__,
        )
        self.model_dtype = next(self.adapter.parameters()).dtype
        self.decoder = QwenNTPDecoder(
            input_hidden_size=self.adapter.output_hidden_size,
            **config.llm_decoder.__dict__,
        )
        self.tokenizer_repo = token_list

    def forward(self, batch) -> dict[str, torch.Tensor]:
        audio_input = self._unwrap_batch(batch)
        labels = audio_input.get("text")
        label_lengths = audio_input.get("text_lengths")
        if labels is None or label_lengths is None:
            raise RuntimeError("Qwen QFormer ASR training requires text and text_lengths in the batch")

        hidden_states, padding_mask = self._encode_and_adapt(audio_input)
        _, lm_loss = self.decoder(
            audio_hidden_states=hidden_states,
            audio_attention_mask=padding_mask,
            text_input_ids=labels.to(self.device).long(),
            text_lengths=label_lengths.to(self.device).long(),
        )
        return {
            "loss": lm_loss,
            "lm_loss": lm_loss,
        }

    @torch.inference_mode()
    def generate_greedy(
        self,
        batch,
        max_new_tokens: int = 128,
    ) -> tuple[list[list[int]], list[float], torch.Tensor | None]:
        audio_input = self._unwrap_batch(batch)
        hidden_states, padding_mask = self._encode_and_adapt(audio_input)
        token_ids, scores = self.decoder.greedy_generate(
            audio_hidden_states=hidden_states,
            audio_attention_mask=padding_mask,
            max_new_tokens=max_new_tokens,
        )
        return token_ids, scores, None

    def ids_to_text(self, ids):
        return self.decoder.ids_to_text(ids)

    def get_last_expert_usage(self) -> torch.Tensor | None:
        return None

    def load_for_inference(
        self,
        state_dict: dict[str, torch.Tensor],
        checkpoint_format: str,
        data_type: list[str],
    ) -> None:
        if checkpoint_format not in {"ckpt", "pth"}:
            raise ValueError(f"Unsupported checkpoint format for inference: {checkpoint_format}")
        try:
            self.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError as exc:
            if not self._should_allow_partial_frontend_load(data_type, state_dict):
                raise
            self._load_backend_only_for_inference(state_dict, original_error=exc)

    def _encode_and_adapt(
        self,
        audio_input: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_states, padding_mask = self._encode_inputs(audio_input)
        return self.adapter(
            hidden_states=fused_states,
            padding_mask=padding_mask,
        )

    def _encode_inputs(
        self,
        audio_input: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_input = audio_input.get("fused")
        fused_lengths = audio_input.get("fused_lengths")
        if fused_input is not None:
            fused_states = fused_input.to(self.device, dtype=self.model_dtype)
            if fused_lengths is None:
                fused_lengths = torch.full(
                    (fused_states.shape[0],),
                    fused_states.shape[1],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                fused_lengths = fused_lengths.to(self.device).long()
            padding_mask = build_padding_mask_from_lengths(
                fused_lengths,
                max_len=fused_states.shape[1],
            )
            return fused_states, padding_mask

        if self.frontend is None:
            raise RuntimeError("Frontend is not initialized for online feature extraction.")
        frontend_batch = self.frontend(audio_input["sound"], audio_input["sound_lengths"])
        fused_states = frontend_batch.fused_states.to(self.device, dtype=self.model_dtype)
        padding_mask = frontend_batch.padding_mask.to(self.device)
        return fused_states, padding_mask

    def _should_allow_partial_frontend_load(
        self,
        data_type: list[str],
        state_dict: dict[str, torch.Tensor],
    ) -> bool:
        if self.frontend is None or "sound" not in data_type:
            return False
        return not any(key.startswith("frontend.") for key in state_dict)

    def _load_backend_only_for_inference(
        self,
        state_dict: dict[str, torch.Tensor],
        original_error: RuntimeError,
    ) -> None:
        backend_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("frontend.")
        }
        incompatible = self.load_state_dict(backend_state_dict, strict=False)
        missing_keys = [
            key for key in incompatible.missing_keys if not key.startswith("frontend.")
        ]
        unexpected_keys = [
            key for key in incompatible.unexpected_keys if not key.startswith("frontend.")
        ]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Qwen QFormer ASR inference backend-only load failed. "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            ) from original_error

    @staticmethod
    def _unwrap_batch(batch) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[1], dict):
            return batch[1]
        raise TypeError(f"Unsupported batch type for LitQwenQFormerASR: {type(batch)}")
