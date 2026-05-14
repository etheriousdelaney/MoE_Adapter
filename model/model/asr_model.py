from __future__ import annotations

from pathlib import Path

import torch
import lightning as L
import torch.nn.functional as F

from inference.expert_heatmap_utils import expert_usage_from_selected_experts
from model.adapter.MoEadapter import MoEAdapter
from model.decoder.asr_ctc import ASRCTCDecoder
from model.encoder.kimi_audio_encoder import KimiAudioFrontend
from model.encoder.kimi_audio_encoder import build_padding_mask_from_lengths
from train.config import ModelConfig


class LitMoEASR(L.LightningModule):
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
        self.hidden_size = (
            self.frontend.hidden_size if self.frontend is not None else 3584
        )
        self.aux_loss_weight = config.aux_loss_weight
        self.supports_expert_heatmap = True

        self.adapter = MoEAdapter(
            hidden_size=self.hidden_size,
            **config.adapter.__dict__,
        )
        self.model_dtype = next(self.adapter.parameters()).dtype

        with Path(token_list).open("r", encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f if line.rstrip("\n")]
        self.tokens = tokens
        if "<blank>" not in tokens:
            raise ValueError("token_list must contain <blank> for CTC")
        self.blank_id = tokens.index("<blank>")

        self.decoder = ASRCTCDecoder(
            hidden_size=self.hidden_size,
            vocab_size=len(tokens),
            blank_id=self.blank_id,
            **config.asr_decoder.__dict__,
        )

    def forward(
        self,
        batch,
    ) -> dict[str, torch.Tensor]:
        audio_input = self._unwrap_batch(batch)
        labels = audio_input.get("text")
        label_lengths = audio_input.get("text_lengths")
        logits, padding_mask = self.compute_logits(batch)
        if labels is not None:
            labels = labels.to(self.device).long()
        if label_lengths is not None:
            label_lengths = label_lengths.to(self.device).long()

        _, ctc_loss = self.decoder(
            logits,
            text=labels,
            text_lengths=label_lengths,
            padding_mask=padding_mask,
            logits_are_pre_projected=True,
        )
        if ctc_loss is None:
            raise RuntimeError("ASR training requires text and text_lengths in the batch")

        aux_loss = self._last_aux_loss
        total_loss = ctc_loss + (self.aux_loss_weight * aux_loss)
        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss,
            "aux_loss": aux_loss,
            "expert_usage": self._last_expert_usage,
        }

    def compute_logits(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_input = self._unwrap_batch(batch)
        fused_states, padding_mask = self._encode_inputs(audio_input)
        hidden_states, aux_loss, selected_experts = self.adapter(
            hidden_states=fused_states,
            padding_mask=padding_mask,
        )
        self._last_aux_loss = aux_loss
        self._last_expert_usage = expert_usage_from_selected_experts(
            selected_experts=selected_experts,
            padding_mask=padding_mask,
            top_k=self.adapter.top_k,
        )
        logits, _ = self.decoder(
            hidden_states,
            text=None,
            text_lengths=None,
            padding_mask=padding_mask,
        )
        return logits, padding_mask

    def compute_log_probs(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, padding_mask = self.compute_logits(batch)
        log_probs = F.log_softmax(logits, dim=-1)
        input_lengths = padding_mask.to(torch.long).sum(dim=1)
        return log_probs, input_lengths

    def get_last_expert_usage(self) -> torch.Tensor | None:
        return getattr(self, "_last_expert_usage", None)

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
                "ASR inference backend-only load failed. "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            ) from original_error

    @staticmethod
    def _unwrap_batch(batch) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[1], dict):
            return batch[1]
        raise TypeError(f"Unsupported batch type for LitMoEASR: {type(batch)}")
