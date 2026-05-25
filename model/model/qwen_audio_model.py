from __future__ import annotations

import lightning as L
import torch

from inference.expert_heatmap_utils import expert_usage_from_selected_experts
from model.encoder.kimi_audio_encoder import build_padding_mask_from_lengths


class LitQwenAudioModel(L.LightningModule):
    def __init__(
        self,
        frontend,
        adapter,
        decoder,
        token_list: str,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["frontend", "adapter", "decoder"])
        self.frontend = frontend
        self.adapter = adapter
        self.decoder = decoder
        self.aux_loss_weight = float(aux_loss_weight)
        self.supports_expert_heatmap = hasattr(adapter, "top_k") and hasattr(adapter, "experts")
        self.supports_generation_inference = True
        self.model_dtype = next(self.adapter.parameters()).dtype
        self.tokenizer_repo = token_list

    def forward(self, batch) -> dict[str, torch.Tensor]:
        audio_input = self._unwrap_batch(batch)
        hidden_states, padding_mask = self._encode_and_adapt(audio_input)
        labels = audio_input.get("answer")
        label_lengths = audio_input.get("answer_lengths")
        if labels is None or label_lengths is None:
            raise RuntimeError("Qwen audio model requires answer/answer_lengths in the batch")
        _, lm_loss = self.decoder(
            audio_hidden_states=hidden_states,
            audio_attention_mask=padding_mask,
            answer_input_ids=labels.to(self.device).long(),
            answer_lengths=label_lengths.to(self.device).long(),
            **self._audio_context_decoder_kwargs(audio_input),
        )

        aux_loss = getattr(self, "_last_aux_loss", torch.zeros((), device=self.device))
        total_loss = lm_loss + (self.aux_loss_weight * aux_loss if self.supports_expert_heatmap else 0.0)
        outputs = {
            "loss": total_loss,
            "lm_loss": lm_loss,
        }
        if self.supports_expert_heatmap:
            outputs["aux_loss"] = aux_loss
            outputs["expert_usage"] = self._last_expert_usage
        return outputs

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
            **self._audio_context_decoder_kwargs(audio_input),
        )
        expert_usage = getattr(self, "_last_expert_usage", None)
        return token_ids, scores, expert_usage

    def ids_to_text(self, ids):
        return self.decoder.ids_to_text(ids)

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

    def _encode_and_adapt(
        self,
        audio_input: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_states, padding_mask = self._encode_inputs(audio_input)
        adapter_output = self.adapter(
            hidden_states=fused_states,
            padding_mask=padding_mask,
        )
        if isinstance(adapter_output, tuple) and len(adapter_output) == 3:
            hidden_states, aux_loss, selected_experts = adapter_output
            self._last_aux_loss = aux_loss
            self._last_expert_usage = expert_usage_from_selected_experts(
                selected_experts=selected_experts,
                padding_mask=padding_mask,
                top_k=self.adapter.top_k,
            )
            return hidden_states, padding_mask

        if isinstance(adapter_output, tuple) and len(adapter_output) == 2:
            return adapter_output
        return adapter_output, padding_mask

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

    def _audio_context_decoder_kwargs(
        self,
        audio_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        required_names = (
            "audio_prefix",
            "audio_prefix_lengths",
            "audio_suffix",
            "audio_suffix_lengths",
        )
        if not all(name in audio_input for name in required_names):
            raise RuntimeError(
                "Qwen audio model requires audio_prefix/audio_suffix fields. "
                "Use data_type with audio_context so DeSTA-style prompt context is available."
            )
        return {
            "audio_prefix_input_ids": audio_input["audio_prefix"].to(self.device).long(),
            "audio_prefix_lengths": audio_input["audio_prefix_lengths"].to(self.device).long(),
            "audio_suffix_input_ids": audio_input["audio_suffix"].to(self.device).long(),
            "audio_suffix_lengths": audio_input["audio_suffix_lengths"].to(self.device).long(),
        }

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
                "Qwen audio model inference backend-only load failed. "
                f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            ) from original_error

    @staticmethod
    def _unwrap_batch(batch) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[1], dict):
            return batch[1]
        raise TypeError(f"Unsupported batch type for LitQwenAudioModel: {type(batch)}")
