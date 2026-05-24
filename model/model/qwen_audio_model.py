from __future__ import annotations

import lightning as L
import torch

from inference.expert_heatmap_utils import expert_usage_from_selected_experts
from model.adapter.Denseadapter import DenseAdapter
from model.adapter.MoEadapter import MoEAdapter
from model.adapter.Qformer import QFormerAdapter
from model.decoder.qwen_frozen_ntp import FrozenQwenNTPDecoder
from model.decoder.qwen_ntp import QwenNTPDecoder
from model.encoder.kimi_audio_encoder import KimiAudioFrontend, build_padding_mask_from_lengths
from train.config import ModelConfig


DECODER_ALIASES = {
    "ntp": "qwen",
    "qwen": "qwen",
    "frozen_ntp": "frozen_qwen",
    "frozen_qwen": "frozen_qwen",
}

ADAPTER_ALIASES = {
    "moe": "moe",
    "dense": "dense",
    "qformer": "qformer",
}


class LitQwenAudioModel(L.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        token_list: str,
        use_frontend: bool = True,
        adapter_type: str | None = None,
        decoder_type: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.adapter_type = self._normalize_adapter_type(adapter_type or config.adapter_type)
        self.decoder_type = self._normalize_decoder_type(decoder_type or config.decoder_type)
        freeze_frontend = bool(config.freeze_frontend or self.decoder_type == "frozen_qwen")

        self.frontend = (
            KimiAudioFrontend(
                model_repo=config.model_repo,
                tokenizer_repo=config.tokenizer_repo,
                sample_rate=config.sample_rate,
                freeze_frontend=freeze_frontend,
            )
            if use_frontend
            else None
        )
        self.hidden_size = self.frontend.hidden_size if self.frontend is not None else 3584
        self.aux_loss_weight = config.aux_loss_weight
        self.supports_expert_heatmap = self.adapter_type == "moe"
        self.supports_generation_inference = True
        self.inference_prompt_text = (
            config.llm_decoder.prompt_text.strip()
            if config.llm_decoder.prompt_text.strip()
            else "Transcribe the following speech:"
        )

        self.adapter = self._build_adapter(config)
        self.model_dtype = next(self.adapter.parameters()).dtype
        self.decoder_input_size = getattr(self.adapter, "output_hidden_size", self.hidden_size)
        self.decoder = self._build_decoder(config, input_hidden_size=self.decoder_input_size)
        self.tokenizer_repo = token_list

    @staticmethod
    def _normalize_adapter_type(adapter_type: str) -> str:
        normalized = str(adapter_type).strip().lower()
        if normalized not in ADAPTER_ALIASES:
            raise ValueError(f"Unsupported adapter_type: {adapter_type}")
        return ADAPTER_ALIASES[normalized]

    @staticmethod
    def _normalize_decoder_type(decoder_type: str) -> str:
        normalized = str(decoder_type).strip().lower()
        if normalized not in DECODER_ALIASES:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        return DECODER_ALIASES[normalized]

    def _build_adapter(self, config: ModelConfig):
        if self.adapter_type == "moe":
            return MoEAdapter(
                hidden_size=self.hidden_size,
                **config.adapter.__dict__,
            )
        if self.adapter_type == "dense":
            return DenseAdapter(
                hidden_size=self.hidden_size,
                ffn_dim=config.adapter.expert_ffn_dim,
            )
        if self.adapter_type == "qformer":
            return QFormerAdapter(
                speech_width=self.hidden_size,
                **config.qformer.__dict__,
            )
        raise ValueError(f"Unsupported adapter_type: {self.adapter_type}")

    def _build_decoder(self, config: ModelConfig, input_hidden_size: int):
        if self.decoder_type == "qwen":
            return QwenNTPDecoder(
                input_hidden_size=input_hidden_size,
                **config.llm_decoder.__dict__,
            )
        if self.decoder_type == "frozen_qwen":
            return FrozenQwenNTPDecoder(
                input_hidden_size=input_hidden_size,
                **config.llm_decoder.__dict__,
            )
        raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")

    def forward(self, batch) -> dict[str, torch.Tensor]:
        audio_input = self._unwrap_batch(batch)
        hidden_states, padding_mask = self._encode_and_adapt(audio_input)

        if self.decoder_type == "qwen":
            labels = audio_input.get("answer")
            label_lengths = audio_input.get("answer_lengths")
            if labels is None or label_lengths is None:
                raise RuntimeError(
                    "Qwen audio model requires answer/answer_lengths in the batch"
                )
            _, lm_loss = self.decoder(
                audio_hidden_states=hidden_states,
                audio_attention_mask=padding_mask,
                answer_input_ids=labels.to(self.device).long(),
                answer_lengths=label_lengths.to(self.device).long(),
                **self._audio_context_decoder_kwargs(audio_input),
            )
        elif self.decoder_type == "frozen_qwen":
            labels = audio_input.get("answer")
            label_lengths = audio_input.get("answer_lengths")
            if labels is None or label_lengths is None:
                raise RuntimeError(
                    "Frozen Qwen audio model requires answer/answer_lengths in the batch"
                )
            _, lm_loss = self.decoder(
                audio_hidden_states=hidden_states,
                audio_attention_mask=padding_mask,
                answer_input_ids=labels.to(self.device).long(),
                answer_lengths=label_lengths.to(self.device).long(),
                **self._audio_context_decoder_kwargs(audio_input),
            )
        else:
            raise RuntimeError(f"Unsupported decoder_type during forward: {self.decoder_type}")

        aux_loss = getattr(self, "_last_aux_loss", torch.zeros((), device=self.device))
        total_loss = lm_loss + (self.aux_loss_weight * aux_loss if self.adapter_type == "moe" else 0.0)
        outputs = {
            "loss": total_loss,
            "lm_loss": lm_loss,
        }
        if self.adapter_type == "moe":
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
        if self.decoder_type == "qwen":
            token_ids, scores = self.decoder.greedy_generate(
                audio_hidden_states=hidden_states,
                audio_attention_mask=padding_mask,
                max_new_tokens=max_new_tokens,
                **self._audio_context_decoder_kwargs(audio_input),
            )
        elif self.decoder_type == "frozen_qwen":
            context_kwargs = self._audio_context_decoder_kwargs(audio_input)
            token_ids, scores = self.decoder.greedy_generate(
                audio_hidden_states=hidden_states,
                audio_attention_mask=padding_mask,
                max_new_tokens=max_new_tokens,
                **context_kwargs,
            )
        else:
            raise RuntimeError(f"Unsupported decoder_type during inference: {self.decoder_type}")
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
        if self.adapter_type == "moe":
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
            return hidden_states, padding_mask

        if self.adapter_type == "qformer":
            return self.adapter(
                hidden_states=fused_states,
                padding_mask=padding_mask,
            )

        hidden_states = self.adapter(
            hidden_states=fused_states,
            padding_mask=padding_mask,
        )
        return hidden_states, padding_mask

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
            return {}
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

    def _build_inference_prompts(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.decoder.tokenizer(
            self.inference_prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if len(prompt_ids) == 0:
            prompt = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            return prompt, lengths

        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        prompt_tensor = prompt_tensor.expand(batch_size, -1).contiguous()
        prompt_lengths = torch.full(
            (batch_size,),
            fill_value=len(prompt_ids),
            dtype=torch.long,
            device=device,
        )
        return prompt_tensor, prompt_lengths

    @staticmethod
    def _unwrap_batch(batch) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[1], dict):
            return batch[1]
        raise TypeError(f"Unsupported batch type for LitQwenAudioModel: {type(batch)}")
