from __future__ import annotations

import os
from typing import Iterable

import torch
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from model.decoder.qwen_audio_prompt import QwenAudioPromptMixin


class FrozenQwenDecoder(QwenAudioPromptMixin, nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        llm_repo: str = "Qwen/Qwen3-1.7B",
        projector_hidden_dim: int = 2048,
        projector_num_layers: int = 2,
        dropout: float = 0.1,
        max_target_length: int = 256,
        prompt_text: str = "",
    ):
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        hf_disable_progress_bars()
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_repo, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_repo,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        self.llm_hidden_size = int(self.llm.config.hidden_size)
        self.max_target_length = int(max_target_length)
        self.prompt_text = prompt_text
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0)
        )
        self.eos_token_id = self.tokenizer.eos_token_id

        layers = [nn.LayerNorm(input_hidden_size)]
        in_features = input_hidden_size
        for _ in range(max(1, projector_num_layers) - 1):
            layers.extend(
                [
                    nn.Linear(in_features, projector_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = projector_hidden_dim
        layers.append(nn.Linear(in_features, self.llm_hidden_size))
        self.projector = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self._init_fixed_prompt_ids(prompt_text)

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        response_input_ids: torch.Tensor | None = None,
        response_lengths: torch.Tensor | None = None,
        answer_input_ids: torch.Tensor | None = None,
        answer_lengths: torch.Tensor | None = None,
        audio_prefix_input_ids: torch.Tensor | None = None,
        audio_prefix_lengths: torch.Tensor | None = None,
        audio_suffix_input_ids: torch.Tensor | None = None,
        audio_suffix_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_input_ids = answer_input_ids if answer_input_ids is not None else response_input_ids
        target_lengths = answer_lengths if answer_lengths is not None else response_lengths
        if target_input_ids is None or target_lengths is None:
            raise RuntimeError("FrozenQwenDecoder requires response or answer target ids and lengths")

        inputs_embeds, attention_mask, labels = self._build_training_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_lengths=prompt_lengths,
            target_input_ids=target_input_ids,
            target_lengths=target_lengths,
            audio_prefix_input_ids=audio_prefix_input_ids,
            audio_prefix_lengths=audio_prefix_lengths,
            audio_suffix_input_ids=audio_suffix_input_ids,
            audio_suffix_lengths=audio_suffix_lengths,
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits, outputs.loss

    @torch.inference_mode()
    def greedy_generate(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        audio_prefix_input_ids: torch.Tensor | None = None,
        audio_prefix_lengths: torch.Tensor | None = None,
        audio_suffix_input_ids: torch.Tensor | None = None,
        audio_suffix_lengths: torch.Tensor | None = None,
    ) -> tuple[list[list[int]], list[float]]:
        prefix_embeds, prefix_mask = self._build_prefix_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_lengths=prompt_lengths,
            audio_prefix_input_ids=audio_prefix_input_ids,
            audio_prefix_lengths=audio_prefix_lengths,
            audio_suffix_input_ids=audio_suffix_input_ids,
            audio_suffix_lengths=audio_suffix_lengths,
        )
        batch_size = prefix_embeds.shape[0]
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        scores = [0.0 for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prefix_embeds.device)

        current_embeds = prefix_embeds
        current_mask = prefix_mask
        past_key_values = None
        input_embeddings = self.llm.get_input_embeddings()

        for _ in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            next_token_ids = torch.argmax(log_probs, dim=-1)
            next_token_ids = torch.where(
                finished,
                torch.full_like(next_token_ids, self.pad_token_id),
                next_token_ids,
            )
            next_token_scores = log_probs.gather(1, next_token_ids.unsqueeze(1)).squeeze(1)
            for batch_idx, token_id in enumerate(next_token_ids.tolist()):
                if finished[batch_idx]:
                    continue
                if self.eos_token_id is not None and token_id == self.eos_token_id:
                    finished[batch_idx] = True
                    continue
                generated[batch_idx].append(int(token_id))
                scores[batch_idx] += float(next_token_scores[batch_idx].item())

            if bool(torch.all(finished)):
                break

            past_key_values = outputs.past_key_values
            current_embeds = input_embeddings(next_token_ids.unsqueeze(1)).to(
                device=prefix_embeds.device,
                dtype=self.model_dtype,
            )
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(batch_size, 1, device=current_mask.device, dtype=current_mask.dtype),
                ],
                dim=1,
            )

        return generated, scores

    def ids_to_text(self, ids: Iterable[int]) -> tuple[list[str], str]:
        token_ids = [int(idx) for idx in ids]
        if not token_ids:
            return [], ""
        token_items = self.tokenizer.convert_ids_to_tokens(token_ids)
        filtered_token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in set(self.tokenizer.all_special_ids)
        ]
        text = self.tokenizer.decode(filtered_token_ids, skip_special_tokens=True).strip()
        return token_items, text
