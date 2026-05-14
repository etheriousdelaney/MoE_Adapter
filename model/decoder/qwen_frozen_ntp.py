from __future__ import annotations

import os
from typing import Iterable

import torch
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


class FrozenQwenNTPDecoder(nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        llm_repo: str = "Qwen/Qwen3-1.7B",
        projector_hidden_dim: int = 2048,
        projector_num_layers: int = 2,
        dropout: float = 0.1,
        max_target_length: int = 256,
        prompt_text: str = "Transcribe the following speech:",
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

    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds, attention_mask, labels = self._build_training_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_lengths=prompt_lengths,
            response_input_ids=response_input_ids,
            response_lengths=response_lengths,
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
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[list[list[int]], list[float]]:
        prefix_embeds, prefix_mask = self._build_prefix_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_lengths=prompt_lengths,
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

    def _build_prefix_inputs(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_audio = self.dropout(self.projector(audio_hidden_states)).to(self.model_dtype)
        prompt_ids, prompt_mask = self._build_prompt_ids(prompt_input_ids, prompt_lengths)
        prompt_embeds = self._prompt_embeds(prompt_ids).to(
            device=projected_audio.device,
            dtype=self.model_dtype,
        )
        prompt_mask = prompt_mask.to(device=projected_audio.device, dtype=torch.long)
        prefix_embeds = torch.cat([prompt_embeds, projected_audio], dim=1)
        attention_mask = torch.cat(
            [prompt_mask, audio_attention_mask.to(device=projected_audio.device, dtype=torch.long)],
            dim=1,
        )
        return prefix_embeds, attention_mask

    def _build_training_inputs(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        response_input_ids: torch.Tensor,
        response_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embeds, attention_mask = self._build_prefix_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_lengths=prompt_lengths,
        )
        target_ids, target_mask = self._build_target_ids(response_input_ids, response_lengths)
        target_embeddings = self.llm.get_input_embeddings()(target_ids).to(
            device=prefix_embeds.device,
            dtype=self.model_dtype,
        )

        inputs_embeds = torch.cat([prefix_embeds, target_embeddings], dim=1)
        attention_mask = torch.cat(
            [attention_mask, target_mask.to(device=attention_mask.device, dtype=torch.long)],
            dim=1,
        )

        prefix_labels = torch.full(
            (prefix_embeds.shape[0], prefix_embeds.shape[1]),
            -100,
            device=prefix_embeds.device,
            dtype=torch.long,
        )
        target_labels = target_ids.masked_fill(~target_mask, -100)
        labels = torch.cat([prefix_labels, target_labels.to(prefix_embeds.device)], dim=1)
        return inputs_embeds, attention_mask, labels

    def _build_prompt_ids(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = prompt_input_ids.shape[0]
        max_prompt_length = int(prompt_lengths.max().item()) if prompt_lengths.numel() > 0 else 0
        prompt_ids = torch.full(
            (batch_size, max_prompt_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
            device=prompt_input_ids.device,
        )
        prompt_mask = torch.zeros(
            batch_size,
            max_prompt_length,
            dtype=torch.bool,
            device=prompt_input_ids.device,
        )
        for batch_idx in range(batch_size):
            length = int(prompt_lengths[batch_idx].item())
            if length <= 0:
                continue
            token_ids = prompt_input_ids[batch_idx, :length]
            prompt_ids[batch_idx, :length] = token_ids
            prompt_mask[batch_idx, :length] = True
        return prompt_ids, prompt_mask

    def _build_target_ids(
        self,
        text_input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_input_ids.shape[0]
        target_sequences: list[list[int]] = []
        for batch_idx in range(batch_size):
            length = int(text_lengths[batch_idx].item())
            token_ids = text_input_ids[batch_idx, :length].tolist()
            token_ids = token_ids[: self.max_target_length]
            if self.eos_token_id is not None:
                token_ids = token_ids + [int(self.eos_token_id)]
            if not token_ids:
                token_ids = [int(self.pad_token_id)]
            target_sequences.append([int(token_id) for token_id in token_ids])

        max_length = max(len(sequence) for sequence in target_sequences)
        padded = torch.full(
            (batch_size, max_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
            device=text_input_ids.device,
        )
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=text_input_ids.device)
        for batch_idx, sequence in enumerate(target_sequences):
            padded[batch_idx, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=text_input_ids.device,
            )
            mask[batch_idx, : len(sequence)] = True
        return padded, mask

    def _prompt_embeds(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        if prompt_ids.shape[1] == 0:
            return torch.zeros(
                prompt_ids.shape[0],
                0,
                self.llm_hidden_size,
                device=prompt_ids.device,
                dtype=self.model_dtype,
            )
        return self.llm.get_input_embeddings()(prompt_ids).to(dtype=self.model_dtype)
