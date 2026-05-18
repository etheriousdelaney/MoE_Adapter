from __future__ import annotations

import torch


class QwenAudioNTPPromptMixin:
    def _init_fixed_prompt_ids(self, prompt_text: str) -> None:
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        self.register_buffer(
            "prompt_ids",
            torch.tensor(prompt_ids, dtype=torch.long),
            persistent=False,
        )

    def _build_prefix_inputs(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        audio_prefix_input_ids: torch.Tensor | None = None,
        audio_prefix_lengths: torch.Tensor | None = None,
        audio_suffix_input_ids: torch.Tensor | None = None,
        audio_suffix_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_audio = self.dropout(self.projector(audio_hidden_states)).to(self.model_dtype)
        audio_mask = audio_attention_mask.to(device=projected_audio.device, dtype=torch.long)
        has_audio_context = (
            audio_prefix_input_ids is not None
            and audio_prefix_lengths is not None
            and audio_suffix_input_ids is not None
            and audio_suffix_lengths is not None
        )
        if has_audio_context:
            prefix_embeds, prefix_mask = self._sequence_embeds_and_mask(
                audio_prefix_input_ids,
                audio_prefix_lengths,
                device=projected_audio.device,
            )
            suffix_embeds, suffix_mask = self._sequence_embeds_and_mask(
                audio_suffix_input_ids,
                audio_suffix_lengths,
                device=projected_audio.device,
            )
            prefix_embeds = torch.cat([prefix_embeds, projected_audio, suffix_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, audio_mask, suffix_mask], dim=1)
            return prefix_embeds, attention_mask

        if prompt_input_ids is not None and prompt_lengths is not None:
            prompt_embeds, prompt_mask = self._sequence_embeds_and_mask(
                prompt_input_ids,
                prompt_lengths,
                device=projected_audio.device,
            )
        else:
            prompt_embeds = self._fixed_prompt_embeds(
                batch_size=projected_audio.shape[0],
                device=projected_audio.device,
            )
            prompt_mask = self._fixed_prompt_mask(
                batch_size=projected_audio.shape[0],
                device=projected_audio.device,
            )
        prefix_embeds = torch.cat([prompt_embeds, projected_audio], dim=1)
        attention_mask = torch.cat([prompt_mask, audio_mask], dim=1)
        return prefix_embeds, attention_mask

    def _build_training_inputs(
        self,
        audio_hidden_states: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_lengths: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        audio_prefix_input_ids: torch.Tensor | None = None,
        audio_prefix_lengths: torch.Tensor | None = None,
        audio_suffix_input_ids: torch.Tensor | None = None,
        audio_suffix_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embeds, attention_mask = self._build_prefix_inputs(
            audio_hidden_states=audio_hidden_states,
            audio_attention_mask=audio_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_lengths=prompt_lengths,
            audio_prefix_input_ids=audio_prefix_input_ids,
            audio_prefix_lengths=audio_prefix_lengths,
            audio_suffix_input_ids=audio_suffix_input_ids,
            audio_suffix_lengths=audio_suffix_lengths,
        )
        target_ids, target_mask = self._build_target_ids(target_input_ids, target_lengths)
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

    def _sequence_embeds_and_mask(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, token_mask = self._build_token_ids(input_ids, lengths)
        token_embeds = self._token_embeds(token_ids).to(device=device, dtype=self.model_dtype)
        token_mask = token_mask.to(device=device, dtype=torch.long)
        return token_embeds, token_mask

    def _build_token_ids(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]
        max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0
        token_ids = torch.full(
            (batch_size, max_length),
            fill_value=int(self.pad_token_id),
            dtype=torch.long,
            device=input_ids.device,
        )
        token_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=input_ids.device)
        for batch_idx in range(batch_size):
            length = int(lengths[batch_idx].item())
            if length <= 0:
                continue
            token_ids[batch_idx, :length] = input_ids[batch_idx, :length]
            token_mask[batch_idx, :length] = True
        return token_ids, token_mask

    def _build_target_ids(
        self,
        target_input_ids: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = target_input_ids.shape[0]
        target_sequences: list[list[int]] = []
        for batch_idx in range(batch_size):
            length = int(target_lengths[batch_idx].item())
            token_ids = target_input_ids[batch_idx, :length].tolist()
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
            device=target_input_ids.device,
        )
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=target_input_ids.device)
        for batch_idx, sequence in enumerate(target_sequences):
            padded[batch_idx, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=target_input_ids.device,
            )
            mask[batch_idx, : len(sequence)] = True
        return padded, mask

    def _token_embeds(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[1] == 0:
            return torch.zeros(
                token_ids.shape[0],
                0,
                self.llm_hidden_size,
                device=token_ids.device,
                dtype=self.model_dtype,
            )
        return self.llm.get_input_embeddings()(token_ids).to(dtype=self.model_dtype)

    def _fixed_prompt_embeds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.prompt_ids.numel() == 0:
            return torch.zeros(
                batch_size,
                0,
                self.llm_hidden_size,
                device=device,
                dtype=self.model_dtype,
            )
        prompt_embeds = self.llm.get_input_embeddings()(self.prompt_ids.to(device))
        prompt_embeds = prompt_embeds.to(dtype=self.model_dtype)
        return prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)

    def _fixed_prompt_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones(
            batch_size,
            int(self.prompt_ids.numel()),
            device=device,
            dtype=torch.long,
        )
