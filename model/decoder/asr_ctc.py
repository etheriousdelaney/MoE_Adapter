from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ASRCTCDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        blank_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_are_pre_projected: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = hidden_states if logits_are_pre_projected else self.proj(hidden_states)
        ctc_loss = None
        if text is not None and text_lengths is not None:
            if padding_mask is None:
                input_lengths = torch.full(
                    (hidden_states.shape[0],),
                    hidden_states.shape[1],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            else:
                input_lengths = padding_mask.to(torch.long).sum(dim=1)

            flat_targets = []
            for idx, target_length in enumerate(text_lengths.tolist()):
                flat_targets.append(text[idx, :target_length])
            targets = torch.cat(flat_targets, dim=0)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            ctc_loss = self.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                text_lengths.to(torch.long),
            )

        return logits, ctc_loss
