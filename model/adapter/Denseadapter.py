from __future__ import annotations

import torch
from torch import nn


class DenseAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adapted_outputs = self.adapter(hidden_states)
        adapted_outputs = self.output_projection(adapted_outputs)

        if padding_mask is not None:
            adapted_outputs = adapted_outputs * padding_mask.unsqueeze(-1).to(adapted_outputs.dtype)

        return adapted_outputs
