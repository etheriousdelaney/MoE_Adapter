from __future__ import annotations

import torch


def build_padding_mask_from_lengths(
    lengths: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1-D, but got shape={tuple(lengths.shape)}")
    if max_len is None:
        max_len = int(lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
