import torch
import torch.nn.functional as F
from torch import nn


class MoEExpert(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class AggregationBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        ffn_dim = hidden_size * 4
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class MoEERCLossAdapter(nn.Module):
    SUPPORTED_ERC_FEATURES = {"w1", "w1_silu", "w2_direct", "full_output"}

    def __init__(
        self,
        hidden_size: int,
        expert_ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        load_balancing_loss_weight: float = 1.0,
        erc_loss_weight: float = 1.0,
        erc_alpha: float = 1.0,
        erc_noisy_router: bool = True,
        erc_feature: str = "w1",
        erc_eps_min: float = 1.0e-6,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        erc_feature = str(erc_feature).lower()
        if erc_feature not in self.SUPPORTED_ERC_FEATURES:
            raise ValueError(
                f"Unsupported erc_feature={erc_feature!r}; expected one of {sorted(self.SUPPORTED_ERC_FEATURES)}"
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_loss_weight = float(load_balancing_loss_weight)
        self.erc_loss_weight = float(erc_loss_weight)
        self.erc_alpha = float(erc_alpha)
        self.erc_noisy_router = bool(erc_noisy_router)
        self.erc_feature = erc_feature
        self.erc_eps_min = float(erc_eps_min)

        self.R = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.R, a=5**0.5)
        self.experts = nn.ModuleList(
            [MoEExpert(hidden_size, expert_ffn_dim) for _ in range(num_experts)]
        )
        self.aggregation = AggregationBlock(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        forced_expert_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = torch.matmul(hidden_states, self.R.t())
        _, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        selected_experts = torch.zeros(
            router_logits.size(),
            dtype=torch.float32,
            device=router_logits.device,
        )
        selected_experts.scatter_(2, topk_idx, 1)
        expert_load = selected_experts.clone()
        masked_router_logits = router_logits.masked_fill(selected_experts == 0, float("-inf"))
        routing_probs = F.softmax(masked_router_logits, dim=-1, dtype=torch.float32)

        adapted_outputs = self._dispatch_to_experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            routing_probs=routing_probs,
            padding_mask=padding_mask,
        )
        adapted_outputs = self.aggregation(adapted_outputs)

        if padding_mask is not None:
            adapted_outputs = adapted_outputs * padding_mask.unsqueeze(-1).to(adapted_outputs.dtype)

        lb_loss = self._load_balancing_loss(routing_probs, expert_load, padding_mask)
        erc_loss = self._compute_erc_loss() if self.training else lb_loss.new_zeros(())
        aux_loss = (
            self.load_balancing_loss_weight * lb_loss
            + self.erc_loss_weight * erc_loss
        )
        self.last_lb_loss = lb_loss
        self.last_erc_loss = erc_loss

        return adapted_outputs, aux_loss, selected_experts

    def _load_balancing_loss(
        self,
        routing_probs: torch.Tensor,
        selected_expert: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if padding_mask is None:
            mask = torch.ones(
                routing_probs.shape[:2] + (1,),
                dtype=torch.float32,
                device=routing_probs.device,
            )
        else:
            mask = padding_mask.float().unsqueeze(-1)

        valid_count = mask.sum().clamp_min(1.0)
        p_bar = (routing_probs * mask).sum(dim=(0, 1)) / valid_count
        f_bar = (selected_expert * mask).sum(dim=(0, 1)) / valid_count
        return self.num_experts * torch.sum(p_bar * f_bar)

    def _compute_erc_loss(self) -> torch.Tensor:
        router = self._get_noisy_router(self.R) if self.erc_noisy_router else self.R
        matrix = self._erc_matrix(router)
        diag = torch.diag(matrix)
        row_diff = matrix - self.erc_alpha * diag.unsqueeze(1)
        col_diff = matrix - self.erc_alpha * diag.unsqueeze(0)
        row_diff = torch.clamp(row_diff, min=0.0)
        col_diff = torch.clamp(col_diff, min=0.0)
        mask = torch.ones_like(matrix) - torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return ((row_diff + col_diff) * mask).mean()

    def _get_noisy_router(self, router: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            router_norm = torch.norm(router, dim=1).clamp_min(self.erc_eps_min)
            distances = torch.cdist(router.float(), router.float(), p=2).to(router.dtype)
            distances.fill_diagonal_(float("inf"))
            min_dist, _ = torch.min(distances, dim=1)
            eps = (min_dist / (2.0 * router_norm)).clamp_min(self.erc_eps_min)
            low = (1.0 - eps).unsqueeze(1)
            high = (1.0 + eps).unsqueeze(1)
            noise = torch.rand_like(router)
        return (low + noise * (high - low)) * router

    def _erc_matrix(self, router: torch.Tensor) -> torch.Tensor:
        if self.erc_feature == "w1":
            w1 = torch.stack([expert.net[1].weight for expert in self.experts])
            return torch.norm(torch.einsum("jfh,ih->ijf", w1, router), dim=-1)

        if self.erc_feature == "w2_direct":
            w2 = torch.stack([expert.net[3].weight for expert in self.experts])
            return torch.norm(torch.einsum("ih,jhf->ijf", router, w2), dim=-1)

        values = []
        for expert in self.experts:
            normalized = expert.net[0](router)
            first_hidden = expert.net[1](normalized)
            first_hidden = expert.net[2](first_hidden)
            if self.erc_feature == "w1_silu":
                values.append(torch.norm(first_hidden, dim=-1))
            elif self.erc_feature == "full_output":
                output = expert.net[3](first_hidden)
                values.append(torch.norm(output, dim=-1))
            else:
                raise ValueError(f"Unsupported erc_feature={self.erc_feature!r}")
        return torch.stack(values, dim=1)

    def _dispatch_to_experts(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        routing_probs: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(batch_size * seq_len, hidden_size)
        topk_flat = topk_idx.reshape(batch_size * seq_len, self.top_k)
        routing_flat = routing_probs.reshape(batch_size * seq_len, self.num_experts)
        output_flat = hidden_flat.new_zeros(hidden_flat.shape)

        if padding_mask is None:
            valid_tokens = torch.ones(
                hidden_flat.size(0),
                dtype=torch.bool,
                device=hidden_flat.device,
            )
        else:
            valid_tokens = padding_mask.reshape(-1).bool()

        for expert_idx, expert in enumerate(self.experts):
            token_indices = (topk_flat == expert_idx).any(dim=-1) & valid_tokens
            if not token_indices.any():
                continue
            expert_output = expert(hidden_flat[token_indices])
            expert_weight = routing_flat[token_indices, expert_idx].to(expert_output.dtype)
            output_flat.index_add_(
                0,
                token_indices.nonzero(as_tuple=False).squeeze(-1),
                expert_output * expert_weight.unsqueeze(-1),
            )

        return output_flat.reshape(batch_size, seq_len, hidden_size)
