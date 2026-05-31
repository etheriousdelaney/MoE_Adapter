from torch import nn
import torch
import torch.nn.functional as F

class MoEExpert(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)

class AggregationBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        ffn_dim = hidden_size * 4
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),  # ← expansion
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)
    
class MoEAdapter(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            expert_ffn_dim: int,
            num_experts: int = 8,
            top_k: int = 2
            ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoEExpert(hidden_size, expert_ffn_dim) for _ in range(num_experts)]
        )
        self.aggregation = AggregationBlock(hidden_size)
    
    def _load_balancing_loss(
        self,
        routing_probs: torch.Tensor,
        selected_expert: torch.Tensor,
        padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if padding_mask is None:
            mask = torch.ones(
                routing_probs.shape[:2] + (1,),
                dtype=torch.float32,
                device=routing_probs.device,
            )
        else:
            mask = padding_mask.float().unsqueeze(-1)

        B = mask.sum().clamp_min(1.0)
        P_bar = (routing_probs * mask).sum(dim=(0,1)) / B
        F_bar = (selected_expert * mask).sum(dim=(0,1)) / B
        
        return self.num_experts * torch.sum(P_bar * F_bar)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        forced_expert_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
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

        aux_loss = self._load_balancing_loss(routing_probs, expert_load, padding_mask)
        
        return adapted_outputs, aux_loss, selected_experts

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
