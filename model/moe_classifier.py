from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MoEConfig:
    hidden_size: int
    num_classes: int
    num_experts: int = 8
    top_k: int = 1
    expert_ffn_dim: int | None = None
    classifier_hidden_dim: int | None = None
    aux_loss_weight: float = 0.01
    pooling_type: str = "attention"
    classifier_type: str = "mlp"
    dropout: float = 0.1


class ExpertMLP(nn.Module):
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


class DenseAdapter(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.proj = ExpertMLP(hidden_size, ffn_dim)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        forced_expert_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adapted = self.output_projection(self.proj(hidden_states))
        if padding_mask is not None:
            adapted = adapted * padding_mask.unsqueeze(-1).to(adapted.dtype)
        return {
            "hidden_states": adapted,
            "router_logits": torch.empty(0, device=hidden_states.device),
            "routing_probs": torch.empty(0, device=hidden_states.device),
            "aux_loss": hidden_states.new_zeros(()),
        }


class SparseMoEAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 1,
        expert_ffn_dim: int | None = None,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_ffn_dim = expert_ffn_dim or max(hidden_size // 4, 256)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, self.expert_ffn_dim) for _ in range(num_experts)]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

    def _load_balancing_loss(
        self,
        routing_probs: torch.Tensor,
        selected_experts: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if padding_mask is None:
            valid_mask = torch.ones(
                routing_probs.shape[:2],
                device=routing_probs.device,
                dtype=torch.bool,
            )
        else:
            valid_mask = padding_mask.to(torch.bool)

        if not valid_mask.any():
            return routing_probs.new_zeros(())

        valid_probs = routing_probs[valid_mask]
        expert_importance = valid_probs.mean(dim=0)

        selected = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=-2)
        selected = selected.to(valid_probs.dtype)[valid_mask]
        expert_load = selected.mean(dim=0)
        return self.num_experts * torch.sum(expert_importance * expert_load)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        forced_expert_indices: torch.Tensor | None = None,
        sample_expert_indices: torch.Tensor | None = None,
        sample_routing_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        effective_top_k = self.top_k
        if sample_expert_indices is None:
            raise ValueError("sample_expert_indices must be provided for SparseMoEAdapter.")
        if sample_routing_probs is None:
            raise ValueError("sample_routing_probs must be provided for SparseMoEAdapter.")

        selected_sample_experts = sample_expert_indices
        if forced_expert_indices is not None:
            allowed_mask = torch.zeros_like(routing_probs, dtype=torch.bool)
            allowed_mask.scatter_(
                dim=-1,
                index=forced_expert_indices.unsqueeze(1).expand(-1, routing_probs.shape[1], -1),
                value=True,
            )
            routing_probs = routing_probs.masked_fill(~allowed_mask, 0.0)
            routing_probs = routing_probs / routing_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            effective_top_k = min(self.top_k, forced_expert_indices.shape[-1])
            selected_sample_experts = forced_expert_indices[:, :effective_top_k]
            sample_routing_probs = sample_routing_probs.gather(dim=-1, index=selected_sample_experts)
            sample_routing_probs = sample_routing_probs / sample_routing_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            selected_sample_experts = selected_sample_experts[:, :effective_top_k]
            sample_routing_probs = sample_routing_probs.gather(dim=-1, index=selected_sample_experts)
            sample_routing_probs = sample_routing_probs / sample_routing_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_votes = F.one_hot(selected_sample_experts, num_classes=self.num_experts).to(torch.float32)
        expert_vote_distribution = expert_votes.sum(dim=1)
        expert_vote_distribution = expert_vote_distribution / expert_vote_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        batch_selected_experts = selected_sample_experts
        selected_experts = batch_selected_experts.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        topk_probs = sample_routing_probs.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.experts],
            dim=2,
        )
        gather_index = selected_experts.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)
        selected_outputs = expert_outputs.gather(dim=2, index=gather_index)
        adapted = torch.sum(
            selected_outputs * topk_probs.unsqueeze(-1).to(selected_outputs.dtype),
            dim=2,
        )
        adapted = self.output_projection(adapted)

        if padding_mask is not None:
            adapted = adapted * padding_mask.unsqueeze(-1).to(adapted.dtype)

        aux_loss = self._load_balancing_loss(routing_probs, selected_experts, padding_mask)
        return {
            "hidden_states": adapted,
            "router_logits": router_logits,
            "routing_probs": routing_probs,
            "expert_votes": expert_votes,
            "expert_vote_distribution": expert_vote_distribution,
            "batch_selected_experts": batch_selected_experts,
            "selected_experts": selected_experts,
            "aux_loss": aux_loss,
        }


def masked_mean_pool(hidden_states: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return hidden_states.mean(dim=1)
    weights = padding_mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denom


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_scores = self.score(hidden_states).squeeze(-1)
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(~padding_mask.to(torch.bool), torch.finfo(attn_scores.dtype).min)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        return pooled, attn_weights


class AudioClipClassifier(nn.Module):
    def __init__(self, config: MoEConfig, adapter_type: str = "moe"):
        super().__init__()
        self.config = config
        adapter_hidden = config.expert_ffn_dim or max(config.hidden_size // 4, 256)
        if adapter_type == "dense":
            self.adapter = DenseAdapter(config.hidden_size, adapter_hidden)
            self.sample_router = None
        elif adapter_type == "moe":
            self.adapter = SparseMoEAdapter(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                expert_ffn_dim=adapter_hidden,
            )
            self.sample_router = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.num_experts),
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        if config.pooling_type == "mean":
            self.pooling = None
        elif config.pooling_type == "attention":
            self.pooling = AttentionPooling(config.hidden_size)
        else:
            raise ValueError(f"Unknown pooling_type: {config.pooling_type}")

        classifier_hidden_dim = config.classifier_hidden_dim or max(config.hidden_size // 2, 256)
        if config.classifier_type == "linear":
            self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        elif config.classifier_type == "mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, classifier_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(classifier_hidden_dim, config.num_classes),
            )
        else:
            raise ValueError(f"Unknown classifier_type: {config.classifier_type}")

    def forward(
        self,
        fused_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        include_aux_loss: bool = True,
        forced_expert_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_dtype = next(self.adapter.parameters()).dtype
        fused_states = fused_states.to(model_dtype)
        sample_router_logits = torch.empty(0, device=fused_states.device)
        sample_routing_probs = torch.empty(0, device=fused_states.device)
        sample_selected_experts = None
        if self.sample_router is not None:
            router_inputs = masked_mean_pool(fused_states, padding_mask)
            sample_router_logits = self.sample_router(router_inputs)
            sample_routing_probs = F.softmax(sample_router_logits, dim=-1, dtype=torch.float32)
            sample_selected_experts = torch.topk(sample_routing_probs, self.config.top_k, dim=-1).indices
        adapter_kwargs = {
            "padding_mask": padding_mask,
            "forced_expert_indices": forced_expert_indices,
        }
        if self.sample_router is not None:
            adapter_kwargs["sample_expert_indices"] = sample_selected_experts
            adapter_kwargs["sample_routing_probs"] = sample_routing_probs
        adapter_outputs = self.adapter(fused_states, **adapter_kwargs)
        if self.pooling is None:
            pooled = masked_mean_pool(adapter_outputs["hidden_states"], padding_mask)
            attn_weights = None
        else:
            pooled, attn_weights = self.pooling(adapter_outputs["hidden_states"], padding_mask)
        logits = self.classifier(pooled)

        ce_loss = None
        total_loss = adapter_outputs["aux_loss"] if include_aux_loss else fused_states.new_zeros(())
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            total_loss = ce_loss
            if include_aux_loss:
                total_loss = total_loss + (self.config.aux_loss_weight * adapter_outputs["aux_loss"])

        return {
            **adapter_outputs,
            "sample_router_logits": sample_router_logits,
            "sample_routing_probs": sample_routing_probs,
            "sample_selected_experts": sample_selected_experts,
            "pooled_states": pooled,
            "attn_weights": attn_weights,
            "logits": logits,
            "ce_loss": ce_loss,
            "loss": total_loss,
        }
