from torch import nn
import torch
import torch.nn.functional as F


class AudioClassfier(nn.Module):
    """Utterance-level classification head with masked temporal pooling."""

    def __init__(
            self,
            hidden_size,
            classifier_hidden_dim,
            dropout,
            num_classes,
            ):
        
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def _masked_mean_pool(
        self,
        feature: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if feature.ndim == 2:
            return feature
        if padding_mask is None:
            return feature.mean(dim=1)

        mask = padding_mask.unsqueeze(-1).to(feature.dtype)
        return (feature * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
            self,
            feature: torch.Tensor,
            labels: torch.Tensor | None = None,
            padding_mask: torch.Tensor | None = None):
        pooled_feature = self._masked_mean_pool(feature, padding_mask)
        logits = self.classifier(pooled_feature)
        ce_loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels.view(-1).long())

        return logits, ce_loss
