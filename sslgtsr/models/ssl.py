from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE for aligned pairs across two views.
    z1, z2: [B, D] (assumed already sampled corresponding pairs)
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / float(temperature)  # [B, B]
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


class FeatureDropout(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate <= 0.0:
            return x
        mask = torch.rand_like(x) >= self.drop_rate
        return x * mask


