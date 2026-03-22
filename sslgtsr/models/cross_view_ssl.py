from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossViewSSLConfig:
    enabled: bool = False
    weight: float = 0.1
    n_pairs: int = 2048
    leaky_relu_slope: float = 0.2


class LearnableSimilarity(nn.Module):
    """
    Eq.(16)-style learnable similarity function:
      s_hat = sigm( w^T * sigma( W * [x_i ; x_j] + b ) + c )
    where sigma is LeakyReLU and sigm is Sigmoid.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 128, slope: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, int(hidden_dim)),
            nn.LeakyReLU(negative_slope=float(slope)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # x_i, x_j: [B, D]
        h = torch.cat([x_i, x_j], dim=-1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)  # [B] in (0,1)


def cosine_to_unit_interval(sim: torch.Tensor) -> torch.Tensor:
    """Map cosine similarity [-1,1] -> [0,1]."""
    return 0.5 * (sim + 1.0)


def cross_view_align_loss(
    user_ui: torch.Tensor,
    user_uu: torch.Tensor,
    sim_fn: LearnableSimilarity,
    n_pairs: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """
    Eq.(17)-style alignment loss:
    - sample random user pairs (u_i, u_i')
    - predict similarity from interaction-view embeddings via learnable sim_fn
    - target is the social-view similarity (cosine) mapped to [0,1]
    - minimize MSE between predicted and target
    """
    U = user_ui.size(0)
    if U <= 1:
        return torch.zeros((), device=user_ui.device)

    n = int(min(n_pairs, U * 4))
    i = torch.randint(0, U, (n,), device=user_ui.device, generator=rng)
    j = torch.randint(0, U, (n,), device=user_ui.device, generator=rng)
    # avoid i==j where possible
    eq = i == j
    if bool(eq.any()):
        j[eq] = (j[eq] + 1) % U

    ui_i, ui_j = user_ui[i], user_ui[j]
    uu_i, uu_j = user_uu[i], user_uu[j]

    pred = sim_fn(ui_i, ui_j)  # [n]
    target_cos = F.cosine_similarity(uu_i, uu_j, dim=-1)  # [-1,1]
    target = cosine_to_unit_interval(target_cos).detach()
    return F.mse_loss(pred, target)


