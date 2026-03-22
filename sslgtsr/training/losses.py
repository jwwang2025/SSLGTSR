from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    # -log sigma(pos-neg)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def l2_reg(*tensors: torch.Tensor) -> torch.Tensor:
    reg = torch.zeros((), device=tensors[0].device)
    for t in tensors:
        reg = reg + (t.pow(2).sum() / t.size(0))
    return reg


