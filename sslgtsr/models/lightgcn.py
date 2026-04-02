from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def lightgcn_propagate(
    norm_adj: torch.Tensor, x: torch.Tensor, n_layers: int
) -> torch.Tensor:
    """
    LightGCN 传播：对各层的嵌入向量取平均。
    norm_adj：稀疏归一化邻接矩阵 (N × N)
    x：稠密节点嵌入向量 (N × D)
    """
    out = x
    embs = [x]
    for _ in range(n_layers):
        out = torch.sparse.mm(norm_adj, out)
        embs.append(out)
    return torch.stack(embs, dim=0).mean(dim=0)


class LightGCNEncoder(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = int(n_layers)

    def forward(self, norm_adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return lightgcn_propagate(norm_adj, x, self.n_layers)


