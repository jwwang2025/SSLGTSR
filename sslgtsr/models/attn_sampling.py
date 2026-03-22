from __future__ import annotations

"""

已知：

来自交互图的用户嵌入：U_I​（user_ui）
来自社交图的用户嵌入：U_S​（user_uu）
社交邻接矩阵（用户–用户）：G_S​

我们执行以下步骤：

融合两种视角，构建相似度矩阵 S（对应公式 (1) 形式）；
借助邻接矩阵 G_S​ 聚合邻居偏好，对相似度矩阵 S 进行优化（对应公式 (2) 形式）；
对每个用户 v_i​，采样相似度最高的前 t 个用户作为注意力采样样本 Smp(v_i​)（对应公式 (3)）。

本模块提供轻量、可配置的实现，训练器（Trainer）可在每个轮次（epoch）调用一次，以获取注意力邻居的索引。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class AttnSamplingConfig:
    enabled: bool = False
    topk: int = 20
    beta_view: float = 0.5  # fuse UI / UU similarities
    beta_neighbor: float = 0.5  # fuse original S and neighbor-refined S


class GraphAttentionSampler:
    def __init__(self, cfg: AttnSamplingConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _sim_matrix(user_ui: torch.Tensor, user_uu: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute similarity matrix S from two views, roughly corresponding to Eq.(1):
            S = beta * sim(U_I, U_I) + (1-beta) * sim(U_S, U_S)
        where sim is cosine similarity.
        """
        u_i = F.normalize(user_ui, dim=-1)
        u_s = F.normalize(user_uu, dim=-1)
        S_i = u_i @ u_i.t()  # [U,U]
        S_s = u_s @ u_s.t()  # [U,U]
        S = beta * S_i + (1.0 - beta) * S_s
        return S

    @staticmethod
    def _refine_with_neighbors(
        S: torch.Tensor,
        gs_sparse: torch.Tensor,
        beta_neighbor: float,
    ) -> torch.Tensor:
        r"""Rough implementation of Eq.(2): incorporate neighbors' preferences via
        \hat G_S = G_S + I and propagate S once."""
        U = S.size(0)
        # convert to dense for simplicity (OK for typical U in social rec datasets)
        if gs_sparse.is_sparse:
            G = gs_sparse.to_dense()
        else:
            G = gs_sparse
        I = torch.eye(U, device=S.device, dtype=S.dtype)
        G_hat = G + I
        # neighbor-aware similarity: propagate similarities along social neighbors once
        S_neigh = G_hat @ S  # [U,U]
        # fuse original and neighbor-aggregated similarity
        return beta_neighbor * S_neigh + (1.0 - beta_neighbor) * S

    @staticmethod
    def _topk_indices(S: torch.Tensor, k: int) -> torch.Tensor:
        """
        For each row i in S, pick indices of top‑k most similar users (excluding self).
        Implements Eq.(3): Smp(v_i) = top_t(S_{i,*}).
        """
        U = S.size(0)
        if k <= 0:
            return torch.empty((U, 0), dtype=torch.long, device=S.device)

        S = S.clone()
        diag_idx = torch.arange(U, device=S.device)
        S[diag_idx, diag_idx] = -1e9  # mask self-similarity
        _, idx = torch.topk(S, k=min(k, U - 1), dim=-1)
        return idx  # [U,k]

    @torch.no_grad()
    def sample(
        self,
        user_ui: torch.Tensor,
        user_uu: torch.Tensor,
        gs_sparse: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Main entry: given user embeddings from both views and social adjacency,
        return attention neighbor indices of shape [U, topk], or None if disabled.
        """
        if not self.cfg.enabled:
            return None

        beta_v = float(self.cfg.beta_view)
        beta_n = float(self.cfg.beta_neighbor)

        S = self._sim_matrix(user_ui, user_uu, beta_v)
        S_ref = self._refine_with_neighbors(S, gs_sparse, beta_n)
        attn_idx = self._topk_indices(S_ref, int(self.cfg.topk))
        return attn_idx


