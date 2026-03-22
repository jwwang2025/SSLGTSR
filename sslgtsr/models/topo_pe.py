from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _coo_to_csr_binary(adj: sp.coo_matrix) -> sp.csr_matrix:
    """Ensure unweighted binary adjacency in CSR."""
    adj = adj.tocoo()
    data = np.ones_like(adj.data, dtype=np.float32)
    return sp.coo_matrix((data, (adj.row, adj.col)), shape=adj.shape).tocsr()


def degree_vector(adj: sp.coo_matrix) -> np.ndarray:
    adj_csr = _coo_to_csr_binary(adj)
    deg = np.asarray(adj_csr.sum(axis=1)).squeeze().astype(np.float32)
    return deg


def pagerank_vector(
    adj: sp.coo_matrix,
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Simple PageRank via power iteration on row-stochastic transition matrix.
    Returns PR normalized to sum=1.
    """
    A = _coo_to_csr_binary(adj)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    out_deg = np.asarray(A.sum(axis=1)).squeeze().astype(np.float32)
    # handle dangling nodes
    dangling = out_deg == 0
    out_deg[dangling] = 1.0
    inv_out = 1.0 / out_deg

    pr = np.full((n,), 1.0 / n, dtype=np.float32)
    teleport = (1.0 - alpha) / n

    for _ in range(int(max_iter)):
        pr_old = pr
        # pr_new = alpha * (A^T * (pr_old / out_deg)) + dangling_mass + teleport
        contrib = pr_old * inv_out
        pr = alpha * (A.T @ contrib).astype(np.float32)
        dangling_mass = alpha * float(pr_old[dangling].sum()) / n
        pr = pr + dangling_mass + teleport
        if float(np.abs(pr - pr_old).sum()) < tol:
            break
    pr = pr / (pr.sum() + 1e-12)
    return pr.astype(np.float32)


def spd_anchor_distances(
    adj: sp.coo_matrix,
    anchors: np.ndarray,
    directed: bool,
    max_dist: int,
) -> np.ndarray:
    """
    Compute shortest-path distances from each anchor to all nodes.
    Returns dist matrix of shape [A, N] with values in [1..max_dist] and (max_dist+1) as "unreachable/too-far".
    """
    A = _coo_to_csr_binary(adj)
    # scipy uses 0 for self-distance; we keep 0 as 0 then clamp.
    dist = sp.csgraph.shortest_path(
        A,
        directed=bool(directed),
        unweighted=True,
        indices=anchors.astype(np.int32),
    )
    # dist: float64 with inf for unreachable
    dist = np.asarray(dist)
    dist[np.isinf(dist)] = float(max_dist + 1)
    dist = np.clip(dist, 0, max_dist + 1).astype(np.int64)
    return dist


@dataclass
class TopoPEConfig:
    enabled: bool = False
    spd_enabled: bool = True
    deg_enabled: bool = True
    pr_enabled: bool = True

    # SPD hyperparams
    spd_num_anchors: int = 16
    spd_max_dist: int = 5
    spd_dist_emb_dim: int = 16
    spd_mlp_hidden: int = 64
    directed: bool = False

    # PR hyperparams
    pr_alpha: float = 0.85
    pr_max_iter: int = 50


class TopologyPositionEncoder(nn.Module):
    """
    Produce a topology-aware enriched embedding:
        h0 = proj([x_init, pe_spd, pe_deg, pe_pr])
    where pe_* are computed from graph topology (Eq.7 in the user's excerpt).
    """

    def __init__(
        self,
        emb_dim: int,
        cfg: TopoPEConfig,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.cfg = cfg

        in_dim = self.emb_dim
        self.use_spd = bool(cfg.spd_enabled)
        self.use_deg = bool(cfg.deg_enabled)
        self.use_pr = bool(cfg.pr_enabled)

        if self.use_spd:
            # dist ids: 0..max_dist+1 (max_dist+1 is "unreachable")
            self.dist_emb = nn.Embedding(int(cfg.spd_max_dist) + 2, int(cfg.spd_dist_emb_dim))
            self.spd_mlp = nn.Sequential(
                nn.Linear(int(cfg.spd_dist_emb_dim), int(cfg.spd_mlp_hidden)),
                nn.GELU(),
                nn.Linear(int(cfg.spd_mlp_hidden), self.emb_dim),
            )
            in_dim += self.emb_dim

        if self.use_deg:
            self.deg_mlp = nn.Sequential(
                nn.Linear(1, self.emb_dim),
                nn.GELU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )
            in_dim += self.emb_dim

        if self.use_pr:
            self.pr_mlp = nn.Sequential(
                nn.Linear(1, self.emb_dim),
                nn.GELU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )
            in_dim += self.emb_dim

        self.proj = nn.Sequential(
            nn.Linear(in_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
        )

    def encode(
        self,
        x_init: torch.Tensor,  # [N, D]
        spd_dists: Optional[torch.Tensor],  # [A, N] int64, or None
        deg: Optional[torch.Tensor],  # [N] float32, or None
        pr: Optional[torch.Tensor],  # [N] float32, or None
    ) -> torch.Tensor:
        feats = [x_init]

        if self.use_spd:
            if spd_dists is None:
                raise ValueError("SPD enabled but spd_dists is None")
            # [A,N] -> [A,N,E]
            dist_e = self.dist_emb(spd_dists)  # type: ignore[arg-type]
            pe_spd = self.spd_mlp(dist_e).mean(dim=0)  # [N,D]
            feats.append(pe_spd)

        if self.use_deg:
            if deg is None:
                raise ValueError("Degree PE enabled but deg is None")
            feats.append(self.deg_mlp(deg.view(-1, 1)))

        if self.use_pr:
            if pr is None:
                raise ValueError("PageRank PE enabled but pr is None")
            feats.append(self.pr_mlp(pr.view(-1, 1)))

        x = torch.cat(feats, dim=-1)
        return self.proj(x)


