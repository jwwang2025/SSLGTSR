from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch


def _normalize_adj(adj: sp.coo_matrix) -> sp.coo_matrix:
    # Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}
    rowsum = np.asarray(adj.sum(axis=1)).squeeze()
    d_inv_sqrt = np.power(rowsum + 1e-12, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).tocoo()


def coo_to_torch_sparse(adj: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    adj = adj.tocoo()
    indices = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    shape = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def build_ui_bipartite_norm_adj(
    num_users: int, num_items: int, ui_edges: np.ndarray
) -> sp.coo_matrix:
    # UI edges: array of shape [E, 2] with (u, i)
    u = ui_edges[:, 0].astype(np.int64)
    i = ui_edges[:, 1].astype(np.int64)
    # bipartite adjacency in (U+I) x (U+I)
    rows = np.concatenate([u, i + num_users])
    cols = np.concatenate([i + num_users, u])
    data = np.ones_like(rows, dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))
    return _normalize_adj(adj)


def build_uu_norm_adj(num_users: int, uu_edges: np.ndarray, directed: bool) -> sp.coo_matrix:
    a = uu_edges[:, 0].astype(np.int64)
    b = uu_edges[:, 1].astype(np.int64)
    if directed:
        rows = a
        cols = b
    else:
        rows = np.concatenate([a, b])
        cols = np.concatenate([b, a])
    data = np.ones_like(rows, dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_users, num_users))
    return _normalize_adj(adj)


def edge_dropout_coo(adj: sp.coo_matrix, drop_rate: float, rng: np.random.Generator) -> sp.coo_matrix:
    if drop_rate <= 0.0:
        return adj.tocoo()
    adj = adj.tocoo()
    nnz = adj.nnz
    keep = rng.random(nnz) >= drop_rate
    if keep.sum() == 0:
        # keep at least one edge to avoid empty graph
        keep[rng.integers(0, nnz)] = True
    dropped = sp.coo_matrix((adj.data[keep], (adj.row[keep], adj.col[keep])), shape=adj.shape)
    return dropped


@dataclass
class GraphTensors:
    ui_norm_adj: torch.Tensor  # sparse (U+I)x(U+I)
    uu_norm_adj: torch.Tensor  # sparse UxU


