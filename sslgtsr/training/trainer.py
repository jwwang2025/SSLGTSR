from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from sslgtsr.data.graph import build_ui_bipartite_norm_adj, build_uu_norm_adj, coo_to_torch_sparse
from sslgtsr.data.sampling import BPRSampler
from sslgtsr.models.sslgtsr import SSLGTSR
from sslgtsr.models.topo_pe import TopoPEConfig, degree_vector, pagerank_vector, spd_anchor_distances
from sslgtsr.models.attn_sampling import AttnSamplingConfig, GraphAttentionSampler
from sslgtsr.training.losses import bpr_loss, l2_reg
from sslgtsr.training.metrics import evaluate_topk
from sslgtsr.utils.logging import write_json


class Trainer:
    def __init__(
        self,
        model: SSLGTSR,
        ui_norm_adj_coo: sp.coo_matrix,
        uu_norm_adj_coo: sp.coo_matrix,
        device: torch.device,
        lr: float,
        weight_decay: float,
        bpr_reg: float,
        ssl_weight: float,
        seed: int,
        early_stopping_patience: int = 10,
        attn_sample_size: int = 15,
        cfg: Dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.ui_norm_adj_coo = ui_norm_adj_coo.tocoo()
        self.uu_norm_adj_coo = uu_norm_adj_coo.tocoo()
        self.device = device
        self.bpr_reg = float(bpr_reg)
        self.ssl_weight = float(ssl_weight)
        self.rng = np.random.default_rng(int(seed))
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = -1.0
        self.patience_counter = 0

        self.ui_norm_adj = coo_to_torch_sparse(self.ui_norm_adj_coo, device)
        self.uu_norm_adj = coo_to_torch_sparse(self.uu_norm_adj_coo, device)

        # Store config for accessing model parameters
        self.cfg = cfg

        # ---- 拓扑位置编码（论文 3.3 节，可选）----
        topo_cfg = getattr(self.model, "topo_cfg", None)
        if isinstance(topo_cfg, TopoPEConfig) and topo_cfg.enabled:
            self._inject_topology_signals(topo_cfg)

        # ---- 注意力采样（论文 3.2 节）----
        if cfg is not None and "model" in cfg:
            attn_sample_size = cfg["model"].get("attn_sample_size", 15)
        self.attn_cfg = AttnSamplingConfig(enabled=True, topk=attn_sample_size)
        self.attn_sampler = GraphAttentionSampler(self.attn_cfg)
        self.user_attn_indices: torch.Tensor | None = None

        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    def _inject_topology_signals(self, cfg: TopoPEConfig) -> None:
        device = self.device

        ui_adj = self.ui_norm_adj_coo
        uu_adj = self.uu_norm_adj_coo

        ui_deg = torch.from_numpy(degree_vector(ui_adj)).to(device=device, dtype=torch.float32)
        uu_deg = torch.from_numpy(degree_vector(uu_adj)).to(device=device, dtype=torch.float32)

        ui_pr = torch.from_numpy(
            pagerank_vector(ui_adj, alpha=cfg.pr_alpha, max_iter=cfg.pr_max_iter)
        ).to(device=device, dtype=torch.float32)
        uu_pr = torch.from_numpy(
            pagerank_vector(uu_adj, alpha=cfg.pr_alpha, max_iter=cfg.pr_max_iter)
        ).to(device=device, dtype=torch.float32)

        ui_spd_t = None
        uu_spd_t = None
        if cfg.spd_enabled:
            rng = np.random.default_rng(0)
            n_ui = int(ui_adj.shape[0])
            n_uu = int(uu_adj.shape[0])
            a_ui = min(int(cfg.spd_num_anchors), n_ui)
            a_uu = min(int(cfg.spd_num_anchors), n_uu)
            ui_anchors = rng.choice(n_ui, size=a_ui, replace=False).astype(np.int64)
            uu_anchors = rng.choice(n_uu, size=a_uu, replace=False).astype(np.int64)

            ui_spd = spd_anchor_distances(ui_adj, ui_anchors, directed=cfg.directed, max_dist=cfg.spd_max_dist)
            uu_spd = spd_anchor_distances(uu_adj, uu_anchors, directed=cfg.directed, max_dist=cfg.spd_max_dist)

            ui_spd_t = torch.from_numpy(ui_spd).to(device=device, dtype=torch.long)
            uu_spd_t = torch.from_numpy(uu_spd).to(device=device, dtype=torch.long)

        self.model.set_topology_signals(
            ui_spd=ui_spd_t,
            ui_deg=ui_deg if cfg.deg_enabled else None,
            ui_pr=ui_pr if cfg.pr_enabled else None,
            uu_spd=uu_spd_t,
            uu_deg=uu_deg if cfg.deg_enabled else None,
            uu_pr=uu_pr if cfg.pr_enabled else None,
        )

    def train_one_epoch(self, sampler: BPRSampler, steps: int, batch_size: int) -> Dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        bpr_sum = 0.0
        ssl_sum = 0.0
        reg_sum = 0.0

        # ---- 每个 epoch 开始时，用当前模型（detach）采样注意力邻居 ----
        with torch.no_grad():
            self.model.eval()
            out_full = self.model(self.ui_norm_adj, self.uu_norm_adj)
            self.user_attn_indices = self.attn_sampler.sample(
                out_full.user_ui, out_full.user_uu, self.uu_norm_adj
            )
            # 将采样索引注入模型，供后续 forward 使用
            self.model.set_attn_indices(self.user_attn_indices)

        self.model.train()

        for _ in tqdm(range(steps), desc="train", leave=False):
            batch = sampler.sample(batch_size)
            users = torch.from_numpy(batch.users).to(self.device)
            pos = torch.from_numpy(batch.pos_items).to(self.device)
            neg = torch.from_numpy(batch.neg_items).to(self.device)

            out = self.model(self.ui_norm_adj, self.uu_norm_adj)
            pos_scores = self.model.score(out.user_fused, out.item_ui, users, pos)
            neg_scores = self.model.score(out.user_fused, out.item_ui, users, neg)
            bpr = bpr_loss(pos_scores, neg_scores)

            # L2 正则（论文损失函数中的 λ2 ||θ||²）
            reg = l2_reg(out.user_fused[users], out.item_ui[pos], out.item_ui[neg])

            # SSL 对比学习损失（论文 3.5，公式 17）
            ssl = self.model.ssl_loss(
                ui_norm_adj_coo=self.ui_norm_adj_coo,
                uu_norm_adj_coo=self.uu_norm_adj_coo,
                device=self.device,
                user_batch=users,
                rng=self.rng,
            )

            loss = bpr + self.bpr_reg * reg + self.ssl_weight * ssl

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            loss_sum += float(loss.detach().cpu())
            bpr_sum += float(bpr.detach().cpu())
            ssl_sum += float(ssl.detach().cpu())
            reg_sum += float(reg.detach().cpu())

        denom = float(steps)
        return {
            "loss": loss_sum / denom,
            "bpr": bpr_sum / denom,
            "ssl": ssl_sum / denom,
            "reg": reg_sum / denom,
        }

    @torch.no_grad()
    def evaluate(
        self,
        user_pos_train: Sequence[np.ndarray],
        user_pos_eval: Sequence[np.ndarray],
        topk: Sequence[int],
    ) -> Dict[str, float]:
        self.model.eval()
        # 评估时需要 attn_indices，确保已设置
        if self.user_attn_indices is not None:
            self.model.set_attn_indices(self.user_attn_indices)
        scores = self.model.full_sort_scores(self.ui_norm_adj, self.uu_norm_adj).detach().cpu().numpy()
        return evaluate_topk(scores, list(user_pos_train), list(user_pos_eval), topk_list=topk)

    def save(self, path: str | Path, extra: Dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"state_dict": self.model.state_dict()}
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    def load(self, path: str | Path) -> Dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        return payload.get("extra", {})


def build_graphs_from_split(
    num_users: int,
    num_items: int,
    train_ui: np.ndarray,
    uu: np.ndarray,
    directed: bool,
) -> tuple[sp.coo_matrix, sp.coo_matrix]:
    ui_norm = build_ui_bipartite_norm_adj(num_users, num_items, train_ui)
    uu_norm = build_uu_norm_adj(num_users, uu, directed=directed) if uu.size > 0 else sp.eye(num_users, format="coo")
    return ui_norm, uu_norm
