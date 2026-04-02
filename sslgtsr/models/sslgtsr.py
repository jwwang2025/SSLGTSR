from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from sslgtsr.data.graph import coo_to_torch_sparse, edge_dropout_coo
from sslgtsr.models.lightgcn_layer import lightgcn_propagate
from sslgtsr.models.ssl import FeatureDropout, info_nce_loss
from sslgtsr.models.cross_view_ssl import CrossViewSSLConfig, LearnableSimilarity, cross_view_align_loss
from sslgtsr.models.topo_pe import TopoPEConfig, TopologyPositionEncoder
from sslgtsr.models.propagation_block import PropagationBlock
from sslgtsr.models.fusion_layer import TwoViewFusion


@dataclass
class ModelOutputs:
    user_fused: torch.Tensor  # [U, D]
    item_ui: torch.Tensor     # [I, D]
    user_ui: torch.Tensor     # [U, D]
    user_uu: torch.Tensor      # [U, D]
    p_all: list[torch.Tensor]   # len=K+1, each [U,D] 社交视角各层嵌入
    q_all: list[torch.Tensor]   # len=K+1, each [U,D] 交互视角各层嵌入


class SSLGTSR(nn.Module):
    """
    严格遵循论文 3.4 节架构的 SSLGTSR 模型：

    - 输入：位置编码后的用户初始嵌入 p_u^0、物品初始嵌入 e_i^0
    - 堆叠 K 个传播块，每个块包含：
        1. Transformer 层（公式 8-11）：中心节点 + 注意力样本做多头自注意力
        2. GNN 层（公式 12-14）：LightGCN 风格在交互图/社交图上传播
    - 公式 (15)：融合各层嵌入得到最终用户表征
    - 预测：用户表征与物品表征的点积
    - SSL：对比学习（公式 17）
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int,
        n_layers: int,           # K，传播块的数量
        tf_heads: int,
        tf_dropout: float,
        ssl_temperature: float,
        ssl_edge_drop_rate: float,
        ssl_feature_drop_rate: float,
        topo_pe: TopoPEConfig | None = None,
        cross_view_ssl: CrossViewSSLConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.emb_dim = int(emb_dim)
        self.n_layers = int(n_layers)  # K

        # ---- 可学习嵌入（对应论文初始嵌入 e_u^0 / e_i^0 / p_u^0）----
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

        # ---- K 个传播块（论文 3.4，Transformer + GNN 交替）----
        self.propagation_blocks = nn.ModuleList([
            PropagationBlock(emb_dim, tf_heads, tf_dropout)
            for _ in range(self.n_layers)
        ])

        # ---- 拓扑感知位置编码器（对应论文 3.3 节，可选）----
        self.topo_cfg = topo_pe or TopoPEConfig(enabled=False)
        self.topo_encoder: TopologyPositionEncoder | None = None
        if self.topo_cfg.enabled:
            self.topo_encoder = TopologyPositionEncoder(self.emb_dim, self.topo_cfg)

        # ---- 图上缓存的拓扑信号（由 Trainer 注入）----
        self._ui_spd: torch.Tensor | None = None
        self._ui_deg: torch.Tensor | None = None
        self._ui_pr: torch.Tensor | None = None
        self._uu_spd: torch.Tensor | None = None
        self._uu_deg: torch.Tensor | None = None
        self._uu_pr: torch.Tensor | None = None

        # ---- 注意力采样索引（对应论文 3.2 节）----
        self._attn_indices: Optional[torch.Tensor] = None  # [U, T]

        # ---- 多视图融合（公式 15）----
        self.fusion = TwoViewFusion(self.emb_dim)

        # ---- SSL 配置 ----
        self.ssl_temperature = float(ssl_temperature)
        self.ssl_edge_drop_rate = float(ssl_edge_drop_rate)
        self.feature_drop = FeatureDropout(ssl_feature_drop_rate)

        # ---- 跨视图自监督学习（论文 3.5，可选）----
        self.cv_cfg = cross_view_ssl or CrossViewSSLConfig(enabled=False)
        self.cv_sim: LearnableSimilarity | None = None
        if self.cv_cfg.enabled:
            self.cv_sim = LearnableSimilarity(
                self.emb_dim, hidden_dim=self.emb_dim * 2, slope=self.cv_cfg.leaky_relu_slope
            )
            self._cv_rng = torch.Generator()
            self._cv_rng.manual_seed(0)

    def set_topology_signals(
        self,
        *,
        ui_spd: torch.Tensor | None = None,
        ui_deg: torch.Tensor | None = None,
        ui_pr: torch.Tensor | None = None,
        uu_spd: torch.Tensor | None = None,
        uu_deg: torch.Tensor | None = None,
        uu_pr: torch.Tensor | None = None,
    ) -> None:
        self._ui_spd = ui_spd
        self._ui_deg = ui_deg
        self._ui_pr = ui_pr
        self._uu_spd = uu_spd
        self._uu_deg = uu_deg
        self._uu_pr = uu_pr

    def set_attn_indices(self, attn_indices: Optional[torch.Tensor]) -> None:
        """
        设置每个用户的注意力样本索引 Smp(u)。
        Shape: [num_users, topk] 或 None。
        """
        self._attn_indices = attn_indices

    def init_attn_indices(self, num_users: int, topk: int, device: torch.device) -> None:
        """
        初始化注意力索引（用于首次 forward 采样前）。
        使用全零索引作为占位符，采样后会替换。
        """
        self._attn_indices = torch.zeros(num_users, topk, dtype=torch.long, device=device)

    def _init_with_topo(self, ui_x0: torch.Tensor, uu_x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """如果启用了拓扑位置编码，则对初始嵌入进行编码。"""
        if self.topo_encoder is not None:
            ui_x0 = self.topo_encoder.encode(ui_x0, self._ui_spd, self._ui_deg, self._ui_pr)
            uu_x0 = self.topo_encoder.encode(uu_x0, self._uu_spd, self._uu_deg, self._uu_pr)
        return ui_x0, uu_x0

    def forward(
        self,
        ui_norm_adj: torch.Tensor,
        uu_norm_adj: torch.Tensor,
    ) -> ModelOutputs:
        """
        完整前向传播，对应论文算法流程：

        1. 初始化：p_u^0 = q_u^0 = 可学习用户嵌入 + 拓扑编码（可选）
                   e_i^0 = 可学习物品嵌入 + 拓扑编码（可选）
        2. 迭代 K 次：
             - 对交互图（UI）执行 PropagationBlock.forward_ui
               → q_u^(k), e_i^(k)
             - 对社交图（UU）执行 PropagationBlock.forward_uu
               → p_u^(k)
        3. 公式 (15)：融合 p_u^0..p_u^K 和 q_u^0..q_u^K 得到 e_u
        4. 预测：e_u 与 e_i^K 的点积
        """
        attn_idx = self._attn_indices
        if attn_idx is None:
            raise ValueError(
                "Attention indices must be set via set_attn_indices() before forward(). "
                "Set attn_cfg.enabled=True in the config or call attn_sampler.sample() in Trainer."
            )

        # ---- 初始嵌入（论文 e_u^0 / e_i^0 / p_u^0）----
        p_u = self.user_emb.weight.clone()  # [U, D] 社交视角
        q_u = self.user_emb.weight.clone()  # [U, D] 交互视角
        e_i = self.item_emb.weight.clone()  # [I, D] 物品

        # 拓扑位置编码（可选）
        ui_x0 = torch.cat([q_u, e_i], dim=0)  # [U+I, D]
        ui_x0, p_u = self._init_with_topo(ui_x0, p_u)
        q_u = ui_x0[: self.num_users]
        e_i = ui_x0[self.num_users:]

        # ---- 收集各层嵌入用于融合（公式 15）----
        p_all: list[torch.Tensor] = [p_u]
        q_all: list[torch.Tensor] = [q_u]

        # ---- K 个传播块（论文 3.4）----
        for k in range(self.n_layers):
            # --- 交互图上的传播块 ---
            # q_u 和 e_i 通过 GNN 传播；q_u 先经过 Transformer（使用注意力邻居作为样本）
            q_u, e_i = self.propagation_blocks[k].forward_ui(
                q_u, e_i, ui_norm_adj, attn_idx
            )

            # --- 社交图上的传播块 ---
            # p_u 先经过 Transformer（使用注意力邻居作为样本），再通过 GNN 传播
            p_u = self.propagation_blocks[k].forward_uu(
                p_u, uu_norm_adj, attn_idx
            )

            p_all.append(p_u)
            q_all.append(q_u)

        # ---- 公式 (15)：融合各层嵌入 ----
        user_fused = self.fusion(p_all, q_all)  # [U, D]

        return ModelOutputs(
            user_fused=user_fused,
            item_ui=e_i,
            user_ui=q_u,
            user_uu=p_u,
            p_all=p_all,
            q_all=q_all,
        )

    def forward_fallback(self, ui_norm_adj: torch.Tensor, uu_norm_adj: torch.Tensor) -> ModelOutputs:
        """
        当没有提供注意力索引时，退化为纯 LightGCN 的简化版本。
        保持与原版 baseline 的兼容性。
        """
        # 初始嵌入
        p_u = self.user_emb.weight.clone()
        q_u = self.user_emb.weight.clone()
        e_i = self.item_emb.weight.clone()

        ui_x0 = torch.cat([q_u, e_i], dim=0)
        ui_x0, p_u = self._init_with_topo(ui_x0, p_u)
        q_u = ui_x0[: self.num_users]
        e_i = ui_x0[self.num_users:]

        p_all: list[torch.Tensor] = [p_u]
        q_all: list[torch.Tensor] = [q_u]

        for k in range(self.n_layers):
            # GNN 传播
            ui_x = torch.cat([q_u, e_i], dim=0)
            ui_x = torch.sparse.mm(ui_norm_adj, ui_x)
            q_u = ui_x[: self.num_users]
            e_i = ui_x[self.num_users:]

            p_u = torch.sparse.mm(uu_norm_adj, p_u)

            p_all.append(p_u)
            q_all.append(q_u)

        user_fused = self.fusion(p_all, q_all)

        return ModelOutputs(
            user_fused=user_fused,
            item_ui=e_i,
            user_ui=q_u,
            user_uu=p_u,
            p_all=p_all,
            q_all=q_all,
        )

    def score(
        self,
        user_vecs: torch.Tensor,
        item_vecs: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        return (user_vecs[users] * item_vecs[items]).sum(dim=-1)

    @torch.no_grad()
    def full_sort_scores(
        self,
        ui_norm_adj: torch.Tensor,
        uu_norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        out = self.forward(ui_norm_adj, uu_norm_adj)
        return out.user_fused @ out.item_ui.t()  # [U, I]

    def ssl_loss(
        self,
        ui_norm_adj_coo: sp.coo_matrix,
        uu_norm_adj_coo: sp.coo_matrix,
        device: torch.device,
        user_batch: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """
        对比学习损失（论文 3.5，公式 17）。
        对交互图和社交图做边 dropout，得到两个增强视图，
        在采样的用户对上计算 InfoNCE 损失。
        """
        # 边 dropout 创建两个增强视图
        ui_aug = edge_dropout_coo(ui_norm_adj_coo, self.ssl_edge_drop_rate, rng)
        uu_aug = edge_dropout_coo(uu_norm_adj_coo, self.ssl_edge_drop_rate, rng)
        ui_aug_t = coo_to_torch_sparse(ui_aug, device)
        uu_aug_t = coo_to_torch_sparse(uu_aug, device)

        # ---- SSL 前向传播（使用增广图）----
        # 初始嵌入 + 特征 dropout
        p0 = self.feature_drop(self.user_emb.weight)
        q0 = self.feature_drop(self.user_emb.weight)
        e0 = self.feature_drop(self.item_emb.weight)

        # 拓扑编码（复用全局信号）
        ui_x0 = torch.cat([q0, e0], dim=0)
        if self.topo_encoder is not None:
            ui_x0 = self.topo_encoder.encode(ui_x0, self._ui_spd, self._ui_deg, self._ui_pr)
            p0 = self.topo_encoder.encode(p0, self._uu_spd, self._uu_deg, self._uu_pr)
        q0 = ui_x0[: self.num_users]
        e0 = ui_x0[self.num_users:]

        p_aug = p0
        q_aug = q0
        e_aug = e0

        attn_idx = self._attn_indices

        for k in range(self.n_layers):
            # 交互图传播
            q_aug, e_aug = self.propagation_blocks[k].forward_ui(
                q_aug, e_aug, ui_aug_t, attn_idx
            )
            # 社交图传播
            p_aug = self.propagation_blocks[k].forward_uu(
                p_aug, uu_aug_t, attn_idx
            )

        # 取最后一层嵌入用于 SSL 对比
        z1 = q_aug[user_batch]  # 交互视角
        z2 = p_aug[user_batch]  # 社交视角

        loss = info_nce_loss(z1, z2, temperature=self.ssl_temperature)

        # 跨视图对齐损失（可选）
        if self.cv_sim is not None and self.cv_cfg.enabled:
            align = cross_view_align_loss(
                user_ui=q_aug,
                user_uu=p_aug,
                sim_fn=self.cv_sim,
                n_pairs=int(self.cv_cfg.n_pairs),
                rng=self._cv_rng,
            )
            loss = loss + float(self.cv_cfg.weight) * align

        return loss
