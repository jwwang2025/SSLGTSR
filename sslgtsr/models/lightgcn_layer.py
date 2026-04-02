"""
LightGCN 层模块，对应论文 3.4.2 节。

轻量级图卷积网络（LightGCN），通过去除传统 GCN 中的线性变换和非线性激活操作，
简化了图卷积网络的结构。

包含：
- LightGCNPropagation：多层 LightGCN 传播（收集各层嵌入用于融合）
- LightGCNLayer：单层 LightGCN 传播
"""

from __future__ import annotations

import torch
import torch.nn as nn


def lightgcn_propagate(
    norm_adj: torch.Tensor, x: torch.Tensor, n_layers: int
) -> torch.Tensor:
    """
    LightGCN 传播：对各层的嵌入向量取平均。
    
    这是独立的 LightGCN 传播函数，适用于纯 LightGCN baseline。
    
    参数:
        norm_adj: 稀疏归一化邻接矩阵 (N × N)
        x: 稠密节点嵌入向量 (N × D)
        n_layers: 传播层数
    返回:
        [N, D] 各层嵌入的平均
    """
    out = x
    embs = [x]
    for _ in range(n_layers):
        out = torch.sparse.mm(norm_adj, out)
        embs.append(out)
    return torch.stack(embs, dim=0).mean(dim=0)


class LightGCNLayer(nn.Module):
    """
    单层 LightGCN 传播，对应论文公式 (12)-(14)。
    
    公式 (12): q_u^k = Σ_{i∈N_u^I} q_{i}^{k-1} / √(|N_u^I|)·√(|N_i^I|)
    公式 (13): p_u^k = Σ_{v∈N_u^S} p_{v}^{k-1} / √(|N_u^S|)·√(|N_v^S|)
    公式 (14): e_i^k = Σ_{u∈N_i^I} e_{u}^{k-1} / √(|N_i^I|)·√(|N_u^I|)
    
    注意：归一化因子已由传入的 norm_adj 矩阵预处理提供。
    """

    def forward_ui(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        ui_norm_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        用户-物品交互图上的 LightGCN 传播（公式 14）。
        
        参数:
            user_emb:   [U, D] 用户嵌入
            item_emb:   [I, D] 物品嵌入
            ui_norm_adj: (U+I)×(U+I) 稀疏对称归一化邻接矩阵
        返回:
            user_emb: [U, D] 更新后的用户嵌入
            item_emb: [I, D] 更新后的物品嵌入
        """
        x = torch.cat([user_emb, item_emb], dim=0)  # [U+I, D]
        x = torch.sparse.mm(ui_norm_adj, x)         # [U+I, D]
        u = x[: user_emb.size(0)]
        i = x[user_emb.size(0):]
        return u, i

    def forward_uu(
        self,
        user_emb: torch.Tensor,
        uu_norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        社交图上的 LightGCN 传播（公式 13）。
        
        参数:
            user_emb:   [U, D] 用户嵌入
            uu_norm_adj: U×U 稀疏对称归一化邻接矩阵
        返回:
            [U, D] 更新后的用户嵌入
        """
        return torch.sparse.mm(uu_norm_adj, user_emb)


class LightGCNEncoder(nn.Module):
    """
    纯 LightGCN 编码器（不含 Transformer），用于 baseline 对比实验。
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = int(n_layers)

    def forward(self, norm_adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return lightgcn_propagate(norm_adj, x, self.n_layers)
