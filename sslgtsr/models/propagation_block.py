"""
消息传递层模块，对应论文 3.4 节。

将 Transformer 层和 LightGCN 层组合成一个基本传播块（PropagationBlock）。
每个传播块包含：
  1. Transformer 层（多头注意力 + 残差连接 + 层归一化）
  2. LightGCN 层（邻居聚合）

块结构对应论文图 2：
    输入 p^(k-1) / q^(k-1)
      → MultiHeadAttn + AddNorm
      → LightGCN 邻域传播
    输出 p^(k) / q^(k) / e_i^(k)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sslgtsr.models.transformer_layer import TransformerLayer
from sslgtsr.models.lightgcn_layer import LightGCNLayer


class PropagationBlock(nn.Module):
    """
    传播块，对应论文 3.4 节的完整传播块。

    将 Transformer 层和 LightGCN 层组合：
    - Transformer 层：对中心节点与其注意力样本进行自注意力，捕捉全局依赖关系
    - LightGCN 层：在交互图或社交图上做邻居聚合，保留局部结构信息
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.tf_layer = TransformerLayer(emb_dim, n_heads, dropout)
        self.gnn_layer = LightGCNLayer()

    def forward_ui(
        self,
        q_u: torch.Tensor,            # [U, D]  交互视角用户嵌入
        e_i: torch.Tensor,            # [I, D]  物品嵌入
        ui_norm_adj: torch.Tensor,    # (U+I)×(U+I) 稀疏归一化邻接矩阵
        attn_indices: torch.Tensor,   # [U, T] 每个用户的注意力样本索引（来自社交邻接）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        交互图上的一个传播块。

        对 q_u 中的每个用户，用其在社交图中的注意力邻居作为 Transformer 的样本，
        然后在交互图上做 LightGCN 传播得到更新后的 q_u^(k) 和 e_i^(k)。

        参数:
            q_u:           [U, D]  交互视角用户嵌入（来自上一块或初始嵌入）
            e_i:           [I, D]  物品嵌入
            ui_norm_adj:   (U+I)×(U+I) 稀疏归一化邻接矩阵
            attn_indices:  [U, T]  每个用户的注意力样本索引（在 U 个用户上）

        返回:
            q_u_next:  [U, D] 更新后的交互视角用户嵌入
            e_i_next:  [I, D] 更新后的物品嵌入
        """
        sampled = q_u[attn_indices]  # [U, T, D]
        q_u_tf = self.tf_layer(q_u, sampled)  # [U, D]

        q_u_next, e_i_next = self.gnn_layer.forward_ui(q_u_tf, e_i, ui_norm_adj)
        return q_u_next, e_i_next

    def forward_uu(
        self,
        p_u: torch.Tensor,            # [U, D]  社交视角用户嵌入
        uu_norm_adj: torch.Tensor,    # U×U 稀疏归一化邻接矩阵
        attn_indices: torch.Tensor,   # [U, T] 每个用户的注意力样本索引
    ) -> torch.Tensor:
        """
        社交图上的一个传播块。

        对 p_u 中的每个用户，用其在社交图中的注意力邻居作为 Transformer 的样本，
        然后在社交图上做 LightGCN 传播得到更新后的 p_u^(k)。

        参数:
            p_u:           [U, D]  社交视角用户嵌入（来自上一块或初始嵌入）
            uu_norm_adj:   U×U 稀疏归一化邻接矩阵
            attn_indices:  [U, T]  每个用户的注意力样本索引（在 U 个用户上）

        返回:
            p_u_next:  [U, D] 更新后的社交视角用户嵌入
        """
        sampled = p_u[attn_indices]  # [U, T, D]
        p_u_tf = self.tf_layer(p_u, sampled)  # [U, D]

        p_u_next = self.gnn_layer.forward_uu(p_u_tf, uu_norm_adj)
        return p_u_next
