from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制，对应论文公式 (8)-(10)。

    对于中心节点 p_i 及其注意力样本 Smp(p_i) = {p_j1, ..., p_jT}，
    将 {p_i} ∪ Smp(p_i) 作为序列输入，计算 Q/K/V 投影，
    然后通过缩放点积注意力聚合邻居信息。
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.emb_dim = int(emb_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.emb_dim // self.n_heads
        self.scale = self.head_dim**-0.5

        self.W_Q = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_K = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_V = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_O = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, center: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        参数:
            center:      [B, D] 中心节点嵌入（对应论文中的 p_i 或 q_u）
            neighbors:   [B, T, D] 注意力样本嵌入（对应 Smp(p_i) 或 Smp(q_u)）
        返回:
            [B, D] 经过注意力聚合后的中心节点表征
        """
        B = center.size(0)
        T = neighbors.size(1)

        # tokens: [B, 1+T, D]，第一个 token 是中心节点
        tokens = torch.cat([center.unsqueeze(1), neighbors], dim=1)  # [B, 1+T, D]

        Q = self.W_Q(tokens)  # [B, 1+T, D]
        K = self.W_K(tokens)
        V = self.W_V(tokens)

        # 多头: reshape 为 [B, n_heads, 1+T, head_dim]
        Q = Q.view(B, 1 + T, self.n_heads, self.head_dim).transpose(1, 2)   # [B, h, 1+T, d]
        K = K.view(B, 1 + T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1 + T, self.n_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力 (公式 8 的核心)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, h, 1+T, 1+T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # [B, h, 1+T, d]
        out = out.transpose(1, 2).contiguous().view(B, 1 + T, self.emb_dim)  # [B, 1+T, D]

        # 取中心节点（第 0 个 token）的输出作为聚合结果
        out = out[:, 0, :]  # [B, D]
        out = self.W_O(out)  # [B, D]
        return out


class TransformerAttentionBlock(nn.Module):
    """
    单个传播块（论文 3.4 节），包含：
    1. Transformer 层（多头注意力 + 残差连接 + 层归一化）
    2. GNN 层（LightGCN 风格的邻居聚合）

    块结构对应论文图 2：
        输入 p^(k-1) / q^(k-1)
          → MultiHeadAttn + AddNorm
          → LightGCN 邻域传播
        输出 p^(k) / q^(k) / e_i^(k)
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward_transformer(
        self,
        center: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """
        先做注意力，再做残差+层归一化（对应公式 11）。
        """
        attn_out = self.attn(center, neighbors)  # [B, D]
        out = self.norm1(center + self.dropout(attn_out))  # [B, D] 残差连接
        return out

    def forward_gnn_ui(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        ui_norm_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        用户-物品交互图上的 LightGCN 传播（公式 14）：
            e_u^(k) = Σ_{i∈N_u^I} e_i^(k-1) / |N_u^I|
            e_i^(k) = Σ_{u∈N_i^I} e_u^(k-1) / |N_i^I|
        ui_norm_adj 是对称归一化的邻接矩阵 (U+I)×(U+I)。
        """
        x = torch.cat([user_emb, item_emb], dim=0)  # [U+I, D]
        x = torch.sparse.mm(ui_norm_adj, x)          # [U+I, D]
        u = x[: user_emb.size(0)]
        i = x[user_emb.size(0):]
        return u, i

    def forward_gnn_uu(
        self,
        user_emb: torch.Tensor,
        uu_norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        社交图上的 LightGCN 传播（公式 12/13）：
            p_u^(k) = Σ_{v∈N_u^S} p_v^(k-1) / |N_u^S|
        uu_norm_adj 是对称归一化的邻接矩阵 U×U。
        """
        return torch.sparse.mm(uu_norm_adj, user_emb)  # [U, D]


class PropagationBlock(nn.Module):
    """
    对应论文 3.4 节的一个完整传播块。
    包含：
      - Transformer 层（对中心节点与其注意力样本进行自注意力）
      - GNN 层（在交互图或社交图上做 LightGCN 传播）
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.tf_block = TransformerAttentionBlock(emb_dim, n_heads, dropout)

    def forward_ui(
        self,
        q_u: torch.Tensor,       # [U, D]  交互视角用户嵌入
        e_i: torch.Tensor,       # [I, D]  物品嵌入
        ui_norm_adj: torch.Tensor,  # (U+I)×(U+I) 稀疏归一化邻接
        attn_indices: torch.Tensor, # [U, T] 每个用户的注意力样本索引（来自社交邻接）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        交互图上的一个传播块。

        对 q_u 中的每个用户，用其在社交图中的注意力邻居作为 Transformer 的样本，
        然后在交互图上做 LightGCN 传播得到更新后的 q_u^(k) 和 e_i^(k)。

        参数:
            q_u:           [U, D]  交互视角用户嵌入 (来自上一块或初始嵌入)
            e_i:           [I, D]  物品嵌入
            ui_norm_adj:   (U+I)×(U+I) 稀疏归一化邻接矩阵
            attn_indices:  [U, T]  每个用户的注意力样本索引（在 U 个用户上）

        返回:
            q_u_next:  [U, D] 更新后的交互视角用户嵌入
            e_i_next:  [I, D] 更新后的物品嵌入
        """
        # ---- Transformer 层：在注意力样本上做自注意力 ----
        # 注意力样本来自社交邻接，取对应索引的 q_u
        sampled = q_u[attn_indices]  # [U, T, D]
        q_u_tf = self.tf_block.forward_transformer(q_u, sampled)  # [U, D]

        # ---- GNN 层：在交互图上做 LightGCN 传播 ----
        q_u_next, e_i_next = self.tf_block.forward_gnn_ui(q_u_tf, e_i, ui_norm_adj)
        return q_u_next, e_i_next

    def forward_uu(
        self,
        p_u: torch.Tensor,       # [U, D]  社交视角用户嵌入
        uu_norm_adj: torch.Tensor,  # U×U 稀疏归一化邻接矩阵
        attn_indices: torch.Tensor, # [U, T] 每个用户的注意力样本索引
    ) -> torch.Tensor:
        """
        社交图上的一个传播块。

        对 p_u 中的每个用户，用其在社交图中的注意力邻居作为 Transformer 的样本，
        然后在社交图上做 LightGCN 传播得到更新后的 p_u^(k)。

        参数:
            p_u:           [U, D]  社交视角用户嵌入 (来自上一块或初始嵌入)
            uu_norm_adj:   U×U 稀疏归一化邻接矩阵
            attn_indices:  [U, T]  每个用户的注意力样本索引（在 U 个用户上）

        返回:
            p_u_next:  [U, D] 更新后的社交视角用户嵌入
        """
        # ---- Transformer 层 ----
        sampled = p_u[attn_indices]  # [U, T, D]
        p_u_tf = self.tf_block.forward_transformer(p_u, sampled)  # [U, D]

        # ---- GNN 层：在社交图上做 LightGCN 传播 ----
        p_u_next = self.tf_block.forward_gnn_uu(p_u_tf, uu_norm_adj)
        return p_u_next


class TwoViewFusion(nn.Module):
    """
    融合两个视角的用户表征，得到最终用户嵌入。
    对应论文公式 (15)：
        e_u^k = σ( Σ_{k'=0}^{K} α_{k'} * W_4 · [p_u^{k'}, q_u^{k'}] )
    其中 α_k = 1/(k+1)，|·| 表示拼接。
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.W_4 = nn.Linear(emb_dim * 2, emb_dim, bias=False)

    def forward(
        self,
        p_list: list[torch.Tensor],  # len = K+1, each [U, D]
        q_list: list[torch.Tensor],  # len = K+1, each [U, D]
    ) -> torch.Tensor:
        """
        参数:
            p_list: 各层社交视角嵌入列表，p_list[0] 为初始嵌入，p_list[k] 为第 k 层输出
            q_list: 各层交互视角嵌入列表，格式同上
        返回:
            [U, D] 融合后的最终用户嵌入
        """
        assert len(p_list) == len(q_list), "p_list and q_list must have the same length"
        K = len(p_list) - 1  # 总共 K 层

        fused_list = []
        for k, (p, q) in enumerate(zip(p_list, q_list)):
            alpha = 1.0 / (k + 1)  # 论文公式 α_k = 1/(k+1)
            concat = torch.cat([p, q], dim=-1)           # [U, 2D]
            fused = torch.tanh(self.W_4(concat))          # [U, D]
            fused_list.append(alpha * fused)              # 加权

        return sum(fused_list)  # [U, D]
