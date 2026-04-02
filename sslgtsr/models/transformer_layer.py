"""
Transformer 层模块，对应论文 3.4.1 节。

包含：
- MultiHeadAttention：多头自注意力机制（公式 8-10）
- TransformerLayer：多头注意力 + 残差连接 + 层归一化（公式 11）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制，对应论文公式 (8)-(10)。

    对于中心节点 p_i 及其注意力样本 Smp(p_i) = {p_j1, ..., p_jT}，
    将 {p_i} ∪ Smp(p_i) 作为序列输入，计算 Q/K/V 投影，
    然后通过缩放点积注意力聚合邻居信息。

    公式 (8): p_i' = Σ_{j=1}^{m} α_{ij} · s_{i,j} · W^V
    公式 (9): α_{ij} = exp(d^{-1}(p_i W^Q)(s_{i,j} W^K)^T) / Σ_{t=1}^{m} exp(d^{-1}(p_i W^Q)(s_{i,t} W^K)^T)
    公式 (10): p_i'' = Concat(p_{i,1}', ..., p_{i,h}') · W^h
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

        tokens = torch.cat([center.unsqueeze(1), neighbors], dim=1)  # [B, 1+T, D]

        Q = self.W_Q(tokens)  # [B, 1+T, D]
        K = self.W_K(tokens)
        V = self.W_V(tokens)

        Q = Q.view(B, 1 + T, self.n_heads, self.head_dim).transpose(1, 2)   # [B, h, 1+T, d]
        K = K.view(B, 1 + T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1 + T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, h, 1+T, 1+T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # [B, h, 1+T, d]
        out = out.transpose(1, 2).contiguous().view(B, 1 + T, self.emb_dim)  # [B, 1+T, D]

        out = out[:, 0, :]  # [B, D]
        out = self.W_O(out)  # [B, D]
        return out


class TransformerLayer(nn.Module):
    """
    Transformer 层，对应论文公式 (11)。

    进行残差连接和层归一化：
        p_i^out = LayerNorm(p_i + p_i'')
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, center: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        先做注意力，再做残差+层归一化（对应公式 11）。

        参数:
            center:    [B, D] 中心节点嵌入
            neighbors: [B, T, D] 注意力样本嵌入
        返回:
            [B, D] 融合了位置编码和注意力机制后的表征
        """
        attn_out = self.attn(center, neighbors)
        out = self.norm1(center + self.dropout(attn_out))
        return out
