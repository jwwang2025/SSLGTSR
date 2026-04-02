"""
融合层模块，对应论文公式 (15)。

融合交互视角和社交视角的用户嵌入，生成最终的用户表征。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoViewFusion(nn.Module):
    """
    融合两个视角的用户表征，得到最终用户嵌入。
    对应论文公式 (15)（重编号后为公式 3-12）：
        p_u = Σ_{k=0}^{K} α_k · p_u^k
        q_u = Σ_{k=0}^{K} α_k · q_u^k
        e~u = W_6( σ(W_4·q_u) ∥ σ(W_5·p_u) )
        e_u = ||e~u||₂ · e~u
    其中 α_k = 1/(k+1)，|·| 表示拼接，σ 为 tanh 激活函数，
    ||·||₂ 表示 L2 归一化。
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.W_4 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_5 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_6 = nn.Linear(emb_dim * 2, emb_dim, bias=False)

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

        p_accum = torch.zeros_like(p_list[0])
        q_accum = torch.zeros_like(q_list[0])
        for k, (p, q) in enumerate(zip(p_list, q_list)):
            alpha = 1.0 / (k + 1)  # 论文公式 α_k = 1/(k+1)
            p_accum = p_accum + alpha * p
            q_accum = q_accum + alpha * q

        fused = torch.cat([torch.tanh(self.W_4(q_accum)), torch.tanh(self.W_5(p_accum))], dim=-1)  # [U, 2D]
        fused = self.W_6(fused)  # [U, D]

        fused_norm = fused / fused.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)  # L2 归一化
        return fused_norm
