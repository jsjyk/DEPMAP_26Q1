"""
GlobalPooling: HMOT Layer 4.

Pathway embeddings (B, P, d_model) → 세포주 임베딩 Z (B, d_out).

설계 (Multi-head Attention Pooling):
  - 헤드별 학습된 global query 벡터 1개가 P개 pathway를 cross-attend
  - Q : (H, d_k) learned  →  각 헤드가 서로 다른 "중요한 경로" 학습
  - K, V : pathway embeddings 에서 projection
  - 출력 : 헤드 concat → d_model → optional linear to d_out

해석 가능성:
  attn_weights (B, H, P) 반환
  → 어떤 pathway가 이 세포주 임베딩에 가장 많이 기여했나
  → 헤드별로 서로 다른 "생물학적 시각" 포착 기대
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPooling(nn.Module):
    """
    HMOT Layer 4: (B, P, d_model) → Z (B, d_out).

    Args:
        d_model : pathway 임베딩 차원 (PathwayPooling 출력과 일치)
        n_heads : attention 헤드 수
        d_out   : 최종 세포주 임베딩 차원 (None 이면 d_model 유지)
        dropout : dropout 비율
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int  = 8,
        d_out:    Optional[int] = None,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_out   = d_out or d_model

        # 헤드별 전역 query 벡터 — 어떤 pathway에 주목할지 학습
        self.global_query = nn.Parameter(
            torch.empty(n_heads, self.d_head)
        )
        nn.init.normal_(self.global_query, std=0.02)

        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, self.d_out, bias=True)

        self.norm      = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.drop      = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        pathway_emb: torch.Tensor,  # (B, P, d_model)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Z            : (B, d_out)  — 세포주 임베딩
            attn_weights : (B, n_heads, P) — pathway별 기여도
        """
        B, P, _  = pathway_emb.shape
        H, dk    = self.n_heads, self.d_head

        # Pre-LN
        x = self.norm(pathway_emb)  # (B, P, d_model)

        # K, V: pathway embeddings 기반
        K = self.k_proj(x).view(B, P, H, dk).permute(0, 2, 1, 3)  # (B, H, P, dk)
        V = self.v_proj(x).view(B, P, H, dk).permute(0, 2, 1, 3)

        # Q: (H, dk) → (B, H, 1, dk) broadcast
        Q = self.global_query.unsqueeze(0).unsqueeze(2).expand(B, -1, 1, -1)

        # Scaled dot-product attention over pathways
        scores       = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        # scores: (B, H, 1, P)
        attn_weights = F.softmax(scores, dim=-1)          # (B, H, 1, P)
        attn_weights = self.attn_drop(attn_weights)

        # 가중 합산 → concat heads
        out = torch.matmul(attn_weights, V)               # (B, H, 1, dk)
        out = out.squeeze(2)                              # (B, H, dk)
        out = out.contiguous().view(B, self.d_model)      # (B, d_model)
        out = self.drop(out)

        Z = self.out_proj(out)                            # (B, d_out)

        return Z, attn_weights.squeeze(2)                 # Z, (B, H, P)
