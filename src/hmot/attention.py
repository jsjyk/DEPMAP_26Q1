"""
PPITransformerLayer: PPI-guided multi-head self-attention (HMOT Layer 2).

어텐션 점수 공식:
  score[h, i, j] = (Q_i · K_j) / √d_k  +  α_h · ppi_score[i, j]

  α_h : 헤드별 학습 스칼라 (PPI가 어텐션에 미치는 영향 강도 조절)
        → 양수면 PPI 이웃을 더 주목, 학습을 통해 결정

  비연결 유전자쌍은 ppi_score = 0 이므로 표준 self-attention과 동일.

전체 레이어 구조 (Pre-LN Transformer):
  x → LN → PPIAttention → residual
    → LN → FFN           → residual
    → output (B, N, d_model)

해석 가능성:
  forward()가 attn_weights (B, n_heads, N, N)도 반환.
  → 어떤 유전자가 어떤 유전자에 주목했는지 사후 분석 가능.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPIAttention(nn.Module):
    """
    PPI bias가 추가된 Multi-head Self-Attention.

    Args:
        d_model  : 입력/출력 차원
        n_heads  : 어텐션 헤드 수 (d_model % n_heads == 0)
        dropout  : 어텐션 dropout
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 헤드별 PPI bias 스케일 (학습 가능)
        # 초기값 0 → 초반에 표준 self-attention처럼 동작, 점진적으로 PPI 반영
        self.ppi_head_scale = nn.Parameter(torch.zeros(n_heads))

        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x:            torch.Tensor,            # (B, N, d_model)
        ppi_bias:     torch.Tensor,            # (N, N) FloatTensor
        padding_mask: torch.Tensor | None = None,  # (B, N) True=padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output       : (B, N, d_model)
            attn_weights : (B, n_heads, N, N) — 해석 가능성용
        """
        B, N, _ = x.shape
        H, d_k  = self.n_heads, self.d_head

        # ── Q, K, V 프로젝션 + multi-head reshape ────────────────────
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # ── Scaled dot-product scores ─────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: (B, H, N, N)

        # ── PPI bias 추가 ─────────────────────────────────────────────
        # ppi_bias: (N, N) → (1, 1, N, N)
        # ppi_head_scale: (H,) → (1, H, 1, 1)
        ppi_contribution = (
            self.ppi_head_scale.view(1, H, 1, 1)
            * ppi_bias.unsqueeze(0).unsqueeze(0)
        )  # (1, H, N, N)
        scores = scores + ppi_contribution

        # ── Padding 마스킹 ────────────────────────────────────────────
        if padding_mask is not None:
            # 패딩 위치의 key를 -inf로 → softmax 후 0 가중치
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, N)
                float("-inf"),
            )

        # ── Softmax + Dropout ─────────────────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)       # (B, H, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # ── 출력 집계 ─────────────────────────────────────────────────
        output = torch.matmul(attn_weights, V)         # (B, H, N, d_k)
        output = output.transpose(1, 2).contiguous().view(B, N, self.d_model)
        output = self.out_proj(output)                 # (B, N, d_model)

        return output, attn_weights


class PPITransformerLayer(nn.Module):
    """
    HMOT Layer 2: PPI-guided Transformer block (Pre-LN 구조).

    구조:
      x → LayerNorm → PPIAttention → x (residual)
        → LayerNorm → FFN          → x (residual)

    Pre-LN은 학습 안정성이 좋아 Post-LN보다 권장됨.

    Args:
        d_model   : 토큰 임베딩 차원
        n_heads   : 어텐션 헤드 수
        ffn_dim   : FFN 중간 차원 (보통 4 × d_model)
        dropout   : dropout 비율
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ffn_dim:  int | None = None,
        dropout:  float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model

        self.norm1   = nn.LayerNorm(d_model)
        self.attn    = PPIAttention(d_model, n_heads, dropout)
        self.drop1   = nn.Dropout(dropout)

        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2   = nn.Dropout(dropout)

    def forward(
        self,
        x:            torch.Tensor,            # (B, N, d_model)
        ppi_bias:     torch.Tensor,            # (N, N)
        padding_mask: torch.Tensor | None = None,  # (B, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x            : (B, N, d_model) — 업데이트된 유전자 임베딩
            attn_weights : (B, n_heads, N, N) — 해석 가능성용
        """
        # ── Self-Attention block (Pre-LN) ─────────────────────────────
        residual = x
        attn_out, attn_weights = self.attn(
            self.norm1(x), ppi_bias, padding_mask
        )
        x = residual + self.drop1(attn_out)

        # ── FFN block (Pre-LN) ────────────────────────────────────────
        x = x + self.drop2(self.ffn(self.norm2(x)))

        return x, attn_weights
