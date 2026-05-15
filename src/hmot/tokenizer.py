"""
GeneTokenizer: (gene_ids, omics, omics_mask) → gene token embeddings.

설계:
  token_i = LayerNorm(
      Gene_Emb(gene_id_i)                            ← 유전자 identity
    + OmicsProj([expr, cnv, mut, m_e, m_c, m_m])    ← 오믹스 값 + 측정 여부
  )

  - Gene_Emb     : nn.Embedding (vocab_size × d_model)
  - OmicsProj    : Linear(6→d_model) → GELU → Linear(d_model→d_model)
  - 입력 6 = 값 3개 [expr, cnv, mut] + 마스크 3개 [has_expr, has_cnv, has_mut]
    → 마스크를 값과 함께 넣어 "측정 안 됨"과 "측정값=0"을 구분

Masked Gene Modeling (pre-training용):
  - mask_genes()  : 랜덤하게 유전자 토큰을 마스킹
  - <MASK> 토큰 인덱스는 vocab의 두 번째 특수 토큰(index=2)으로 예약
"""

from __future__ import annotations
import torch
import torch.nn as nn


class GeneTokenizer(nn.Module):
    """
    Args:
        vocab_size : GeneVocab의 전체 크기 (len(vocab))
        d_model    : 토큰 임베딩 차원 (기본 256)
        dropout    : dropout 비율
        pad_idx    : 패딩 인덱스 (gradient 없음, 기본 0)
    """

    MASK_IDX = 2  # vocab 인덱스 2를 <MASK> 토큰으로 예약

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Gene ID 임베딩 ───────────────────────────────────────────
        self.gene_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # ── 오믹스 값 프로젝션 ────────────────────────────────────────
        # 입력: [expr, cnv, mut, has_expr, has_cnv, has_mut] = 6 dims
        self.omics_proj = nn.Sequential(
            nn.Linear(6, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        for module in self.omics_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        gene_ids:   torch.Tensor,  # (B, N)  LongTensor
        omics:      torch.Tensor,  # (B, N, 3) FloatTensor [expr, cnv, mut]
        omics_mask: torch.Tensor,  # (B, N, 3) BoolTensor
    ) -> torch.Tensor:             # (B, N, d_model)
        """
        Returns:
            tokens : (B, N, d_model) — 패딩 위치는 0 벡터로 유지됨
        """
        id_emb = self.gene_emb(gene_ids)  # (B, N, d_model)

        # 값 + 마스크를 함께 넣어 "측정 안 됨 vs 값=0" 구분
        omics_input = torch.cat(
            [omics, omics_mask.float()], dim=-1
        )  # (B, N, 6)
        omics_emb = self.omics_proj(omics_input)  # (B, N, d_model)

        tokens = self.layer_norm(id_emb + omics_emb)  # (B, N, d_model)
        tokens = self.dropout(tokens)
        return tokens

    # ── Masked Gene Modeling 유틸 ─────────────────────────────────────────

    def mask_genes(
        self,
        gene_ids:    torch.Tensor,   # (B, N)
        omics:       torch.Tensor,   # (B, N, 3)
        omics_mask:  torch.Tensor,   # (B, N, 3)
        padding_mask: torch.Tensor,  # (B, N) True=pad
        mask_ratio:  float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        BERT 방식으로 유전자 토큰을 랜덤 마스킹.

        마스킹 전략 (BERT 80/10/10 규칙):
          80% → <MASK> 토큰 + 오믹스 값 0으로 교체
          10% → 랜덤한 다른 유전자 ID로 교체
          10% → 원본 유지 (복원 시 정규화 역할)

        Returns:
            masked_gene_ids  : (B, N)   마스킹 적용된 gene_ids
            masked_omics     : (B, N, 3)
            masked_omics_mask: (B, N, 3)
            label_mask       : (B, N)  True = 복원해야 할 위치
        """
        B, N = gene_ids.shape
        device = gene_ids.device

        # 패딩 위치 제외하고 마스킹 후보 결정
        candidate = ~padding_mask  # (B, N)
        rand = torch.rand(B, N, device=device)
        label_mask = candidate & (rand < mask_ratio)  # (B, N)

        # 복사본 생성
        masked_ids   = gene_ids.clone()
        masked_omics = omics.clone()
        masked_omask = omics_mask.clone()

        # 80%: <MASK> 토큰
        replace_mask = label_mask & (torch.rand(B, N, device=device) < 0.8)
        masked_ids[replace_mask]     = self.MASK_IDX
        masked_omics[replace_mask]   = 0.0
        masked_omask[replace_mask]   = False

        # 10%: 랜덤 유전자 ID로 교체 (나머지 20% 중 절반)
        replace_rand = (
            label_mask
            & ~replace_mask
            & (torch.rand(B, N, device=device) < 0.5)
        )
        vocab_size = self.gene_emb.num_embeddings
        rand_ids = torch.randint(3, vocab_size, (B, N), device=device)
        masked_ids[replace_rand] = rand_ids[replace_rand]

        # 나머지 10%: 원본 유지 (이미 복사본이므로 별도 처리 불필요)

        return masked_ids, masked_omics, masked_omask, label_mask
