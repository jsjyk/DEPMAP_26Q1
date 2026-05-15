"""
PathwayDB  + PathwayPooling: HMOT Layer 3.

PathwayDB
─────────
Pathway → 멤버 유전자 집합 매핑.
지원 형식: GMT (MSigDB / KEGG / Reactome / Hallmark)

권장 데이터:
  - MSigDB Hallmark (H)  : 50개  암 생물학에 curated
  - MSigDB C2 KEGG       : ~186개 KEGG 경로
  - MSigDB C2 Reactome   : ~1,600개 (계층적)
  → script/download_pathways.py 로 자동 다운로드

PathwayPooling
──────────────
HMOT Layer 3: gene embeddings → pathway embeddings.

설계 (multi-head cross-attention):
  Q : 각 pathway의 학습된 query 벡터  (P, d_model)
  K,V : Layer 2 출력 gene embeddings  (B, N, d_model)
  mask: pathway_member[p, n] AND NOT padding[n]

  score[b, h, p, n] = Q_p · K_n / √d_k
  → pathway_mask 외 위치 → -inf
  → softmax (빈 pathway → NaN → 0으로 대체)
  → (B, H, P, N) attn_weights  ← 해석 가능성의 핵심

  출력: (B, P, d_model) pathway 임베딩
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import GeneVocab


# ═══════════════════════════════════════════════════════════════════════════
# PathwayDB
# ═══════════════════════════════════════════════════════════════════════════

class PathwayDB:
    """
    Pathway → 멤버 유전자 (vocab 인덱스) 매핑.

    Args:
        pathway_names : list[str]         길이 P
        members       : list[LongTensor]  pathway별 vocab 인덱스 리스트
        vocab         : GeneVocab
    """

    def __init__(
        self,
        pathway_names: list[str],
        members: list[torch.Tensor],
        vocab: GeneVocab,
    ):
        assert len(pathway_names) == len(members)
        self.pathway_names = pathway_names
        self._members      = members          # list[LongTensor]
        self._vocab        = vocab
        self._vocab_size   = len(vocab)

        # 역방향 인덱스: vocab_idx → [pathway_idx, ...]  (get_mask 속도 향상)
        self._gene_to_pathways: dict[int, list[int]] = defaultdict(list)
        for p_idx, member_ids in enumerate(members):
            for vid in member_ids.tolist():
                self._gene_to_pathways[vid].append(p_idx)

    @property
    def n_pathways(self) -> int:
        return len(self.pathway_names)

    # ── 핵심: pathway × gene 마스크 ──────────────────────────────────────

    def get_mask(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """
        현재 유전자 집합에 대한 (P, N) BoolTensor 반환.
        True = 해당 pathway에 해당 유전자가 속함.

        Args:
            gene_ids : (N,) or (B, N) — vocab 인덱스
        """
        ids = gene_ids[0] if gene_ids.dim() == 2 else gene_ids
        N   = ids.shape[0]
        P   = self.n_pathways

        mask = torch.zeros(P, N, dtype=torch.bool)
        for pos, vid in enumerate(ids.cpu().tolist()):
            for p_idx in self._gene_to_pathways.get(vid, []):
                mask[p_idx, pos] = True

        return mask.to(gene_ids.device)

    # ── 통계 ─────────────────────────────────────────────────────────────

    def coverage(self, gene_ids: torch.Tensor) -> None:
        """현재 유전자 집합에서 커버되는 pathway 수 출력."""
        mask    = self.get_mask(gene_ids)
        covered = mask.any(dim=1).sum().item()
        sizes   = mask.sum(dim=1)
        print(f"PathwayDB coverage:")
        print(f"  커버된 pathway: {covered} / {self.n_pathways}")
        print(f"  유전자 수 per pathway: "
              f"min={sizes.min().item()}  "
              f"mean={sizes.float().mean().item():.1f}  "
              f"max={sizes.max().item()}")

    def stats(self) -> None:
        sizes = [len(m) for m in self._members]
        print(f"PathwayDB: {self.n_pathways}개 pathway")
        print(f"  유전자 수: min={min(sizes)}  "
              f"mean={sum(sizes)/len(sizes):.1f}  max={max(sizes)}")
        print(f"  예시: {self.pathway_names[:5]}")

    # ── 팩토리 ───────────────────────────────────────────────────────────

    @classmethod
    def from_gmt(
        cls,
        path: str,
        vocab: GeneVocab,
        min_genes: int = 5,
        max_genes: int = 500,
    ) -> PathwayDB:
        """
        GMT 파일에서 로드.

        GMT 형식 (탭 구분):
          PATHWAY_NAME\\tDESCRIPTION\\tGENE1\\tGENE2\\t...
        """
        pathway_names, members = [], []
        skipped_size, skipped_unk = 0, 0

        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                name  = parts[0]
                genes = parts[2:]  # index 1 = description, skip

                # vocab 매핑
                ids = []
                for g in genes:
                    idx = vocab[g]
                    if idx != vocab.unk_idx:
                        ids.append(idx)
                    else:
                        skipped_unk += 1

                if not (min_genes <= len(ids) <= max_genes):
                    skipped_size += 1
                    continue

                pathway_names.append(name)
                members.append(torch.tensor(ids, dtype=torch.long))

        print(f"[PathwayDB] GMT 로드: {len(pathway_names)}개 pathway")
        if skipped_size:
            print(f"  크기 필터 제외: {skipped_size}개 (<{min_genes} or >{max_genes})")
        if skipped_unk:
            print(f"  vocab 미등록 유전자: {skipped_unk}개 (무시)")

        return cls(pathway_names, members, vocab)

    @classmethod
    def from_gseapy(
        cls,
        gene_set: str,
        vocab: GeneVocab,
        organism: str = "Human",
        min_genes: int = 5,
        max_genes: int = 500,
    ) -> PathwayDB:
        """
        gseapy 라이브러리로 MSigDB gene set 직접 로드.

        Args:
            gene_set : 예) "MSigDB_Hallmark_2020", "KEGG_2021_Human"
        """
        try:
            import gseapy as gp
        except ImportError:
            raise ImportError("pip install gseapy 후 사용 가능합니다.")

        library = gp.get_library(name=gene_set, organism=organism)
        pathway_names, members = [], []
        for name, genes in library.items():
            ids = [vocab[g] for g in genes if vocab[g] != vocab.unk_idx]
            if min_genes <= len(ids) <= max_genes:
                pathway_names.append(name)
                members.append(torch.tensor(ids, dtype=torch.long))

        print(f"[PathwayDB] gseapy '{gene_set}': {len(pathway_names)}개 pathway")
        return cls(pathway_names, members, vocab)

    @classmethod
    def make_synthetic(
        cls,
        vocab: GeneVocab,
        n_pathways: int = 50,
        genes_per_pathway: int = 100,
        overlap: float = 0.2,
        seed: int = 42,
    ) -> PathwayDB:
        """
        테스트용 랜덤 pathway 생성.
        overlap: 인접 pathway 간 유전자 공유 비율 (실제 biology 모사)
        """
        import numpy as np
        rng   = np.random.default_rng(seed)
        V     = len(vocab)
        gene_pool = np.arange(2, V)  # PAD/UNK 제외

        pathway_names, members = [], []
        prev_genes = None
        for i in range(n_pathways):
            n_overlap = int(genes_per_pathway * overlap) if prev_genes is not None else 0
            shared    = (rng.choice(prev_genes, n_overlap, replace=False).tolist()
                         if n_overlap else [])
            unique_n  = genes_per_pathway - len(shared)
            unique    = rng.choice(gene_pool, unique_n, replace=False).tolist()
            ids       = sorted(set(shared + unique))

            pathway_names.append(f"SYNTHETIC_PATHWAY_{i+1:03d}")
            members.append(torch.tensor(ids, dtype=torch.long))
            prev_genes = ids

        print(f"[PathwayDB] synthetic: {n_pathways}개 pathway "
              f"(~{genes_per_pathway}개/pathway, overlap={overlap:.0%})")
        return cls(pathway_names, members, vocab)

    # ── 저장 / 로드 ──────────────────────────────────────────────────────

    def save_gmt(self, path: str) -> None:
        """GMT 형식으로 저장."""
        with open(path, "w") as f:
            for name, m in zip(self.pathway_names, self._members):
                genes = [self._vocab.idx_to_gene(i) for i in m.tolist()]
                f.write(name + "\tna\t" + "\t".join(genes) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# PathwayPooling
# ═══════════════════════════════════════════════════════════════════════════

class PathwayPooling(nn.Module):
    """
    HMOT Layer 3: gene embeddings → pathway embeddings (multi-head cross-attention).

    전체 구조 (Pre-LN):
      pathway_queries → Q
      gene_emb        → K, V
      cross-attn (pathway_mask 적용) → residual + LN → FFN → residual + LN

    Args:
        n_pathways : pathway 수 (PathwayDB.n_pathways)
        d_model    : 임베딩 차원
        n_heads    : cross-attention 헤드 수
        ffn_dim    : FFN 중간 차원 (기본 4 × d_model)
        dropout    : dropout 비율
    """

    def __init__(
        self,
        n_pathways: int,
        d_model:    int,
        n_heads:    int  = 8,
        ffn_dim:    Optional[int] = None,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_pathways = n_pathways
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        ffn_dim         = ffn_dim or 4 * d_model

        # 학습된 pathway query 벡터
        self.pathway_queries = nn.Parameter(
            torch.empty(n_pathways, d_model)
        )
        nn.init.normal_(self.pathway_queries, std=0.02)

        # Cross-attention projections
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1    = nn.LayerNorm(d_model)  # query 정규화
        self.norm2    = nn.LayerNorm(d_model)  # gene 정규화 (cross-attn 입력)
        self.norm3    = nn.LayerNorm(d_model)  # FFN 앞

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.drop      = nn.Dropout(dropout)

        for lin in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(lin.weight)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        gene_emb:     torch.Tensor,           # (B, N, d_model)
        pathway_mask: torch.Tensor,           # (P, N)  BoolTensor
        padding_mask: Optional[torch.Tensor] = None,  # (B, N) True=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pathway_emb  : (B, P, d_model) — pathway 임베딩
            attn_weights : (B, n_heads, P, N) — 해석 가능성용
        """
        B, N, _  = gene_emb.shape
        P, H, dk = self.n_pathways, self.n_heads, self.d_head

        # ── Query: pathway queries (Pre-LN) ──────────────────────────
        q = self.norm1(self.pathway_queries)              # (P, d_model)
        Q = self.q_proj(q)                                # (P, d_model)
        Q = Q.view(P, H, dk).unsqueeze(0).expand(B, -1, -1, -1)
        Q = Q.permute(0, 2, 1, 3)                         # (B, H, P, dk)

        # ── Key / Value: gene embeddings (Pre-LN) ────────────────────
        g = self.norm2(gene_emb)                          # (B, N, d_model)
        K = self.k_proj(g).view(B, N, H, dk).permute(0, 2, 1, 3)  # (B, H, N, dk)
        V = self.v_proj(g).view(B, N, H, dk).permute(0, 2, 1, 3)

        # ── Scaled dot-product cross-attention ───────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        # scores: (B, H, P, N)

        # pathway 멤버 마스크 적용: (P, N) → (1, 1, P, N)
        member_mask = pathway_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, P, N)
        scores = scores.masked_fill(~member_mask, float("-inf"))

        # padding mask 추가 적용: (B, N) → (B, 1, 1, N)
        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # softmax → 빈 pathway (모두 -inf) 는 NaN → 0으로 대체
        attn_weights = F.softmax(scores, dim=-1).nan_to_num(0.0)  # (B, H, P, N)
        attn_weights = self.attn_drop(attn_weights)

        # ── 집계 → pathway 임베딩 ────────────────────────────────────
        out = torch.matmul(attn_weights, V)               # (B, H, P, dk)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, P, self.d_model)
        out = self.out_proj(out)                          # (B, P, d_model)

        # ── Residual 1: pathway_queries + cross-attn output ──────────
        pathway_emb = self.pathway_queries.unsqueeze(0) + self.drop(out)

        # ── FFN block (Pre-LN) ────────────────────────────────────────
        pathway_emb = pathway_emb + self.drop(self.ffn(self.norm3(pathway_emb)))

        return pathway_emb, attn_weights
