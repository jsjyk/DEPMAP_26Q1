"""
HMOT: Hierarchical Multi-Omics Transformer.

Layer 1  GeneTokenizer        (gene_ids, omics, mask) → (B, N, d_model)
Layer 2  PPITransformerLayer  × n_ppi_layers           → (B, N, d_model)
Layer 3  PathwayPooling                                → (B, P, d_model)
Layer 4  GlobalPooling                                 → Z (B, d_out)

사용 예:
    model = HMOT.from_config(HMOTConfig(...), vocab, pathway_db)
    Z, maps = model.encode(batch, ppi_graph, pathway_db, return_attns=True)

    # 해석: 어떤 pathway가 이 세포주를 특징짓나
    maps.global_attn   # (B, H, P)       Layer 4
    maps.pathway_attn  # (B, H, P, N)    Layer 3
    maps.ppi_attns     # list[(B,H,N,N)] Layer 2 각 레이어
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn

from .tokenizer import GeneTokenizer
from .attention import PPITransformerLayer
from .pathway   import PathwayPooling
from .global_pool import GlobalPooling
from .ppi        import PPIGraph
from .pathway    import PathwayDB


# ── 설정 ─────────────────────────────────────────────────────────────────

@dataclass
class HMOTConfig:
    vocab_size:    int
    n_pathways:    int
    d_model:       int   = 256
    n_heads:       int   = 8
    n_ppi_layers:  int   = 2       # Layer 2 반복 횟수
    ffn_multiplier: int  = 4       # FFN dim = d_model × ffn_multiplier
    d_out:         int   = 512     # 최종 세포주 임베딩 차원
    dropout:       float = 0.1


# ── 해석 가능성 컨테이너 ──────────────────────────────────────────────────

@dataclass
class AttentionMaps:
    """모든 레이어의 attention weight 묶음."""
    ppi_attns:    list[torch.Tensor]  # [(B, H, N, N)] × n_ppi_layers
    pathway_attn: torch.Tensor        # (B, H, P, N)
    global_attn:  torch.Tensor        # (B, H, P)

    def top_pathways(
        self,
        pathway_db: PathwayDB,
        sample_idx: int = 0,
        head_idx:   int = 0,
        top_k:      int = 5,
    ) -> list[tuple[str, float]]:
        """Layer 4 기준 상위 pathway 반환 [(name, score)]."""
        scores = self.global_attn[sample_idx, head_idx]  # (P,)
        topk   = scores.topk(min(top_k, scores.shape[0]))
        return [
            (pathway_db.pathway_names[i], round(v.item(), 4))
            for i, v in zip(topk.indices.tolist(), topk.values)
        ]

    def top_genes_in_pathway(
        self,
        pathway_idx: int,
        gene_list:   list[str],
        sample_idx:  int = 0,
        head_idx:    int = 0,
        top_k:       int = 5,
    ) -> list[tuple[str, float]]:
        """Layer 3 기준 특정 pathway에서 상위 기여 유전자 반환."""
        scores = self.pathway_attn[sample_idx, head_idx, pathway_idx]  # (N,)
        topk   = scores.topk(min(top_k, scores.shape[0]))
        return [
            (gene_list[i], round(v.item(), 4))
            for i, v in zip(topk.indices.tolist(), topk.values)
        ]


# ── HMOT 모델 ─────────────────────────────────────────────────────────────

class HMOT(nn.Module):
    """
    Hierarchical Multi-Omics Transformer.

    Args:
        cfg : HMOTConfig
    """

    def __init__(self, cfg: HMOTConfig):
        super().__init__()
        self.cfg = cfg
        ffn_dim  = cfg.d_model * cfg.ffn_multiplier

        # Layer 1: Gene Tokenizer
        self.tokenizer = GeneTokenizer(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )

        # Layer 2: PPI Transformer (stacked)
        self.ppi_layers = nn.ModuleList([
            PPITransformerLayer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ffn_dim=ffn_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_ppi_layers)
        ])

        # Layer 3: Pathway Pooling
        self.pathway_pooling = PathwayPooling(
            n_pathways=cfg.n_pathways,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_dim=ffn_dim,
            dropout=cfg.dropout,
        )

        # Layer 4: Global Pooling → 세포주 임베딩 Z
        self.global_pooling = GlobalPooling(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_out=cfg.d_out,
            dropout=cfg.dropout,
        )

        # 최종 Layer Norm (Z 출력 전 안정화)
        self.final_norm = nn.LayerNorm(cfg.d_out)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        batch:       dict,
        ppi_graph:   PPIGraph,
        pathway_db:  PathwayDB,
        return_attns: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionMaps]:
        """
        Args:
            batch        : DataLoader 출력 (collate_fn 결과)
            ppi_graph    : PPIGraph 인스턴스
            pathway_db   : PathwayDB 인스턴스
            return_attns : True면 AttentionMaps도 반환

        Returns:
            Z           : (B, d_out) — 세포주 임베딩
            attn_maps   : AttentionMaps (return_attns=True 일 때)
        """
        gene_ids     = batch["gene_ids"]     # (B, N)
        omics        = batch["omics"]        # (B, N, 3)
        omics_mask   = batch["omics_mask"]   # (B, N, 3)
        padding_mask = batch["padding_mask"] # (B, N)

        # ── Layer 1: Gene Tokenization ───────────────────────────────
        x = self.tokenizer(gene_ids, omics, omics_mask)  # (B, N, d_model)

        # ── Layer 2: PPI-guided Transformer ──────────────────────────
        ppi_bias  = ppi_graph.get_bias(gene_ids)         # (N, N)
        ppi_attns = []
        for layer in self.ppi_layers:
            x, attn = layer(x, ppi_bias, padding_mask)
            if return_attns:
                ppi_attns.append(attn)

        # ── Layer 3: Pathway Pooling ──────────────────────────────────
        pathway_mask = pathway_db.get_mask(gene_ids)     # (P, N)
        pathway_emb, l3_attn = self.pathway_pooling(
            x, pathway_mask, padding_mask
        )  # (B, P, d_model)

        # ── Layer 4: Global Pooling ───────────────────────────────────
        Z, l4_attn = self.global_pooling(pathway_emb)   # (B, d_out)
        Z = self.final_norm(Z)

        if return_attns:
            maps = AttentionMaps(
                ppi_attns=ppi_attns,
                pathway_attn=l3_attn,
                global_attn=l4_attn,
            )
            return Z, maps

        return Z

    # ── 편의 메서드 ──────────────────────────────────────────────────────

    def encode(
        self,
        batch:        dict,
        ppi_graph:    PPIGraph,
        pathway_db:   PathwayDB,
        return_attns: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionMaps]:
        """forward의 alias — 추론/임베딩 추출 시 명시적으로 사용."""
        return self.forward(batch, ppi_graph, pathway_db, return_attns)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── 팩토리 ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: HMOTConfig) -> HMOT:
        return cls(cfg)
