"""
PPIGraph: Protein-Protein Interaction 그래프 관리자.

데이터 소스 (STRING DB v12.0, Human):
  9606.protein.links.v12.0.txt   — 단백질 쌍 + combined_score (0–1000)
  9606.protein.info.v12.0.txt    — preferred_name (= gene symbol)

권장 파이프라인:
  1. script/download_ppi.py 실행 → data/ppi/ppi_edges.csv 생성
  2. PPIGraph.from_csv("data/ppi/ppi_edges.csv", vocab) 로드
  3. ppi_graph.get_bias(gene_ids)  →  (N, N) attention bias

내부 저장 형태:
  edge_index : LongTensor  (E, 2)  — vocab 인덱스 쌍 [idx_a, idx_b]
  edge_score : FloatTensor (E,)    — [0, 1] 정규화된 상호작용 신뢰도
"""

from __future__ import annotations
import os
from typing import Optional

import torch
import numpy as np

from .vocab import GeneVocab


class PPIGraph:

    def __init__(
        self,
        edge_index: torch.Tensor,  # (E, 2) LongTensor
        edge_score: torch.Tensor,  # (E,)   FloatTensor ∈ [0, 1]
        vocab: GeneVocab,
    ):
        assert edge_index.shape[1] == 2
        assert edge_index.shape[0] == edge_score.shape[0]

        self.vocab      = vocab
        self.edge_index = edge_index  # CPU
        self.edge_score = edge_score  # CPU
        self._vocab_size = len(vocab)

    # ── 핵심: attention bias 계산 ─────────────────────────────────────────

    def get_bias(self, gene_ids: torch.Tensor) -> torch.Tensor:
        """
        현재 배치의 유전자 집합에 대한 (N, N) PPI bias 행렬 반환.

        Args:
            gene_ids : (N,) or (B, N) LongTensor — vocab 인덱스
        Returns:
            ppi_bias : (N, N) FloatTensor — 연결된 쌍은 score, 비연결은 0
        """
        # (B, N) → (N,) : 배치 내 모든 샘플은 같은 유전자 집합을 공유
        ids = gene_ids[0] if gene_ids.dim() == 2 else gene_ids
        ids_cpu = ids.cpu()
        N = ids_cpu.shape[0]
        V = self._vocab_size

        # vocab_idx → position 매핑 (없으면 -1)
        pos_map = torch.full((V,), -1, dtype=torch.long)
        pos_map[ids_cpu] = torch.arange(N)

        # 엣지 양 끝점이 모두 현재 유전자 집합에 있는지 확인
        a_pos = pos_map[self.edge_index[:, 0]]  # (E,)
        b_pos = pos_map[self.edge_index[:, 1]]  # (E,)
        valid = (a_pos >= 0) & (b_pos >= 0)

        # (N, N) bias 행렬 구성 (대칭)
        bias = torch.zeros(N, N, dtype=torch.float32)
        a_v, b_v, s_v = a_pos[valid], b_pos[valid], self.edge_score[valid]
        bias[a_v, b_v] = s_v
        bias[b_v, a_v] = s_v  # 무방향 그래프

        return bias.to(gene_ids.device)

    # ── 통계 출력 ─────────────────────────────────────────────────────────

    def stats(self) -> None:
        E = self.edge_index.shape[0]
        n_genes = self._vocab_size - 2  # PAD, UNK 제외
        print(f"PPIGraph:")
        print(f"  엣지 수       : {E:,}")
        print(f"  연결된 유전자 : "
              f"{self.edge_index.unique().numel():,} / {n_genes:,}")
        print(f"  Score 분포    : "
              f"min={self.edge_score.min():.3f}  "
              f"mean={self.edge_score.mean():.3f}  "
              f"max={self.edge_score.max():.3f}")

    # ── 생성자 팩토리 ─────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str,
        vocab: GeneVocab,
        min_score: float = 0.7,
    ) -> PPIGraph:
        """
        CSV 파일에서 로드.

        형식 (헤더 포함):
            gene_a,gene_b,score
            TP53,MDM2,0.98
            ...

        Args:
            path      : CSV 파일 경로
            vocab     : GeneVocab 인스턴스
            min_score : 이 값 미만 엣지 제거 (0–1)
        """
        import pandas as pd
        df = pd.read_csv(path)
        df = df[df["score"] >= min_score]

        rows, cols, scores = [], [], []
        skipped = 0
        for _, row in df.iterrows():
            a, b, s = row["gene_a"], row["gene_b"], float(row["score"])
            ia, ib = vocab[a], vocab[b]
            if ia == vocab.unk_idx or ib == vocab.unk_idx:
                skipped += 1
                continue
            rows.append(ia); cols.append(ib); scores.append(s)

        if skipped:
            print(f"  [PPIGraph] {skipped:,}개 엣지 skip (vocab 없는 유전자)")

        edge_index = torch.tensor([rows, cols], dtype=torch.long).T  # (E, 2)
        edge_score = torch.tensor(scores, dtype=torch.float32)

        print(f"  [PPIGraph] 로드 완료: {edge_index.shape[0]:,}개 엣지 "
              f"(score ≥ {min_score})")
        return cls(edge_index, edge_score, vocab)

    @classmethod
    def from_string_files(
        cls,
        links_path: str,
        info_path: str,
        vocab: GeneVocab,
        min_score: int = 700,
    ) -> PPIGraph:
        """
        STRING DB raw 파일에서 직접 로드.

        Args:
            links_path : 9606.protein.links.v12.0.txt
            info_path  : 9606.protein.info.v12.0.txt
            min_score  : combined_score 최솟값 (0–1000, 권장: 700)
        """
        import pandas as pd

        print("STRING DB 로딩 중...")
        # protein ID → gene symbol 매핑
        info = pd.read_csv(info_path, sep="\t",
                           usecols=["#string_protein_id", "preferred_name"])
        id2gene: dict[str, str] = dict(
            zip(info["#string_protein_id"], info["preferred_name"])
        )

        # 엣지 로드
        links = pd.read_csv(links_path, sep=" ")
        links = links[links["combined_score"] >= min_score]
        links["score"] = links["combined_score"] / 1000.0

        rows, cols, scores = [], [], []
        skipped = 0
        for _, row in links.iterrows():
            ga = id2gene.get(row["protein1"], "")
            gb = id2gene.get(row["protein2"], "")
            ia, ib = vocab[ga], vocab[gb]
            if ia == vocab.unk_idx or ib == vocab.unk_idx:
                skipped += 1
                continue
            rows.append(ia); cols.append(ib)
            scores.append(float(row["score"]))

        print(f"  로드: {len(rows):,}개 엣지 / {skipped:,}개 skip")
        edge_index = torch.tensor([rows, cols], dtype=torch.long).T
        edge_score = torch.tensor(scores, dtype=torch.float32)
        return cls(edge_index, edge_score, vocab)

    @classmethod
    def make_synthetic(
        cls,
        vocab: GeneVocab,
        n_edges: int = 50_000,
        seed: int = 42,
    ) -> PPIGraph:
        """
        테스트용 랜덤 PPI 그래프 생성.
        실제 학습에는 사용하지 말 것.
        """
        rng = np.random.default_rng(seed)
        V = len(vocab)
        # PAD(0), UNK(1) 제외하고 샘플링
        gene_range = np.arange(2, V)
        idx = rng.choice(gene_range, size=(n_edges, 2), replace=True)
        # 자기 자신 엣지 제거
        mask = idx[:, 0] != idx[:, 1]
        idx = idx[mask]
        scores = rng.uniform(0.4, 1.0, size=len(idx)).astype(np.float32)

        edge_index = torch.from_numpy(idx).long()
        edge_score = torch.from_numpy(scores)
        print(f"  [PPIGraph] synthetic: {len(edge_index):,}개 엣지 생성")
        return cls(edge_index, edge_score, vocab)

    # ── 저장 / 로드 ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {"edge_index": self.edge_index, "edge_score": self.edge_score},
            path,
        )

    @classmethod
    def load(cls, path: str, vocab: GeneVocab) -> PPIGraph:
        data = torch.load(path, map_location="cpu")
        return cls(data["edge_index"], data["edge_score"], vocab)
