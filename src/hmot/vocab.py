"""
GeneVocab: gene symbol → integer index mapping.

특수 토큰:
  0: <PAD>  padding (gradient 없음)
  1: <UNK>  vocab에 없는 유전자
"""

from __future__ import annotations
import json
import os
from typing import Iterable


class GeneVocab:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, genes: Iterable[str]):
        unique_genes = sorted(set(genes))
        self._genes = [self.PAD, self.UNK] + unique_genes
        self._stoi: dict[str, int] = {g: i for i, g in enumerate(self._genes)}

    # ── 기본 인터페이스 ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._genes)

    def __contains__(self, gene: str) -> bool:
        return gene in self._stoi

    def __getitem__(self, gene: str) -> int:
        """gene symbol → index. 없으면 UNK 반환."""
        return self._stoi.get(gene, self._stoi[self.UNK])

    def idx_to_gene(self, idx: int) -> str:
        return self._genes[idx]

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def unk_idx(self) -> int:
        return 1

    @property
    def genes(self) -> list[str]:
        """특수 토큰 제외한 유전자 목록."""
        return self._genes[2:]

    # ── 파일 I/O ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._genes, f, indent=2)

    @classmethod
    def load(cls, path: str) -> GeneVocab:
        with open(path) as f:
            all_tokens = json.load(f)
        # 특수 토큰 제외하고 재구성
        specials = {cls.PAD, cls.UNK}
        genes = [t for t in all_tokens if t not in specials]
        return cls(genes)

    # ── 전처리 파일에서 자동 구성 ─────────────────────────────────────────

    @classmethod
    def from_preprocessed(cls, data_dir: str) -> GeneVocab:
        """
        전처리 디렉터리의 3개 오믹스 파일에서 유전자명을 읽어 vocab 구성.
        prefix(EXP_/CNV_/MUT_)를 제거한 유전자 심볼의 union을 사용.
        """
        import pandas as pd

        prefix_map = {
            "expression_zscore.csv": "EXP_",
            "cnv_log2.csv":          "CNV_",
            "mutation_binary.csv":   "MUT_",
        }
        genes: set[str] = set()
        for fname, prefix in prefix_map.items():
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                print(f"  [경고] 파일 없음, 건너뜀: {fname}")
                continue
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            genes.update(c[len(prefix):] for c in cols if c.startswith(prefix))

        print(f"GeneVocab: {len(genes):,}개 유전자 (union of 3 modalities)")
        return cls(genes)
