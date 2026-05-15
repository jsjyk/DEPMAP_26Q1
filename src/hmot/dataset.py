"""
OmicsDataset: 전처리된 DepMap CSV → PyTorch Dataset.

각 샘플(세포주)은 다음을 반환:
  gene_ids    LongTensor  (N,)     유전자 vocab 인덱스
  omics       FloatTensor (N, 3)   [expr_zscore, cnv_log2, mut_binary]
  omics_mask  BoolTensor  (N, 3)   True = 해당 모달리티 값 존재
  model_id    str                  세포주 ID

가변 N 처리:
  - 유전자 서브셋을 지정하면 그 N개만 사용 (실험용)
  - 지정 없으면 3개 파일 union 전체 (~19,427개)
  - collate_fn이 배치 내 max_N으로 동적 패딩
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .vocab import GeneVocab


# ── 파일 설정 ────────────────────────────────────────────────────────────

_OMICS_FILES = [
    ("expression_zscore.csv", "EXP_"),
    ("cnv_log2.csv",          "CNV_"),
    ("mutation_binary.csv",   "MUT_"),
]


class OmicsDataset(Dataset):
    """
    멀티오믹스 DepMap 데이터셋.

    Args:
        data_dir  : 전처리 CSV 디렉터리 경로
        vocab     : GeneVocab 인스턴스
        genes     : 사용할 유전자 목록 (None = union 전체)
        samples   : 사용할 세포주 목록 (None = 공통 전체)
    """

    def __init__(
        self,
        data_dir: str,
        vocab: GeneVocab,
        genes: Optional[list[str]] = None,
        samples: Optional[list[str]] = None,
    ):
        self.vocab = vocab

        # ── 1. CSV 로드 ──────────────────────────────────────────────
        frames: list[pd.DataFrame] = []
        for fname, prefix in _OMICS_FILES:
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            df = pd.read_csv(path, index_col=0)
            # prefix 제거 → 유전자 심볼만 남김
            df.columns = [
                c[len(prefix):] if c.startswith(prefix) else c
                for c in df.columns
            ]
            frames.append(df)
            print(f"  로드: {fname}  ({df.shape[0]:,} samples × {df.shape[1]:,} genes)")

        expr_df, cnv_df, mut_df = frames

        # ── 2. 공통 샘플 확정 ────────────────────────────────────────
        common = sorted(
            set(expr_df.index) & set(cnv_df.index) & set(mut_df.index)
        )
        if samples:
            sample_set = set(samples)
            common = [s for s in common if s in sample_set]
        self.samples: list[str] = common
        print(f"  공통 세포주: {len(self.samples):,}개")

        # ── 3. 유전자 집합 확정 ──────────────────────────────────────
        all_genes = sorted(
            set(expr_df.columns) | set(cnv_df.columns) | set(mut_df.columns)
        )
        if genes:
            gene_set = set(genes)
            self.genes: list[str] = [g for g in all_genes if g in gene_set]
        else:
            self.genes = all_genes
        print(f"  사용 유전자: {len(self.genes):,}개")

        # ── 4. gene_ids 미리 계산 (고정 순서) ───────────────────────
        self.gene_ids = torch.tensor(
            [vocab[g] for g in self.genes], dtype=torch.long
        )

        # ── 5. 각 모달리티를 (samples × genes) NumPy 배열로 캐싱 ────
        #    없는 유전자 열 → NaN 으로 채움
        self._expr = (
            expr_df
            .reindex(index=self.samples, columns=self.genes)
            .to_numpy(dtype=np.float32)
        )
        self._cnv = (
            cnv_df
            .reindex(index=self.samples, columns=self.genes)
            .to_numpy(dtype=np.float32)
        )
        self._mut = (
            mut_df
            .reindex(index=self.samples, columns=self.genes)
            .to_numpy(dtype=np.float32)
        )

    # ── Dataset 인터페이스 ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        e = torch.from_numpy(self._expr[idx])  # (N,)
        c = torch.from_numpy(self._cnv[idx])
        m = torch.from_numpy(self._mut[idx])

        # mask: True = 해당 모달리티에서 측정된 유전자
        mask_e = ~torch.isnan(e)
        mask_c = ~torch.isnan(c)
        mask_m = ~torch.isnan(m)

        # NaN → 0.0 (mask로 구분되므로 정보 손실 없음)
        omics = torch.stack(
            [e.nan_to_num(0.0), c.nan_to_num(0.0), m.nan_to_num(0.0)],
            dim=1,
        )  # (N, 3)
        omics_mask = torch.stack([mask_e, mask_c, mask_m], dim=1)  # (N, 3)

        return {
            "gene_ids":   self.gene_ids,  # (N,)  LongTensor
            "omics":      omics,          # (N, 3) FloatTensor
            "omics_mask": omics_mask,     # (N, 3) BoolTensor
            "model_id":   self.samples[idx],
        }

    # ── 편의 메서드 ──────────────────────────────────────────────────────

    def modality_coverage(self) -> dict[str, float]:
        """각 모달리티에서 측정된 유전자 비율 반환 (평균)."""
        return {
            "expr": float(~np.isnan(self._expr).all(axis=0)).mean()
                    if self._expr.shape[0] else 0.0,
            "cnv":  float(np.isnan(self._cnv).mean()),
            "mut":  float(np.isnan(self._mut).mean()),
        }


# ── collate_fn ───────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    가변 N 배치를 배치 내 max_N으로 동적 패딩.

    padding_mask: True = 패딩 위치 (Transformer key_padding_mask 규격)
    """
    max_n = max(b["gene_ids"].size(0) for b in batch)

    gene_ids_list    = []
    omics_list       = []
    omics_mask_list  = []
    padding_mask_list = []

    for b in batch:
        n   = b["gene_ids"].size(0)
        pad = max_n - n

        gene_ids_list.append(
            torch.nn.functional.pad(b["gene_ids"], (0, pad), value=0)
        )
        omics_list.append(
            torch.nn.functional.pad(b["omics"], (0, 0, 0, pad), value=0.0)
        )
        omics_mask_list.append(
            torch.nn.functional.pad(b["omics_mask"], (0, 0, 0, pad), value=False)
        )
        pm = torch.zeros(max_n, dtype=torch.bool)
        pm[n:] = True  # 패딩 위치 마킹
        padding_mask_list.append(pm)

    return {
        "gene_ids":     torch.stack(gene_ids_list),     # (B, max_N)
        "omics":        torch.stack(omics_list),        # (B, max_N, 3)
        "omics_mask":   torch.stack(omics_mask_list),   # (B, max_N, 3)
        "padding_mask": torch.stack(padding_mask_list), # (B, max_N) True=pad
        "model_ids":    [b["model_id"] for b in batch],
    }
