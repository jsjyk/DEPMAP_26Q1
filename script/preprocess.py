"""
DepMap 26Q1 전처리 스크립트
입력: /mnt/c/Users/Admin/Desktop/데이터/DepMap_26Q1/
출력: /tmp/DEPMAP_26Q1/DepMap_26Q1_preprocessed/
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import os

DATA_DIR = "/mnt/c/Users/Admin/Desktop/데이터/DepMap_26Q1"
OUT_DIR  = "/tmp/DEPMAP_26Q1/DepMap_26Q1_preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Step 0. 파일 로드
# ──────────────────────────────────────────────
print("[1/6] 파일 로딩 중...")

model_meta = pd.read_csv(f"{DATA_DIR}/Model.csv", index_col="ModelID")

cnv = pd.read_csv(f"{DATA_DIR}/OmicsCNGeneWGS.csv", index_col=0)
print(f"  CNV raw: {cnv.shape}")

expr = pd.read_csv(f"{DATA_DIR}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", index_col=0)
print(f"  Expression raw: {expr.shape}")

# Mutation은 크기가 크므로 필요한 컬럼만 로드
mut_cols = [
    "SequencingID", "ModelID", "ModelConditionID",
    "IsDefaultEntryForModel", "IsDefaultEntryForMC",
    "HugoSymbol", "AF", "DP",
    "VepImpact", "VariantType",
    "LikelyLoF", "Hotspot",
    "OncogeneHighImpact", "TumorSuppressorHighImpact"
]
mut = pd.read_csv(f"{DATA_DIR}/OmicsSomaticMutations.csv", index_col=0, usecols=lambda c: c in mut_cols + [""])
# unnamed index 포함 처리
if "ModelID" not in mut.columns:
    mut = pd.read_csv(f"{DATA_DIR}/OmicsSomaticMutations.csv", index_col=0)
    mut = mut[["ModelID", "ModelConditionID", "IsDefaultEntryForModel", "IsDefaultEntryForMC",
               "HugoSymbol", "AF", "DP", "VepImpact", "VariantType",
               "LikelyLoF", "Hotspot", "OncogeneHighImpact", "TumorSuppressorHighImpact"]]
print(f"  Mutation raw: {mut.shape}")

# ──────────────────────────────────────────────
# Step 1. Default entry 필터링 → ModelID 인덱스
# ──────────────────────────────────────────────
print("[2/6] Default entry 필터링 중...")

cnv_default  = cnv[cnv["IsDefaultEntryForModel"] == "Yes"].set_index("ModelID")
expr_default = expr[expr["IsDefaultEntryForModel"] == "Yes"].set_index("ModelID")
mut_default  = mut[mut["IsDefaultEntryForModel"] == "Yes"]

print(f"  CNV default: {cnv_default.shape[0]} 세포주")
print(f"  Expression default: {expr_default.shape[0]} 세포주")
print(f"  Mutation default: {mut_default['ModelID'].nunique()} 세포주")

# 공통 세포주 확정
common_models = (
    set(cnv_default.index)
    & set(expr_default.index)
    & set(mut_default["ModelID"].unique())
    & set(model_meta.index)
)
common_models = sorted(common_models)
print(f"  공통 세포주 수: {len(common_models)}")

# ──────────────────────────────────────────────
# Step 2. 메타데이터 전처리
# ──────────────────────────────────────────────
print("[3/6] 메타데이터 전처리 중...")

meta_cols = [
    "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease",
    "OncotreeSubtype", "OncotreeCode",
    "Sex", "Age", "AgeCategory", "PrimaryOrMetastasis",
    "SampleCollectionSite", "ModelType", "GrowthPattern", "TissueOrigin",
    "PatientTreatmentDetails"
]
meta_raw = model_meta.loc[common_models, meta_cols].copy()

# Age: 수치형 변환
meta_raw["Age"] = pd.to_numeric(meta_raw["Age"], errors="coerce")

# 저장: raw 메타
meta_raw.to_csv(f"{OUT_DIR}/metadata_raw.csv")

# 범주형 컬럼 NaN → 'Unknown' 처리 후 One-hot 인코딩 (학습용)
onehot_cols = [
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
    "OncotreeCode", "Sex", "AgeCategory", "PrimaryOrMetastasis",
    "SampleCollectionSite", "ModelType", "GrowthPattern", "TissueOrigin",
    "PatientTreatmentDetails"
]
meta_to_encode = meta_raw.drop(columns=["CellLineName"]).copy()
for col in onehot_cols:
    meta_to_encode[col] = meta_to_encode[col].fillna("Unknown")

meta_encoded = pd.get_dummies(meta_to_encode, columns=onehot_cols, dummy_na=False)
meta_encoded["Age"] = meta_encoded["Age"].fillna(meta_encoded["Age"].median())

meta_encoded.to_csv(f"{OUT_DIR}/metadata_encoded.csv")
print(f"  메타데이터: {meta_encoded.shape} (세포주 × 피처)")

# ──────────────────────────────────────────────
# Step 3. Copy Number 전처리
# ──────────────────────────────────────────────
print("[4/6] Copy Number 전처리 중...")

drop_cols_cnv = [c for c in ["SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
                 if c in cnv_default.columns]
cnv_mat = cnv_default.drop(columns=drop_cols_cnv).loc[common_models].copy()

# 유전자 컬럼명 정리: "TSPAN6 (7105)" → "TSPAN6"
cnv_mat.columns = [col.split(" (")[0] for col in cnv_mat.columns]

# NaN 비율 > 20% 유전자 제거
nan_ratio = cnv_mat.isna().mean()
cnv_mat = cnv_mat.loc[:, nan_ratio < 0.2]

# 나머지 NaN: 열 median imputation
cnv_mat = cnv_mat.fillna(cnv_mat.median())

# 분산 필터링 (VarianceThreshold)
selector = VarianceThreshold(threshold=0.01)
cnv_filtered = pd.DataFrame(
    selector.fit_transform(cnv_mat),
    index=cnv_mat.index,
    columns=cnv_mat.columns[selector.get_support()]
)

# Log2 변환 (linear scale → log2 scale, 0값 보정)
cnv_log2 = np.log2(cnv_filtered + 1e-3)
cnv_log2.columns = ["CNV_" + c for c in cnv_log2.columns]

cnv_log2.to_csv(f"{OUT_DIR}/cnv_log2.csv")
print(f"  CNV (log2): {cnv_log2.shape} (세포주 × 유전자)")

# ──────────────────────────────────────────────
# Step 4. Expression 전처리
# ──────────────────────────────────────────────
print("[5/6] Expression 전처리 중...")

drop_cols_expr = [c for c in ["SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
                  if c in expr_default.columns]
expr_mat = expr_default.drop(columns=drop_cols_expr).loc[common_models].copy()

# 유전자 컬럼명 정리
expr_mat.columns = [col.split(" (")[0] for col in expr_mat.columns]

# 저발현 유전자 제거: 20% 이상 세포주에서 발현(>0.5)
expressed = (expr_mat > 0.5).mean(axis=0)
expr_mat = expr_mat.loc[:, expressed >= 0.2]

# 분산 하위 10% 제거
var_threshold = expr_mat.var().quantile(0.10)
expr_mat = expr_mat.loc[:, expr_mat.var() > var_threshold]

# Z-score 정규화 (유전자별)
expr_zscore = (expr_mat - expr_mat.mean()) / expr_mat.std()
expr_zscore = expr_zscore.fillna(0)  # 분산 0인 유전자 대비
expr_zscore.columns = ["EXP_" + c for c in expr_zscore.columns]

expr_zscore.to_csv(f"{OUT_DIR}/expression_zscore.csv")

# log TPM 원본도 저장 (정규화 전)
expr_mat_out = expr_mat.copy()
expr_mat_out.columns = ["EXP_" + c for c in expr_mat_out.columns]
expr_mat_out.to_csv(f"{OUT_DIR}/expression_logTPM.csv")
print(f"  Expression (z-score): {expr_zscore.shape} (세포주 × 유전자)")

# ──────────────────────────────────────────────
# Step 5. Mutation 전처리 → 이진 행렬
# ──────────────────────────────────────────────
print("[6/6] Mutation 전처리 중...")

mut_filt = mut_default[mut_default["ModelID"].isin(common_models)].copy()

# 품질 필터
mut_filt["AF"] = pd.to_numeric(mut_filt["AF"], errors="coerce")
mut_filt["DP"] = pd.to_numeric(mut_filt["DP"], errors="coerce")
mut_filt = mut_filt[
    (mut_filt["AF"] >= 0.1) &
    (mut_filt["DP"] >= 10) &
    (mut_filt["VepImpact"].isin(["HIGH", "MODERATE"]))
]

# 전체 이진 행렬 (HIGH+MODERATE)
mut_binary = (
    mut_filt.groupby(["ModelID", "HugoSymbol"])
    .size().unstack(fill_value=0).clip(upper=1)
    .reindex(index=common_models, fill_value=0)
)

# LikelyLoF 행렬
mut_lof = (
    mut_filt[mut_filt["LikelyLoF"] == True]
    .groupby(["ModelID", "HugoSymbol"])
    .size().unstack(fill_value=0).clip(upper=1)
    .reindex(index=common_models, fill_value=0)
)

# Hotspot 행렬
mut_hotspot = (
    mut_filt[mut_filt["Hotspot"] == True]
    .groupby(["ModelID", "HugoSymbol"])
    .size().unstack(fill_value=0).clip(upper=1)
    .reindex(index=common_models, fill_value=0)
)

# 희귀 유전자 제거: 최소 1% 세포주에서 발견
min_freq = max(1, int(0.01 * len(common_models)))
mut_binary  = mut_binary.loc[:,  mut_binary.sum()  >= min_freq]
mut_lof     = mut_lof.loc[:,     mut_lof.sum()     >= min_freq]
mut_hotspot = mut_hotspot.loc[:, mut_hotspot.sum() >= min_freq]

# prefix 부여
mut_binary.columns  = ["MUT_"     + c for c in mut_binary.columns]
mut_lof.columns     = ["MUT_LoF_" + c for c in mut_lof.columns]
mut_hotspot.columns = ["MUT_HOT_" + c for c in mut_hotspot.columns]

mut_binary.to_csv(f"{OUT_DIR}/mutation_binary.csv")
mut_lof.to_csv(f"{OUT_DIR}/mutation_lof.csv")
mut_hotspot.to_csv(f"{OUT_DIR}/mutation_hotspot.csv")
print(f"  Mutation binary:  {mut_binary.shape}")
print(f"  Mutation LoF:     {mut_lof.shape}")
print(f"  Mutation Hotspot: {mut_hotspot.shape}")

# ──────────────────────────────────────────────
# Step 6. 최종 통합 행렬
# ──────────────────────────────────────────────
print("\n[최종] 통합 행렬 생성 중...")

final_df = pd.concat([
    meta_encoded,
    cnv_log2,
    expr_zscore,
    mut_binary,
    mut_lof,
    mut_hotspot,
], axis=1).loc[common_models]

assert final_df.isna().sum().sum() == 0, "NaN 존재! 확인 필요"
final_df.to_csv(f"{OUT_DIR}/final_integrated.csv")
print(f"  최종 통합 행렬: {final_df.shape} (세포주 × 전체 피처)")

# ──────────────────────────────────────────────
# 요약 저장
# ──────────────────────────────────────────────
summary = {
    "공통 세포주 수": len(common_models),
    "메타데이터 피처": meta_encoded.shape[1],
    "CNV 유전자 수": cnv_log2.shape[1],
    "Expression 유전자 수": expr_zscore.shape[1],
    "Mutation binary 유전자 수": mut_binary.shape[1],
    "Mutation LoF 유전자 수": mut_lof.shape[1],
    "Mutation Hotspot 유전자 수": mut_hotspot.shape[1],
    "최종 통합 피처 수": final_df.shape[1],
}
pd.Series(summary).to_csv(f"{OUT_DIR}/preprocessing_summary.csv", header=["value"])

print("\n====== 전처리 완료 ======")
for k, v in summary.items():
    print(f"  {k}: {v:,}")
print(f"\n출력 경로: {OUT_DIR}")
