# DepMap 26Q1 데이터 구조 및 ML 전처리 가이드

> **작성일**: 2026-05-15  
> **대상 파일**: Copy Number / Mutation / Expression + 메타데이터  
> **목적**: 모델 학습에 바로 사용 가능한 형태로 데이터 정리

---

## 1. 파일별 데이터 구조 요약

### 1-1. 메타데이터

#### `Model.csv` — 세포주 정보
| 항목 | 내용 |
|------|------|
| 행 수 | 2,154 (세포주) |
| 인덱스 | `ModelID` (예: `ACH-000839`) |
| 역할 | 모든 Omics 파일의 기준 ID |

**ML에 유용한 주요 컬럼**

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `ModelID` | str | 고유 세포주 ID (Join 키) |
| `CellLineName` | str | 세포주 명칭 |
| `OncotreeLineage` | str (categorical) | 암종 계통 (예: Lung, Breast) |
| `OncotreePrimaryDisease` | str (categorical) | 주요 질환명 |
| `OncotreeSubtype` | str (categorical) | 세부 암종 서브타입 |
| `OncotreeCode` | str | OncoTree 코드 (예: LUAD, BRCA) |
| `Sex` | str (categorical) | Female / Male / Unknown |
| `Age` | float | 샘플 채취 시 나이 |
| `AgeCategory` | str (categorical) | Adult / Pediatric / Fetus / Unknown |
| `PrimaryOrMetastasis` | str (categorical) | Primary / Metastatic / Recurrence |
| `SampleCollectionSite` | str (categorical) | 샘플 채취 부위 |
| `ModelType` | str (categorical) | Cell Line / Organoid |
| `GrowthPattern` | str (categorical) | Adherent / Suspension / Spheroid |
| `TissueOrigin` | str (categorical) | Human / Mouse |
| `EngineeredModel` | bool | 유전적으로 조작된 모델 여부 |
| `PatientTreatmentDetails` | str (categorical) | 환자 치료 이력 (약물 조합 등, 2,154개 중 13개만 비어있지 않음) |

**제외 권장 컬럼** (ML 무관 또는 중복)
- `PatientID`, `RRID`, `CatalogNumber`, `FormulationID`, `OnboardedMedia`
- `WTSIMasterCellID`, `SangerModelID`, `COSMICID`, `ModelIDAlias`
- `PublicComments`, `EngineeredModelDetails`

---

#### `ModelCondition.csv` — 실험 조건
| 항목 | 내용 |
|------|------|
| 행 수 | ~2,000+ (ModelID별 1개 이상 가능) |
| 인덱스 | `ModelConditionID` |
| Join 키 | `ModelID` → Model.csv 연결 |

**주요 컬럼**

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `ModelConditionID` | str | 실험 조건 고유 ID |
| `ModelID` | str | 세포주 ID (Join 키) |
| `CellFormat` | str (categorical) | Adherent / Suspension / Dome / Spheroid |
| `GrowthMedia` | str | 배지 조성 |
| `PlateCoating` | str (categorical) | Laminin / Matrigel / None |
| `SerumFreeMedia` | bool | 무혈청 배지 여부 |
| `AnchorDrug` | str | Anchor 스크린 약물명 (없으면 NaN) |

---

### 1-2. Omics 데이터

#### `OmicsCNGeneWGS.csv` — Copy Number (WGS)
| 항목 | 내용 |
|------|------|
| 행 수 | 1,132 (프로파일) |
| 열 수 | 19,961 (메타 5열 + 유전자 19,956개) |
| 값 범위 | 양수 실수 (linear scale copy ratio, 정상=1.0) |
| 파이프라인 | HMMcopy + PureCN, hg38 기준 |

**컬럼 구조**
```
unnamed_index | SequencingID | ModelConditionID | ModelID | IsDefaultEntryForMC | IsDefaultEntryForModel | TSPAN6 (7105) | TNMD (64102) | ...
```

- 유전자 컬럼 형식: `"HUGO Symbol (Entrez ID)"` (예: `TSPAN6 (7105)`)
- 값 예시: `0.72`, `1.47`, `2.83` (정상=1.0, 결실<1, 증폭>1)
- 행당 하나의 시퀀싱 프로파일. `IsDefaultEntryForModel=Yes`인 행이 세포주 대표값

---

#### `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` — RNA 발현 (log TPM)
| 항목 | 내용 |
|------|------|
| 행 수 | 1,775 (프로파일) |
| 열 수 | 19,221 (메타 5열 + 유전자 19,216개) |
| 값 범위 | 0 이상 실수 (log₂(TPM+1)) |
| 파이프라인 | Salmon v1.10.0 (unstranded), Gencode v38 |

**컬럼 구조**
```
unnamed_index | SequencingID | ModelConditionID | ModelID | IsDefaultEntryForMC | IsDefaultEntryForModel | TSPAN6 (7105) | ...
```

- 유전자 컬럼 형식: CNV와 동일 (`"HUGO Symbol (Entrez ID)"`)
- 값 예시: `0.71`, `1.47`, `5.23` (0에 가까울수록 저발현)
- 단백질 코딩 유전자만 포함 (~19,216개)

---

#### `OmicsSomaticMutations.csv` — 체세포 변이 (MAF 형식)
| 항목 | 내용 |
|------|------|
| 행 수 | 1,172,688 (변이 레코드) |
| 열 수 | 67 |
| 형태 | Long format (변이 1개 = 1행) |
| 파이프라인 | Mutect2, hg38 기준 |

**컬럼 구조**
```
unnamed_index | SequencingID | ModelID | ModelConditionID | IsDefaultEntryForModel | IsDefaultEntryForMC |
Chrom | Pos | Ref | Alt | AF | DP | RefCount | AltCount | GT | PS |
VariantType | VariantInfo | DNAChange | ProteinChange | HugoSymbol | Exon | Intron |
EnsemblGeneID | EnsemblFeatureID | HgncName | HgncFamily | UniprotID | DbsnpRsID |
GcContent | NMD | MolecularConsequence | VepImpact | VepBiotype | VepHgncID |
VepExistingVariation | VepManeSelect | VepENSP | VepSwissprot |
Sift | Polyphen | GnomadeAF | GnomadgAF | VepClinSig | VepSomatic |
VepPliGeneValue | VepLofTool | OncogeneHighImpact | TumorSuppressorHighImpact |
TranscriptLikelyLof | Brca1FuncScore | CivicID | CivicDescription | CivicScore |
LikelyLoF | HessDriver | HessSignature | RevelScore | PharmgkbId |
GwasDisease | GwasPmID | GtexGene | ProveanPrediction | AMClass | AMPathogenicity |
Rescue | RescueReason | Hotspot | EntrezGeneID
```

**주요 컬럼 설명**

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `ModelID` | str | 세포주 ID (Join 키) |
| `HugoSymbol` | str | 유전자 심볼 |
| `AF` | float [0,1] | Allele Frequency (변이 비율) |
| `VariantType` | str | SNV / insertion / deletion |
| `VepImpact` | str | HIGH / MODERATE / LOW / MODIFIER |
| `MolecularConsequence` | str | missense_variant, stop_gained 등 |
| `LikelyLoF` | bool | Loss-of-Function 예측 여부 |
| `Hotspot` | bool | 알려진 암 hotspot 변이 여부 |
| `OncogeneHighImpact` | bool | 암유전자 고영향 변이 |
| `TumorSuppressorHighImpact` | bool | 종양억제인자 고영향 변이 |
| `Sift` | str | SIFT 기능 예측 (tolerated/deleterious) |
| `Polyphen` | str | PolyPhen 기능 예측 |
| `AMPathogenicity` | float | AlphaMissense 병원성 점수 |
| `RevelScore` | float | REVEL 병원성 점수 |
| `GnomadeAF` | float | gnomAD exome population AF |
| `HessDriver` | bool | Hess 2019 driver mutation 여부 |

---

## 2. ID 체계 및 연결 구조

```
Model.csv
    ModelID (ACH-xxxxxx)
        │
        ├── ModelCondition.csv
        │       ModelConditionID → ModelID
        │
        └── Omics 파일들
                OmicsCNGeneWGS.csv        → ModelID (+ IsDefaultEntryForModel)
                OmicsExpressionTPM...csv  → ModelID (+ IsDefaultEntryForModel)
                OmicsSomaticMutations.csv → ModelID (+ IsDefaultEntryForModel)
```

**핵심**: 세포주 하나(`ModelID`)에 여러 시퀀싱 프로파일이 존재할 수 있음  
→ `IsDefaultEntryForModel == Yes`인 행을 선택하여 1 세포주 = 1 행으로 통일

---

## 3. 전처리 파이프라인

### Step 0. 공통 세포주 집합 확정

```python
import pandas as pd

model_meta = pd.read_csv("Model.csv", index_col="ModelID")
cnv         = pd.read_csv("OmicsCNGeneWGS.csv", index_col=0)
expr        = pd.read_csv("OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", index_col=0)
mut         = pd.read_csv("OmicsSomaticMutations.csv", index_col=0)

# 각 파일에서 default entry만 선택 → ModelID를 인덱스로
cnv_default  = cnv[cnv["IsDefaultEntryForModel"] == "Yes"].set_index("ModelID")
expr_default = expr[expr["IsDefaultEntryForModel"] == "Yes"].set_index("ModelID")

# 세 Omics 데이터에 모두 존재하는 공통 세포주
mut_models = mut[mut["IsDefaultEntryForModel"] == "Yes"]["ModelID"].unique()
common_models = (
    set(cnv_default.index)
    & set(expr_default.index)
    & set(mut_models)
    & set(model_meta.index)
)
print(f"공통 세포주 수: {len(common_models)}")
```

---

### Step 1. 메타데이터 전처리 (Model.csv)

```python
# ML에 사용할 컬럼 선택
meta_cols = [
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype", "OncotreeCode",
    "Sex", "Age", "AgeCategory", "PrimaryOrMetastasis",
    "SampleCollectionSite", "ModelType", "GrowthPattern"
]
meta = model_meta.loc[list(common_models), meta_cols].copy()

# 결측치 처리
meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
meta["Sex"] = meta["Sex"].replace("Unknown", pd.NA)

# 범주형 인코딩 (One-hot 또는 Label)
meta_encoded = pd.get_dummies(meta, columns=[
    "OncotreeLineage", "Sex", "AgeCategory",
    "PrimaryOrMetastasis", "ModelType", "GrowthPattern"
], dummy_na=False)
```

**주의사항**
- `OncotreePrimaryDisease`, `OncotreeSubtype`은 카테고리 수가 많아 Label Encoding 또는 타겟 기반 그룹핑 권장
- `Age`는 약 30~40% NaN 존재 → median imputation 또는 별도 결측 플래그 컬럼 추가

---

### Step 2. Copy Number 전처리 (OmicsCNGeneWGS.csv)

```python
# 메타 컬럼 제거, 유전자 행렬만 추출
meta_cols_cnv = ["SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
cnv_mat = cnv_default.drop(columns=meta_cols_cnv).loc[list(common_models)]

# 유전자 컬럼명 정리: "TSPAN6 (7105)" → "TSPAN6"
cnv_mat.columns = [col.split(" (")[0] for col in cnv_mat.columns]
```

**변환 옵션**

| 방법 | 코드 | 설명 |
|------|------|------|
| Log2 변환 | `np.log2(cnv_mat + 1e-3)` | 분포 정규화, 0값 보정 필요 |
| 이진화 (결실/정상/증폭) | `cnv_mat.apply(discretize_cn)` | 값 범주화 (아래 참조) |
| 그대로 사용 | — | 일부 선형 모델에 적합 |

```python
def discretize_cn(series):
    # 0: 결실(deep del), 1: 손실, 2: 정상, 3: 저증폭, 4: 고증폭
    return pd.cut(series,
                  bins=[-np.inf, 0.3, 0.7, 1.3, 2.5, np.inf],
                  labels=[0, 1, 2, 3, 4]).astype(float)
```

**결측치 처리**
```python
# NaN 비율 확인
nan_ratio = cnv_mat.isna().mean()
# 결측 비율 > 20% 유전자 제거
cnv_mat = cnv_mat.loc[:, nan_ratio < 0.2]
# 나머지 결측: 열(유전자) median으로 imputation
cnv_mat = cnv_mat.fillna(cnv_mat.median())
```

**분산 필터링 (선택)**
```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
cnv_filtered = pd.DataFrame(
    selector.fit_transform(cnv_mat),
    index=cnv_mat.index,
    columns=cnv_mat.columns[selector.get_support()]
)
```

---

### Step 3. Expression 전처리 (OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv)

```python
meta_cols_expr = ["SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
expr_mat = expr_default.drop(columns=meta_cols_expr).loc[list(common_models)]

# 유전자 컬럼명 정리
expr_mat.columns = [col.split(" (")[0] for col in expr_mat.columns]
```

**값 특성**
- 이미 log₂(TPM+1) 변환된 상태 → 추가 log 변환 불필요
- 0에 매우 가까운 값 = 발현 없음 (NaN이 아닌 실제 0)

**저발현 유전자 필터링**
```python
# 전체 세포주의 20% 이상에서 발현(>0.5)되지 않는 유전자 제거
expressed = (expr_mat > 0.5).mean(axis=0)
expr_mat = expr_mat.loc[:, expressed >= 0.2]
print(f"필터링 후 유전자 수: {expr_mat.shape[1]}")
```

**분산 필터링**
```python
# 분산 하위 10% 유전자 제거
var_threshold = expr_mat.var().quantile(0.10)
expr_mat = expr_mat.loc[:, expr_mat.var() > var_threshold]
```

**정규화 옵션**

| 방법 | 코드 | 권장 상황 |
|------|------|-----------|
| Z-score (유전자별) | `(expr_mat - expr_mat.mean()) / expr_mat.std()` | 유전자 간 스케일 통일 |
| Quantile normalization | `sklearn`의 `QuantileTransformer` | 분포 차이 제거 |
| 그대로 (log TPM) | — | 트리 기반 모델 |

---

### Step 4. Mutation 전처리 (OmicsSomaticMutations.csv)

MAF 형식(long format)을 ML용 행렬로 변환해야 합니다.

#### 4-1. 기본 필터링

```python
# default entry만 선택
mut_filt = mut[mut["IsDefaultEntryForModel"] == "Yes"].copy()
mut_filt = mut_filt[mut_filt["ModelID"].isin(common_models)]

# 품질 필터
mut_filt = mut_filt[
    (mut_filt["AF"] >= 0.1) &          # 최소 allele frequency
    (mut_filt["DP"] >= 10)             # 최소 read depth
]

# 단백질에 영향 있는 변이만
mut_filt = mut_filt[mut_filt["VepImpact"].isin(["HIGH", "MODERATE"])]
```

#### 4-2. 행렬 변환 방법 (3가지 선택지)

**옵션 A — 이진 행렬 (유전자별 변이 유무)**
```python
# 세포주 × 유전자 이진 행렬 (0/1)
mut_binary = (
    mut_filt.groupby(["ModelID", "HugoSymbol"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)  # 여러 변이 → 1로 통일
    .reindex(index=list(common_models), fill_value=0)
)
```

**옵션 B — AF 기반 연속 행렬 (최대 AF)**
```python
# 같은 유전자에 여러 변이 있을 경우 최대 AF 사용
mut_af = (
    mut_filt.groupby(["ModelID", "HugoSymbol"])["AF"]
    .max()
    .unstack(fill_value=0.0)
    .reindex(index=list(common_models), fill_value=0.0)
)
```

**옵션 C — 변이 유형별 분리 행렬 (권장)**
```python
# LoF, Hotspot, Missense 별도 행렬 생성
def make_binary_mat(df, flag_col):
    sub = df[df[flag_col] == True]
    return (
        sub.groupby(["ModelID", "HugoSymbol"])
        .size().unstack(fill_value=0).clip(upper=1)
        .reindex(index=list(common_models), fill_value=0)
    )

mut_lof     = make_binary_mat(mut_filt, "LikelyLoF")
mut_hotspot = make_binary_mat(mut_filt, "Hotspot")
```

#### 4-3. 희귀 변이 유전자 제거

```python
# 전체 세포주의 1% 미만에서만 발견되는 유전자 제거
min_freq = 0.01 * len(common_models)
mut_binary = mut_binary.loc[:, mut_binary.sum() >= min_freq]
print(f"필터링 후 유전자 수: {mut_binary.shape[1]}")
```

---

### Step 5. 최종 통합

```python
# 공통 인덱스(ModelID) 기준으로 합치기
final_df = pd.concat([
    meta_encoded,    # 메타데이터 (one-hot)
    cnv_filtered,    # CNV (log2 또는 이진화)
    expr_mat,        # 발현 (log TPM)
    mut_binary,      # 변이 이진 행렬
], axis=1).loc[list(common_models)]

print(f"최종 행렬 크기: {final_df.shape}")
# 예: (1050, 40000+)
```

**컬럼명 충돌 방지 (유전자 이름이 겹칠 경우)**
```python
cnv_filtered.columns  = ["CNV_" + c for c in cnv_filtered.columns]
expr_mat.columns      = ["EXP_" + c for c in expr_mat.columns]
mut_binary.columns    = ["MUT_" + c for c in mut_binary.columns]
```

---

## 4. 최종 행렬 구조 요약

| 데이터 | 행 (샘플) | 열 (피처) | 값 타입 | 주요 처리 |
|--------|-----------|-----------|---------|-----------|
| 메타데이터 | 공통 세포주 수 | ~20–50 | int/float | One-hot encoding |
| CNV | 공통 세포주 수 | ~15,000–19,000 | float | Log2 변환, median imputation |
| Expression | 공통 세포주 수 | ~10,000–19,000 | float | 저발현/저분산 필터 |
| Mutation | 공통 세포주 수 | ~5,000–15,000 | int (0/1) | 희귀 유전자 필터 |
| **최종 통합** | **~1,000–1,100** | **~30,000–50,000** | mixed | 모달리티별 prefix |

---

## 5. 추가 권장 처리

### 5-1. 차원 축소 (선택)
```python
from sklearn.decomposition import PCA

# 발현 데이터 PCA (Top 500 PC)
pca = PCA(n_components=500, random_state=42)
expr_pca = pd.DataFrame(
    pca.fit_transform(expr_mat),
    index=expr_mat.index,
    columns=[f"EXP_PC{i+1}" for i in range(500)]
)
print(f"설명 분산 비율: {pca.explained_variance_ratio_.sum():.3f}")
```

### 5-2. 암종별 층화 분할 (Train/Val/Test)
```python
from sklearn.model_selection import StratifiedShuffleSplit

labels = meta["OncotreeLineage"].loc[final_df.index]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(final_df, labels))
```

### 5-3. 클래스 불균형 확인
```python
# 암종 분포 확인
print(meta["OncotreeLineage"].value_counts())
# 희귀 암종(n<10)은 'Other'로 묶거나 제외 권장
```

---

## 6. 전처리 체크리스트

- [x] `IsDefaultEntryForModel == Yes` 필터링 완료
- [x] 3개 Omics 공통 세포주 확정
- [x] CNV: NaN 처리 및 분산 필터링
- [x] Expression: 저발현 유전자 제거 및 정규화
- [x] Mutation: AF/DP 품질 필터 → 이진 행렬 변환 → 희귀 유전자 제거
- [x] 컬럼명 prefix 부여 (`CNV_`, `EXP_`, `MUT_`)
- [x] 최종 NaN 없음 확인 (`final_df.isna().sum().sum() == 0`)
- [ ] Train/Val/Test 암종 층화 분할

---

## 7. 전처리 실행 결과 (2026-05-15)

### 7-1. 전처리 요약

| 항목 | 값 |
|------|----|
| 공통 세포주 수 | **1,105** |
| 메타데이터 피처 수 | 532 (PatientTreatmentDetails +4 추가) |
| CNV 유전자 수 | 18,853 |
| Expression 유전자 수 | 12,928 |
| Mutation binary 유전자 수 | 12,175 |
| Mutation LoF 유전자 수 | 1,055 |
| Mutation Hotspot 유전자 수 | 21 |
| **최종 통합 피처 수** | **45,564** |
| 최종 NaN 수 | 0 |

**입력 → 공통 세포주 확정 과정**

| 데이터 | 원본 행 수 | Default entry | 공통 세포주 |
|--------|-----------|--------------|------------|
| CNV | 1,132 프로파일 | 1,118 세포주 | |
| Expression | 1,775 프로파일 | 1,719 세포주 | → **1,105** |
| Mutation | 1,172,688 변이 | 1,968 세포주 | |
| Model 메타 | 2,154 세포주 | — | |

> Expression 파일의 세포주 수가 가장 적어 병목이 됨

---

### 7-2. 출력 파일 목록

출력 경로: `DepMap_26Q1_preprocessed/`

| 파일명 | 크기 | 행 × 열 | 설명 |
|--------|------|---------|------|
| `metadata_raw.csv` | 154K | 1,105 × 14 | 세포주 메타데이터 원본 (PatientTreatmentDetails 포함) |
| `metadata_encoded.csv` | 3.4M | 1,105 × 532 | One-hot 인코딩 완료 (PatientTreatmentDetails 포함) |
| `cnv_log2.csv` | 401M | 1,105 × 18,853 | CNV log2 변환, prefix: `CNV_` |
| `expression_logTPM.csv` | 132M | 1,105 × 12,928 | 발현 log(TPM+1) 원본, prefix: `EXP_` |
| `expression_zscore.csv` | 268M | 1,105 × 12,928 | 발현 Z-score 정규화, prefix: `EXP_` |
| `mutation_binary.csv` | 26M | 1,105 × 12,175 | 변이 이진 행렬 (HIGH+MODERATE), prefix: `MUT_` |
| `mutation_lof.csv` | 2.3M | 1,105 × 1,055 | LoF 변이 이진 행렬, prefix: `MUT_LoF_` |
| `mutation_hotspot.csv` | 58K | 1,105 × 21 | Hotspot 변이 이진 행렬, prefix: `MUT_HOT_` |
| `final_integrated.csv` | 700M | 1,105 × 45,564 | 전체 통합 행렬 |
| `preprocessing_summary.csv` | 1K | — | 전처리 결과 요약 |

---

### 7-3. 각 전처리 단계별 피처 변화

```
[CNV]
  원본:    1,132 profiles × 19,956 genes
  default: 1,118 cells
  NaN>20% 제거: 19,956 → (유지)
  분산 필터 (threshold=0.01): → 18,853 genes  (-5.5%)

[Expression]
  원본:    1,775 profiles × 19,216 genes
  default: 1,719 cells
  저발현 필터 (발현율<20%): 19,216 → ~13,500
  분산 하위 10% 제거: → 12,928 genes  (-32.7% 총)

[Mutation]
  원본:    1,172,688 변이 레코드
  품질 필터 (AF≥0.1, DP≥10, VepImpact∈{HIGH,MODERATE}): 대폭 감소
  이진 행렬 변환 후 희귀 유전자 제거 (빈도<1%): → 12,175 genes

[Metadata]
  원본:    14 컬럼 (PatientTreatmentDetails 포함)
  One-hot 인코딩 (12 범주형 컬럼): → 532 피처
```

---

## 9. 참고

| 파일 | 파이프라인 | 기준 유전체 |
|------|-----------|------------|
| OmicsCNGeneWGS.csv | HMMcopy + PureCN | hg38 |
| OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv | Salmon v1.10.0 | Gencode v38 |
| OmicsSomaticMutations.csv | Mutect2 + VEP | hg38 |

- DepMap Portal: https://depmap.org/portal/
- 변이 파이프라인 문서: https://storage.googleapis.com/shared-portal-files/Tools/26Q1_Mutation_Pipeline_Documentation.pdf
