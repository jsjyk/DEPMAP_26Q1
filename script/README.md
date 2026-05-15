# Script 사용 가이드

| 스크립트 | 역할 |
|----------|------|
| `preprocess.py` | 원본 DepMap 파일 → 전처리 CSV 생성 |
| `extract.py` | 전처리 CSV에서 원하는 피처·샘플 추출 |

---

## preprocess.py

원본 DepMap 26Q1 파일을 읽어 ML용 전처리 CSV를 `DepMap_26Q1_preprocessed/`에 저장합니다.

**입력 경로 (스크립트 내 상수)**
```
DATA_DIR = /mnt/c/Users/Admin/Desktop/데이터/DepMap_26Q1/
```

**필요 원본 파일**
- `Model.csv` — 세포주 메타데이터
- `OmicsCNGeneWGS.csv` — Copy Number (WGS)
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` — RNA 발현
- `OmicsSomaticMutations.csv` — 체세포 변이 (MAF)

**실행**
```bash
python preprocess.py
```

**출력 파일**

| 파일 | 크기 | 형태 | 설명 |
|------|------|------|------|
| `metadata_raw.csv` | ~154K | 1,105 × 14 | 메타데이터 원본 (문자열 포함) |
| `metadata_encoded.csv` | ~3.4M | 1,105 × 532 | One-hot 인코딩 완료 |
| `cnv_log2.csv` | ~401M | 1,105 × 18,853 | CNV log2 변환, prefix `CNV_` |
| `expression_logTPM.csv` | ~132M | 1,105 × 12,928 | 발현 log(TPM+1), prefix `EXP_` |
| `expression_zscore.csv` | ~268M | 1,105 × 12,928 | 발현 Z-score 정규화, prefix `EXP_` |
| `mutation_binary.csv` | ~26M | 1,105 × 12,175 | 변이 이진 행렬, prefix `MUT_` |
| `mutation_lof.csv` | ~2.3M | 1,105 × 1,055 | LoF 변이, prefix `MUT_LoF_` |
| `mutation_hotspot.csv` | ~58K | 1,105 × 21 | Hotspot 변이, prefix `MUT_HOT_` |
| `final_integrated.csv` | ~700M | 1,105 × 45,564 | 전체 통합 행렬 |
| `preprocessing_summary.csv` | ~1K | — | 전처리 결과 요약 |

**전처리 주요 단계**
1. `IsDefaultEntryForModel == Yes` 행만 선택 → 세포주당 1개 프로파일
2. CNV / Expression / Mutation 3개 파일에 모두 존재하는 공통 세포주 확정 (1,105개)
3. 메타데이터: `PatientTreatmentDetails` 포함 12개 범주형 컬럼 One-hot 인코딩
4. CNV: NaN 비율 >20% 유전자 제거 → median imputation → 분산 필터 → log2 변환
5. Expression: 저발현 유전자 제거 → 분산 하위 10% 제거 → Z-score 정규화
6. Mutation: AF ≥ 0.1, DP ≥ 10, VepImpact ∈ {HIGH, MODERATE} 필터 → 이진 행렬 → 희귀 유전자 제거

---

## extract.py

전처리된 개별 CSV 파일에서 원하는 피처 유형·유전자·샘플을 선택해 추출합니다.  
`final_integrated.csv` 전체를 로드하지 않으므로 메모리 효율적입니다.

**실행 위치**: 프로젝트 루트 또는 `script/` 폴더 어디서든 가능

```bash
python script/extract.py [옵션]
```

### 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--features` | 추출할 피처 종류 (복수 가능) | `--features cnv expr mut` |
| `--genes` | 유전자 심볼 지정 (생략 시 전체) | `--genes TP53 KRAS EGFR` |
| `--lineage` | 암종 계통 필터 | `--lineage Lung Breast` |
| `--disease` | PrimaryDisease 필터 | `--disease "Lung Adenocarcinoma"` |
| `--model-ids` | 특정 세포주 ID 지정 | `--model-ids ACH-000001` |
| `--output` | 출력 파일 경로 | `--output result.csv` |
| `--data-dir` | 전처리 파일 디렉터리 경로 변경 | `--data-dir /path/to/dir` |
| `--list-lineages` | 암종 계통 목록 + 세포주 수 출력 | |
| `--list-diseases` | PrimaryDisease 목록 출력 | |

### 피처 이름

| 이름 | 소스 파일 | 컬럼 prefix |
|------|----------|-------------|
| `cnv` | `cnv_log2.csv` | `CNV_` |
| `expr` | `expression_zscore.csv` | `EXP_` |
| `expr_tpm` | `expression_logTPM.csv` | `EXP_` |
| `mut` | `mutation_binary.csv` | `MUT_` |
| `mut_lof` | `mutation_lof.csv` | `MUT_LoF_` |
| `mut_hot` | `mutation_hotspot.csv` | `MUT_HOT_` |
| `meta` | `metadata_encoded.csv` | — |

### 사용 예시

```bash
# 사용 가능한 암종 계통 목록 확인
python script/extract.py --list-lineages

# Lung + Breast 세포주에서 CNV + 발현(z-score), 5개 유전자 추출
python script/extract.py \
  --features cnv expr \
  --genes TP53 KRAS EGFR BRAF PIK3CA \
  --lineage Lung Breast \
  --output subset.csv

# Pancreas 세포주의 LoF 변이 + 메타데이터 전체 추출
python script/extract.py \
  --features mut_lof meta \
  --lineage Pancreas \
  --output pancreas_lof.csv

# 특정 세포주 지정 + 전체 피처 유형 추출
python script/extract.py \
  --features cnv expr mut mut_lof mut_hot meta \
  --model-ids ACH-000001 ACH-000312 ACH-001282 \
  --output selected_models.csv

# 전처리 파일이 다른 경로에 있을 때
python script/extract.py \
  --features expr \
  --genes MYC MYCN \
  --data-dir /path/to/preprocessed \
  --output myc_expr.csv
```
