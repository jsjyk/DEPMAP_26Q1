# DepMap Public 26Q1 데이터셋 파일 목록

> **릴리즈**: DepMap Public 26Q1  
> **총 파일 수**: 22개  
> **총 용량**: 약 4.8GB  
> **로컬 경로**: `C:/Users/Admin/Desktop/데이터/DepMap_26Q1/`  
> **작성일**: 2026-05-15

---

## 개요

DepMap 26Q1 릴리즈는 암 세포주에 대한 다음 데이터를 포함합니다:
- 전장유전체/엑솜 시퀀싱 기반 **Copy Number & Mutation** 데이터
- RNA 시퀀싱 기반 **발현(Expression)** 데이터
- 전장 유전체 규모 **CRISPR knockout 스크리닝** 데이터
- 세포주 및 실험 조건 **메타데이터**

---

## 1. 메타데이터 파일

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `Model.csv` | 682K | 암 모델/세포주 메타데이터. ModelID, 암종(OncotreeLineage), 환자정보(나이, 성별, 인종), 샘플 수집 부위, 모델 타입 등 포함 |
| `ModelCondition.csv` | 219K | 모델 실험 조건. 배지, 배양 형태(Adherent/Suspension 등), passage number, anchor drug 정보 포함 |
| `Gene.csv` | 17M | 전체 유전자 메타데이터. HGNC 기반 gene symbol, Entrez ID, Ensembl ID 등 identifier 매핑 |
| `PortalCompounds.csv` | 684K | 약물 메타데이터. CompoundID, 타겟 유전자, ChEMBL ID, SMILES, InChIKey, PubChemCID 등 |
| `OmicsProfiles.csv` | 656K | Omics 프로파일 ID 매핑. ModelID ↔ SequencingID ↔ ProfileID 연결, 시퀀싱 플랫폼 및 날짜 포함 |

---

## 2. 유전자 발현 데이터 (RNA-seq)

> **행**: ProfileID (또는 ModelID) | **열**: 유전자  
> Salmon v1.10.0으로 TPM 계산, STAR v2.7.11b으로 raw count 산출 | Gencode v38 기반

### TPM (Log 변환)

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `OmicsExpressionTPMLogp1HumanAllGenes.csv` | 697M | log(TPM+1), **unstranded**, 전체 유전자 |
| `OmicsExpressionTPMLogp1HumanAllGenesStranded.csv` | 689M | log(TPM+1), **stranded**, 전체 유전자 ✅ 권장 |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 291M | log(TPM+1), **unstranded**, 단백질 코딩 유전자만 |

### Expected Count

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `OmicsExpressionExpectedCountHumanAllGenes.csv` | 277M | Expected count, **unstranded**, 전체 유전자 |
| `OmicsExpressionExpectedCountHumanAllGenesStranded.csv` | 276M | Expected count, **stranded**, 전체 유전자 ✅ 권장 |
| `OmicsExpressionExpectedCountHumanProteinCodingGenes.csv` | 124M | Expected count, **unstranded**, 단백질 코딩 유전자 |

### Raw Read Count

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `OmicsExpressionRawReadCountHumanAllGenesStranded.csv` | 278M | Raw read count, **stranded**, 전체 유전자 |
| `OmicsExpressionRawReadCountHumanProteinCodingGenesStranded.csv` | 124M | Raw read count, **stranded**, 단백질 코딩 유전자 |

---

## 3. Copy Number 데이터 (WGS)

> WGS 기반 copy number. HMMcopy 및 PureCN 파이프라인 사용, hg38 기준

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `OmicsCNGeneWGS.csv` | 391M | 유전자 수준 copy number (linear scale). 행=ModelID, 열=유전자 |
| `OmicsCNSegmentsWGS.csv` | 61M | 세그먼트 수준 copy number. CONTIG, START, END, SEGMENT_COPY_NUMBER 등 포함 |
| `PortalOmicsCNGeneLog2.csv` | 385M | log2(CN+1) 변환 버전 (포털 Data Explorer용) |

---

## 4. 체세포 변이 데이터 (Somatic Mutations)

> Mutect2로 변이 콜링, hg38 기준. VEP 기반 기능적 주석 포함

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `OmicsSomaticMutations.csv` | 554M | MAF 형식 전체 체세포 변이. 100개 이상 컬럼 포함 (Sift, Polyphen, gnomAD AF, ClinSig, OncoKB 등) |
| `OmicsSomaticMutationsMatrixDamaging.csv` | 228M | 유전자×세포주 행렬. Damaging mutation 유무 (0/1/2). LikelyLoF=True 기준 |
| `OmicsSomaticMutationsMatrixHotspot.csv` | 6.6M | 유전자×세포주 행렬. Hotspot mutation 유무 (0/1/2). OncoKB/COSMIC Tier1 기준 |

### OmicsSomaticMutations.csv 주요 컬럼
- **변이 정보**: Chrom, Pos, Ref, Alt, AF, DP, VariantType, DNAChange, ProteinChange
- **유전자 정보**: HugoSymbol, EnsemblGeneID, HgncFamily, UniprotID
- **기능 예측**: VepImpact, Sift, Polyphen, AMClass, AMPathogenicity, RevelScore
- **임상 정보**: VepClinSig, CivicDescription, HessDriver, Hotspot
- **ID 매핑**: ModelID, ModelConditionID, SequencingID

---

## 5. Genomic Signatures

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `OmicsGlobalSignatures.csv` | 321K | 게놈 시그니처 행렬. 행=SequencingID |

### 포함 지표
| 컬럼 | 설명 | 도구 |
|------|------|------|
| `MSIScore` | Microsatellite Instability score (≥20이면 MSI) | MSIsensor2 |
| `Ploidy` | 배수성 | PureCN |
| `CIN` | Chromosomal Instability | PureCN |
| `WGD` | Whole Genome Doubling (0/1) | PureCN |
| `LoHFraction` | Loss of Heterozygosity 비율 | PureCN |
| `Aneuploidy` | Aneuploidy score (Ben-David 2021 기준) | - |

---

## 6. CRISPR 스크리닝 데이터

> Achilles (Avana Cas9 + Humagne-CD Cas12) + Sanger Project SCORE (KY Cas9) + BioGRID (Brunello, TKOv3) 통합  
> Chronos 알고리즘으로 gene effect 계산

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `ScreenGeneEffectUncorrected.csv` | 494M | Screen-level gene effect (copy number 보정 및 스케일링 전). 행=ScreenID, 열=유전자 |

---

## 7. 기타

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `README.txt` | 47K | 전체 파일 상세 설명 및 파이프라인 문서 |

---

## 데이터 구조 요약

```
행(Row) 인덱스 종류:
- ModelID       : ACH-xxxxxx 형식, 세포주 수준 대표값
- ProfileID     : 개별 시퀀싱 프로파일 수준 (한 세포주에 여러 개 가능)
- SequencingID  : 원시 시퀀싱 ID
- ScreenID      : CRISPR 스크리닝 ID

열(Column) 인덱스:
- 유전자: "HUGO Symbol (Entrez ID)" 형식 (예: TP53 (7157))
```

---

## 주요 파일 선택 가이드

| 분석 목적 | 권장 파일 |
|-----------|-----------|
| 유전자 발현 분석 (표준) | `OmicsExpressionTPMLogp1HumanAllGenesStranded.csv` |
| DEG 분석 (raw count 필요) | `OmicsExpressionRawReadCountHumanAllGenesStranded.csv` |
| Copy number 분석 | `OmicsCNGeneWGS.csv` |
| 변이 기반 분류 | `OmicsSomaticMutationsMatrixDamaging.csv` |
| Hotspot 변이 확인 | `OmicsSomaticMutationsMatrixHotspot.csv` |
| 세포주 정보 조회 | `Model.csv` |
| 게놈 안정성 지표 | `OmicsGlobalSignatures.csv` |
| CRISPR 필수 유전자 분석 | `ScreenGeneEffectUncorrected.csv` |

---

## 참고 링크

- DepMap Portal: https://depmap.org/portal/
- 변이 파이프라인 문서: https://storage.googleapis.com/shared-portal-files/Tools/26Q1_Mutation_Pipeline_Documentation.pdf
- Copy number 파이프라인: https://github.com/broadinstitute/depmap-omics-wgs
- RNA-seq 파이프라인: https://github.com/broadinstitute/depmap-omics-rna
