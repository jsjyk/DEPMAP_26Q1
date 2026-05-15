# Feature Extractor AI 모델 설계 노트

> **작성일**: 2026-05-15  
> **목적**: DepMap 기반 멀티오믹스 Feature Extractor 아키텍처 구상 및 토론 기록  
> **핵심 목표**: 유전자 수 변동에 유연하고, 생물학적 구조를 반영하며, 세포주 → 오가노이드 → 임상 도메인 전이가 가능한 모델

---

## 1. 설계 요구사항

| 요구사항 | 설명 |
|----------|------|
| **가변 유전자 수 대응** | 데이터셋마다 profiling된 유전자 수가 다름. 고정 차원 입력 불가 |
| **멀티오믹스 통합** | Expression / Copy Number / Mutation / PPI / Pathway 동시 활용 |
| **도메인 전이** | 세포주(대량) → 오가노이드(소량, ~수십 개) → 임상(미접촉) |
| **해석 가능성** | 어떤 유전자·경로가 예측에 기여했는지 역추적 가능해야 함 |

---

## 2. 핵심 아이디어: 유전자를 "토큰"으로 처리

고정 길이 벡터 대신 **각 유전자를 하나의 토큰**으로 취급한다.

- 유전자 ID → 학습된 임베딩 (HGNC 기준, ~30,000개 유전자 어휘)
- 오믹스 값(expression, CNV, mutation) → 같은 임베딩 공간에 추가
- 없는 유전자는 패딩 없이 그냥 제외 → **Set 처리**

이 방식은 Geneformer(Nature 2023), scGPT(Nature Methods 2024)에서 검증된 접근법이다.

---

## 3. 전체 아키텍처: Hierarchical Multi-Omics Transformer (HMOT)

```
입력: N개 유전자 (N은 샘플마다 다를 수 있음)
  각 유전자 g_i: expression_i, cnv_i, mutation_i

      │
      ▼
┌─────────────────────────────┐
│  Layer 1: Gene Tokenizer    │   유전자 수 가변성 해결
│                             │
│  token_i = LayerNorm(        │
│    Gene_Emb(gene_id_i)      │   ID 임베딩 (학습)
│  + MLP([expr, cnv, mut])    │   오믹스 값 → 같은 공간
│  )                          │
│                             │
│  출력: N × D  (N 가변)       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 2: PPI Graph         │   분자 상호작용 문맥 반영
│  Attention                  │
│                             │
│  STRING DB / BioGRID 엣지를  │
│  attention bias로 사용        │
│  → 상호작용 유전자끼리 정보 교환  │
│                             │
│  출력: N × D' (contextualized)│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 3: Pathway Pooling   │   유전자 → 생물학적 경로
│                             │
│  KEGG / Reactome 기반        │
│  pathway_j =                │
│    AttentionPool(           │
│      { token_i | g_i ∈ p_j }│
│    )                        │
│                             │
│  출력: P × D''  (P ≈ 300)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 4: Global Pooling    │   세포주 전체 표현
│                             │
│  Pathway 임베딩들을           │
│  attention pooling으로 집계   │
│                             │
│  출력: Z ∈ R^512 (고정 dim)  │ ← downstream에 사용되는 표현
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│  Task Head   │  │  Domain Adapter  │
│  (drug resp/ │  │  (도메인 전이용)   │
│   survival)  │  └──────────────────┘
└──────────────┘
```

---

## 4. 설계 토론 및 결론

### 4-1. Pooling이 해석 가능성을 해치지 않는가?

**결론: 오히려 계층적으로 더 해석 유리하다.**

평탄한(flat) 고차원 벡터와 달리, attention pooling은 각 집계 단계에서 가중치를 저장한다.

```
예측 결과 → Pathway attention weight → 어떤 pathway가 중요했나
                    ↓
           Gene attention weight → 그 pathway 안에서 어떤 유전자가 주도했나
```

추가 해석 도구:
- **Integrated Gradients**: 출력에서 입력 유전자 토큰까지 기여도 역추적
- **Attention Rollout**: 레이어간 attention weight 곱으로 end-to-end 기여도 계산
- **Pathway-level 요약**: 300개 pathway 중 상위 K개 → 임상의가 읽을 수 있는 수준

핵심: pooling이 정보를 "뭉개는" 게 아니라 **생물학적 계층 구조를 따라 압축**하므로,
압축 과정에서 사용된 가중치 자체가 해석의 재료가 된다.

---

### 4-2. PPI → Pathway 전환에서 기대할 수 있는 이점

**핵심: 유전자 토큰이 "분자적 상태"를 담고 나서 pathway 집계가 일어난다.**

PPI attention 없이 pathway pooling을 하면:
> "KRAS가 이 pathway에 속한다" (구조적 사실)

PPI attention 후 pathway pooling을 하면:
> "KRAS가 EGFR, RAF와 함께 활성화된 허브 상태에서 이 pathway에 기여한다" (기능적 상태)

구체적 이점:

| 이점 | 설명 |
|------|------|
| **Hub 효과 반영** | PPI에서 연결이 많은 유전자(hub)는 더 풍부한 문맥을 담아 pathway에 기여 |
| **Cross-pathway 크로스토크** | 여러 pathway에 속하는 유전자가 PPI 이웃 정보를 pathway 간에 전파 |
| **노이즈 분리** | 고립된 변이(PPI 연결 적음) vs. 네트워크 허브 변이를 구별 가능 |
| **생물학적 계층 자연스럽게 구현** | 분자 상호작용(PPI) → 기능적 모듈(Pathway) 순서가 실제 세포 신호전달과 일치 |

---

### 4-3. 오가노이드 수십 개 + 임상 zero-shot 일반화의 현실적 가능성

**난이도: 높음. 단, 생물학적 inductive bias가 핵심 안전망.**

상황 정리:
```
세포주:    ~1,105개  (DepMap, 레이블 풍부)
오가노이드:  ~수십 개  (레이블 일부)
임상:       0개     (학습 시 미접촉)
```

#### 전략 A: 메타러닝 (MAML / Prototypical Networks)
- 세포주 데이터로 "적은 예시로 빠르게 적응하는 능력" 자체를 학습
- 각 에피소드 = 한 암종의 일부만 보고 나머지를 예측
- 오가노이드 수십 개를 few-shot adaptation의 support set으로 활용

#### 전략 B: 생물학적 불변성에 의존
- KRAS 변이 → RAS 신호 교란은 세포주/오가노이드/임상 모두 공통
- PPI + Pathway 구조를 고정 prior로 사용 → 도메인 무관하게 작동하는 표현 학습
- 모델이 "배양 환경 특성"이 아닌 "분자 생물학 법칙"을 학습하도록 유도

#### 전략 C: 레이블 없는 임상 데이터 활용 (권장)
- 임상 데이터를 완전히 zero-shot으로 보는 건 현실적 한계 존재
- **최소한 레이블 없는 임상 샘플(TCGA 등)이라도** domain alignment에 활용
- TCGA는 공개 데이터로 접근 가능 → 실질적 선택지

**현실적 기대치:**
> 세포주 → 오가노이드: 충분히 가능  
> 세포주+오가노이드 → 임상: 생물학적 prior + TCGA unlabeled 활용 시 의미있는 수준 기대 가능  
> 완전 zero-shot 임상 일반화: 매우 도전적, 점진적 검증 필요

---

### 4-4. DANN과 Progressive Fine-tuning의 데이터 중복 문제

**결론: 문제없음. 예측 타겟이 다르고, 오히려 시너지가 있다.**

```
DANN (전략 1)
  목적: 인코더가 도메인을 구별 못 하게 만들기
  타겟: 도메인 레이블 (세포주/오가노이드/임상 중 어디?)
  시점: pre-training과 동시에 (regularization)

Progressive Fine-tuning (전략 3)
  목적: 순차적으로 도메인을 적응
  타겟: task label (약물 반응, 생존율 등)
  시점: pre-training 이후 순차 적용
```

시너지 구조:
```
Phase 1: DANN으로 도메인 불변 백본 학습
              ↓
Phase 2: 그 위에 오가노이드로 fine-tune (출발점이 이미 좋음)
              ↓
Phase 3: 임상 데이터로 마지막 레이어만 fine-tune
```

DANN이 미리 분포를 정렬해놨기 때문에, Progressive fine-tuning의 각 단계 적응 거리가 짧아진다.

---

## 5. 도메인 적응 전략 종합

### 전략 1: Adversarial Training (DANN)

```
Z ──→ [Gradient Reversal Layer (λ)] ──→ Domain Classifier
                                          (세포주 / 오가노이드 / 임상)

L_total = L_task - λ · L_domain
```

- 인코더는 task를 잘 풀면서도 도메인을 구별 못 하는 표현 학습
- λ를 학습 중 점진적으로 증가 (초반엔 task에 집중, 후반엔 정렬 강화)

### 전략 2: MMD / CORAL Loss

```
L_total = L_task + λ · MMD(Z_cellline, Z_target)
```

- 도메인 간 분포의 통계를 직접 최소화
- CORAL: 2차 통계량(공분산) 정렬 → 계산 간단, 효과적

### 전략 3: Progressive Fine-tuning

```
세포주 (대량, DepMap)     → 전체 모델 pre-training
        ↓
오가노이드 (수십 개)       → Layer 1~2 동결, Layer 3~4 + Head fine-tune
        ↓
임상 (소량 또는 unlabeled) → Head만 fine-tune (또는 MMD alignment)
```

### 전략 4 (추가): 메타러닝 (MAML)

오가노이드 few-shot 적응 전용:
```python
# 각 에피소드: 한 암종의 일부 support set → query set 예측
# 메타 목적: 적은 gradient step으로 새 도메인에 적응하는 초기화 학습
for episode in meta_train:
    support, query = sample_episode(cell_lines, cancer_type)
    adapted_params = inner_loop_update(model, support)
    meta_loss += task_loss(adapted_params, query)
```

---

## 6. 학습 파이프라인

| 단계 | 데이터 | 목적 | 손실 함수 |
|------|--------|------|-----------|
| **Pre-training** | 세포주 1,105개 | Masked Gene Modeling (BERT 방식) | Reconstruction loss |
| **Contrastive** | 세포주 (augmentation) | 같은 세포주의 다른 오믹스 조합을 가깝게 | NT-Xent (SimCLR) |
| **Domain-aware training** | 세포주 + 오가노이드 | DANN으로 도메인 불변 표현 | L_task + L_DANN |
| **Few-shot adaptation** | 오가노이드 수십 개 | MAML fine-tuning | Meta-gradient |
| **Clinical alignment** | TCGA 등 unlabeled | 임상 분포 정렬 | MMD / CORAL |

---

## 7. 참고 논문

| 모델 | 학술지 | 핵심 기여 | 참고 포인트 |
|------|--------|-----------|------------|
| Geneformer | Nature 2023 | 유전자 토큰 Transformer pre-training | Gene Tokenizer 설계 |
| scGPT | Nature Methods 2024 | 단세포 foundation model | 가변 유전자 set 처리 |
| MOGONET | Nature Comm 2021 | 멀티오믹스 GNN fusion | 오믹스 통합 구조 |
| DANN | JMLR 2016 | Gradient reversal domain adaptation | 도메인 정렬 이론 |
| MAML | ICML 2017 | Model-Agnostic Meta-Learning | Few-shot 적응 |
| scVI | Nature Methods 2018 | VAE + 배치효과 제거 | 도메인 표현 분리 |
| Graphormer | NeurIPS 2021 | 그래프 구조를 Transformer에 통합 | PPI attention bias |
| PathDSP | Nature Comm 2021 | Pathway 기반 약물 민감도 예측 | Pathway pooling 설계 |

---

## 8. 구현 로드맵

```
Phase 1 — 기반 모델 (우선순위 ★★★)
  ├─ Gene Tokenizer 구현
  ├─ Transformer encoder (variable-length input)
  └─ 세포주 데이터로 Masked Gene Modeling pre-training

Phase 2 — 생물학적 구조 통합 (우선순위 ★★★)
  ├─ PPI 그래프 로드 (STRING DB)
  ├─ PPI-guided attention bias 추가
  ├─ KEGG/Reactome pathway 매핑 로드
  └─ Pathway attention pooling 추가

Phase 3 — 도메인 전이 (우선순위 ★★☆)
  ├─ DANN (Gradient Reversal Layer) 구현
  ├─ MAML few-shot adaptation (오가노이드)
  └─ TCGA unlabeled 데이터 MMD alignment

Phase 4 — 해석 가능성 (우선순위 ★★☆)
  ├─ Attention weight 저장 및 시각화
  ├─ Integrated Gradients 구현
  └─ Pathway-level 중요도 리포트 생성
```

---

## 9. 미결 사항 및 다음 논의 포인트

- [ ] Pathway 데이터 소스 결정: KEGG vs Reactome vs MSigDB (Hallmark)
- [ ] PPI threshold: STRING DB confidence score 기준값
- [ ] 오가노이드 데이터 확보 계획 (현재 DepMap 내 수십 개)
- [ ] TCGA unlabeled 데이터 활용 방식 구체화
- [ ] Task head 목적 함수 확정 (약물 반응? 생존 예측? 바이오마커?)
- [ ] 평가 지표: 도메인 전이 성능을 어떻게 측정할 것인가
