# 00. Plan — 혼합변수 다목적 Black-box 최적화: 벤치마크 & 알고리즘 개발

> 이 문서는 실제 calculator를 직접 최적화하기 전에, **싼 대체 문제(benchmark)로 최적화 알고리즘을 개발·검증**하고, 그 결과를 실제 문제에 적용하기 위한 작업 계획이다.

---

## 1. 문제 정의

### X (입력, 30열)
- **Categorical 10개 + Ordinal 20개**, 전체 조합 ≈ **5.5×10¹⁴** (10¹⁴~10¹⁵ 범위)
- 무효(invalid) 조합 없음 — feature engineering으로 사전 제거됨
- 변수 간 교호작용(interaction)이 있을 수도, 없을 수도 있음

### Y (출력, 6개)
- `y11, y12, y13, y21, y22, y23`
- **최대화:** y11, y12, y21, y22 / **최소화:** y13, y23
- 데이터 흐름: `X → black box → 2 set of (2D array, y값) → (y_1, y_2 from 2D array, y_3)`
  - set1 → (y11, y12, y13), set2 → (y21, y22, y23)
- **effect sparsity**: 각 Y는 일부 X에 크게, 일부 X에 적게 의존
- **노이즈**: 주효과(main effect)의 **약 5%** 수준

### 평가 비용 / 예산
- 실제 calculator: **1 eval = 1분**, **순차 평가만 가능(병렬 불가)**
- 최적화 예산: **180 / 780 / 2400 iteration** (각각 3h / 13h / 40h)

---

## 2. Phase 0 — 확정된 설계 결정

| 항목 | 결정 |
|---|---|
| 다목적 처리 | **단일 best 타협해** + 서로 다른 X 영역의 **다양한 top-3 추천** (Pareto front 전체 탐색 X) |
| 효용(=score) 정의 | 정규화 후 **3종 scalarization** (아래 §4) |
| 병렬 평가 | 불가 → **순차 sample-efficient BO** 중심 |
| feasibility | 무효 조합 없음 → 마스킹 불필요 |
| 노이즈 | 주효과 5%, **noisy-GP** 사용. **반복측정은 탐색 중엔 하지 않고 confirmation 단계에만** |
| 예산 분할 | 탐색 + 끝단 확정 (예: 178 탐색 + 12 확정) |

### Driver 구조 (X 블록 분할) — 실제 시스템 구조 반영

| 블록 | 컬럼 | 변량(조합 수) | 영향 대상 |
|---|---|---|---|
| **공통(common)** | 10 | ~10⁶ | 6목적 전부 |
| **set1 전용** | 5 | ~10³ | y11, y12, y13 |
| **set2 전용** | 15 | ~10⁶ | y21, y22, y23 |

- 검산: 10+5+15=30컬럼, 10⁶×10³×10⁶ = **10¹⁵** (목표 범위 상단)
- **유효 차원 비대칭:** set1 = 공통10+전용5 = 15컬럼(~10⁹, 쉬움) / set2 = 공통10+전용15 = 25컬럼(~10¹², 어려움)
  → **set2 목적(y21,y22,y23)이 set1보다 3자릿수 더 어려움 → 단일 효용의 병목.** k-bottom OWA·Chebyshev가 이 병목을 보호.

### Cardinality 배정 (config 기본값, 실제 값으로 교체 가능)
- **공통(10):** `{30,10,5,4,3,3,2,2,2,2}` → ~10⁶ (cat 4 / ord 6)
- **set1(5):** `{10,6,3,3,2}` → ~10³ (cat 2 / ord 3)
- **set2(15):** `{6,5,4,3,3,2,2,2,2,2,2,2,2,2,2}` → ~10⁶ (cat 4 / ord 11)
- 합계: categorical 10 / ordinal 20, 전체 ~10¹⁵
- categorical은 고cardinality 허용(최대 30), ordinal은 중간 이하(2~10)
- **BM1/2/3는 동일한 X 구조/블록 분할을 공유** → 난이도는 교호작용·다봉성·목적충돌·노이즈·공유율로만 조절(알고리즘 비교 공정성 확보)

### 목적 충돌(conflict) 위치 (기본값)
- max군(y11,y12,y21,y22) vs min군(y13,y23) 상충은 주로 **공통 블록의 공유 변수**에 인코딩(set 간 결합도 여기서 발생), set 전용 블록엔 약한 내부 상충만
- 충돌 강도 ρ는 **ladder**: BM1=약 → BM3=강

---

## 3. 작업 순서 (Phase 1 → 5)

### Phase 1. 벤치마크 생성기 + BM1/2/3
실제 black box를 흉내 내되 **즉시 평가되고, 정답(참 최적 효용)을 알 수 있는** 합성 함수 생성기.

**난이도 조절 knob**
- effect sparsity: Y별 강/약 의존 X 배정
- interaction order: 없음(0) → 2차 → 고차
- multimodality / 험준함: 봉우리 개수, 기만성(deceptive)
- objective conflict: 6개 Y 간 trade-off 강도 (max군 vs min군 충돌)
- mixed-variable coupling: cat–ord 결합 효과
- noise: 분산 (기본 주효과의 5%)

**난이도 설계**
- **BM1 (easy):** 거의 가법적 main effect, 교호작용 약, 매끈, 목적 충돌 약
- **BM2 (medium):** 2차 교호작용 + 중간 multimodal + 목적 충돌 존재
- **BM3 (hard):** 고차 교호작용 + 기만적/다봉 + 강한 목적 충돌 + 강한 sparsity

**정답(ground truth)**: 공간이 5.5×10¹⁴라 brute-force 불가 →
함수를 **참 최적이 설계상 알려지도록(plant the optimum)** 구성하고, 대량 랜덤/국소 탐색으로 교차검증.

### Phase 2. Score 체계 (Phase 1 truth 위에 구축)
- **품질 지표**: best 효용의 *참 최적 대비 gap*(정규화). 3종 scalarization 각각.
- **anytime 성능**: best-효용 vs iteration 곡선의 AUC (수렴 속도 — 예산이 제약이라 핵심)
- **robustness**: 여러 seed 평균±표준편차 + worst-case
- **집계**: BM1/2/3 × {180,780,2400} × seeds × 3 scalarization → 알고리즘별 랭킹표
- 예산 분할(탐색/확정)도 변수로 포함해 검증

### Phase 3. 알고리즘 포트폴리오 탐색
예산이 작아 sample-efficient 계열 중심:
- **Bayesian Optimization (최우선):** Optuna(TPE/MOTPE), Ax·BoTorch(qNEHVI, **SAASBO** — 고차원+sparsity 적합), **TuRBO**(예산 큰 쪽), SMAC3(RF surrogate, 혼합변수 강건)
- **EA baseline:** pymoo NSGA-II/III (180 예산엔 약할 것으로 예상, 비교 기준용)
- **surrogate-assisted EA:** 중간 절충
- Phase 2 score로 BM1/2/3 전 예산에서 경쟁 → 우승안 선정

### Phase 4. 문제 구조 활용 (우승안 강화)
- **screening**: Morris elementary effects / RF·fANOVA 변수중요도로 Y별 핵심 X 식별 → 축소 공간 최적화 (180 예산에선 screening 비용 대비 효과를 benchmark로 검증)
- 축소본을 다시 score로 검증

### Phase 5. 실제 calculator 적용 + 확정
- 동결한 알고리즘+하이퍼파라미터를 실제 calculator에 예산 내 적용
- **confirmation**: top-3 후보를 각 4~5회 재측정(총 12~15 eval)해 노이즈 하에서 순위 확정

---

## 4. Score 체계 — 3종 Scalarization

모든 목적을 정규화(min목적은 부호 반전)하여 [0,1], 1=best 로 변환한 후:

1. **정규화 동일가중** — 균형 baseline (한 지표 폭락이 가려질 수 있음 → 비교 기준용)
2. **Augmented Chebyshev** — `max_i [wᵢ·(zᵢ*−zᵢ)] + ρ·Σ(zᵢ*−zᵢ)` (기본 `ρ=0.01`). 최악 성분을 지배 항으로 삼아 "망한 지표" 억제, 증강항으로 weakly-dominated 회피
3. **k-bottom OWA** — 정규화 점수 오름차순 정렬 후 **하위 k개 평균** (기본 `k=2`). 하위 목적들의 동반 향상 강제

- 2·3번은 "한 지표만 폭락" 방지용 안전장치 → 실제 배포 후보. 1번은 비교 baseline.
- 정규화 기준: 벤치마크는 알려진 range로 [0,1]. 실문제는 BO가 관측치로 적응 정규화.
- `ρ`, `k`는 knob로 노출.

---

## 5. 기술 스택
- **언어**: Python
- **생성기**: numpy
- **알고리즘 포트폴리오**: Optuna, Ax/BoTorch, SMAC3, pymoo

---

## 6. "무엇을 먼저" 요약
**Phase 0(완료) → Phase 1(생성기 + BM1/2/3 + truth) → Phase 2(score) → Phase 3(알고리즘) → Phase 4(구조 활용) → Phase 5(실문제 적용+확정)**
benchmark와 score가 갖춰져야 알고리즘 탐색이 의미를 가진다.
