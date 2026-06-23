# 03 — 초기 설계 (cold start): screening DoE

## 왜 첫 배치는 "무작위"가 아니라 "설계"인가
- 데이터 0건 + 점당 비용 큼 → **첫 40~80점을 어디에 쓰느냐**가 이후 대리모델 품질을 좌우.
- 목표: 적은 실험으로 **주효과(main effect)** 와 가능하면 **저차(2-factor) 상호작용** 신호를 최대한 확보.
- 전제 원리: **effect sparsity(소수 변수만 크게 작용)** 와 **effect heredity(상호작용은 주효과가 있는 변수들 사이에서 주로 발생)**.

## 후보 설계

### Plackett–Burman (PB)
- **N개 실험으로 최대 N−1개 요인** 스크리닝 (N은 4의 배수: 4,8,12,16,20,24,…).
- Resolution III: **주효과끼리는 분리**되나 **2-요인 상호작용과 교락(confounding)**.
- 용도: "어떤 변수가 중요한가"를 **최소 실험**으로. 상호작용이 작다고 가정될 때.
- 본 문제: N>60이면 **64런 PB**로 ~63요인 주효과 스크리닝 가능 → 첫 1.5~2일치 배치와 부합.

### Fractional Factorial (2^(k−p))
- 런 수가 **2의 거듭제곱**(4,8,16,32,64,…). Resolution(III/IV/V)로 교락 구조 제어.
  - Res IV: 주효과 ⟂ 2-요인 상호작용(상호작용끼리는 교락).
  - Res V: 2-요인 상호작용까지 분리(런 수↑).
- 상호작용을 일부라도 보고 싶으면 PB보다 유리하나 런 수 증가.

### D-optimal (컴퓨터 생성)
- 원하는 모델(주효과+선택된 2차항)과 **제약**, 임의 런 수에 맞춰 정보행렬 최적화.
- **배타/그룹 제약(불가능 조합 제외)** 을 직접 반영 가능 → 04 문서의 제약과 결합하기 좋음.
- 본 문제처럼 **제약 + 특정 상호작용 관심 + 런 수 고정(40의 배수)** 이면 가장 유연.

## 본 문제 권고
1. **첫 배치(=하루 40, 또는 64런)**: 제약을 반영한 **D-optimal**(주효과 + 의심되는 소수 2차항) 우선.
   - 제약 표현이 아직 없으면 **PB/Res III**로 시작 → 제약 확보 후 D-optimal로 전환.
2. 첫 배치 분석으로 **중요 변수 솎기** → 이후 BO 루프는 축소된/가중된 공간에서 진행(표본효율↑, 반복시간↓).
3. 구현: `pyDOE2`(factorial/PB), D-optimal은 `dexpy`/직접 구현/`scikit-optimize`/상용(JMP) 참고.

## 주의
- PB의 주효과–상호작용 교락 때문에, **상호작용이 큰 변수**가 의심되면 PB 단독은 위험 → Res IV/V 또는 D-optimal로 보강.
- screening은 BO를 대체하지 않고 **BO의 사전 데이터(prior batch)** 를 만드는 단계.

---

## ★ 제약(group/mutex/conditional) 하에서 DoE — DoE는 인코딩된 X에서 한다
> 핵심 통찰: **raw X0(60bit)에 PB를 그대로 쓰면 안 된다.** PB/분수계획은 "서로 독립인 2수준 요인"을 가정하는데,
> 여러 비트가 하나의 변수(그룹)·배타(one-hot/mutex)·조건부(conditional)이면 이 가정이 깨진다.
> → **먼저 X0→X 인코딩**(그룹/배타→범주형 요인, 조건부→계층 요인)으로 **합법 요인공간 X**를 만든 뒤, **X 위에서 설계**한다.
> (표현 파이프라인 X0→X→…→Y의 X 단계에서 DoE 수행; research/10과 정합)

이렇게 인코딩하면 요인이 2수준 binary가 아니라 **다수준 범주형 + 조건부**가 되므로, 고전적 직교설계(PB/분수계획)보다
**제약-aware 최적설계(D-/Ds-optimal)** 가 자연스럽다. 아래 4개 설계질문에 답한다.

### 4A. D-optimal을 한다면 candidate set은 어떻게 만드나
- 두 갈래:
  1. **후보집합 행교환(candidate-set / row-exchange)**: 인코딩된 X의 **실현가능(feasible) 점만** 후보 풀로 생성(그룹·배타·조건부를 만족하는 조합만) → D-optimal이 그 안에서 정보행렬 |X′X| 최대가 되는 부분집합 선택.
     - 후보 풀 생성: 요인 수가 작으면 **완전열거**, 크면 **제약을 지키는 무작위/space-filling 샘플링**으로 feasible 후보 다수 생성.
  2. **좌표교환(coordinate-exchange)**: **후보집합을 명시적으로 만들지 않음** — 설계공간 자체가 후보. 각 좌표를 이웃 feasible 값으로 교환하며 |X′X| 개선. **요인 수가 많아 후보 풀이 지수폭발할 때 유리**(본 문제처럼 N이 큰 경우 권장).
- 어느 쪽이든 **불가능 조합(disallowed)을 후보/이동에서 원천 배제** → 인코딩이 이미 feasible만 만들면 자동 충족.
- conditional 변수: 비활성일 때는 **기준값(baseline)으로 고정**해 모델행렬에 일관되게 반영.
- **feasible 후보 생성의 구체 방법은 research/04 §4.2** 참조: rejection이 아니라 **구성적 샘플링**(mutex=범주형, conditional=DAG 위상정렬로 parent→child) 또는 **coordinate-exchange + feasibility 오라클**. 이 풀/오라클은 **DoE·EA·서빙이 공유하는 단일 소스**여야 함.

### 4B. constraint·repair가 DoE의 균형성을 깨뜨리지 않나
- **깨뜨린다 — 그래서 "직교설계 + 사후 repair"는 피한다.** PB/분수계획의 강점은 **직교성(별칭구조 통제, 최소분산)** 인데,
  생성 후 비트를 뒤집어 제약을 맞추면(repair) **직교성·균형이 망가지고 별칭구조가 통제 불능**이 된다(주효과가 엉뚱한 효과와 섞임).
- 올바른 길: **제약을 설계단계에 내장**한 **최적설계(D-optimal)** 사용. 최적설계는 직교성에 의존하지 않고 **정보행렬을 직접 최대화**하므로,
  완전 직교를 일부 포기하는 대신 **feasible 영역에서 최선의 균형**을 찾는다.
- 잔여 불균형은 불가피 → **사후 점검**: 효과 간 상관/VIF, 별칭 행렬, 검정력(power)을 확인해 추정 가능성을 검증.
- 참고: 직교설계는 **n이 4의 배수(분수계획은 2의 거듭제곱)** 일 때만 존재 → 제약·임의 런수에선 최적설계가 현실적.

### 4C. 어떤 효과까지 screening 할 것인가 + **DoE의 목표는 main effect인가 interaction 발굴인가**
> 직접 답: **초기 DoE의 1차 목표 = 활성 주효과 식별(factor screening) — "어떤 요인이 죽었고 어떤 요인이 사는가".**
> interaction은 **"완전 추정"이 아니라 "후보 발굴(flag)"** 까지만 노린다. 상호작용의 본격 모델링은 **이후 적응형 루프(대리모델/EA)** 의 몫.
- 이유:
  - 본 문제는 "상호작용이 크다"고 알려져 있음 → **주효과 추정이 2차와 심하게 별칭(aliased)되면 안 됨** → 적어도 **Res IV 유사**(주효과 ⟂ 2차)로 설계해 주효과를 보호하고 **어떤 2차가 살아있는지 단서**를 얻는다.
  - 그러나 전체 2차 C(k,2)를 한 번에 추정하려 들면 런 폭발 → **표적 2차만**(도메인 의심쌍 + effect heredity로 주효과 큰 쌍)을 모델에 포함.
- 정리(효과 범위): **필수 = 모든 인코딩 요인의 주효과(범주형은 수준 대비)** / **선택 = 표적 2차** / **3차↑ = screening 범위 밖**(루프가 암묵 포착).
- 해상도: 예산 충분 → **Res IV 유사**(주효과를 2차와 분리), 빠듯 → 주효과 우선(Res III 유사) + **후속 D-optimal augmentation**으로 의심 2차 해소.
- 한 줄: **DoE = "활성 주효과 솎기 + 상호작용 후보 깃발 꽂기"**, 상호작용 확정은 적응형 루프로 이연.

### 4D. 초기 DoE 예산은 얼마나
- 하한: D-optimal은 **런 수 n ≥ 추정할 계수 수 p** 면 성립. p = 1(절편) + Σ(주효과 자유도) + (표적 2차항 수).
  - 예: 인코딩 후 유효 주효과 자유도 ~60, 표적 2차 ~20~30이면 p ≈ 80~90.
- 실무 권장: **n ≈ 1.5~2 × p**(오차 자유도·검정력 확보) → 위 예에서 **~120~180 런**.
  - MAIN(320/일): **1일 미만**. SUB(40/일): **3~4일**.
- **전체 예산의 소수(≈5~15%)만** 초기 DoE에 배정 — 대부분은 최적화 루프로. **순차 증강**(작게 시작 → D-optimal augmentation)으로 낭비 최소화.
- 일부 런은 **반복측정(replication)** 에 배정해 **노이즈 추정**(Q10)과 순수오차 자유도 확보.

> 요약: **인코딩으로 feasible 요인공간 X를 먼저 만들고 → X에서 (coordinate-exchange) D-optimal로 주효과+표적 2차를 ~1.5~2×p 런으로 추정 → 직교설계+repair는 금지 → 예산은 전체의 5~15%, 순차 증강.**

## Sources
- [Plackett–Burman Designs (JMP)](https://www.jmp.com/en/statistics-knowledge-portal/design-of-experiments/screening-designs/plackett-burman-designs)
- [Plackett–Burman (PSU STAT 503)](https://online.stat.psu.edu/stat503/lesson/8/8.4)
- [When and How to Use Plackett–Burman (iSixSigma)](https://www.isixsigma.com/design-of-experiments-doe/when-and-how-to-use-plackett-burman-experimental-design/)
- [2^k-p Fractional Factorial vs Plackett–Burman (JMP Community)](https://community.jmp.com/t5/Discussions/2k-p-fractional-factorial-designs-vs-Plackett-Burman-designs/td-p/238587)
- [MOODE: Multi-Objective Optimal Design of Experiments (R)](https://arxiv.org/pdf/2412.17158)
- 제약-aware 최적설계: [The Coordinate-Exchange Algorithm for Exact Optimal Designs](https://www.researchgate.net/publication/241736151_The_Coordinate-Exchange_Algorithm_for_Constructing_Exact_Optimal_Experimental_Designs) · [D-Optimal Designs (MATLAB: cordexch/candexch)](https://www.mathworks.com/help/stats/d-optimal-designs.html) · [D-Optimal in Constrained Experiments (MetricGate)](https://metricgate.com/docs/d-optimal-design-construction) · [Coordinate-Exchange for Irregular Design Space (Clemson)](https://open.clemson.edu/all_dissertations/1928/)
- screening 런수/효과: [D- and A-optimal Screening Designs](https://arxiv.org/pdf/2210.13943) · [Optimal screening designs (patent)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10535422)
