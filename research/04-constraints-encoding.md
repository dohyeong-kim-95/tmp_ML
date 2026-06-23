# 04 — 제약/구조 인코딩 (그룹·배타·조건부·상호작용)

> 01_Problem_def.md의 구조: (a) 여러 binary가 합쳐져 하나의 변량(그룹), (b) 서로 배타적(최대 1개=1),
> (c) 독립, (d) 상호작용 큼. 규칙 자체는 대외비 → **인코딩 슬롯**만 설계해 둔다.

## 1. 배타(mutually exclusive) / 그룹(합성 범주)
- **재파라미터화가 1순위**: 배타적 binary 묶음은 **하나의 범주형 변수**로 합치는 것이 가장 깔끔.
  - 예: b1,b2,b3가 배타적·최대 1개 → 범주형 g ∈ {none,1,2,3}. 차원↓, 불가능 조합 원천 제거.
- **one-hot + 제약**: 범주를 one-hot으로 두고 "합=1" 제약을 acquisition 최적화에 부과.
  - 단점: one-hot은 **모든 범주쌍에 동일 공분산 가정**(실제 상관 무시), acquisition 표면이 **평평한 영역** 많아 최적화 난이도↑, 라운딩으로 **중복 추천** 발생 가능.
- **정수 인코딩 + 초소 길이척도**: 범주를 정수로 두고 ARD GP 길이척도를 매우 작게 → 이웃 정수 간 영향 제거(one-hot 제약 회피 트릭).

## 2. 조건부(conditional) 관계
- "A가 켜져야만 B가 유효" 류는 **조건부 탐색공간**으로 표현.
- **SMAC3** 가 조건부/계층 파라미터를 native 지원(ConfigSpace) → 구조가 조건부 중심이면 SMAC3가 강점.
- Optuna도 `suggest_*`를 코드 분기로 조건부 표현 가능.

### 2.1 parent–child 그래프는 DAG인가? (질문)
- **반드시 비순환(acyclic)이어야 한다.** "A의 활성이 B를 활성, B의 활성이 A를 활성"처럼 **순환이면 활성 정의가 모순** → 불가.
- 일반형은 **DAG**: child가 **여러 parent**를 가질 수 있음(예: "A=1 **그리고** C=2일 때만 D 활성" = 합집합/교집합 조건).
- 흔한 특수형은 **트리/포레스트**(각 child가 parent 1개). 대부분의 HPO 도구(ConfigSpace 등)는 **DAG(비순환) 가정**, 다수는 트리.
- ⚠️ **본 문제의 실제 구조가 트리인지 일반 DAG인지는 도메인 입력**(대외비) → 현재 보류. 단 **프레임워크 요건 = 비순환 DAG**.
- 구현 함의: 조건은 DAG로 모델링하고, 표본 생성·모델행렬·교배는 **위상정렬(topological order)** 로 처리(아래 4.2).

## 3. 상호작용(interaction)
- **BOCS형 선형 대리모델**: pairwise 항을 명시적으로 모델 → 상호작용을 **계수로 발견·해석**. 추후 도메인지식화에 직결.
- **COMBO diffusion kernel / RF**: 고차 상호작용을 비모수적으로 포착(해석성은 낮음).
- effect heredity 가정으로 **상호작용 후보를 주효과 큰 변수쌍으로 제한** → 항 수 폭발 억제(N=60이면 전체 pairwise ≈ 1770항).

## 4. 불가능(infeasible) 조합 처리
- 가능하면 **인코딩 단계에서 원천 차단**(범주형 재파라미터화, D-optimal 제약).
- 차단 못 하면 **feasibility 분류모델**(가능/불가능 예측) + acquisition에 곱하거나, 페널티 부여.
- 제약을 acquisition 최적화기(정수계획/SA)에 hard constraint로 넣는 방법도.

### 4.2 feasible candidate pool은 어떻게 생성하나 (질문)
> 원칙: **"무작위 생성 후 버리기(rejection)"가 아니라 "처음부터 합법만 만들기(feasibility-by-construction)".**
> 제약이 빡빡하면 rejection은 대부분 버려져 낭비 → 구성적 샘플링이 정석.

- **구성적(constructive) 샘플링 — 1순위**:
  1. **배타/그룹**: 각 mutex 그룹을 **하나의 범주형**으로 보고 {none, 옵션1, …} 중 하나를 뽑음(자동으로 "최대 1개=1" 충족).
  2. **조건부**: DAG **위상정렬 순서로 parent 먼저 표본 → 활성 조건 만족 시에만 child 표본**(비활성 child는 baseline/NA 고정).
  3. **독립 요인**: 자유롭게 표본.
  → 이렇게 만든 점은 **항상 feasible**. 풀 크기는 필요 수보다 크게 잡아 D-optimal 선택 여유 확보.
- **공간 채움(space-filling)**: 구성적 샘플링에 Latin-hypercube/최대최소거리 기준을 얹어 후보가 **고루 퍼지게**.
- **명시적 풀이 부담될 때 — coordinate-exchange**: 후보집합을 만들지 않고 **feasibility 오라클**(점이 합법인지 판정)만 두고
  좌표를 **합법 이웃으로만 교환** → 큰 요인공간에서 풀 폭발 회피(research/03 §4A와 동일 권고).
- **제약이 복잡(논리식 다수)할 때**: SAT/CP 솔버로 **uniform feasible 표본** 생성도 가능(고급).
- 핵심 연결: 이 풀/오라클은 **DoE(§03)·EA 교배·서빙 후보 생성**이 **모두 같은 feasibility 규칙**을 쓰게 하는 단일 소스가 되어야 함.

### 4.3 optimizer가 invalid 조합을 만들면 시스템은 어떻게 반응하나 (질문 Q1)
> 답: **단일 정책이 아니라 제약 성격별 계층(layered) 정책.** "재추첨 vs X0→X 재설계"의 이분법이 아니다.
> 핵심은 **feasibility 규칙을 한 곳(데이터 명세/계약)에 두고**, 위반을 **가능한 한 발생 자체를 막되, 못 막으면 단계적으로 처리**.

| 우선순위 | 방법 | 적용 대상 | 비고 |
|---------|------|-----------|------|
| 1 | **feasibility-by-construction (인코딩)** | **hard·구조적 제약**(mutex/그룹/조건부) | X0→X 인코딩으로 **애초에 invalid가 안 나오게**. mutex=범주형, conditional=DAG 위상정렬(§4.2). EA 연산자도 X에서 동작. |
| 2 | **repair (가장 가까운 feasible로 투영)** | 루프 중 연산(EA 교배/변이)이 경계를 넘을 때 | **최적화 루프에선 OK**(DoE에선 균형 깨므로 금지 — §03 4B와 구분). 결정적 repair 연산자 사용. |
| 3 | **rejection / 재추첨** | 위 둘로 안 되는 드문 경우 | 단순하지만 제약 빡빡하면 낭비 → fallback. |
| 4 | **penalty / feasibility 분류모델** | **soft·미지(아직 규칙 모름) 제약** | Y에 페널티 또는 가능/불가능 예측모델 곱. 규칙이 데이터로만 드러날 때. |

- **"X0→X 재설계가 복잡해진다"** 우려에 대해: **hard 구조 제약만 인코딩**하면 복잡도는 관리 가능(이미 표현 파이프라인의 일부). 모든 제약을 표현에 욱여넣지 말고, **드물거나 미지인 제약은 2~4로** 처리 → 표현 단순성 유지.
- 원칙: **invalid를 "생성 후 거르기"보다 "생성 단계에서 막기"가 1순위**, repair는 루프 한정 허용, penalty는 미지 제약용.

## 5. 전문가 지식 주입 경로 (추후 단계)
- **Gryffin**: 범주형 BO에 **전문가 지식(사전분포)** 을 주입하도록 설계된 알고리즘 → 도메인지식 단계 참고.
- 일반 경로: 사전분포(prior), 커스텀 feature(그룹 합성변수), 커널 설계, 제약, 탐색공간 축소.

## 본 문제 권고
1. **배타·그룹 → 범주형 재파라미터화**(차원·불가능조합 동시 해결)를 기본.
2. **조건부가 많으면 SMAC3**, **상호작용 해석이 중요하면 BOCS형**.
3. baseline은 제약 없이도 돌게 만들되, **위 슬롯(범주화/조건부/feasibility)** 을 함수 경계로 분리해 추후 규칙 주입이 쉽게.

## Sources
- 조건부 DAG/계층공간: [Hyperparameter Optimization: Foundations, Algorithms, Best Practices (조건부=parent/child, 트리/DAG)](https://arxiv.org/pdf/2107.05847) · [Conditional PED-ANOVA: Hierarchical & Dynamic Search Spaces](https://arxiv.org/pdf/2601.20800) · [Auto-WEKA (조건부 계층 탐색공간)](https://arxiv.org/pdf/1208.3719)
- [Dealing with Categorical and Integer Variables in BO with GPs (Garrido-Merchán & Hernández-Lobato)](https://arxiv.org/abs/1805.03463)
- [BO for Categorical and Category-Specific Continuous Inputs (AAAI)](https://arxiv.org/pdf/1911.12473)
- [BO over Multiple Continuous and Categorical Inputs (CoCaBO)](https://meta-learn.github.io/2019/papers/metalearn2019-ru.pdf)
- [Gryffin: BO of categorical variables informed by expert knowledge](https://arxiv.org/pdf/2003.12127)
- [Combinatorial BO with Random Mapping to Convex Polytopes](https://arxiv.org/pdf/2011.13094)
- [SMAC3 (ConfigSpace, conditional params)](https://www.researchgate.net/publication/354766153_SMAC3_A_Versatile_Bayesian_Optimization_Package_for_Hyperparameter_Optimization)
