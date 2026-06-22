# 01 — 문제 프레이밍: 왜 베이지안 최적화(BO) 계열인가

## 1. 이 문제의 본질
- 입력: binary N>60 → 탐색공간 ≥ 2⁶⁰ ≈ 1.15×10¹⁸
- 평가: 생산설비 필요, **하루 ~40개** 배치, 15주 → 누적 **O(10³)** 표본
- 즉 **"점당 비용이 매우 큰 black-box 함수"** 를 **극소수 평가**로 최대화하는 문제.

이런 설정에서 핵심 성능지표는 정확도가 아니라 **표본 효율(sample efficiency)** —
*몇 번의 실제 측정으로 좋은 해에 도달하는가* 이다.

## 2. 후보 접근별 적합성

| 접근 | 평가 예산 가정 | 본 문제 적합성 |
|------|----------------|----------------|
| 전수/그리드 탐색 | 무한 | ❌ 10¹⁸ 공간, 불가능 |
| 랜덤 서치 | 큼 | ❌ 표본효율 너무 낮음 (예산 O(10³)) |
| 순수 지도학습(대량 라벨) | 大 데이터 선확보 | ❌ 데이터 0건·점당 비용 큼 → 전제 불성립 |
| 유전알고리즘(GA)/진화 | 수천~수만 평가 | ⚠️ 비싼 평가에 부적합(아래) |
| **베이지안 최적화(BO)/능동학습** | **수십~수백 평가** | ✅ **비싼 평가에 설계된 방법** |

## 3. BO가 표본효율이 높은 이유
- 관측으로 **대리모델(surrogate)** 을 만들고, **획득함수(acquisition)** 로 *다음에 어디를 측정할지* 능동적으로 결정.
- 탐색(exploration)과 활용(exploitation)을 명시적으로 균형 → 적은 평가로 전역 최적에 수렴.
- "비싼 물리 실험(며칠~몇 주, 고비용)에서 수천 번이 아니라 수십 번의 실험으로 near-optimal을 찾는다"고 보고됨.

## 4. BO vs 유전알고리즘 (설득 포인트)
- GA는 expensive 평가에서 비효율적 — 많은 세대×개체 평가 필요.
- 비교 연구 정리: 선택 기준은 **(1) 점당 비용 (2) 차원 (3) 목적함수 지형** 세 가지이며, 어느 쪽도 보편 우위는 아님.
  본 문제는 **점당 비용이 지배적**이라 BO가 유리.
- 다만 **surrogate-assisted EA**(대리모델로 적합도 근사)는 GA 예산을 **한 자릿수(order of magnitude) 절감** → 즉 "대리모델로 평가를 아끼는" BO의 아이디어가 GA에도 이식될 만큼 핵심.
- 주의: 표준 GP-BO는 통상 "연속 20~30차원까지 신뢰 가능"으로 언급됨 → **N>60 binary는 표준 GP-BO의 컴포트존 밖** → 고차원/조합 특화 변형(02 문서)이 필요.

## 5. 본 문제에 BO를 쓸 때 추가로 신경 쓸 점
1. **고차원 binary** → 표준 GP 커널·연속화로는 한계 → 조합 특화(BOCS/COMBO) 또는 트리 기반(SMAC) 필요.
2. **배치 평가(q≈40)** → 순차 BO가 아니라 **batch BO** 필요(05 문서).
3. **반복시간 제약** → HMC/SDP/대형 GP는 wall-clock 부담 → 경량 대리모델 선호(02·06).
4. **구조(상호작용) 미지** → 상호작용을 *발견*하는 모델이 추후 도메인지식화에 유리(BOCS형).

## Sources
- [A Tutorial on Bayesian Optimization of Expensive Cost Functions (Brochu et al.)](https://arxiv.org/pdf/1012.2599)
- [Bayesian optimization vs genetic algorithms for materials (PatSnap)](https://www.patsnap.com/resources/blog/articles/bayesian-optimization-vs-genetic-algorithms-for-materials/)
- [A comparison study between genetic algorithms and Bayesian optimization (ResearchGate)](https://www.researchgate.net/publication/220742974_A_comparison_study_between_genetic_algorithms_and_bayesian_optimize_algorithms_by_novel_indices)
- [How Bayesian Should Bayesian Optimisation Be?](https://arxiv.org/pdf/2105.00894)
