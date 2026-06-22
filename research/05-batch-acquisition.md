# 05 — 배치(q≈40) 제안 전략

> 본 문제는 **하루 ~40점을 한 번에** 측정한다(설비 배치). 순차 BO가 아니라 **batch BO** 필요.
> 핵심 난점: 한 배치 안의 40점이 **서로 몰리지 않고**(다양성) 동시에 **유망**해야 함.

## 후보 전략

### q-EI (joint q-batch Expected Improvement)
- 시퀀셜 EI를 q점 동시 선택으로 일반화. 이론적으로 정석.
- **한계: q>4면 결합 최적화가 사실상 비현실적** → q=40에는 그대로 쓰기 어려움.
- BoTorch는 q-batch를 **joint**(한 번에) vs **sequential greedy**(이전 후보 조건부로 하나씩) 두 방식 제공. q 클수록 sequential이 더 강건.

### Local Penalization (LP)
- 이미 고른 점 주변에서 acquisition을 **순차적으로 페널티** → 배치 다양성 확보. Lipschitz 상수로 페널티 반경 추정.
- q가 커도 동작, 구현 단순. **q≈40에 실용적.**

### Batch Thompson Sampling (TS) ★ 본 문제 적합
- 사후분포에서 **여러 함수 표본**을 뽑아("전문가 패널"), 각 표본의 최적점을 모아 배치 구성.
- **자연스럽게 q개 병렬**, 다양성 내장, q가 커도 확장 양호 → **q≈40에 가장 매끄러움**.
- BOCS형(사후 선형회귀 표본)·GP 모두에 적용 가능. 조합공간에선 각 표본을 SA/정수계획으로 최적화.

### 비동기(asynchronous) 변형
- 일부 결과가 먼저 나오는 환경이면 비동기 TS/ε-greedy가 시간제약 하에서 더 낮은 regret.
- 본 문제는 "하루 40개 동기 배치"에 가까우므로 1순위는 동기 배치, 운영상 결과가 산발 도착하면 비동기 고려.

## 본 문제 권고
1. **1순위: Batch Thompson Sampling** — q≈40에 가장 적합, 대리모델 무관(BOCS형/RF/GP 다 가능).
2. **대안: Local Penalization** — GP-EI 계열을 쓸 때 배치 다양성 확보.
3. q-EI joint는 q가 작을 때만(여기선 부적합). BoTorch 쓰면 **sequential greedy q-batch**로.
4. **반복시간 주의**: 배치 1회 = (대리모델 적합) + (q개 후보를 위한 획득 최적화 q회). TS면 표본 q개 각각 SA → **병렬화** 가능하므로 wall-clock 관리 용이.

## 운영 팁 (15주 루프)
- 매일: ① 누적 데이터로 대리모델 적합 → ② 배치 TS로 40점 제안 → ③ 설비 측정 → ④ 데이터 누적.
- 초반 배치는 **탐색 비중↑**(다양성), 후반은 **활용 비중↑**(best 주변 정밀).
- 배치 일부(예: 5~10%)는 **순수 탐색/검증용**으로 고정 배정하면 모델 과신 방지.

## Sources
- [Sampling Acquisition Functions for Batch BO](https://www.researchgate.net/publication/331978415_Sampling_Acquisition_Functions_for_Batch_Bayesian_Optimization)
- [Asynchronous Batch BO with Improved Local Penalisation](https://ar5iv.labs.arxiv.org/html/1901.10452)
- [Parallel and Distributed Thompson Sampling](https://arxiv.org/pdf/1706.01825)
- [Asynchronous ε-Greedy Bayesian Optimisation](https://arxiv.org/pdf/2010.07615)
- [Batched BO with correlated candidate uncertainties](https://arxiv.org/html/2410.06333v1)
- [Efficient and Scalable Batch BO using K-Means](https://arxiv.org/pdf/1806.01159)
- [BoTorch q-batch optimization (joint vs sequential)](https://botorch.org/docs/tutorials/saasbo)
