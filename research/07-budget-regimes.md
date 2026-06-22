# 07 — 예산 체급(regime) 변화: 비용 대폭 인하 시나리오

> 트리거: 생산성이 **40개/시간 × 8시간 = 320개/일** 로 상향(기존 ~40개/일의 8배).

## 1. 예산 재계산
| 가정 | 일 생산 | 15주 누적(주5일) | 15주 누적(주7일) |
|------|---------|-------------------|-------------------|
| 기존 | ~40 | ~3,000 (O(10³)) | ~4,200 |
| **신규** | **320** | **~24,000 (O(10⁴))** | **~33,600** |
| 야간/연장 시 | 더 큼 | → O(10⁵) 근접 가능 | |

→ 누적 예산 **O(10³) → O(10⁴~10⁵)**, **10~30배**.
(여전히 2⁶⁰≈10¹⁸의 극히 일부(10⁻¹³)라 전수는 불가. 하지만 "방법의 선택지"는 완전히 달라짐.)

## 2. regime이 바뀐다 — 핵심
| 항목 | 저예산 O(10³) | **중예산 O(10⁴~10⁵)** |
|------|----------------|------------------------|
| 지배 지표 | 표본효율(거의 전부) | 표본효율 + **탐색 알고리즘 효율** 균형 |
| 지도학습 대리모델 | 데이터 부족 | **수만 라벨 → 강한 예측모델 학습 가능** |
| 진화/메타휴리스틱 | 평가 예산 부족 | **수만 평가 굴릴 수 있음 → 본령 진입** |
| 권고 주력 | BO(BOCS형/SMAC/TPE) | **모델기반 진화알고리즘 + (선택)대리모델 보조** |
| N>60 binary | BO엔 부담 | **연관학습 EA의 정확한 타깃** |

## 3. 중예산에서 떠오르는 방법

### (A) 모델기반/연관학습 진화알고리즘 ★ 본 구조에 최적 적합
binary + "변수 묶음(building block)" + "큰 상호작용"은 이 계열이 **직접 겨냥**하는 문제다.
변수들 사이의 **연관구조(linkage)를 데이터로 학습**해 그 묶음 단위로 교배 → 구조를 깨지 않고 탐색.

- **DSMGA-II**: 쌍별 연관탐지를 **Dependency Structure Matrix(DSM)** 에 저장, 클러스터링으로 **building block** 추출 후 mixing. trap/NK-landscape/Ising/MAX-SAT에서 **LT-GOMEA·hBOA보다 적은 평가**로 우수.
- **LT-GOMEA / LTGA**: 매 세대 **linkage tree**(계층적 클러스터링)로 변수 묶음 학습 후 optimal mixing.
- **hBOA (Hierarchical BOA, EDA계열)**: 선택된 해로 **베이지안 네트워크**를 학습해 변수 의존성 모델링 후 샘플링. 거의 분해가능(nearly-decomposable) 문제를 약 **O(n^1.55 log n) ~ O(n²) 평가**로 해결. n=60이면 **수천 평가 규모** → O(10⁴) 예산 안에서 여유.

> ⭐ 부가가치: 이 알고리즘들이 **학습한 linkage/DSM/베이지안 네트워크 = "어떤 binary가 묶이고 상호작용하는가"의 데이터 기반 추정**. goal.md의 "구조는 추후 도메인지식으로 주입" 과 **양방향으로 맞물림**(모델이 구조를 제안 → 전문가가 검증/보강).

### (B) 지도학습 대리모델 + 오프라인 최적화 (이제 가능)
- 수만 개 (X,Y)로 **GBM/RandomForest/NN** 학습 → Y=f(X) 예측모델.
- 모델 위에서 대규모 탐색(EA/정수계획/local search)으로 유망 후보 다수 생성 → **실설비로 검증 배치**.
- N=60 binary + 수만 샘플이면 트리부스팅이 상호작용을 잘 포착. SHAP 등으로 변수중요도·상호작용 해석.

### (C) 대리모델 보조 진화 (SAEA) — (A)+(B) 결합
- EA가 후보를 많이 생성 → **대리모델로 사전선별(triage)** → 상위 320개만 실설비 측정(=하루 배치) → 데이터 누적·모델 갱신.
- "각 실측을 최대한 값지게" + "하루 320 배치"와 자연스럽게 부합. 중·대규모 expensive 최적화의 표준 패턴.

## 4. 중예산 권고(요약)
1. **주력: 모델기반 연관학습 EA (DSMGA-II 또는 LT-GOMEA)** — N>60 binary·구조·중예산에 가장 부합, 구조까지 부산물로 추정.
2. **보조: 지도학습 대리모델(GBM)** 로 (a) 후보 사전선별(SAEA), (b) 변수중요도/상호작용 해석.
3. **배치(320/일)**: EA 세대 크기를 배치에 맞추거나, 대리모델로 상위 320 선별해 실측.
4. BO(BOCS/SMAC/TPE)는 **초기 데이터가 적은 첫 1~2주의 워밍업** 또는 비교 baseline으로 잔존.

## 5. 단, 주의할 점
- 여전히 **noisy expensive 평가** → EA에 노이즈 견딘 선택(쌍별 연관탐지가 노이즈에 강함), 반복측정 일부 필요.
- 320/일이라도 **2⁶⁰는 못 훑음** → "구조 활용"이 본질. 구조를 못 쓰면 중예산도 부족.
- hBOA/DSMGA-II는 **near-decomposable 가정**에서 강함. 구조가 매우 얽히면(deep epistasis) 평가 수 급증.

## 6. 의사결정 포인트 (사용자 확인 필요)
- 이 **320/일이 실제 계획**이면 → goal.md의 제약·접근(주력 방법)을 **모델기반 EA 중심으로 개정**해야 함.
- 단순 가정 탐색이면 → 본 문서를 시나리오로 보관, 기존 BO 권고 유지.

## Sources
- [Surrogate-assisted EAs for expensive combinatorial optimization: a survey](https://link.springer.com/article/10.1007/s40747-024-01465-5)
- [A survey of surrogate-assisted evolutionary algorithms for expensive optimization](https://link.springer.com/article/10.1007/s41965-024-00165-w)
- [SAEA for Medium-Scale Expensive Multi-Objective Problems (≤50 vars)](https://arxiv.org/pdf/2002.03150)
- [DSMGA-II: Pairwise Linkage Detection, Incremental Linkage Set, Back Mixing](https://arxiv.org/pdf/1807.11669)
- [The Linkage Tree Genetic Algorithm (LTGA)](https://www.researchgate.net/publication/226752922_The_Linkage_Tree_Genetic_Algorithm)
- [Linkage Neighbors, Optimal Mixing (GOMEA)](https://www.researchgate.net/publication/254461772_Linkage_Neighbors_Optimal_Mixing_and_Forced_Improvements_in_Genetic_Algorithms)
- [Two-edge graphical linkage model for DSMGA-II](https://www.researchgate.net/publication/318067716_Two-edge_graphical_linkage_model_for_DSMGA-II)
- [Hierarchical BOA (hBOA)](https://link.springer.com/content/pdf/10.1007/978-3-540-34954-9_4.pdf) · [Parameter-less hierarchical BOA](https://arxiv.org/pdf/cs/0402031) · [BOA overview](https://cleveralgorithms.com/nature-inspired/probabilistic/boa.html)
