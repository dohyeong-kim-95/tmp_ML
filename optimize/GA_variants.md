# GA(유전 알고리즘) 갈래 리서치 — 우리 문제 맞춤

> 문제 맥락: 혼합 이산 40변수(binary 32/ordinal 4/categorical 4), 교호작용 2~4차,
> 다봉성(Case3 등), 목적 = ΣY 최대화. "어떤 GA가 있고, 무엇을 다음에 시도할까".

## 1. GA는 4~5개 '축'의 조합으로 갈린다

### (A) 인코딩(표현)
| 종류 | 설명 | 우리 문제 적합성 |
|---|---|---|
| Binary-coded GA | 고전 Holland, 비트열 | binary엔 맞으나 ordinal/categorical은 비효율 |
| Real-coded GA (RCGA) | 실수 벡터, SBX/BLX 교차 | 연속용 → 부적합 |
| **Integer/Categorical GA** | 정수·범주 유전자 | **우리가 쓰는 방식. 가장 자연스러움** |
| Permutation GA | 순열(PMX/OX/CX) | 순서문제(TSP)용 → 무관 |

### (B) 선택(selection)
roulette(적합도비례) · **tournament**(우리 사용) · rank-based · SUS · truncation · Boltzmann.
→ tournament는 스케일 불변·견고. 무난한 기본값.

### (C) 교차(crossover)
1-point · 2-point · k-point · **uniform**(우리 사용) · 실수전용(SBX, BLX-α, arithmetic).
→ 변수간 위치 상관이 없으면 uniform이 적절(우리 경우 맞음).

### (D) 대치/생존(replacement)
| 종류 | 설명 |
|---|---|
| Generational | 세대 전체 교체 |
| **Steady-state (SSGA)** | 매 스텝 1~2개만 교체 → 빠른 수렴, 다양성 관리 중요 |
| Elitism / (μ+λ),(μ,λ) | 최상위 보존 (우리: elitism 1개) |

### (E) 다양성/니칭(다봉 대응)
fitness sharing · crowding · **clearing** · **Restricted Tournament Selection(RTS)**.
→ 우리 복잡도 분석에서 **다봉성이 난이도 핵심**(Case3)이었으므로 중요한 축.

## 2. 이름 붙은 주요 GA 계열

| 알고리즘 | 핵심 아이디어 | 강점 |
|---|---|---|
| **Simple GA (SGA)** | 고전 generational+roulette | 베이스라인 |
| **Steady-State GA** | 1개씩 교체 | 빠른 수렴, 적은 메모리 |
| **CHC** (Eshelman) | 엘리트 교차세대 선택 + 이질재조합(HUX) + cataclysmic 재시작 | **작은 예산·이진문제에 강하고 견고** |
| **Micro-GA (μGA)** | 초소형 pop + 수렴시 재시작 | 빠른 탐색 |
| **Memetic/Hybrid GA** | GA + **국소탐색**(Lamarckian) | **국소최적 정밀화 — 매우 강력** |
| **Island model** | 여러 부분개체군 + 이주(migration) | 병렬·다양성 |
| **Cellular GA** | 격자 공간구조 교배 | 다양성 보존 |
| **Adaptive GA** | pm/pc 자기적응 | 튜닝 부담↓ |
| **NSGA-II / SPEA2 / MOEA/D** | 다목적 Pareto | **Y 4개 개별 최적화 시 정답** |

## 3. ★ 교호작용을 '학습'하는 GA 계열 (우리 문제에 특히 유망)

일반 GA의 uniform crossover는 변수간 **linkage(연관구조)를 모름** → 교호작용을
깨뜨릴 수 있다. 이를 명시적으로 학습·활용하는 계열:

| 알고리즘 | 메커니즘 |
|---|---|
| **EDA 계열** (UMDA, PBIL, **cGA**, **BOA/hBOA**) | 교차 대신 **확률모델**을 세워 샘플링. BOA는 베이지안망으로 변수 의존성 포착 |
| **Linkage-learning** (mGA, **LT-GOMEA**, **DSMGA-II**, **CGOMEA**) | DSM/linkage-tree로 **변수 묶음(building block)**을 찾아 통째로 교환(optimal mixing) |

벤치마크상 DSMGA-II·GOMEA 계열은 trap/NK-landscape/Ising/MAX-SAT 등
**교호작용·중첩 구조 문제에서 일반 GA를 큰 차이로 능가**(함수평가 횟수 기준).
우리 Case3~6(고차 교호작용)이 정확히 이 구조에 해당.

## 4. 우리 문제용 추천 (다음 실험 후보)

현재 우리 GA = "tournament + uniform crossover + elitism + 정수/범주 인코딩"
(generational에 가까움). 개선 유망 순위:

1. **Memetic GA** (GA + 좌표상승 국소탐색)
   - 우리 좌표상승만으로도 거의 전역최적 → GA의 탐색 + 국소탐색 결합이면 가장 안정적·강력할 가능성. 구현 쉬움.
2. **DSMGA-II / GOMEA (linkage-learning)**
   - 교호작용을 명시적으로 학습 → Case3~6에서 평가효율 크게 개선 기대. 구현 복잡도 높음.
3. **CHC**
   - 작은 예산(우리 2000)에서 견고. diversity 재시작으로 다봉(Case3) 대응. 구현 중간.
4. **RTS/crowding 니칭 추가**
   - 기존 GA에 니칭만 얹어 다봉 Case3 trap 회피. 저비용.
5. **NSGA-II** (방향 전환 시)
   - ΣY 단일 스칼라 대신 **Y4개 Pareto front**를 원하면 정답.

## 참고
- 변형 개관: Variations of Genetic Algorithms (arXiv:1911.00490)
- Linkage/Optimal Mixing: DSMGA-II (arXiv:1807.11669), Parameterless GOMEA (arXiv:2109.05259)
