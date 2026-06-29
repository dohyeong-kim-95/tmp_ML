# Benchmark (Phase 1) — 합성 black-box 생성기

실제 calculator(1 eval=1분)를 흉내 내되 **즉시 평가되고 참조 최적이 알려지는** 합성 문제.
최적화 알고리즘을 싸게 개발·검증하기 위한 BM1<BM2<BM3 난이도 ladder.

## 구조
- `generator.py` — `BlackBoxBenchmark`: X(정수 레벨 30개) → 6개 Y, 노이즈/정규화/3종 효용/참조최적
- `configs.py` — BM1/2/3 난이도 설정
- `build.py` — 인스턴스 생성 + 참조최적/정규화범위 산출 → `artifacts/<BM>.json`

## X 레이아웃 (BM 공통)
| 블록 | 컬럼 | 변량 | 영향 |
|---|---|---|---|
| 공통 | 0–9 (10) | ~1e6 | 6목적 전부 |
| set1 | 10–14 (5) | ~1e3 | y11,y12,y13 |
| set2 | 15–29 (15) | ~1e6 | y21,y22,y23 |

전체 ~1e15. set1 유효차원 15(쉬움) / set2 유효차원 25(어려움, 병목).

## Y 생성 방식 (functional-ANOVA)
```
g_m(X) = Σ_j a_{m,j}·φ_{m,j}(x_j)            # 주효과(영향변수별 효과)
       + Σ_{(j,k)} b·ψ(x_j,x_k)             # 2차 교호작용
       + (BM3) 3차 항
y_m = g_m + ε,  ε~N(0, (0.05·spread)²)
```
- φ: ordinal=매끄러운 곡선(성분수=봉우리수↔다봉성), categorical=레벨별 랜덤효과
- effect sparsity: 목적당 strong driver 소수 + weak 다수
- 충돌: 공통블록 base shape를 `conflict_rho`로 공유 → max/min 좋은방향 상충
- 정규화: 목적별 goodness(최소화는 부호반전)를 [0,1], 1=best

## 효용(scalarization) 3종
- `equal` — 정규화 동일가중 평균
- `chebyshev` — augmented Chebyshev(최악 성분 보호, ρ=0.01)
- `owa` — k-bottom OWA(하위 k=2 평균)

## 난이도 ladder
| | n_harmonics | interaction(density/strength) | 3-way | conflict_rho | noise_frac |
|---|---|---|---|---|---|
| BM1 | 1 | 없음 | 0 | 0.15 | 0.03 |
| BM2 | 4 | 0.22 / 0.8 | 0 | 0.55 | 0.05 |
| BM3 | 8 | 0.45 / 1.3 | 12 | 0.85 | 0.08 |

X 구조/크기는 셋이 동일 공유 → 난이도는 위 knob으로만 조절(비교 공정성).

**난이도 검증(예산제한 local-search gap-closure, equal, build.py):**
| BM | closure@180 | closure@2400 |
|---|---|---|
| BM1 | 100% (완전히 풀림) | 100% |
| BM2 | 89% | 96% |
| BM3 | 82% | 95% |

타이트한 예산(180)에서 100→89→82%로 단조 분리. BM3는 추가로 측정노이즈가
가장 크므로(noiseless probe는 과소평가) 노이즈 포함 실제 최적화에서 최난도다.
난이도 변별은 Phase 3에서 BO 등 실제 최적화기로 최종 검증한다.

## 사용
```python
from benchmark.generator import BlackBoxBenchmark
from benchmark import configs
bm = BlackBoxBenchmark(configs.BM1)
y  = bm.evaluate(X)              # 관측(노이즈 포함) — 최적화기가 보는 값
u  = bm.utility(X, "owa")        # 효용
x_star, u_star = bm.reference_optimum("owa")   # 참조 최적
```
빌드: `python -m benchmark.build`
