# 발표용 요약 — 저예산 혼합이산 최적화, 알고리즘 리서치

> 범위: BM3(가장 어려운 현실적 칸) × 예산 {180, 780, 2400} × scalarization {sum, chebyshev, owa},
> **각 셀 10 seed**. closure = (best_true − floor)/(ref_opt − floor), 0=무작위 평균, 1=참조최적.
> 성능 = 방문점의 **참(노이즈 없는) 점수 누적최댓값**(추천정책 무관, 공정).
> **reference = block_coord_local@20000**(global maxima; 좌표상승·GA-200k보다 타이트한 천장).
> 풀: SF(=random/sobol/mlhs 중 best) / block_coord_local / SA / GA / PSO / ACO
> (메타휴리스틱은 flat·blk 중 per-cell best). BO(tpe/smac)는 이번 라운드 제외.

---

## Lesson learned (핵심 교훈)

1. **저예산(180~780)의 본질 = "등반 속도 한계"이지 "교호 한계"가 아니다.** 주효과가 교호를
   지배하면(effect sparsity) 달성 가능한 최적값 대부분이 주효과를 그리디하게 맞추는 것만으로
   도달 가능하고, 교호의 잔여 이득이 작아 그걸 모델링/탐색하는 기법(BO surrogate, GA/SA의
   모집단·온도)은 탐색세만 낸다. **GA에 200,000회를 줘도 좌표상승을 못 이긴다** = 이 함수클래스가
   본질적으로 좌표-친화적(교호를 "못 봐서"가 아니라 "최적이 애초에 좌표-도달가능"해서).
2. **그래서 저예산에선 "블록 구조 + 그리디 좌표 local search"가 GA/SA/PSO/ACO/BO를 앞선다.**
   블록 지식을 모두에게 줘도(공정비교) block_coord_local이 전 예산·전 지표 1위(평균 82%, worst로도
   1위). 동력은 블록 지식 자체가 아니라 **블록 + 매 평가를 단조 전진으로 바꾸는 그리디 좌표 효율**.
3. **예산이 커지면 격차는 줄지만(이번엔 역전 아님), 더 큰 예산에선 역전 가능.** 2400에서 우위
   +17pp→+4~6pp로 축소. 도전자 순서도 바뀐다(PSO 저예산↑·고예산↓, **ACO 고예산↑** = 범주형
   페로몬 학습). >2400 예산이면 ACO류가 좌표법을 위협할 수 있다.
4. **고예산을 막는 건 BO가 아니라 예산 자체다.** 실문제(60s/eval)에선 RF-BO(SMAC) 오버헤드는
   무시 가능 → 2400 소화 가능. 단 **GP-BO는 surrogate O(n³)** 라 고예산에서 그 자체로 불리.
   2400=40h라 한계는 옵티마이저가 아니라 wall-clock 예산.
5. **(전이 조건) 이 결론은 "주효과 지배(effect sparsity)" 가정 위에서만 실문제로 전이된다.**
   SI 물리(전송선/EQ/임피던스는 매끈·effect-sparse, XOR/parity 기만은 희소)가 이 가정과 일치 →
   전이 기대 가능. 실문제에서 교호가 주효과를 *압도*하면 좌표법이 깨지고 그땐 ACO/BO가 답.

---

## 한 장 요약 (말할 것)

### 1. block_coord_local이 모든 예산·모든 지표에서 1위 — 운이 아니다
종합 평균 closure **82.1%**, worst-seed 평균 **69.5%** (2위 ACO 69.8%/58.2% 대비 +12pp).
**전 9셀(3예산×3지표)에서 평균·worst 모두 1위(★◆).** 10 seed라 "최악 seed로 평가해도 1위".

### 2. 우위는 *블록을 알아서*가 아니라 **저예산 sample-efficiency + 함수구조 적합** 때문
- 블록 지식은 SA/GA/PSO/ACO에도 `_blk`로 동일하게 줬다(공정). 그래도 block_coord_local이 이긴다.
- **예산이 클수록 격차가 줄어든다**: block_coord_local 우위 = **@180 +17pp → @2400 +4~6pp**.
  → "좌표법의 이점은 *저예산(=당신의 180~780 영역)에서 가장 크다*. 고예산에선 메타휴리스틱이 따라온다."

### 3. budget × 방법 **crossover** (← "2400에서 순위 바뀌나?"의 답: 그렇다)
예산별 메타휴리스틱 랭킹(전 지표 평균):
| 예산 | 1위(챔피언) | 메타휴리스틱 순위 |
|---|---|---|
| @180 | block_coord_local (71%) | **PSO**(54) > GA(51) > ACO(50) > SA(49) |
| @780 | block_coord_local (85%) | **ACO**(74) > SA(72) > GA(71) > PSO(71) |
| @2400 | block_coord_local (90%) | **ACO**(86) > SA(83) > GA/PSO(80) |

- **PSO**: 저예산 최강 → 고예산 꼴찌로 추락(범주형에 *가짜순서* 강제하는 약점이 후반에 누적).
- **ACO**: 저예산 약함 → 고예산 최강(레벨별 페로몬 = 범주형 친화, 예산 줄수록 학습). cheby@2400에선
  worst-case 1위(◆, 86%)로 block_coord_local(88%)에 근접.
- 시사: **예산이 정말 크면(>2400) ACO류가 좌표법을 위협**할 수 있으나, 우리 목표 예산(180~780)에선 block_coord_local이 확고.

### 4. 초기 space-filling 단독은 무의미
SF(=random/sobol/mlhs 최강)가 **전 구간 꼴찌(48%)**. 이득은 초기설계가 아니라 **탐색 연산자 + 블록 구조**.

---

## "overfit 아니냐"에 대한 정면 반박 (Q&A 대비)
- **벤치마크를 깨려고 만든 비분리/다봉/고밀도 교호에서도 좌표-계열이 이긴다.** 더 나아가:
- **GA에 200,000 evals를 줘도 다중시작 좌표상승을 못 이긴다**(BM3·BM4 전 kind에서 coord 승;
  GA-200k ≈ block_coord_local@780). → 이건 알고리즘 overfit이 아니라 **함수클래스가 본질적으로
  좌표-친화적**(functional-ANOVA: 주효과 지배 + 저차 교호)이라는 구조적 사실.
- **그 구조는 SI 도메인 물리와 일치**(전송선/EQ/임피던스는 매끈·effect-sparse; XOR/parity류 기만은
  아날로그 SI에 거의 없음). → 실문제가 *주효과 지배(effect sparsity)* 가정만 만족하면 **결과가 transfer**.
- 유일한 리스크: 실문제에서 교호가 주효과를 *압도*하면 좌표법이 깨질 수 있음(그땐 ACO/BO 검토).

## 권장 (실문제 적용)
- **1순위 baseline = block_coord_local** (블록-인지 좌표탐색 + random-restart + 평가 캐시).
- 배포 점수 = **owa 또는 chebyshev**(한 지표 폭락 방지; 둘 다에서 1위).
- 예산 분할: 탐색 + 끝단 **confirmation**(top-3 각 4~5회 재측정).
- 고예산(>2400) 여유가 생기면 **ACO**를 보조 후보로(후반 강세).

## 그림 (optim/figs/)
- `pool_{180,780,2400}.png` — 6항목 풀 × BM3, 오차막대=seed min~max, 막대위=선택변형. **crossover를 예산별로**.
- `block_lift_{sum,owa}.png` — 공정비교: flat→+block, block_coord_local 천장선(이번엔 pso/aco 포함).
- 전체 수치표: `optim/RESULTS_pool.md` (평균±표준편차·worst·선택변형).

## 한계 / 다음
- BO(tpe/smac) 미포함(다음 라운드, "GP/RF로도 못 이긴다" 보강용).
- BM4는 *BM3보다 쉬워서*(교호 늘려도 좌표가 안 막힘) 이번 stress 라운드에서 제외. 진짜 더 어려운
  현실적 칸이 필요하면 XOR가 아니라 **categorical regime-switch**(토폴로지가 최적설정을 바꾸는 구조)로.
