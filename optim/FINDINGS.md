# 발표용 요약 — 저예산 혼합이산 최적화, 알고리즘 리서치

---

## ⚠ 2026-07 방법론 수정(Fable_feedback A-tier) 및 재검증 — 이 섹션이 최신

아래 기존 본문/표/그림은 **수정 전 방법론**(좌표-전용 천장, 교호 포함 노이즈)
기준이다. 수정 전 상태는 커밋 `c02eae1`(로컬 태그 pre-methodology-fix)에 보존.
재검증 상세 표는 `RESULTS_v2.md`, 그림은 `figs/closure_v2_*.png`,
원데이터는 `results_v2.json`(이번 라운드부터 커밋, 피드백 C3).

### 무엇을 고쳤나

1. **A1 — 천장의 순환성**: ref_opt 를 좌표 계열(좌표상승+block_coord)만으로 정하면
   비분리 BM에서 천장이 과소평가되고, 그 편향이 챔피언(block_coord_local)과 같은
   방향이라 closure 비교가 순환적이었다. → `reference_ceiling` = **좌표 2종 + SA
   (mixed-move) + GA-lite(재조합) 앙상블의 max** 로 재정의, 탐색기별 값·편차를
   artifact 에 기록.
2. **A2 — 노이즈 정의**: 문서는 "주효과 스프레드의 noise_frac"인데 구현은 교호
   포함 전체 스프레드 기준이었다. → 주효과 기준 닫힌형(√ΣVar(tab))으로 수정.
3. **A3 — 정규화 clip**: knob 로 노출(기본 유지), 포화율 리포트 추가.
4. **A4 — artifact 정합성**: config 해시+fingerprint 를 저장·검증, stale artifact
   는 run.py 가 명확한 에러로 중단.

### 수정이 실측으로 확인해 준 것

**천장(ref_opt) 전후** — 비분리 칸일수록 비좌표 탐색기가 천장을 올렸다(=과소평가 실증):

| 칸 | 구(좌표-전용) | 신(앙상블) | 승자 | 비고 |
|---|---|---|---|---|
| BM4 chebyshev | -0.323 | **-0.296** | GA | 가장 큰 보정 — closure>1 나오던 칸 |
| BM4 owa | 0.710 | **0.731** | SA | |
| BM3 chebyshev | -0.350 | **-0.349** | SA | |
| BM1 (전 kind) | = | = | coord | 4개 탐색기 전원 합의(spread≈0) — 가법 BM 검산 |

**노이즈 전후**(목적 평균 noise_scale) — 교호가 강할수록 실효 노이즈가 의도보다
크게 부풀어 있었다:

| BM | 구(전체 스프레드 기준) | 신(주효과 기준) | 의도(noise_frac) |
|---|---|---|---|
| BM1 | 0.068 | 0.068 (불변) | 3% |
| BM2 | 0.281 | 0.108 | 5% |
| BM3 | 1.050 | 0.195 | 8% |
| BM4 | 0.790 | 0.169 | 6% |

**포화(A3)**: 랜덤 20k 표본에서 z-clip 포화율 ≤0.01% (전 BM) — clip 이 실질적으로
정보를 가리지 않음. 참조최적점의 z 상단 포화 여부는 artifact 에 기록됨.

### 재검증 결과 (random/mlhs/block_coord_local/tpe/tpe_blk × BM1~4 × 3 kind × {180,780} × 5 seed)

1. **closure>100% 셀이 소멸** — 60개 셀 어디에서도 천장 초과 없음(수정 전 BM4에서
   발생하던 현상). run.py 의 누수 경고도 미발동.
2. **핵심 결론 유지: block_coord_local 1위.** 24개 칼럼(BM×kind×budget) **전부에서
   평균·worst 동시 1위(★◆)**. 종합 79.5%(worst 70.9%) vs 2위 tpe_blk 65.8%(56.4%).
   새(공정한) 천장·새(더 작은) 노이즈에서도 순위가 뒤집히지 않음 — 오히려 노이즈가
   정상화되자 그리디 좌표 수용판정의 오염이 줄어 격차가 안정적.
3. **"블록이 레버" 결론도 유지**: tpe_blk ≥ tpe (특히 sum@780: BM2 81% vs 67%,
   BM3 73% vs 63%).
4. **기존 세부 주장 중 하나는 이번 축소 풀에서 재현되지 않음**: "owa/cheby 일부
   칸에서 tpe_blk 가 block_coord_local 추월"은 새 방법론·5 seed 에서 관찰되지 않았다
   (전 칸 block_coord_local 우위). 구 결과는 부풀려진 노이즈(좌표 수용판정에 더
   불리)와 구 천장의 조합 위에서의 관찰이었으므로, owa/cheby 하이브리드(열린 다음
   단계 (a))의 우선순위는 낮춰 잡는 게 맞다.
5. **주의**: 이번 재검증은 축소 풀(5개)이다. SA/GA/PSO/ACO/SMAC/GP-BO 의 전체 풀
   재실행과 2400 예산은 새 방법론 기준으로 미실행(구 표의 해당 칸은 구 방법론 기준).

---

> 범위: BM3(가장 어려운 현실적 칸) × 예산 {180, 780, 2400} × scalarization {sum, chebyshev, owa},
> **각 셀 10 seed**. closure = (best_true − floor)/(ref_opt − floor), 0=무작위 평균, 1=참조최적.
> 성능 = 방문점의 **참(노이즈 없는) 점수 누적최댓값**(추천정책 무관, 공정).
> **reference = block_coord_local@20000**(global maxima; 좌표상승·GA-200k보다 타이트한 천장).
> 풀: SF(=random/sobol/mlhs 중 best) / block_coord_local / SA / GA / PSO / ACO / TPE
> (메타휴리스틱은 flat·blk 중 per-cell best; **공정성 보정 적용** — 아래 표). SMAC/GP-BO는 다음 라운드.

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
3. **예산이 커져도 역전 없음(TPE 포함 검증). 단 격차는 축소.** 2400에서도 block_coord_local이
   전 kind 1위 유지(+17pp@180 → +4pp@2400로 축소). 도전자 순서만 바뀐다(PSO 저예산↑, **ACO/GA
   고예산↑**, TPE 저예산↑·고예산↓). ⚠️ **단일 seed에선 TPE가 sum@2400에서 추월하는 듯 보였으나
   (95.7%) 10-seed로 73.8%로 소멸** — "seed 편차 > 알고리즘 격차"의 실증. >2400이면 ACO류가
   위협할 수 있으나 그건 외삽.
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

### 3. budget × 방법 **crossover** (← "2400에서 순위 바뀌나?"의 답: 챔피언은 안 바뀜, 도전자는 바뀜)
예산별 풀 랭킹(전 지표 평균, 공정성 보정·TPE 포함):
| 예산 | 1위(챔피언) | 도전자 순위 |
|---|---|---|
| @180 | block_coord_local (71%) | **PSO**(55) > TPE(54) > ACO(50) > GA(48) > SA(48) |
| @780 | block_coord_local (85%) | **ACO**(74) > TPE(73) > PSO(72) > GA(71) > SA(70) |
| @2400 | block_coord_local (90%) | **ACO**(86) > GA(84) > SA(83) > PSO(82) > TPE(81) |

- **block_coord_local**: 전 예산 1위 불변(역전 없음).
- **PSO/TPE**: 저예산 강세 → 고예산 상대적 둔화. **ACO/GA**: 저예산 약함 → 고예산 강세.
- **종합 2위 다툼이 빽빽**(ACO 69.8 / PSO 69.7 / TPE 69.3 / GA 67.9 / SA 67.3) → 2위는 사실상 동률,
  여기선 seed·예산에 따라 순위가 뒤섞임. **확실한 건 1위(block_coord_local)와 꼴찌(SF)뿐.**
- 시사: 목표 예산(180~780)에선 block_coord_local 확고. >2400이면 ACO류가 위협 가능(외삽).

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

## 비교 공정성 (cat/ord 처리) — apples-to-apples 보정
"GA/SA를 불구로 만들고 이겼다"는 반론을 막기 위해, 각 알고리즘이 categorical(순서X)/
ordinal(순서O)을 어떻게 다루는지 통일 점검하고 핸디캡을 제거했다:

| 알고리즘 | categorical | ordinal | 비고 |
|---|---|---|---|
| random/sobol/mlhs | 레벨 직접 | 레벨 직접 | 정직(sobol만 약한 floor-순서) |
| block_coord_local | 레벨 전수비교 | 레벨 전수비교 | 가짜순서 없음 |
| TPE/SMAC | `Categorical` | `Integer` | native 구분 |
| **GA** (보정) | Choice(이산 연산자) | Integer(정수 연산자) | **SBX/PM→MixedVariableGA**(categorical 가짜보간 제거) |
| **SA** (보정) | random-reset | ±1 local step | **warm-up best서 시작**(이전엔 버림) + cat/ord 이동 구분 |
| **PSO** (보정) | 레벨별 logit+softmax | integer velocity+round | **run_pso_mixed 신규**(연속완화 가짜순서 제거); 풀엔 mixed/continuous 중 best |
| ACO | 레벨별 페로몬 | (순서 미활용) | categorical 친화 |

→ 보정 후 **GA 64.8%→67.9%, SA 65.0%→67.3%, PSO 68.2%→69.7%** 상승(격차 축소 = 공정해짐).
그럼에도 **block_coord_local 1위는 불변** → "핸디캡 덕"이 아니라 진짜 우위임이 공정하게 확인됨.

## 권장 (실문제 적용)
- **1순위 baseline = block_coord_local** (블록-인지 좌표탐색 + random-restart + 평가 캐시).
- 배포 점수 = **owa 또는 chebyshev**(한 지표 폭락 방지; 둘 다에서 1위).
- 예산 분할: 탐색 + 끝단 **confirmation**(top-3 각 4~5회 재측정).
- 고예산(>2400) 여유가 생기면 **ACO**를 보조 후보로(후반 강세).

## 그림 (optim/figs/)
- **`pool_budgets_BM3.png`** (메인) — 7항목 풀 × 예산(초록180/노랑780/빨강2400), 오차막대=seed min~max,
  막대위=선택변형. budget×방법 crossover를 한 그림에.
- `pool_{180,780,2400}.png` — 예산별 7항목 풀(BM별 막대).
- `block_lift_{sum,owa}.png` — 공정비교: flat→+block, block_coord_local 천장선(pso/aco 포함).
- 전체 수치표: `optim/RESULTS_pool.md` (평균±표준편차·worst·선택변형).

## 한계 / 다음
- TPE 포함 완료. **SMAC(RF-BO)·GP-BO 미포함**(다음 라운드; SMAC은 2400 가능, GP-BO는 O(n³)로 고예산 불리).
- BM4는 *BM3보다 쉬워서*(교호 늘려도 좌표가 안 막힘) 이번 stress 라운드에서 제외. 진짜 더 어려운
  현실적 칸이 필요하면 XOR가 아니라 **categorical regime-switch**(토폴로지가 최적설정을 바꾸는 구조)로.
- screening(RF permutation importance/stepwise): 별도 라운드에서 "구조 복원 + 예산-차감 이득"으로 검증 예정.
