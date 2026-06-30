# AGENTS.md — 프로젝트 맥락 (세션 인수인계용)

## 목표 (한 문단)
이 프로젝트의 목표는, **30개 컬럼(categorical 10 + ordinal 20)으로 이루어진 약 10^15 크기의 혼합 이산 변수공간**에서, 비싼 black-box(`X → calculator → 6개의 Y`, 1회 평가에 약 1분)를 **180 / 780 / 2400 iteration(=3시간 / 13시간 / 40시간)의 작은 예산** 안에서 호출해 **최적의 Y를 내는 X**를 찾는 최적화 알고리즘을 **리서치**하는 것이다. 실제 calculator로는 알고리즘을 튜닝할 수 없으므로(평가가 비쌈), 우리는 실제 문제의 구조를 본뜨되 즉시·노이즈 포함으로 평가되고 정답(참조 최적)을 알 수 있는 **합성 black-box 벤치마크(BM1<BM2<BM3)** 를 만들고, 그 위에서 여러 옵티마이저를 **공정한 점수 체계**로 겨루게 한 뒤, 작은 예산에서 강한 **mixed-variable discrete optimization baseline**을 찾아 실제 문제에 적용하는 것을 지향한다. 핵심 관심사는 "**적은 iteration으로 얼마나 global 최적에 근접하는가**"이며, 도메인 지식(아래 블록 구조)을 활용한 baseline이 범용 옵티마이저(BO/EA 등)를 이기는지를 데이터로 검증해 왔다.

## 문제 정의 (확정됨)
- **X**: 30컬럼. cardinality는 변수마다 다름(2~30 level), 곱 ≈ 5.5×10^14~10^15. 무효 조합 없음.
- **블록 구조(도메인 지식, 활용 가능)**:
  - `common` 10컬럼 → **6목적 전부**에 영향
  - `set1` 5컬럼 → `y11,y12,y13` 에만
  - `set2` 15컬럼 → `y21,y22,y23` 에만
  - ⇒ `set1 ⫫ set2 | common` (공통블록을 통해서만 결합). max/min **충돌(trade-off)은 common 블록에 인코딩**.
  - set1 유효차원 15(쉬움), set2 유효차원 25(어려움 = 병목).
- **Y**: 6개. 최대화 `y11,y12,y21,y22` / 최소화 `y13,y23`. 노이즈 ≈ 주효과의 **5%**.
- **목표 형태**: Pareto 전체가 아니라 **단일 best 타협해 + 다양한 top-3 추천**.
- **평가 점수(scalarization) 3종** (min-max 정규화 후 1=best 방향 통일):
  - `sum` = 정규화 단순합, `chebyshev` = augmented Chebyshev(ρ=0.01), `owa` = bottom-k OWA(k=2).
  - 2·3번은 "한 지표만 폭락"을 막는 안전장치(실전 배포 후보), sum은 baseline.
- **병렬 평가 불가**(순차). 노이즈 대응 반복측정은 탐색 중엔 안 하고 **마지막 confirmation(top-3 재측정)에만**.

## 레포 구조
```
benchmark/                합성 black-box (목적함수/노이즈/scoring은 변경 금지)
  generator.py            BlackBoxBenchmark: X→6Y(functional-ANOVA: 주효과+희소교호)
                          블록상수 COMMON/SET1/SET2, 노이즈, 참조최적 계산
  configs.py              BM1<BM2<BM3 난이도 ladder(다봉성/교호밀도/3차/충돌/노이즈)
  scoring.py              MinMaxNormalizer + 3종 점수(sum/chebyshev/owa). 벤치/실문제 공용
  build.py                BM 인스턴스 + 참조최적/난이도 산출 → artifacts/<BM>.json
optim/                    알고리즘 벤치마킹 하니스
  problem.py              Problem: 노이즈 관측점수 최대화 + 참(true)점수 anytime 곡선
  design.py               marginal_balanced_design(=mlhs): 혼합변수 균형 초기설계
  algos.py                REGISTRY: random/sobol/mlhs/block_coord_local/sa/ga/tpe/smac/botorch
                          + 각 base의 블록-분해판 *_blk
  blockwrap.py            make_block_decomp: 임의 base를 block-coordinate로 감싸는 래퍼
  run.py                  실험 실행기 → results*.json (--seed-list/--merge-extend 지원)
  summarize.py            결과 병합 + 비교표 RESULTS.md
  visualize.py            그림: closure_*, by_kind_*, block_lift_*, vs_global_max
AGENTS.md, 00_Plan.md     계획/맥락 문서
```

## 핵심 개념
- **closure(성능지표)** = `(best_true − floor)/(ref_opt − floor)`. 0=무작위 단일추출 평균(floor), 1=참조최적(ref_opt, artifacts). 부호(특히 chebyshev<0) 무관.
- **anytime/예산**: 한 알고리즘을 max_budget까지 1회 실행하고 곡선에서 180/780 체크포인트 추출. 성능은 방문한 점들의 **참(노이즈 없는) 점수 누적 최댓값**으로 평가(추천정책 무관, 공정).
- **block_coord_local**: 블록-인지 좌표 local search. marginal-balanced 초기점 → `common→set2→set1` 라운드 반복(common 재방문) + best-improvement 1-hop + random-restart, 캐시로 중복평가 회피. (개념상 stepwise/OFAT의 일반화 = 그리디 좌표법. 약점도 동일: 교호작용/비분리에 취약.)
- **global maxima**: `block_coord_local@20000`(BM3 기준 ~52s)으로 정의. 9칸 중 7칸에서 기존 ref_opt 이상 → 더 타이트한 천장.

## 지금까지의 결론 (중요)
1. **block_coord_local이 저예산에서 압도** (sum BM1 ~100%@780 = global max 도달). @780은 대부분 global max의 **85~100%**, @180은 60~99%.
2. **공정 비교**(모든 base에 블록 주입 `*_blk`): **블록 구조가 가장 큰 레버**(random/sobol/ga가 +15~30pp 상승). 그 위에 **좌표 base가 sum에서 추가 이득**. **SA는 블록이 오히려 해로움**(전역탐색이 분해에 제약). **비분리 지표(owa/cheby)에선 tpe_blk가 일부 칸에서 block_coord_local 추월**.
3. **가장 큰 잔여 갭 = owa/cheby, 특히 owa BM2(78%)** → 유망 개선: **block_decomp + 지표별 내부 solver 하이브리드(sum=coordinate, owa/cheby=TPE)**.
4. **오버헤드**: model-free ~6ms/iter, tpe ~0.2s, smac ~1.05s. 실전(60s/eval)에선 모두 <2%로 무시 가능(벤치에서 느린 건 공짜 blackbox 탓).
5. seed는 **평균(best-of 아님)**. seed간 편차가 알고리즘 간 격차보다 클 수 있어 **랭킹엔 다중 seed 필수**(seed=1 위험).

## 실행 방법
```bash
python -m benchmark.build
python -m optim.run --algos random,sobol,mlhs,block_coord_local,sa,tpe \
    --max-budget 780 --budgets 180,780 --seeds 3
python -m optim.summarize --md optim/RESULTS.md
python -m optim.visualize
# 무거운 것: smac/botorch는 별도 파일·작은 seed로. *_blk는 공정비교용.
```
의존성: numpy, scipy, optuna, pymoo, smac(scikit-learn==1.6.1 핀 필요), botorch/torch, matplotlib.

## 제약 / 관례
- **benchmark의 목적함수·노이즈·scoring 정의는 변경 금지.** 기존 비교/저장/요약 흐름 유지.
- 개발은 **main에 직접 커밋·푸시**(사용자 승인). `results*.json`은 gitignore, `figs/`는 커밋.
- `git push -u origin main`, 네트워크 실패 시 지수 백오프 재시도.

## 열린 다음 단계 (택1)
- **(a) 하이브리드 baseline**: `block_decomp` + 지표별 내부 solver(owa/cheby에 TPE)로 owa BM2 갭 메우기.
- **(b) 현실판**: 블록을 *모른다고* 가정하고 screening(민감도/변수중요도)으로 블록을 **발견→활용**(실전 적용 시 이 비용을 빼야 함).
- **(c) 통합표**: 모든 알고리즘을 새 global max(@20000) 기준으로 재정규화한 단일 랭킹.
- **(d) 미완**: smac_blk/botorch_blk 채우기, 2400 예산(대부분 미실행, GP-BO는 780 상한), 실제 calculator 적용 + confirmation(top-3 재측정).
