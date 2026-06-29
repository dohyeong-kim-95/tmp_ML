# optim (Phase 3) — 알고리즘 포트폴리오 벤치마킹

벤치마크(BM1/2/3) × 점수(sum/chebyshev/owa) × 예산 위에서 단일목적 최적화기들을
공정 비교한다. 모든 알고리즘은 **노이즈 관측 점수를 최대화**하고, 성능은 방문한
점들의 **참(노이즈 없는) 점수 누적 최댓값(anytime)** 으로 평가한다.

## 구성
- `problem.py` — `Problem`: (benchmark, kind) 위 최대화 문제 + anytime 곡선
- `design.py` — `marginal_balanced_design`: 혼합변수 marginal-balanced 초기설계
- `algos.py` — 어댑터(`REGISTRY`): 모두 `run(problem, budget, seed)`
- `run.py` — 실험 실행기 → `results*.json`
- `summarize.py` / `visualize.py` — 비교표(`RESULTS.md`) + 그림(`figs/`)

## 포트폴리오
| key | 알고리즘 | 성격 |
|---|---|---|
| `random` | Random search | 정직한 하한선 |
| `sobol` | Scrambled Sobol (floor 매핑) | 저편차 하한선(이산 매핑 한계 비교 기준) |
| **`mlhs`** | **혼합변수 marginal-balanced LHS** | **Sobol floor의 categorical 가짜순서·cardinality 불균형을 피한 강한 초기설계** |
| **`block_coord_local`** | **블록-인지 좌표 local search** | **common/set1/set2 구조 활용. 저예산(180/780)에서 강한 baseline** |
| `sa` | Simulated Annealing | 단일목적 이산공간 홈그라운드 |
| `ga` | pymoo 정수 GA | model-free 이산 대표 |
| `tpe` | Optuna TPESampler | 이산 강함, 경량 |
| `smac` | SMAC3 RF-SMBO | 혼합/이산 native, anchor |
| `botorch` | BoTorch GP-BO (qLogEI, 연속완화) | GP 상한선 reference |

### 신규 baseline 상세
- **`mlhs`** — 각 변수의 level marginal이 균등하도록(그리고 prefix도 균등) 변수별
  독립 순열을 반복해 설계. categorical에 가짜 순서를 강제하지 않고, cardinality가
  제각각이어도 초기 coverage가 균형. Sobol은 비교용으로 유지.
- **`block_coord_local`** — marginal-balanced 초기점에서 시작해, 라운드마다
  `common → set2 → set1` 순서로 각 변수를 1-hop 스윕(best-improvement). common을
  매 라운드 재방문해 블록 간 결합을 흡수하고, 수렴하면 random-restart로 남은 예산
  활용. 같은 X 재평가는 캐시로 회피(예산 절약). 참 점수 anytime 평가는 `Problem` 유지.

## 성능 지표
```
closure = (best_true − floor) / (ref_opt − floor)
  floor   = 무작위 단일추출 점수 평균(BM/kind)
  ref_opt = 참조 최적 (benchmark/artifacts/<BM>.json)
```
0 = 무작위 수준, 1 = 참조최적. 부호(특히 chebyshev<0) 무관하게 동작.

## 비용/예산 정책
- model-free + TPE + SMAC: 최대 780 iter
- **GP-BO(BoTorch): 최대 780**(2400 제외). GP cubic 비용 억제 위해 학습점 상한
  (best+recent ≤256)·주기적 refit 적용.
- 예산 체크포인트: 180 / 780 (단일 실행 곡선에서 추출).

## 실행 (권장 흐름)
```bash
python -m benchmark.build
# 신규 baseline 포함 경량 포트폴리오
python -m optim.run --algos random,sobol,mlhs,block_coord_local,sa,tpe \
    --max-budget 780 --budgets 180,780 --seeds 3
python -m optim.summarize --md optim/RESULTS.md
python -m optim.visualize          # figs/closure_*.png, figs/by_kind_*.png

# (선택) 무거운 anchor/reference는 별도 파일로
python -m optim.run --algos smac    --max-budget 780 --seeds 1 --kinds sum --out optim/results_smac.json
python -m optim.run --algos botorch --max-budget 780 --seeds 1 --out optim/results_botorch.json
```
`--seed-list`/`--merge-extend` 로 기존 seed 결과를 보존하며 seed 추가 가능.
