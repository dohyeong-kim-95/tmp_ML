# optim (Phase 3) — 알고리즘 포트폴리오 벤치마킹

벤치마크(BM1/2/3) × 점수(sum/chebyshev/owa) × 예산 위에서 단일목적 최적화기들을
공정 비교한다. 모든 알고리즘은 **노이즈 관측 점수를 최대화**하고, 성능은 방문한
점들의 **참(노이즈 없는) 점수 누적 최댓값(anytime)** 으로 평가한다.

## 구성
- `problem.py` — `Problem`: (benchmark, kind) 위 최대화 문제 + anytime 곡선
- `algos.py` — 어댑터(`REGISTRY`): 모두 `run(problem, budget, seed)`
- `run.py` — 실험 실행기 → `results*.json`
- `summarize.py` — 결과 병합 + 비교표(`RESULTS.md`)

## 포트폴리오
| key | 알고리즘 | 성격 |
|---|---|---|
| `random` | Random search | 정직한 하한선 |
| `sobol` | Scrambled Sobol | 저편차 하한선 |
| `sa` | Simulated Annealing | 단일목적 이산공간 홈그라운드 (OWA 약진 후보) |
| `ga` | pymoo 정수 GA | model-free 이산 대표 |
| `tpe` | Optuna TPESampler | 이산 강함, 경량 |
| `smac` | SMAC3 RF-SMBO | 혼합/이산 native, **anchor** |
| `botorch` | BoTorch GP-BO (qLogEI, 연속완화) | GP 상한선 reference (sum 강·OWA 약 예상) |

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

## 실행
```bash
python -m optim.run --algos random,sobol,sa,ga,tpe --max-budget 780 --seeds 3
python -m optim.run --algos smac    --max-budget 780 --seeds 2 --out optim/results_smac.json
python -m optim.run --algos botorch --max-budget 780 --seeds 1 --out optim/results_botorch.json
python -m optim.summarize --md optim/RESULTS.md
```
