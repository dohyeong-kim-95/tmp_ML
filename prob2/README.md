# prob2 — model-free 최적화 벤치마크 (노이즈 목적함수)

prob1(전체 탐색: BO/TPE/SMAC 포함 10종, Case1~6)에서 **model-free 방식만 추려**
새로 구성한 벤치마크. 메인 목표는 **TIME 예산에서의 성능**.

## prob1 대비 변경점
| 항목 | prob1 | prob2 |
|---|---|---|
| 알고리즘 | 10종(모델기반 포함) | **model-free 7종**: SA / PSO(binary) / GA / MemGA / CHC / GOMEA / ACO |
| X 차원 | 40열 (binary32/ord4/cat4) | **70열 (binary56/ord7/cat7)** — 비율 80/10/10 유지 |
| Y 개수 | 4 (y11,y12,y21,y22) | **6 (y11,y12,y13,y21,y22,y23)** — family1/2 각 3개 |
| 목적함수 | 노이즈 없음(결정적) | **노이즈 = 주효과의 4%**(`objective_noise_sd`, 자동 산정) |
| 상호작용 구조 | (없음) | **10열 블록 7개: 블록내부 강한 상호작용(계수3~6, 2way/3way), 블록간 약한 상호작용** → 단일출발 좌표상승 도달률 50%→10%로 난이도↑ |
| 변수 구조 | 전 변수 공유 | **y1_=front50열, y2_=back50열**만 영향(중앙30 공유), 같은 family 3개 response는 **서로 다른 열(chunk)** 참조 → 목적함수 난이도↑ |

## 노이즈의 의미 (난이도 상승)
- `objective(x)` = 진짜 ΣY + N(0, sd) → **최적화기는 노이즈 낀 값으로 의사결정**
- 성능은 `true_objective(best_x)`(노이즈 제거)로 평가 → "노이즈에 얼마나 속았나"를 측정
- greedy/단일비교 기반(SA 등)은 노이즈에 약하고, 개체군·반복선택 기반(GA/GOMEA)은
  상대적으로 견고 → model-free 간 변별력 발생.

## 구성
| 파일 | 내용 |
|---|---|
| `generate.py` | 비밀식(ground_truth.json) 생성 (50열/6Y) |
| `ground_truth.json` | 목적함수 정의 + `objective_noise_sd` |
| `problem.py` | 노이즈 objective + 노이즈없는 true_objective + 참조(coordinate_ascent) |
| `{sa,binary_pso,genetic_algorithm,memetic_ga,chc,gomea,aco}.py` | model-free 7종 |
| `run.py` | 벤치마크: 구조난이도 + **TIME sweep만**(메인 목표), 결과 CSV/PNG |

## score = Normalized MSE
`NMSE = mean_seeds[(J* - true_objective(best_x))^2] / Var_random(true)`
- 분자: 추천 X의 진짜값이 전역최적에서 얼마나 떨어졌나(제곱오차, seed 평균)
- 분모: 목적함수의 자연 분산(무작위 X에서의 진짜값 분산) → 스케일 무관
- 0 = 전역최적, 낮을수록 좋음. (참고로 `rmse_true`도 CSV에 함께 기록)
| `results.csv`, `benchmark.png` | 결과 |

## 실행
```bash
python3 prob2/generate.py            # 비밀식 생성
PYTHONPATH=prob2 python3 prob2/run.py  # 7종 벤치마크 (TIME 측정은 단독 실행 권장)
```
