"""
진단: "모두 TIME에서 0%" 가 (1)벤치마크가 쉬워서인가 (2)시간예산이 많아서인가?

측정 1. 구조적 난이도 — 단일출발 좌표상승이 전역최적에 도달하는 데 필요한 평가수/성공률.
        (적은 평가로 풀리면 = 문제 자체가 쉬움)
측정 2. TIME 예산 sweep — T를 0.05~3.0s로 줄이며 gap이 언제 벌어지는지.
        (작은 T에서도 0%면 = 예산이 과한 게 아니라 문제가 쉬움)
"""
import time
import warnings
import numpy as np
from problem import Problem
from simulated_annealing import simulated_annealing
from binary_pso import binary_pso
from genetic_algorithm import genetic_algorithm
from gomea import gomea
from aco import aco

warnings.filterwarnings("ignore")
CASES = {"Case3": "data/case3/ground_truth.json",
         "Case6": "data/case6/ground_truth.json"}
SEEDS = [0, 1, 2]


def single_start_ascent(prob, rng):
    """단일 무작위출발 좌표상승 → (도달 J, 사용 평가수)."""
    x = prob.random_solution(rng)
    improved = True
    while improved:
        improved = False
        for col, (t, dom) in prob.meta.items():
            cur, bv, bestv = x[col], prob.objective(x), x[col]
            for cand in dom:
                if cand == cur:
                    continue
                x[col] = cand
                v = prob.objective(x)
                if v > bv:
                    bv, bestv = v, cand
            x[col] = bestv
            if bestv != cur:
                improved = True
    return prob.objective(x), prob.n_eval


for case, path in CASES.items():
    prob = Problem(gt_path=path)
    # 전역최적 기준
    rng = np.random.default_rng(0)
    _, Jstar = prob.coordinate_ascent(rng, restarts=60)

    print(f"\n===== {case}  (J* = {Jstar:.2f}) =====")

    # 측정 1: 단일출발 좌표상승
    evs, hits = [], 0
    rng = np.random.default_rng(1)
    for _ in range(40):
        prob.n_eval = 0
        Jf, ne = single_start_ascent(prob, rng)
        evs.append(ne)
        if Jf >= Jstar - 1e-6:
            hits += 1
    print(f"[구조적 난이도] 단일출발 좌표상승: 평균 {np.mean(evs):.0f} 평가, "
          f"전역최적 도달률 {hits/40*100:.0f}%  (낮은 평가+높은 도달률 = 쉬움)")

    # 측정 2: TIME sweep
    algos = {"SA": simulated_annealing, "PSO": binary_pso, "GA": genetic_algorithm,
             "GOMEA": gomea, "ACO": aco}
    Ts = [0.05, 0.1, 0.25, 0.5, 1.0, 3.0]
    print(f"[TIME sweep] gap%(평균):")
    print(f"  {'algo':6} " + " ".join(f"T={t:>4}" for t in Ts))
    for name, fn in algos.items():
        row = []
        for T in Ts:
            gaps = []
            for s in SEEDS:
                prob.n_eval = 0
                _, f, _ = fn(prob, max_eval=10_000_000, seed=s, deadline=time.time() + T)
                gaps.append((Jstar - f) / abs(Jstar) * 100)
            row.append(np.mean(gaps))
        print(f"  {name:6} " + " ".join(f"{g:6.2f}" for g in row))

print("\n해석: 작은 T(0.05~0.1s)에서도 gap≈0 이면 '문제가 쉬움'(구조적)."
      "  작은 T에서 gap이 크게 벌어지면 '3s 예산이 과함'.")
