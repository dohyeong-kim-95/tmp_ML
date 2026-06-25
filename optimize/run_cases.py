"""
Case1~6 × 6개 알고리즘(SA / PSO(binary) / GA / BO / TPE / SMAC) 검증.

두 가지 예산 모드:
  (1) EVAL 예산  : SA/PSO/GA = 2000 평가, BO/TPE/SMAC = 500 평가 (샘플효율형)
  (2) TIME 예산  : 모든 알고리즘에 동일 wall-clock T초 부여 → 각자 가능한 만큼 평가

추가:
  - Case별 복잡도 정량화 (complexity.py)
  - 수렴곡선 3col×2row PNG (EVAL/TIME 각각)
  - 범례 표기: EVAL모드 = ALGO(t=평균 wall-time), TIME모드 = ALGO(n=평균 평가수)
"""
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import Problem
from simulated_annealing import simulated_annealing
from binary_pso import binary_pso
from genetic_algorithm import genetic_algorithm
from bayesian_optimization import bayesian_optimization
from tpe import tpe
from smac import smac
from complexity import case_complexity, complexity_index

warnings.filterwarnings("ignore")

CASES = {
    "Case1": "data/ground_truth.json",
    "Case2": "data/case2/ground_truth.json",
    "Case3": "data/case3/ground_truth.json",
    "Case4": "data/case4/ground_truth.json",
    "Case5": "data/case5/ground_truth.json",
    "Case6": "data/case6/ground_truth.json",
}

# 알고리즘: 함수 + EVAL예산 + seed수
ALGOS = {
    "SA":   dict(fn=simulated_annealing, evals=2000, seeds=[0, 1, 2]),
    "PSO":  dict(fn=binary_pso,          evals=2000, seeds=[0, 1, 2]),
    "GA":   dict(fn=genetic_algorithm,   evals=2000, seeds=[0, 1, 2]),
    "BO":   dict(fn=bayesian_optimization, evals=500, seeds=[0, 1]),
    "TPE":  dict(fn=tpe,                 evals=500,  seeds=[0, 1]),
    "SMAC": dict(fn=smac,                evals=500,  seeds=[0, 1]),
}
TIME_BUDGET = 3.0   # TIME 모드 wall-clock 초
GRID = 300


def run_one(fn, prob, seed, max_eval, deadline):
    prob.n_eval = 0
    t0 = time.time()
    x, f, hist = fn(prob, max_eval=max_eval, seed=seed, deadline=deadline)
    return x, f, hist, time.time() - t0, prob.n_eval


# ============================================================
# 1) Case 복잡도 + 전역최적 참조
# ============================================================
print("Case 복잡도 계산 중...")
cx_rows = []
J_star = {}
probs = {}
for case, path in CASES.items():
    p = Problem(gt_path=path)
    probs[case] = p
    cx = case_complexity(p, restarts=60, seed=0)
    cx["case"] = case
    cx_rows.append(cx)
    J_star[case] = cx["J_star"]
cx_index = complexity_index(cx_rows)
for r, ci in zip(cx_rows, cx_index):
    r["complexity_idx"] = ci

print("\n=== Case 복잡도 정량화 ===")
hdr = ["case", "n_terms", "max_order", "n_highorder", "linear_gap%",
       "n_local_opt", "trap_rate%", "complexity_idx", "J_star"]
print("  ".join(f"{h:>12}" for h in hdr))
for r in cx_rows:
    print("  ".join(f"{r[h]:>12}" for h in hdr))

import csv
with open("optimize/complexity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    for r in cx_rows:
        w.writerow({h: r[h] for h in hdr})


# ============================================================
# 2) 두 모드 실행
# ============================================================
def execute(mode):
    """mode='eval' or 'time'. 반환: results(list), curves[case][algo]=(x,y), legends"""
    results, curves = [], {c: {} for c in CASES}
    for case in CASES:
        prob = probs[case]
        for name, cfg in ALGOS.items():
            seeds = cfg["seeds"]
            finals, times, nevals, hist_list, best_x, best_f = [], [], [], [], None, -1e18
            for s in seeds:
                if mode == "eval":
                    x, f, h, wt, ne = run_one(cfg["fn"], prob, s, cfg["evals"], None)
                else:
                    dl = time.time() + TIME_BUDGET
                    x, f, h, wt, ne = run_one(cfg["fn"], prob, s, 10_000_000, dl)
                finals.append(f); times.append(wt); nevals.append(ne); hist_list.append(h)
                if f > best_f:
                    best_f, best_x = f, x
            finals = np.array(finals)
            gap = (J_star[case] - finals.max()) / abs(J_star[case]) * 100
            results.append({"mode": mode, "case": case, "algo": name,
                            "best": round(float(finals.max()), 3),
                            "mean": round(float(finals.mean()), 3),
                            "std": round(float(finals.std()), 3),
                            "gap_pct": round(float(gap), 2),
                            "wall_s": round(float(np.mean(times)), 2),
                            "n_eval": int(np.mean(nevals))})
            # 곡선: 각 seed history를 공통 그리드로 보간 후 평균
            if mode == "eval":
                xaxis = np.linspace(0, cfg["evals"], GRID)
            else:
                xaxis = np.linspace(0, TIME_BUDGET, GRID)
            interp = []
            for h in hist_list:
                interp.append(np.interp(np.linspace(0, 1, GRID),
                                        np.linspace(0, 1, len(h)), h))
            curves[case][name] = (xaxis, np.mean(interp, axis=0),
                                  float(np.mean(times)), int(np.mean(nevals)))
    return results, curves


print(f"\nEVAL 모드 실행...")
res_eval, cur_eval = execute("eval")
print(f"TIME 모드 실행 (T={TIME_BUDGET}s)...")
res_time, cur_time = execute("time")

with open("optimize/results_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res_eval[0].keys()))
    w.writeheader(); [w.writerow(r) for r in res_eval]
with open("optimize/results_time.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res_time[0].keys()))
    w.writeheader(); [w.writerow(r) for r in res_time]


# ============================================================
# 3) 곡선 데이터 캐시 + 플롯 (3col × 2row)
# ============================================================
import pickle
from plot_cases import make_plots

cx_idx = {r["case"]: r["complexity_idx"] for r in cx_rows}
plot_data = {"cases": list(CASES), "algos": list(ALGOS),
             "curves": {"eval": cur_eval, "time": cur_time},
             "J_star": J_star, "cx_idx": cx_idx, "time_budget": TIME_BUDGET}
with open("optimize/curves.pkl", "wb") as f:
    pickle.dump(plot_data, f)

make_plots(plot_data)

# best solutions
best_json = {"J_star": J_star, "complexity": {r["case"]: r for r in cx_rows}}
with open("optimize/best_solutions_cases.json", "w", encoding="utf-8") as f:
    json.dump(best_json, f, ensure_ascii=False, indent=2)


# ============================================================
# 4) gap 요약 (두 모드)
# ============================================================
def gap_table(results, mode):
    print(f"\n=== gap(%) [{mode} 모드] — 전역최적 대비 ===")
    print(f"{'algo':5} " + " ".join(f"{c:>7}" for c in CASES))
    for name in ALGOS:
        g = {r["case"]: r["gap_pct"] for r in results if r["algo"] == name}
        print(f"{name:5} " + " ".join(f"{g[c]:7.2f}" for c in CASES))


gap_table(res_eval, "EVAL")
gap_table(res_time, "TIME")
print("\n완료.")
