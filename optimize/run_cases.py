"""
Case 1~5 (난이도 점증)에 대해 SA / PSO / GA / BO 검증.

A 패러다임: 목적함수 = 각 Case 의 ground_truth 비밀식 (dataset CSV 불필요).
  Case1 = data/ground_truth.json (최저 난이도)
  Case2~5 = data/case{n}/ground_truth.json (점점 복잡·노이즈↑)

예산 budget=3000, 각 Case 마다:
  - 전역최적 참조(좌표상승 다중 재시작)
  - 4 알고리즘 best/mean/std/gap
산출: results_cases.csv, best_solutions_cases.json, convergence_cases.png
"""
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import Problem
from simulated_annealing import simulated_annealing
from particle_swarm import particle_swarm
from genetic_algorithm import genetic_algorithm
from bayesian_optimization import bayesian_optimization

warnings.filterwarnings("ignore")
BUDGET = 3000
SEEDS = [0, 1, 2]       # SA/PSO/GA
SEEDS_BO = [0, 1]       # BO는 비용이 커 2회

CASES = {
    "Case1": "data/ground_truth.json",
    "Case2": "data/case2/ground_truth.json",
    "Case3": "data/case3/ground_truth.json",
    "Case4": "data/case4/ground_truth.json",
    "Case5": "data/case5/ground_truth.json",
}

ALGOS = {
    "SA":  (lambda p, s: simulated_annealing(p, max_eval=BUDGET, seed=s), SEEDS),
    "PSO": (lambda p, s: particle_swarm(p, max_eval=BUDGET, seed=s), SEEDS),
    "GA":  (lambda p, s: genetic_algorithm(p, max_eval=BUDGET, seed=s), SEEDS),
    "BO":  (lambda p, s: bayesian_optimization(p, max_eval=BUDGET, seed=s), SEEDS_BO),
}

all_rows = []
all_curves = {}
all_best = {}
ref = {}

for case, gt_path in CASES.items():
    prob = Problem(gt_path=gt_path)
    # 전역최적 참조
    rng = np.random.default_rng(123)
    x_opt, J_opt = prob.coordinate_ascent(rng, restarts=60)
    ref[case] = J_opt
    all_curves[case] = {}
    all_best[case] = {}
    print(f"\n===== {case}  (전역최적 J* = {J_opt:.3f}) =====")

    for name, (fn, seeds) in ALGOS.items():
        finals, best_f, best_x, hist_list = [], -1e18, None, []
        for s in seeds:
            prob.n_eval = 0
            x, f, hist = fn(prob, s)
            finals.append(f); hist_list.append(hist)
            if f > best_f:
                best_f, best_x = f, x
        finals = np.array(finals)
        gap = (J_opt - finals.max()) / abs(J_opt) * 100
        all_rows.append({"case": case, "algo": name, "J_opt": round(J_opt, 3),
                         "best": round(finals.max(), 3), "mean": round(finals.mean(), 3),
                         "std": round(finals.std(), 3), "gap_pct": round(gap, 2)})
        all_best[case][name] = {"J": float(finals.max()), "x": best_x}
        arr = np.array([np.interp(np.linspace(0, 1, 400),
                                  np.linspace(0, 1, len(h)), h) for h in hist_list])
        all_curves[case][name] = arr.mean(axis=0)
        print(f"  {name:3}  best={finals.max():8.3f}  mean={finals.mean():8.3f}"
              f"  std={finals.std():6.3f}  gap={gap:6.2f}%")

# ---------------- 저장 ----------------
import csv
with open("optimize/results_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "algo", "J_opt", "best",
                                      "mean", "std", "gap_pct"])
    w.writeheader()
    for r in all_rows:
        w.writerow(r)

with open("optimize/best_solutions_cases.json", "w", encoding="utf-8") as f:
    json.dump({"budget": BUDGET, "reference_global_opt": ref,
               "results": all_best}, f, ensure_ascii=False, indent=2)

# ---------------- 수렴곡선 (Case별 subplot) ----------------
xs = np.linspace(0, BUDGET, 400)
fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=False)
for ax, case in zip(axes, CASES):
    for name in ALGOS:
        ax.plot(xs, all_curves[case][name], label=name, lw=1.8)
    ax.axhline(ref[case], ls="--", c="k", lw=1, label="global opt")
    ax.set_title(f"{case}  (J*={ref[case]:.1f})")
    ax.set_xlabel("evaluations"); ax.grid(alpha=0.3)
axes[0].set_ylabel("best J(X)=ΣY")
axes[0].legend(fontsize=8)
plt.suptitle(f"SA vs PSO vs GA vs BO across Case1~5  (budget={BUDGET})")
plt.tight_layout()
plt.savefig("optimize/convergence_cases.png", dpi=110)
print("\nsaved: results_cases.csv, best_solutions_cases.json, convergence_cases.png")

# ---------------- gap 요약 표 ----------------
print("\n=== gap(%) 요약: 전역최적 대비 (낮을수록 좋음) ===")
print(f"{'algo':4} " + " ".join(f"{c:>8}" for c in CASES))
for name in ALGOS:
    gaps = {r["case"]: r["gap_pct"] for r in all_rows if r["algo"] == name}
    print(f"{name:4} " + " ".join(f"{gaps[c]:8.2f}" for c in CASES))
