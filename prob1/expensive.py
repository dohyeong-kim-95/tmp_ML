"""prob1(Case1~6, 노이즈 없음)을 '비싼 목적함수' 영역(10분 @20s/eval = 30 evals)에서 재시도.
   score = 전역최적 대비 gap%(6개 Case 평균)."""
import sys, numpy as np
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from expensive_common import run_algo, CAP, random_search
from problem import Problem
from simulated_annealing import simulated_annealing
from binary_pso import binary_pso
from genetic_algorithm import genetic_algorithm
from memetic_ga import memetic_ga
from chc import chc
from gomea import gomea
from aco import aco
from bayesian_optimization import bayesian_optimization
from tpe import tpe
from smac import smac

CASES = {"Case1": "prob1/data/ground_truth.json",
         "Case2": "prob1/data/case2/ground_truth.json",
         "Case3": "prob1/data/case3/ground_truth.json",
         "Case4": "prob1/data/case4/ground_truth.json",
         "Case5": "prob1/data/case5/ground_truth.json",
         "Case6": "prob1/data/case6/ground_truth.json"}
SEEDS = list(range(5))
E = CAP

J = {}
for c, p in CASES.items():
    rng = np.random.default_rng(0)
    _, J[c] = Problem(gt_path=p).coordinate_ascent(rng, restarts=50)
print("prob1 J*:", {c: round(v, 1) for c, v in J.items()})

ALGOS = {
    "Random": random_search,
    "SA":   lambda p, s: simulated_annealing(p, max_eval=E, seed=s),
    "PSO":  lambda p, s: binary_pso(p, max_eval=E, n_particles=8, seed=s),
    "GA":   lambda p, s: genetic_algorithm(p, max_eval=E, pop_size=10, seed=s),
    "MemGA": lambda p, s: memetic_ga(p, max_eval=E, pop_size=6, n_ls=1, seed=s),
    "CHC":  lambda p, s: chc(p, max_eval=E, pop_size=10, seed=s),
    "GOMEA": lambda p, s: gomea(p, max_eval=E, pop_size=10, seed=s),
    "ACO":  lambda p, s: aco(p, max_eval=E, n_ants=10, seed=s),
    "BO":   lambda p, s: bayesian_optimization(p, max_eval=E, n_init=10, seed=s),
    "TPE":  lambda p, s: tpe(p, max_eval=E, n_startup=10, seed=s),
    "SMAC": lambda p, s: smac(p, max_eval=E, n_init=10, seed=s),
}

results = {}
for name, fn in ALGOS.items():
    gaps = []
    for c, path in CASES.items():
        for s in SEEDS:
            bt, n = run_algo(Problem(gt_path=path), fn, s)
            gaps.append((J[c] - bt) / abs(J[c]) * 100)
    results[name] = (float(np.mean(gaps)), float(np.std(gaps)))

print(f"\n=== prob1 (6 Case 평균) — expensive  (CAP={E} evals = 10min @ 20s/eval) ===")
order = sorted(results, key=lambda k: results[k][0])
for k in order:
    print(f"  {k:8} gap={results[k][0]:7.2f}%  (±{results[k][1]:.2f})")

plt.figure(figsize=(9, 5))
plt.bar(order, [results[k][0] for k in order],
        yerr=[results[k][1] for k in order], capsize=4, color="#1f77b4")
plt.ylabel("gap to optimum (%, mean over 6 cases)")
plt.title("prob1 — expensive objective\n10min budget @ 20s/eval = 30 evals (lower=better)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("prob1/expensive.png", dpi=120)
print("saved: prob1/expensive.png")
