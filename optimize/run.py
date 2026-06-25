"""
네 메타휴리스틱(SA / PSO / GA / BO)으로 J(X)=Σ Y 최대화 → 최적 X 탐색·비교.

- 평가예산 max_eval = 5000 (동일)
- 참조: 데이터셋 최대 / 선형근사 최적 / 좌표상승 best-known(near-global)
- 수렴곡선 plot + 결과 CSV + 최적 X(JSON) 저장
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
MAX_EVAL = 5000
SEEDS = [0, 1, 2, 3, 4]
SEEDS_BO = [0, 1, 2]   # BO는 비용이 커 seed 3회

prob = Problem()

# ---------------- 참조 기준선 ----------------
# A 패러다임: 목적함수 = ground_truth 비밀식 자체. dataset은 불필요.
# (1) 선형근사 최적, (2) 좌표상승 best-known(전역최적)
x_lin = prob.linear_optimum(); J_lin = prob.objective(x_lin)
rng = np.random.default_rng(123)
x_best, J_best = prob.coordinate_ascent(rng, restarts=40)

print("=" * 66)
print("참조 기준선")
print(f"  선형근사 최적(교호작용 무시)      = {J_lin:8.3f}")
print(f"  좌표상승 best-known (전역최적)    = {J_best:8.3f}   <- 비교 기준")
print("=" * 66)

ALGOS = {
    "SA":  (lambda s: simulated_annealing(prob, max_eval=MAX_EVAL, seed=s), SEEDS),
    "PSO": (lambda s: particle_swarm(prob, max_eval=MAX_EVAL, seed=s), SEEDS),
    "GA":  (lambda s: genetic_algorithm(prob, max_eval=MAX_EVAL, seed=s), SEEDS),
    "BO":  (lambda s: bayesian_optimization(prob, max_eval=MAX_EVAL, seed=s), SEEDS_BO),
}

results, curves, best_solutions = {}, {}, {}

for name, (fn, seeds) in ALGOS.items():
    finals, best_run_f, best_run_x, hist_list = [], -1e18, None, []
    for s in seeds:
        prob.n_eval = 0
        x, f, hist = fn(s)
        finals.append(f); hist_list.append(hist)
        if f > best_run_f:
            best_run_f, best_run_x = f, x
    finals = np.array(finals)
    results[name] = {
        "best": float(finals.max()), "mean": float(finals.mean()),
        "std": float(finals.std()), "worst": float(finals.min()),
        "gap_to_best_pct": float((J_best - finals.max()) / abs(J_best) * 100),
    }
    best_solutions[name] = best_run_x
    arr = np.array([np.interp(np.linspace(0, 1, 500),
                              np.linspace(0, 1, len(h)), h) for h in hist_list])
    curves[name] = arr.mean(axis=0)
    print(f"[{name}]  best={finals.max():.3f}  mean={finals.mean():.3f}"
          f"  std={finals.std():.3f}  gap={results[name]['gap_to_best_pct']:.2f}%")

# ---------------- 저장 ----------------
import csv
with open("optimize/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algorithm", "best", "mean", "std", "worst", "gap_to_best_pct"])
    for name, r in results.items():
        w.writerow([name, f"{r['best']:.3f}", f"{r['mean']:.3f}",
                    f"{r['std']:.3f}", f"{r['worst']:.3f}",
                    f"{r['gap_to_best_pct']:.2f}"])

out = {"reference": {"J_linear": J_lin, "J_global_optimum": J_best,
                     "x_global_optimum": x_best},
       "algorithms": {}}
for name in ALGOS:
    out["algorithms"][name] = {
        "J": results[name]["best"], "x": best_solutions[name],
        "per_response": {y: prob.response(best_solutions[name], y)
                         for y in prob.y_cols}}
with open("optimize/best_solutions.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# 수렴곡선 캐시(재플롯 비용 절감)
xs = np.linspace(0, MAX_EVAL, 500)
np.savez("optimize/curves.npz", xs=xs, J_best=J_best, J_lin=J_lin,
         **{name: curves[name] for name in ALGOS})

# ---------------- 수렴곡선 ----------------
plt.figure(figsize=(8, 5))
for name in ALGOS:
    plt.plot(xs, curves[name], label=name, lw=2)
plt.axhline(J_best, ls="--", c="k", lw=1, label="global optimum")
plt.axhline(J_lin, ls=":", c="gray", lw=1, label="linear approx.")
plt.xlabel("objective evaluations"); plt.ylabel("best J(X) = Σ Y")
plt.title(f"SA vs PSO vs GA vs BO  (budget={MAX_EVAL})")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("optimize/convergence.png", dpi=120)
print("\nsaved: optimize/results.csv, best_solutions.json, convergence.png")

print("\n각 알고리즘 최적 X에서의 반응값:")
print(f"{'algo':5} {'J':>9} " + " ".join(f"{y:>8}" for y in prob.y_cols))
for name in ALGOS:
    x = best_solutions[name]
    ys = [prob.response(x, y) for y in prob.y_cols]
    print(f"{name:5} {results[name]['best']:9.3f} " + " ".join(f"{v:8.3f}" for v in ys))
ys = [prob.response(x_best, y) for y in prob.y_cols]
print(f"{'OPT':5} {J_best:9.3f} " + " ".join(f"{v:8.3f}" for v in ys))
