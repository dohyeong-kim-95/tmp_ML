"""
세 메타휴리스틱(SA / PSO / GA)으로 J(X)=Σ Y 최대화 → 최적 X 탐색 후 비교.

- 동일 평가예산(max_eval) 하에서 비교
- 여러 seed 반복으로 평균/최고/표준편차
- 참조 기준선: 선형근사 최적(linear), 좌표상승 best-known(near-global)
- 수렴곡선 plot + 결과 CSV + 최적 X(JSON) 저장
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import Problem
from simulated_annealing import simulated_annealing
from particle_swarm import particle_swarm
from genetic_algorithm import genetic_algorithm

MAX_EVAL = 20000
SEEDS = [0, 1, 2, 3, 4]

prob = Problem()

# ---------------- 참조 기준선 ----------------
x_lin = prob.linear_optimum()
J_lin = prob.objective(x_lin)

rng = np.random.default_rng(123)
x_best, J_best = prob.coordinate_ascent(rng, restarts=40)   # near-global
print("=" * 64)
print(f"참조 기준선")
print(f"  선형근사 최적 J(linear)        = {J_lin:8.3f}")
print(f"  좌표상승 best-known J(near-opt) = {J_best:8.3f}   <- 비교 기준")
print("=" * 64)

ALGOS = {
    "SA":  lambda s: simulated_annealing(prob, max_eval=MAX_EVAL, seed=s),
    "PSO": lambda s: particle_swarm(prob, max_eval=MAX_EVAL, seed=s),
    "GA":  lambda s: genetic_algorithm(prob, max_eval=MAX_EVAL, seed=s),
}

results = {}
curves = {}
best_solutions = {}

for name, fn in ALGOS.items():
    finals, best_run_f, best_run_x, hist_list = [], -1e18, None, []
    for s in SEEDS:
        prob.n_eval = 0
        x, f, hist = fn(s)
        finals.append(f)
        hist_list.append(hist)
        if f > best_run_f:
            best_run_f, best_run_x = f, x
    finals = np.array(finals)
    results[name] = {
        "mean": float(finals.mean()),
        "std": float(finals.std()),
        "best": float(finals.max()),
        "worst": float(finals.min()),
        "gap_to_best_pct": float((J_best - finals.max()) / abs(J_best) * 100),
    }
    best_solutions[name] = best_run_x
    # 수렴곡선: seed별 히스토리를 같은 평가축으로 보간 평균
    maxlen = max(len(h) for h in hist_list)
    arr = np.array([np.interp(np.linspace(0, 1, 500),
                              np.linspace(0, 1, len(h)), h) for h in hist_list])
    curves[name] = arr.mean(axis=0)
    print(f"\n[{name}]  best={finals.max():.3f}  mean={finals.mean():.3f}"
          f"  std={finals.std():.3f}  gap={results[name]['gap_to_best_pct']:.2f}%")

# ---------------- 저장: 결과 표 ----------------
import csv
with open("optimize/results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algorithm", "best", "mean", "std", "worst",
                "gap_to_best_pct"])
    for name, r in results.items():
        w.writerow([name, f"{r['best']:.3f}", f"{r['mean']:.3f}",
                    f"{r['std']:.3f}", f"{r['worst']:.3f}",
                    f"{r['gap_to_best_pct']:.2f}"])

# ---------------- 최적 X 저장 ----------------
out = {"reference": {"J_linear": J_lin, "J_near_optimal": J_best,
                     "x_near_optimal": x_best},
       "algorithms": {}}
for name in ALGOS:
    out["algorithms"][name] = {
        "J": results[name]["best"],
        "x": best_solutions[name],
        "per_response": {y: prob.response(best_solutions[name], y)
                         for y in prob.y_cols},
    }
with open("optimize/best_solutions.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---------------- 수렴곡선 plot ----------------
plt.figure(figsize=(8, 5))
xs = np.linspace(0, MAX_EVAL, 500)
for name in ALGOS:
    plt.plot(xs, curves[name], label=name, lw=2)
plt.axhline(J_best, ls="--", c="k", lw=1, label="near-optimal (coord.ascent)")
plt.axhline(J_lin, ls=":", c="gray", lw=1, label="linear approx.")
plt.xlabel("objective evaluations")
plt.ylabel("best J(X) = Σ Y")
plt.title("SA vs PSO vs GA  (maximize ΣY)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("optimize/convergence.png", dpi=120)
print("\nsaved: optimize/results.csv, optimize/best_solutions.json, optimize/convergence.png")

# ---------------- 각 알고리즘 최적 X별 Y값 ----------------
print("\n각 알고리즘이 찾은 최적 X에서의 반응값:")
print(f"{'algo':5} {'J':>9} " + " ".join(f"{y:>8}" for y in prob.y_cols))
for name in ALGOS:
    x = best_solutions[name]
    ys = [prob.response(x, y) for y in prob.y_cols]
    print(f"{name:5} {results[name]['best']:9.3f} " +
          " ".join(f"{v:8.3f}" for v in ys))
xb = x_best
ys = [prob.response(xb, y) for y in prob.y_cols]
print(f"{'OPT':5} {J_best:9.3f} " + " ".join(f"{v:8.3f}" for v in ys))
