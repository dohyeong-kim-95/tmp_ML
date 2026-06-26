"""
prob2 벤치마크: model-free 7종(SA/PSO/GA/MemGA/CHC/GOMEA/ACO) 비교.

- 목적함수는 노이즈 낀 값을 반환(최적화기가 봄). 성능은 '추천 X의 진짜값'으로 평가.
- 메인 목표 = TIME 예산 성능 → TIME sweep + EVAL 비교.
- gap% = (J* - true_objective(best_x)) / |J*| * 100   (낮을수록 좋음)
"""
import time
import warnings
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import Problem
from simulated_annealing import simulated_annealing
from binary_pso import binary_pso
from genetic_algorithm import genetic_algorithm
from memetic_ga import memetic_ga
from chc import chc
from gomea import gomea
from aco import aco

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
EVAL_BUDGET = 2000
TIME_SWEEP = [0.25, 0.5, 1.0, 2.0]
MAIN_T = 1.0

ALGOS = {"SA": simulated_annealing, "PSO": binary_pso, "GA": genetic_algorithm,
         "MemGA": memetic_ga, "CHC": chc, "GOMEA": gomea, "ACO": aco}

prob = Problem()
Y = prob.y_cols

# ---------- 참조: 전역최적(노이즈 없는 진짜 목적) ----------
rng = np.random.default_rng(0)
x_opt, J_star = prob.coordinate_ascent(rng, restarts=60)
print(f"X 50열, Y {len(Y)}개, 목적함수 노이즈 sd={prob.noise_sd}")
print(f"전역최적 J* (노이즈 제거) = {J_star:.3f}\n")

# ---------- 구조적 난이도: 단일출발 좌표상승(진짜목적) ----------
def single_start_true(rng):
    x = prob.random_solution(rng)
    improved = True
    while improved:
        improved = False
        for col, (t, dom) in prob.meta.items():
            cur, bv, bestv = x[col], prob._true(x), x[col]
            for cand in dom:
                if cand == cur:
                    continue
                x[col] = cand
                v = prob._true(x)
                if v > bv:
                    bv, bestv = v, cand
            x[col] = bestv
            if bestv != cur:
                improved = True
    return prob._true(x)

rng = np.random.default_rng(1)
hits = sum(1 for _ in range(40) if single_start_true(rng) >= J_star - 1e-6)
print(f"[구조적 난이도] 단일출발 좌표상승 전역최적 도달률 = {hits/40*100:.0f}%")
print("  (노이즈는 별도 난이도 — 최적화기는 노이즈 낀 값으로 의사결정)\n")


def gap_of(best_x):
    return (J_star - prob.true_objective(best_x)) / abs(J_star) * 100


def run_algo(fn, mode, budget_or_T, seed):
    prob.n_eval = 0
    if mode == "eval":
        x, f, h = fn(prob, max_eval=budget_or_T, seed=seed, deadline=None)
    else:
        x, f, h = fn(prob, max_eval=10_000_000, seed=seed,
                     deadline=time.time() + budget_or_T)
    return gap_of(x), prob.n_eval


# ---------- EVAL 예산 ----------
print(f"=== EVAL 예산({EVAL_BUDGET}) — gap%(진짜값 기준) ===")
rows = []
eval_gap = {}
for name, fn in ALGOS.items():
    gs = [run_algo(fn, "eval", EVAL_BUDGET, s)[0] for s in SEEDS]
    eval_gap[name] = np.mean(gs)
    rows.append({"mode": "eval", "budget": EVAL_BUDGET, "algo": name,
                 "gap_mean": round(float(np.mean(gs)), 3),
                 "gap_std": round(float(np.std(gs)), 3)})
    print(f"  {name:6} gap={np.mean(gs):6.2f}%  (std {np.std(gs):.2f})")

# ---------- TIME sweep ----------
print(f"\n=== TIME sweep — gap%(진짜값 기준) ===")
print(f"  {'algo':6} " + " ".join(f"T={t:>4}" for t in TIME_SWEEP))
time_gap = {t: {} for t in TIME_SWEEP}
for name, fn in ALGOS.items():
    line = []
    for T in TIME_SWEEP:
        gs = [run_algo(fn, "time", T, s)[0] for s in SEEDS]
        time_gap[T][name] = np.mean(gs)
        rows.append({"mode": "time", "budget": T, "algo": name,
                     "gap_mean": round(float(np.mean(gs)), 3),
                     "gap_std": round(float(np.std(gs)), 3)})
        line.append(np.mean(gs))
    print(f"  {name:6} " + " ".join(f"{g:6.2f}" for g in line))

with open("prob2/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "budget", "algo", "gap_mean", "gap_std"])
    w.writeheader(); [w.writerow(r) for r in rows]

# ---------- 그래프: TIME sweep 곡선 + 메인 T 막대 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
for name in ALGOS:
    ax1.plot(TIME_SWEEP, [time_gap[T][name] for T in TIME_SWEEP],
             marker="o", label=name, lw=1.8)
ax1.set_xlabel("time budget (s)"); ax1.set_ylabel("gap to true optimum (%)")
ax1.set_title("TIME budget sweep (noisy objective)"); ax1.grid(alpha=0.3)
ax1.legend(fontsize=11)

names = list(ALGOS)
vals = [time_gap[MAIN_T][n] for n in names]
order = np.argsort(vals)
ax2.bar([names[i] for i in order], [vals[i] for i in order], color="#1f77b4")
ax2.set_ylabel("gap to true optimum (%)")
ax2.set_title(f"TIME budget = {MAIN_T}s  (lower = better)")
ax2.grid(axis="y", alpha=0.3)
plt.suptitle(f"prob2 model-free benchmark: 50 cols, 6 Y, noisy objective (sd={prob.noise_sd})",
             fontsize=13)
plt.tight_layout()
plt.savefig("prob2/benchmark.png", dpi=120)
print("\nsaved: prob2/results.csv, prob2/benchmark.png")
