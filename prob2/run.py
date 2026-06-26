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
TIME_SWEEP = [0.25, 0.5, 1.0, 2.0]   # 메인 목표 = TIME 예산 (EVAL sweep 미사용)
MAIN_T = 1.0

ALGOS = {"SA": simulated_annealing, "PSO": binary_pso, "GA": genetic_algorithm,
         "MemGA": memetic_ga, "CHC": chc, "GOMEA": gomea, "ACO": aco}

prob = Problem()
Y = prob.y_cols

# ---------- 참조: 전역최적(노이즈 없는 진짜 목적) ----------
rng = np.random.default_rng(0)
x_opt, J_star = prob.coordinate_ascent(rng, restarts=60)
print(f"X {len(prob.vars)}열, Y {len(Y)}개, 목적함수 노이즈 sd={prob.noise_sd}")
print(f"전역최적 J* (노이즈 제거) = {J_star:.3f}")

# 정규화 분모: 목적함수의 자연 분산(무작위 X에서의 진짜값 분산)
rng = np.random.default_rng(7)
samp = np.array([prob.true_objective(prob.random_solution(rng)) for _ in range(5000)])
BASE_VAR = float(samp.var())
print(f"정규화 분모 Var_random(true) = {BASE_VAR:.2f}")
print("score = NMSE = mean_seeds[(J* - true(best_x))^2] / Var_random  (낮을수록 좋음)\n")

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


def sq_err(best_x):
    return (J_star - prob.true_objective(best_x)) ** 2


def run_time(fn, T, seed):
    prob.n_eval = 0
    x, f, h = fn(prob, max_eval=10_000_000, seed=seed, deadline=time.time() + T)
    return sq_err(x)


# ---------- TIME sweep (메인) — score = normalized MSE ----------
rows = []
print(f"=== TIME sweep — NMSE (낮을수록 좋음) ===")
print(f"  {'algo':6} " + " ".join(f"T={t:>5}" for t in TIME_SWEEP))
time_nmse = {t: {} for t in TIME_SWEEP}
for name, fn in ALGOS.items():
    line = []
    for T in TIME_SWEEP:
        se = [run_time(fn, T, s) for s in SEEDS]      # seed별 제곱오차
        nmse = float(np.mean(se)) / BASE_VAR
        time_nmse[T][name] = nmse
        rows.append({"budget_s": T, "algo": name,
                     "nmse": round(nmse, 5),
                     "rmse_true": round(float(np.sqrt(np.mean(se))), 3)})
        line.append(nmse)
    print(f"  {name:6} " + " ".join(f"{g:7.4f}" for g in line))

with open("prob2/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["budget_s", "algo", "nmse", "rmse_true"])
    w.writeheader(); [w.writerow(r) for r in rows]

# ---------- 그래프: TIME sweep 곡선 + 메인 T 막대 (score = NMSE) ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
for name in ALGOS:
    ax1.plot(TIME_SWEEP, [max(time_nmse[T][name], 1e-6) for T in TIME_SWEEP],
             marker="o", label=name, lw=1.8)
ax1.set_yscale("log")
ax1.set_xlabel("time budget (s)"); ax1.set_ylabel("normalized MSE (log)")
ax1.set_title("TIME budget sweep (noisy objective)"); ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=11)

names = list(ALGOS)
vals = [time_nmse[MAIN_T][n] for n in names]
order = np.argsort(vals)
ax2.bar([names[i] for i in order], [vals[i] for i in order], color="#1f77b4")
ax2.set_ylabel("normalized MSE"); ax2.set_yscale("log")
ax2.set_title(f"TIME budget = {MAIN_T}s  (lower = better)")
ax2.grid(axis="y", alpha=0.3, which="both")
plt.suptitle(f"prob2 model-free benchmark: {len(prob.vars)} cols, {len(Y)} Y, "
             f"noisy objective (sd={prob.noise_sd}, =4% of main effect)",
             fontsize=13)
plt.tight_layout()
plt.savefig("prob2/benchmark.png", dpi=120)
print("\nsaved: prob2/results.csv, prob2/benchmark.png")
