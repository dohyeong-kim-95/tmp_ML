"""
가설 타당성 평가:
 "이미 쌓인 데이터로 BO surrogate를 warm-start 하면 BO가 GA를 역전할 수 있나?"

설정:
 - 과거 데이터 = 각 Case의 dummy_data.csv (노이즈 포함 실측 ΣY). 이건 '공짜'(과거 실험).
 - warm-BO : 과거 데이터의 상위 gp_cap개로 GP를 사전 시드(예산 0 소모)하고 최적화 시작.
 - 비교군 : cold-BO, GA, ACO
 - 두 예산 모드(EVAL=500/2000, TIME=3s)에서 전역최적 대비 gap 측정.

핵심 질문 2개:
 (a) warm-start가 BO의 '평가효율'을 올리는가? (cold vs warm, EVAL 예산)
 (b) warm-start가 'TIME 예산'에서 GA를 역전시키는가?
"""
import time
import warnings
import numpy as np
import pandas as pd

from problem import Problem
from bayesian_optimization import bayesian_optimization
from genetic_algorithm import genetic_algorithm
from aco import aco
from complexity import case_complexity

warnings.filterwarnings("ignore")

CASES = {
    "Case1": ("data/ground_truth.json", "data/dummy_data.csv"),
    "Case3": ("data/case3/ground_truth.json", "data/case3/dummy_data.csv"),
    "Case6": ("data/case6/ground_truth.json", "data/case6/dummy_data.csv"),
}
GP_CAP = 180
SEEDS = [0, 1, 2]


def load_warm(prob, csv_path, k=GP_CAP):
    """과거 데이터에서 상위 k개 (X, 노이즈 ΣY) 를 warm 관측으로."""
    df = pd.read_csv(csv_path)
    sumY = df[prob.y_cols].sum(axis=1).to_numpy()
    top = np.argsort(sumY)[::-1][:k]
    wx = [{c: (df.iloc[i][c] if prob.meta[c][0] == "cat" else int(df.iloc[i][c]))
           for c in prob.vars} for i in top]
    wy = [float(sumY[i]) for i in top]
    return wx, wy, float(sumY.max())


def run(fn, prob, seed, max_eval, deadline, **kw):
    prob.n_eval = 0
    x, f, h = fn(prob, max_eval=max_eval, seed=seed, deadline=deadline, **kw)
    return f, prob.n_eval


print(f"{'Case':6} {'J*':>8} {'data_max':>9} | "
      f"{'mode':5} {'cold-BO':>9} {'warm-BO':>9} {'GA':>9} {'ACO':>9}")
print("-" * 88)

for case, (gt, csv) in CASES.items():
    prob = Problem(gt_path=gt)
    cx = case_complexity(prob, restarts=40, seed=0)
    Jstar = cx["J_star"]
    wx, wy, data_max = load_warm(prob, csv)

    def gap(vals):
        return (Jstar - np.mean(vals)) / abs(Jstar) * 100

    # ---- EVAL 예산 ----
    cold = [run(bayesian_optimization, prob, s, 500, None)[0] for s in SEEDS]
    warm = [run(bayesian_optimization, prob, s, 500, None,
                warm=(wx, wy))[0] for s in SEEDS]
    ga = [run(genetic_algorithm, prob, s, 2000, None)[0] for s in SEEDS]
    ac = [run(aco, prob, s, 2000, None)[0] for s in SEEDS]
    print(f"{case:6} {Jstar:8.2f} {data_max:9.2f} | "
          f"{'EVAL':5} {gap(cold):8.2f}% {gap(warm):8.2f}% {gap(ga):8.2f}% {gap(ac):8.2f}%")

    # ---- TIME 예산 (3s) ----
    def timed(fn, **kw):
        out = []
        for s in SEEDS:
            dl = time.time() + 3.0
            out.append(run(fn, prob, s, 10_000_000, dl, **kw)[0])
        return out
    cold_t = timed(bayesian_optimization)
    warm_t = timed(bayesian_optimization, warm=(wx, wy))
    ga_t = timed(genetic_algorithm)
    ac_t = timed(aco)
    print(f"{'':6} {'':8} {'':9} | "
          f"{'TIME':5} {gap(cold_t):8.2f}% {gap(warm_t):8.2f}% {gap(ga_t):8.2f}% {gap(ac_t):8.2f}%")

print("\n(gap%: 전역최적 대비, 낮을수록 좋음. 평균 over seeds)")
