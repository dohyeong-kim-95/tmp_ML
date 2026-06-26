"""
'비싼 목적함수(expensive objective)' 공용 러너 — pushed(가상) wall time.

전제: 1 evaluation = SEC_PER_EVAL(20)초, 예산 = WALL_MIN(10)분.
  → 가상 wall time = n_eval × 20s ≤ 600s  ⇒  정확히 CAP = 30 evaluations.

모든 알고리즘이 '정확히 30번'만 평가하도록 강제(초과 호출 시 BudgetExhausted로 중단),
그때까지 평가한 점들 중 '진짜값(노이즈 제거) 최선'을 그 알고리즘의 성과로 본다.
(노이즈가 있어도 공정: 표본의 best-true 로 비교)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEC_PER_EVAL = 20
WALL_MIN = 10
CAP = WALL_MIN * 60 // SEC_PER_EVAL          # 30


class BudgetExhausted(Exception):
    pass


class Budget:
    """objective 호출을 CAP회로 제한하고, 평가점들의 best-true 를 추적."""
    def __init__(self, prob, cap):
        self.prob = prob
        self.cap = cap
        self.n = 0
        self.best_true = -1e18
        self.best_x = None
        self._orig = prob.objective
        # 노이즈 없는 진짜값: prob2/3은 true_objective, prob1은 objective자체(노이즈 없음)
        self._true = getattr(prob, "true_objective", None)

    def objective(self, x):
        if self.n >= self.cap:
            raise BudgetExhausted
        self.n += 1
        val = self._orig(x)                  # 노이즈 낀 값(최적화기가 봄)
        t = self._true(x) if self._true is not None else val
        if t > self.best_true:
            self.best_true, self.best_x = t, dict(x)
        return val


def run_algo(prob, fn, seed):
    b = Budget(prob, CAP)
    prob.objective = b.objective             # 인스턴스 메서드 오버라이드
    try:
        fn(prob, seed)
    except BudgetExhausted:
        pass
    finally:
        prob.objective = b._orig             # 복원
    return b.best_true, b.n


def random_search(prob, seed):
    rng = np.random.default_rng(seed)
    while True:
        prob.objective(prob.random_solution(rng))   # BudgetExhausted 로 종료됨


def benchmark(prob_provider, algos, metric, seeds, title, out_png,
              ylabel="score (lower=better)"):
    """
    prob_provider(): 매 호출마다 새 Problem 인스턴스(누적 상태 없게).
    algos: {name: fn(prob, seed)}   (random_search 포함 가능)
    metric(best_true): -> score (낮을수록 좋음)
    """
    results = {}
    for name, fn in algos.items():
        scores = []
        for s in seeds:
            prob = prob_provider(s)          # seed별 새 Problem(노이즈도 달라짐)
            bt, n = run_algo(prob, fn, s)
            scores.append(metric(bt))
        results[name] = (float(np.mean(scores)), float(np.std(scores)))
    # 출력
    print(f"\n=== {title}  (CAP={CAP} evals = {WALL_MIN}min @ {SEC_PER_EVAL}s/eval) ===")
    order = sorted(results, key=lambda k: results[k][0])
    for k in order:
        print(f"  {k:8} {results[k][0]:10.4f}  (±{results[k][1]:.4f})")
    # 그래프
    plt.figure(figsize=(9, 5))
    vals = [results[k][0] for k in order]
    errs = [results[k][1] for k in order]
    plt.bar(order, vals, yerr=errs, capsize=4, color="#1f77b4")
    plt.ylabel(ylabel); plt.yscale("log")
    plt.title(f"{title}\n10min budget @ 20s/eval = {CAP} evaluations (lower=better)")
    plt.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(out_png, dpi=120)
    print(f"saved: {out_png}")
    return results
