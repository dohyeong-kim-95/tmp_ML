"""
Case별 최적화 난이도(복잡도) 정량화.

지표:
  n_terms      : 총 항 수 (strong+weak+interactions, 4 Y 합)
  max_order    : 최대 교호작용 차수 (1=선형, 2=2차, 3=3차, 4=4차)
  n_highorder  : 3차 이상 교호작용 개수
  linear_gap%  : (J* - J_linear)/J* ×100  → 교호작용이 만드는 추가이득(비선형성)
  n_local_opt  : 좌표상승 R회 재시작에서 나온 '서로 다른 국소최적' 개수 (다봉성)
  trap_rate%   : 국소최적에 갇혀 전역최적에 도달 못한 재시작 비율
  complexity   : 위 지표들을 0~100으로 정규화 합산한 종합 복잡도 지수
"""
import numpy as np


def case_complexity(prob, restarts=60, seed=0):
    rng = np.random.default_rng(seed)

    # --- 구조적 지표 ---
    n_terms, max_order, n_highorder = 0, 1, 0
    for y in prob.y_cols:
        s = prob.gt["responses"][y]
        n_terms += len(s["strong"]) + len(s["weak"]) + len(s["interactions"])
        for it in s["interactions"]:
            vs = it["vars"] if "vars" in it else [it["a"], it["b"]]
            order = len(set(vs)) if it.get("kind") != "quad" else 2
            order = len(vs)  # quad도 곱 차수는 2(같은 변수 2회) -> len=2
            max_order = max(max_order, order)
            if order >= 3:
                n_highorder += 1

    # --- 전역최적 vs 선형근사 ---
    x_lin = prob.linear_optimum(); J_lin = prob.objective(x_lin)

    # --- 다봉성: 좌표상승 재시작들의 국소최적 분포 ---
    local_opts = []
    for _ in range(restarts):
        x = prob.random_solution(rng)
        improved = True
        while improved:
            improved = False
            for col, (t, dom) in prob.meta.items():
                cur, bv = x[col], prob.objective(x)
                best_local = cur
                for cand in dom:
                    if cand == cur:
                        continue
                    x[col] = cand
                    v = prob.objective(x)
                    if v > bv:
                        bv, best_local = v, cand
                x[col] = best_local
                if best_local != cur:
                    improved = True
        local_opts.append(prob.objective(x))
    local_opts = np.array(local_opts)
    J_star = local_opts.max()
    n_local = len(np.unique(np.round(local_opts, 1)))
    trap_rate = float(np.mean(local_opts < J_star - 1e-6) * 100)
    linear_gap = float((J_star - J_lin) / abs(J_star) * 100)

    return {"n_terms": n_terms, "max_order": max_order,
            "n_highorder": n_highorder, "J_star": float(J_star),
            "linear_gap%": round(linear_gap, 2),
            "n_local_opt": int(n_local), "trap_rate%": round(trap_rate, 1)}


def complexity_index(rows):
    """여러 case 지표를 0~100 정규화 합산한 종합 복잡도 지수."""
    keys = ["n_terms", "max_order", "n_highorder", "linear_gap%",
            "n_local_opt", "trap_rate%"]
    arr = {k: np.array([r[k] for r in rows], float) for k in keys}
    norm = {}
    for k in keys:
        v = arr[k]
        rng = v.max() - v.min()
        norm[k] = (v - v.min()) / rng if rng > 1e-9 else np.zeros_like(v)
    idx = sum(norm[k] for k in keys) / len(keys) * 100
    return [round(float(x), 1) for x in idx]
