"""
Ant Colony Optimization (MAX-MIN Ant System 스타일) — 이산/혼합 변수용.

각 변수의 각 값에 페로몬 τ[col][value] 를 둔다.
개미는 변수별로 값을 페로몬 확률 ∝ τ^α 로 선택해 해를 구성.
매 반복: 증발 τ←(1-ρ)τ, 전역최적 해의 선택값에 보상, τ를 [τmin,τmax]로 클램프.
(휴리스틱 η는 사전정보가 없어 균일 가정 → 순수 페로몬 기반)
"""
import time
import numpy as np


def aco(prob, max_eval=2000, n_ants=20, alpha=1.0, rho=0.1,
        tau_min=0.05, tau_max=5.0, seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    cols = prob.vars
    doms = {c: list(prob.meta[c][1]) for c in cols}
    idx_of = {c: {v: i for i, v in enumerate(doms[c])} for c in cols}

    tau = {c: np.full(len(doms[c]), tau_max) for c in cols}   # MMAS: τmax 초기화

    best_x, best_f, history = None, -1e18, []

    def build():
        x = {}
        for c in cols:
            w = tau[c] ** alpha
            p = w / w.sum()
            x[c] = doms[c][rng.choice(len(doms[c]), p=p)]
        return x

    n_iter = max_eval // n_ants
    for _ in range(n_iter):
        for _ in range(n_ants):
            x = build()
            f = prob.objective(x)
            if f > best_f:
                best_f, best_x = f, dict(x)
            history.append(best_f)

        # 증발
        for c in cols:
            tau[c] *= (1.0 - rho)
        # 전역최적 보상 (MMAS: global-best 만 deposit)
        for c in cols:
            j = idx_of[c][best_x[c]]
            tau[c][j] += rho * 1.0
            np.clip(tau[c], tau_min, tau_max, out=tau[c])

        if deadline and time.time() > deadline:
            break
    return best_x, best_f, history
