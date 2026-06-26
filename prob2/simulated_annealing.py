"""Simulated Annealing (혼합 이산공간)."""
import time
import numpy as np


def neighbor(x, prob, rng):
    """변수 하나를 무작위 변경한 이웃해."""
    nx = dict(x)
    col = prob.vars[rng.integers(len(prob.vars))]
    t, dom = prob.meta[col]
    if t == "bin":
        nx[col] = 1 - x[col]
    elif t == "ord":
        # 70% 인접 ±1 step, 30% 임의 레벨
        if rng.random() < 0.7:
            step = 1 if rng.random() < 0.5 else -1
            nx[col] = int(np.clip(x[col] + step, 0, len(dom) - 1))
        else:
            nx[col] = dom[rng.integers(len(dom))]
    else:  # cat: 다른 레벨로
        choices = [d for d in dom if d != x[col]]
        nx[col] = choices[rng.integers(len(choices))]
    return nx


def simulated_annealing(prob, max_eval=20000, T0=10.0, Tend=1e-3, seed=0,
                        deadline=None):
    rng = np.random.default_rng(seed)
    x = prob.random_solution(rng)
    fx = prob.objective(x)
    best_x, best_f = dict(x), fx
    history = [best_f]

    # 진행도(progress) 기반 통합 냉각: T = T0 * (Tend/T0)^progress
    #   - EVAL 예산: progress = step / max_eval
    #   - TIME 예산: progress = 경과시간 / 시간예산   (max_eval에 안 묶임 → 공정)
    ratio = Tend / T0
    start = time.time()
    total_t = (deadline - start) if deadline else None
    n_steps = max_eval - 1
    step = 0
    while step < n_steps:
        if deadline:
            progress = min(1.0, (time.time() - start) / total_t) if total_t > 0 else 1.0
        else:
            progress = step / n_steps
        T = T0 * (ratio ** progress)

        nx = neighbor(x, prob, rng)
        fn = prob.objective(nx)
        d = fn - fx  # 최대화
        if d >= 0 or rng.random() < np.exp(d / T):
            x, fx = nx, fn
            if fx > best_f:
                best_f, best_x = fx, dict(x)
        history.append(best_f)
        step += 1
        if deadline and time.time() > deadline:
            break
    return best_x, best_f, history
