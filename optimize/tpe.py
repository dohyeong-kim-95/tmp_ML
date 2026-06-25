"""
TPE (Tree-structured Parzen Estimator) — 이산/혼합 공간용 바닐라 구현 (numpy 벡터화).

관측을 좋은(good, 상위 γ분위)/나쁜(bad) 집단으로 나누고,
변수별 범주분포 l(x|good), g(x|bad) 를 라플라스 평활로 추정.
후보를 l 에서 샘플 → Σ log(l/g) 최대 후보 평가 (Expected Improvement 근사).
모든 변수(binary/ordinal/categorical)를 '범주형'으로 취급.
"""
import time
import numpy as np


def tpe(prob, max_eval=500, n_startup=20, gamma=0.2,
        n_candidates=80, seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    cols = prob.vars
    doms = {c: list(prob.meta[c][1]) for c in cols}
    ksize = {c: len(doms[c]) for c in cols}
    idx_of = {c: {v: i for i, v in enumerate(doms[c])} for c in cols}

    Xi = []   # 관측의 카테고리 인덱스 (list of np.array[len(cols)])
    Y = []
    best_x, best_f, history = None, -1e18, []

    def observe(x):
        nonlocal best_x, best_f
        f = prob.objective(x)
        Xi.append(np.array([idx_of[c][x[c]] for c in cols]))
        Y.append(f)
        if f > best_f:
            best_f, best_x = f, dict(x)
        history.append(best_f)

    for _ in range(min(n_startup, max_eval)):
        observe(prob.random_solution(rng))
        if deadline and time.time() > deadline:
            return best_x, best_f, history

    Xi_arr = None
    while prob.n_eval < max_eval:
        M = np.array(Xi)                       # (n_obs, n_cols)
        Yv = np.array(Y)
        n = len(Yv)
        n_good = max(1, int(np.ceil(gamma * n)))
        good = np.argsort(Yv)[::-1][:n_good]
        good_mask = np.zeros(n, bool); good_mask[good] = True

        # 후보 샘플 + 점수 (열별 벡터화)
        cand_idx = np.empty((n_candidates, len(cols)), int)
        score = np.zeros(n_candidates)
        for ci, c in enumerate(cols):
            k = ksize[c]
            gc = np.bincount(M[good_mask, ci], minlength=k) + 1.0
            bc = np.bincount(M[~good_mask, ci], minlength=k) + 1.0
            l = gc / gc.sum()
            g = bc / bc.sum()
            picks = rng.choice(k, size=n_candidates, p=l)
            cand_idx[:, ci] = picks
            score += np.log(l[picks]) - np.log(g[picks])

        best = int(np.argmax(score))
        cand = {c: doms[c][cand_idx[best, ci]] for ci, c in enumerate(cols)}
        observe(cand)
        if deadline and time.time() > deadline:
            break
    return best_x, best_f, history
