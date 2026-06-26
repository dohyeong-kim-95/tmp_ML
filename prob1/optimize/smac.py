"""
SMAC 스타일 최적화 — Random Forest 대리모델 + Expected Improvement.

SMAC의 핵심은 GP 대신 랜덤포레스트를 surrogate로 쓰는 것.
  - 예측평균 = 트리 평균, 불확실성 = 트리간 표준편차
  - 획득함수 = EI, 후보풀(무작위+현재best mutation)에서 최대 EI 선택
이산/혼합 변수는 binary + ordinal(scaled) + categorical(one-hot) 로 인코딩.
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

CAT_LEVELS = ["A", "B", "C", "D"]


class Enc:
    def __init__(self, prob):
        self.prob = prob

    def encode(self, x):
        v = []
        for c in self.prob.bin_cols:
            v.append(float(x[c]))
        for c in self.prob.ord_cols:
            v.append(float(x[c]) / (self.prob.ord_levels[c] - 1))
        for c in self.prob.cat_cols:
            v.extend([1.0 if x[c] == lv else 0.0 for lv in CAT_LEVELS])
        return np.array(v)


def _mutate(prob, x, rng, k):
    nx = dict(x)
    for col in rng.choice(prob.vars, size=k, replace=False):
        t, dom = prob.meta[col]
        if t == "bin":
            nx[col] = 1 - x[col]
        elif t == "ord":
            nx[col] = int(np.clip(x[col] + (1 if rng.random() < .5 else -1),
                                  0, len(dom) - 1))
        else:
            ch = [d for d in dom if d != x[col]]
            nx[col] = ch[rng.integers(len(ch))]
    return nx


def _rf_predict(rf, Xc):
    preds = np.stack([t.predict(Xc) for t in rf.estimators_])
    return preds.mean(0), preds.std(0)


def smac(prob, max_eval=500, n_init=20, gp_cap=300, n_estimators=60,
         refit_every=5, seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    enc = Enc(prob)

    Xr, Xf, Y = [], [], []
    best_x, best_f = None, -1e18
    history = []

    def observe(x):
        nonlocal best_x, best_f
        f = prob.objective(x)
        Xr.append(x); Xf.append(enc.encode(x)); Y.append(f)
        if f > best_f:
            best_f, best_x = f, dict(x)
        history.append(best_f)

    for _ in range(min(n_init, max_eval)):
        observe(prob.random_solution(rng))
        if deadline and time.time() > deadline:
            return best_x, best_f, history

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=None,
                               min_samples_leaf=2, n_jobs=-1, random_state=seed)
    step = 0
    while prob.n_eval < max_eval:
        if len(Y) > gp_cap:
            order = np.argsort(Y)[::-1]
            idx = sorted(set(order[:gp_cap // 2].tolist())
                         | set(range(len(Y) - gp_cap // 2, len(Y))))
        else:
            idx = list(range(len(Y)))
        if step % refit_every == 0:
            rf.fit(np.array([Xf[i] for i in idx]), np.array([Y[i] for i in idx]))

        cands = [prob.random_solution(rng) for _ in range(300)]
        cands += [_mutate(prob, best_x, rng, rng.integers(1, 4)) for _ in range(300)]
        Cf = np.array([enc.encode(c) for c in cands])
        mu, sd = _rf_predict(rf, Cf)
        sd = np.maximum(sd, 1e-9)
        z = (mu - best_f) / sd
        ei = (mu - best_f) * norm.cdf(z) + sd * norm.pdf(z)
        observe(cands[int(np.argmax(ei))])
        step += 1
        if deadline and time.time() > deadline:
            break
    return best_x, best_f, history
