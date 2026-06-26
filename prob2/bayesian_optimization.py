"""
Vanilla Bayesian Optimization (GP + Expected Improvement).

혼합 이산공간 처리:
  - 특징벡터 = binary(0/1) + ordinal(scaled 0~1) + categorical(one-hot)
대리모델 : Gaussian Process (Matern 2.5 + WhiteKernel), y 정규화
획득함수 : Expected Improvement (EI)
후보생성 : 매 스텝 무작위 해 + 현재 best 의 mutation 풀에서 EI 최대 선택

주의: GP는 관측수 n 에 대해 O(n^3). 5000 eval 예산을 위해
      학습점을 best+recent 일부(cap)로 제한하는 표준 실무 기법을 쓴다.
"""
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm

CAT_LEVELS = ["A", "B", "C", "D"]


class Encoder:
    def __init__(self, prob):
        self.prob = prob

    def encode(self, x):
        v = []
        for c in self.prob.bin_cols:
            v.append(float(x[c]))
        for c in self.prob.ord_cols:
            L = self.prob.ord_levels[c]
            v.append(float(x[c]) / (L - 1))
        for c in self.prob.cat_cols:
            v.extend([1.0 if x[c] == lv else 0.0 for lv in CAT_LEVELS])
        return np.array(v)


def _mutate(prob, x, rng, k=2):
    nx = dict(x)
    cols = rng.choice(prob.vars, size=k, replace=False)
    for col in cols:
        t, dom = prob.meta[col]
        if t == "bin":
            nx[col] = 1 - x[col]
        elif t == "ord":
            step = 1 if rng.random() < 0.5 else -1
            nx[col] = int(np.clip(x[col] + step, 0, len(dom) - 1))
        else:
            ch = [d for d in dom if d != x[col]]
            nx[col] = ch[rng.integers(len(ch))]
    return nx


def _candidates(prob, best_x, rng, n_rand=400, n_mut=400):
    cands = [prob.random_solution(rng) for _ in range(n_rand)]
    if best_x is not None:
        cands += [_mutate(prob, best_x, rng, k=rng.integers(1, 4))
                  for _ in range(n_mut)]
    return cands


def bayesian_optimization(prob, max_eval=5000, n_init=20,
                          gp_cap=180, refit_every=25, seed=0, deadline=None,
                          warm=None):
    rng = np.random.default_rng(seed)
    enc = Encoder(prob)

    # --- 초기 관측 ---
    X_raw, X_feat, Y = [], [], []
    if warm is not None:
        # warm = (X_dicts, Y_values): 과거에 쌓인 '공짜' 관측으로 GP 사전 시드
        wx, wy = warm
        for x, y in zip(wx, wy):
            X_raw.append(dict(x)); X_feat.append(enc.encode(x)); Y.append(float(y))
    else:
        for _ in range(n_init):
            x = prob.random_solution(rng)
            X_raw.append(x); X_feat.append(enc.encode(x))
            Y.append(prob.objective(x))   # cold-start init은 예산 소모
    X_feat = list(X_feat); Y = list(Y)

    if warm is not None:
        # warm 라벨은 노이즈 포함 → incumbent/보고에 직접 쓰지 않는다.
        # best_*는 BO가 '실제(clean) 목적함수로 평가한 점'만으로 추적.
        best_x, best_f = None, -1e18
        mut_seed = dict(X_raw[int(np.argmax(Y))])   # 후보 생성용 시드(과거 best)
        history = []
    else:
        best_i = int(np.argmax(Y))
        best_x, best_f = dict(X_raw[best_i]), Y[best_i]
        mut_seed = best_x
        history = [max(Y[:i + 1]) for i in range(len(Y))]

    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-1, (1e-3, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=0, random_state=seed)

    step = 0
    while prob.n_eval < max_eval:
        # 학습점 cap: best 위주 + 최근점
        if len(Y) > gp_cap:
            order = np.argsort(Y)[::-1]
            keep = list(order[:gp_cap // 2])
            recent = list(range(len(Y) - gp_cap // 2, len(Y)))
            idx = sorted(set(keep) | set(recent))
        else:
            idx = list(range(len(Y)))
        Xt = np.array([X_feat[i] for i in idx])
        Yt = np.array([Y[i] for i in idx])

        # 하이퍼파라미터는 주기적으로만 재최적화(속도)
        if step % refit_every == 0:
            gp.set_params(optimizer="fmin_l_bfgs_b")
        else:
            gp.set_params(optimizer=None)
        gp.fit(Xt, Yt)

        # EI incumbent: clean 평가가 있으면 그 best, 없으면(warm 초기) 사후평균 최댓값(denoised)
        inc = best_f if best_f > -1e17 else float(gp.predict(Xt).max())

        cands = _candidates(prob, mut_seed, rng)
        Cf = np.array([enc.encode(c) for c in cands])
        mu, sd = gp.predict(Cf, return_std=True)
        sd = np.maximum(sd, 1e-9)
        imp = mu - inc
        z = imp / sd
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)
        order = np.argsort(ei)[::-1]

        # EI 최고이며 budget 내인 후보를 평가
        chosen = cands[order[0]]
        f = prob.objective(chosen)
        X_raw.append(chosen); X_feat.append(enc.encode(chosen)); Y.append(f)
        if f > best_f:
            best_f, best_x, mut_seed = f, dict(chosen), dict(chosen)
        history.append(best_f)
        step += 1
        if deadline and time.time() > deadline:
            break

    return best_x, best_f, history
