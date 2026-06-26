"""
Binary PSO (Kennedy & Eberhart, 1997).

모든 결정변수를 비트로 인코딩해 진짜 이진공간에서 탐색:
  - binary      : 1 bit
  - ordinal L   : ceil(log2(L)) bit  (정수 디코딩, L-1 초과는 clamp)
  - categorical : 2 bit (4 level)
속도 갱신은 연속, 위치는 확률적 비트:  x_d = 1 if rand < sigmoid(v_d) else 0
"""
import time
import numpy as np


class BitLayout:
    def __init__(self, prob):
        self.prob = prob
        self.spec = []   # (col, type, n_bits, L)
        d = 0
        for col in prob.vars:
            t, dom = prob.meta[col]
            if t == "bin":
                nb = 1
            elif t == "ord":
                nb = int(np.ceil(np.log2(len(dom))))
            else:
                nb = 2
            self.spec.append((col, t, d, nb, len(dom)))
            d += nb
        self.dim = d

    def decode(self, bits):
        x = {}
        for col, t, start, nb, L in self.spec:
            chunk = bits[start:start + nb]
            val = int("".join(str(int(b)) for b in chunk), 2) if nb > 0 else 0
            if t == "bin":
                x[col] = int(val)
            elif t == "ord":
                x[col] = min(val, L - 1)
            else:
                x[col] = self.prob.meta[col][1][min(val, L - 1)]
        return x


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -60, 60)))


def binary_pso(prob, max_eval=2000, n_particles=30,
               w=0.72, c1=1.49, c2=1.49, vmax=6.0, seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    lay = BitLayout(prob)
    D = lay.dim

    X = (rng.random((n_particles, D)) < 0.5).astype(float)
    V = (rng.random((n_particles, D)) - 0.5) * 2 * vmax

    def evalp(b):
        return prob.objective(lay.decode(b))

    pbest = X.copy()
    pbest_f = np.array([evalp(X[i]) for i in range(n_particles)])
    g = int(np.argmax(pbest_f))
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    history = [gbest_f]

    n_iter = (max_eval - n_particles) // n_particles
    it = 0
    while it < n_iter:
        r1 = rng.random((n_particles, D))
        r2 = rng.random((n_particles, D))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = (rng.random((n_particles, D)) < _sigmoid(V)).astype(float)
        for i in range(n_particles):
            f = evalp(X[i])
            if f > pbest_f[i]:
                pbest_f[i], pbest[i] = f, X[i].copy()
                if f > gbest_f:
                    gbest_f, gbest = f, X[i].copy()
        history.append(gbest_f)
        it += 1
        if deadline and time.time() > deadline:
            break
    return lay.decode(gbest), gbest_f, history
