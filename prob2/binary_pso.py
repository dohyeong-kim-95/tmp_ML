"""
혼합형 PSO (binary/ordinal/categorical 모두 제대로 추적).

이전 순수 비트인코딩의 문제(ordinal Hamming 절벽, categorical 가짜 인접성)를 해결:
  - binary      : 1차원, BPSO 시그모이드 확률 비트  (bit = 1 if rand < sigmoid(v))
  - ordinal     : 1차원 연속위치 ∈ [0, L-1], 반올림 디코딩 → '순서/인접성 보존'
  - categorical : L차원 레벨 score, argmax 디코딩 → 레벨 선호를 연속적으로 추적

속도/관성: Clerc constriction 표준상수(w=0.729, c1=c2=1.49445) — 예산(max_eval/시간)에
비종속 → TIME/EVAL 공정. 속도클램프는 차원 종류별로 스케일 맞춤.
"""
import time
import numpy as np


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -60, 60)))


class Layout:
    def __init__(self, prob):
        self.prob = prob
        self.spec = []          # (col, kind, start, ndim, L)
        lo, hi, vmax, isbin = [], [], [], []
        d = 0
        for col in prob.vars:
            t, dom = prob.meta[col]
            if t == "bin":
                self.spec.append((col, "bin", d, 1, 2))
                lo += [-6.]; hi += [6.]; vmax += [6.]; isbin += [True]; d += 1
            elif t == "ord":
                L = len(dom)
                self.spec.append((col, "ord", d, 1, L))
                lo += [0.]; hi += [L - 1.]; vmax += [max(1.0, 0.5 * (L - 1))]
                isbin += [False]; d += 1
            else:
                L = len(dom)
                self.spec.append((col, "cat", d, L, L))
                lo += [-6.] * L; hi += [6.] * L; vmax += [4.0] * L
                isbin += [False] * L; d += L
        self.dim = d
        self.lo = np.array(lo); self.hi = np.array(hi)
        self.vmax = np.array(vmax); self.is_bin = np.array(isbin)

    def init_pos(self, rng):
        cont = self.lo + rng.random(self.dim) * (self.hi - self.lo)
        bits = (rng.random(self.dim) < 0.5).astype(float)
        return np.where(self.is_bin, bits, cont)

    def step_pos(self, pos, vel, rng):
        """binary 차원은 시그모이드 비트 샘플, 나머지는 위치=clip(위치+속도)."""
        bits = (rng.random(self.dim) < _sigmoid(vel)).astype(float)
        cont = np.clip(pos + vel, self.lo, self.hi)
        return np.where(self.is_bin, bits, cont)

    def decode(self, pos):
        x = {}
        for col, kind, s, nd, L in self.spec:
            if kind == "bin":
                x[col] = int(pos[s])
            elif kind == "ord":
                x[col] = int(np.clip(round(pos[s]), 0, L - 1))
            else:
                x[col] = self.prob.meta[col][1][int(np.argmax(pos[s:s + nd]))]
        return x


def binary_pso(prob, max_eval=2000, n_particles=30,
               w=0.729, c1=1.49445, c2=1.49445, seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    lay = Layout(prob)
    D = lay.dim
    vmax = lay.vmax

    X = np.array([lay.init_pos(rng) for _ in range(n_particles)])
    V = (rng.random((n_particles, D)) - 0.5) * vmax * 0.2

    def evalp(p):
        return prob.objective(lay.decode(p))

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
        X = np.array([lay.step_pos(X[i], V[i], rng) for i in range(n_particles)])
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
