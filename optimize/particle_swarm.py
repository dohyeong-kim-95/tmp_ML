"""
Particle Swarm Optimization (혼합 이산공간).

PSO는 연속공간 알고리즘이므로 연속 위치를 결정변수로 디코딩한다(표준 기법):
  - binary      : 위치 1차원, decode = 1 if pos > 0 else 0
  - ordinal     : 위치 1차원 ∈ [0, L-1], decode = round(clip(pos))
  - categorical : 위치 4차원, decode = argmax (4 level 중 최대)
속도/위치 갱신은 표준 vanilla PSO (관성 w, 인지 c1, 사회 c2).
"""
import numpy as np


class Layout:
    """연속벡터 <-> 결정변수 디코딩 레이아웃."""
    def __init__(self, prob):
        self.prob = prob
        self.dim_index = {}   # col -> (start, length)
        self.bounds_lo, self.bounds_hi = [], []
        d = 0
        for col in prob.vars:
            t, dom = prob.meta[col]
            if t == "bin":
                self.dim_index[col] = (d, 1)
                self.bounds_lo += [-5.0]; self.bounds_hi += [5.0]; d += 1
            elif t == "ord":
                self.dim_index[col] = (d, 1)
                self.bounds_lo += [0.0]; self.bounds_hi += [len(dom) - 1.0]; d += 1
            else:  # cat
                self.dim_index[col] = (d, 4)
                self.bounds_lo += [-5.0] * 4; self.bounds_hi += [5.0] * 4; d += 4
        self.dim = d
        self.bounds_lo = np.array(self.bounds_lo)
        self.bounds_hi = np.array(self.bounds_hi)

    def decode(self, pos):
        x = {}
        for col in self.prob.vars:
            s, L = self.dim_index[col]
            t, dom = self.prob.meta[col]
            if t == "bin":
                x[col] = 1 if pos[s] > 0 else 0
            elif t == "ord":
                x[col] = int(np.clip(round(pos[s]), 0, len(dom) - 1))
            else:
                x[col] = dom[int(np.argmax(pos[s:s + 4]))]
        return x


def particle_swarm(prob, max_eval=20000, n_particles=40,
                   w=0.72, c1=1.49, c2=1.49, seed=0):
    rng = np.random.default_rng(seed)
    lay = Layout(prob)
    dim = lay.dim
    lo, hi = lay.bounds_lo, lay.bounds_hi
    span = hi - lo

    pos = lo + rng.random((n_particles, dim)) * span
    vel = (rng.random((n_particles, dim)) - 0.5) * span * 0.1

    pbest = pos.copy()
    pbest_f = np.array([prob.objective(lay.decode(p)) for p in pos])
    g = int(np.argmax(pbest_f))
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    history = [gbest_f]

    n_iter = (max_eval - n_particles) // n_particles
    for _ in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        # 속도 클램프
        vmax = 0.2 * span
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        # 위치 경계 처리
        pos = np.clip(pos, lo, hi)

        for i in range(n_particles):
            f = prob.objective(lay.decode(pos[i]))
            if f > pbest_f[i]:
                pbest_f[i], pbest[i] = f, pos[i].copy()
                if f > gbest_f:
                    gbest_f, gbest = f, pos[i].copy()
        history.append(gbest_f)

    return lay.decode(gbest), gbest_f, history
