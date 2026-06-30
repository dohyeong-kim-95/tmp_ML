"""최적화기-무관 문제 래퍼.

최적화기는 **노이즈가 있는 관측 점수**(실제 calculator 대응)를 최대화하고,
성능은 방문한 점들의 **노이즈 없는 참(true) 점수의 누적 최댓값(anytime)**으로 평가한다.
→ 알고리즘의 추천정책에 휘둘리지 않는 일관된 비교.
"""
from __future__ import annotations

import numpy as np

from benchmark.generator import N_VARS


class Problem:
    """단일 (benchmark, score-kind) 위에서 최대화 문제를 정의."""

    def __init__(self, bm, kind, seed):
        self.bm = bm
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.levels = np.asarray(bm.levels, dtype=int)
        self.is_cat = list(bm.is_cat)
        self.dim = N_VARS
        self.n = 0
        self.best_true = -np.inf
        self.best_x = None
        self.curve = []          # 각 eval 후 best_true (anytime)

    def evaluate(self, x) -> float:
        """관측(노이즈) 점수 반환. 내부적으로 참 점수 anytime 곡선 기록.

        raw(x)를 1회만 계산해 노이즈 관측·참 점수에 함께 사용(이전엔 bm.evaluate와
        bm.score가 각각 raw를 호출 → 2회). 의미 동일, eval당 ~2배 빠름.
        """
        x = np.asarray(x, dtype=int).reshape(-1)
        y = self.bm.raw(x[None, :])
        y_noisy = y + self.rng.normal(scale=self.bm.noise_scale, size=y.shape)
        s_obs = float(self.bm.scorer.score(y_noisy, self.kind)[0])
        s_true = float(self.bm.scorer.score(y, self.kind)[0])
        self.n += 1
        if s_true > self.best_true:
            self.best_true = s_true
            self.best_x = x.copy()
        self.curve.append(self.best_true)
        return s_obs

    def checkpoints(self, budgets):
        """예산별 best_true (곡선에서 추출)."""
        out = {}
        for b in budgets:
            idx = min(b, len(self.curve)) - 1
            out[b] = self.curve[idx] if idx >= 0 else -np.inf
        return out

    def random_x(self):
        x = np.empty(self.dim, dtype=int)
        for j in range(self.dim):
            x[j] = self.rng.integers(0, self.levels[j])
        return x
