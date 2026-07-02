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

    def __init__(self, bm, kind, seed, budget=None):
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
        self.budget = budget     # 알려진 예산(선택). 초과 호출 시 경고에 사용.
        self._overshoot_warned = False
        self.last_incomplete = {}

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
        if (self.budget is not None and self.n > self.budget
                and not self._overshoot_warned):
            print(f"[Problem] warning: evaluate() 가 budget={self.budget} 을 넘어 "
                  f"호출됨(n={self.n}) — 어댑터가 예산을 초과 평가하고 있음")
            self._overshoot_warned = True
        if s_true > self.best_true:
            self.best_true = s_true
            self.best_x = x.copy()
        self.curve.append(self.best_true)
        return s_obs

    def checkpoints(self, budgets):
        """예산별 best_true (곡선에서 추출).

        len(curve) < b 이면 어댑터가 그 예산만큼 평가를 채우지 못했다는 뜻(조기
        종료/예외) — 경고를 출력하고 self.last_incomplete[b]=True 로 표시한다.
        반환 형식(dict[int, float])은 기존과 동일(하위호환).
        """
        out = {}
        self.last_incomplete = {}
        for b in budgets:
            incomplete = len(self.curve) < b
            self.last_incomplete[b] = incomplete
            if incomplete:
                print(f"[Problem] warning: checkpoint budget={b} 요청했지만 "
                      f"{len(self.curve)}회만 평가됨 — incomplete 로 표시")
            idx = min(b, len(self.curve)) - 1
            out[b] = self.curve[idx] if idx >= 0 else -np.inf
        return out

    def random_x(self):
        x = np.empty(self.dim, dtype=int)
        for j in range(self.dim):
            x[j] = self.rng.integers(0, self.levels[j])
        return x
