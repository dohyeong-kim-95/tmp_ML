"""블록-분해 메타 래퍼 — 임의의 base 옵티마이저에 블록 구조를 '공정하게' 주입.

block_coord_local 만 구조를 알아 유리했다는 지적에 답하기 위해, 어떤 base
알고리즘이든 동일한 블록-분해(block-coordinate) 스킴으로 감싼다:

  - 공통 marginal-balanced 초기점에서 incumbent 시작(모두 동일 초기 이점),
  - 라운드마다 common→set2→set1 블록을 돌며, 각 블록의 변수만 base 옵티마이저로
    최적화(나머지 좌표는 incumbent 고정) → base 는 저차원 부분문제만 본다,
  - common 을 매 라운드 재방문해 블록 간 결합 흡수.

base 어댑터는 SubProblem(부분공간 view) 위에서 '그대로' 동작한다(.dim/.levels/
.is_cat/.evaluate/.n 만 필요). 전역 예산은 SubProblem 이 하드캡으로 보장.

이렇게 하면 block_coord_local 은 사실상 block_decomp(coordinate-descent)이고,
random_blk/sa_blk/tpe_blk 등과 같은 잣대로 비교된다 → '블록 vs base' 분리.
"""
from __future__ import annotations

import numpy as np

from benchmark.generator import COMMON, SET1, SET2
from .design import marginal_balanced_design

BLOCKS = [("common", list(COMMON)), ("set2", list(SET2)), ("set1", list(SET1))]


class SubProblem:
    """전체 Problem 의 한 블록(active 변수)만 노출하는 부분공간 view."""

    def __init__(self, problem, incumbent, active, global_budget):
        self.problem = problem
        self.incumbent = incumbent
        self.active = list(active)
        self.global_budget = global_budget
        self.levels = np.asarray(problem.levels)[self.active]
        self.is_cat = [problem.is_cat[j] for j in self.active]
        self.dim = len(self.active)
        self.n = 0
        self.best_obs = -np.inf
        self.best_sub = None

    def evaluate(self, sub_x):
        # 전역 예산 하드캡: 초과 시 실제 평가 없이 상수 반환(overshoot 방지)
        if self.problem.n >= self.global_budget:
            return self.best_obs if np.isfinite(self.best_obs) else 0.0
        full = self.incumbent.copy()
        full[self.active] = np.asarray(sub_x, dtype=int)
        s = self.problem.evaluate(full)
        self.n += 1
        if s > self.best_obs:
            self.best_obs = s
            self.best_sub = np.asarray(sub_x, dtype=int).copy()
        return s


def make_block_decomp(base_fn, rounds=3):
    """base 어댑터를 블록-분해로 감싼 run(problem, budget, seed) 반환."""

    def run(problem, budget, seed):
        L = np.asarray(problem.levels)
        rng = np.random.default_rng(seed)

        # 공통 초기점(모두 동일 이점): marginal-balanced best
        n_init = max(problem.dim, min(2 * problem.dim, budget // 5))
        init = marginal_balanced_design(L, min(n_init, budget), rng)
        inc, best = None, -np.inf
        for i in range(init.shape[0]):
            if problem.n >= budget:
                break
            s = problem.evaluate(init[i])
            if s > best:
                best, inc = s, init[i].copy()
        if inc is None:
            inc = init[0].copy()

        # 라운드×블록 스케줄, 남은 예산을 블록크기 가중으로 동적 분배
        schedule = [(b, blk, len(blk)) for _ in range(rounds) for (b, blk) in BLOCKS]
        rem_weight = sum(w for _, _, w in schedule)
        for idx, (bname, blk, w) in enumerate(schedule):
            if problem.n >= budget:
                break
            rem_budget = budget - problem.n
            sub_budget = min(rem_budget, max(1, int(round(rem_budget * w / max(rem_weight, 1)))))
            rem_weight -= w
            sub = SubProblem(problem, inc, blk, budget)
            base_fn(sub, sub_budget, seed + 101 * idx + 7)
            if sub.best_sub is not None:
                inc = inc.copy()
                inc[blk] = sub.best_sub

        # 잔여 예산: common 재최적화로 소진(진전 없으면 종료)
        guard = 0
        while problem.n < budget and guard < 20:
            sub = SubProblem(problem, inc, BLOCKS[0][1], budget)
            base_fn(sub, budget - problem.n, seed + 9000 + guard)
            if sub.n == 0:
                break
            if sub.best_sub is not None:
                inc = inc.copy()
                inc[BLOCKS[0][1]] = sub.best_sub
            guard += 1

    return run
