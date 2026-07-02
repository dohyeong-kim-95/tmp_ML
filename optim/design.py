"""혼합변수(categorical+ordinal) 이산공간용 초기설계.

Sobol을 정수공간에 floor 매핑하면 두 가지 문제가 있다:
  (1) categorical에 가짜 순서(fake ordering)를 부여하고,
  (2) 변수별 cardinality가 제각각이라 변수별 level marginal coverage가 불균형.

marginal_balanced_design: 각 변수를 '레벨 전체를 무작위 순열한 시퀀스를 반복'해
채운다. 그러면
  - 변수별 level marginal이 균등(각 레벨이 floor(n/L)~ceil(n/L)회 등장),
  - 임의 prefix(앞 k개)도 거의 균등(prefix-balanced) → 예산 체크포인트마다 공정,
  - 변수마다 독립 순열 → 차원 간 상관 낮음(Latin-hypercube 류),
  - categorical에 순서를 강제하지 않음(레벨을 그냥 순열).
"""
from __future__ import annotations

import numpy as np


def default_n_init(dim, budget):
    """초기설계(marginal-balanced) 크기 공식(공용).

    dim 이상, budget//5 이하, 상한 2*dim. algos.py(run_block_coord_local)와
    blockwrap.py(make_block_decomp)에 동일한 식이 복붙돼 있어 한쪽만 고치면
    초기설계 이점의 공정성이 깨지던 문제를 막기 위해 공용 함수로 추출.
    """
    return max(dim, min(2 * dim, budget // 5))


def marginal_balanced_design(levels, n, rng):
    """(n, dim) 정수 설계. 각 변수 j는 [0..levels[j]-1] 를 균등/순열 반복."""
    levels = np.asarray(levels, dtype=int)
    dim = len(levels)
    X = np.empty((n, dim), dtype=int)
    for j, L in enumerate(levels):
        seq = np.empty(n, dtype=int)
        filled = 0
        while filled < n:
            perm = rng.permutation(int(L))
            take = min(int(L), n - filled)
            seq[filled:filled + take] = perm[:take]
            filled += take
        X[:, j] = seq
    return X
