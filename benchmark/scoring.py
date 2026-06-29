"""점수(scalarization) 체계 — 벤치마크/실문제 공용 독립 모듈.

원시 6목적 Y(일부 최대화, 일부 최소화)를 **min-max 정규화(1=best)** 한 뒤
3종 점수로 단일화한다. 벤치마크(정답 범위 알려짐)와 실제 문제(관측치로
적응 정규화) 양쪽에서 동일하게 쓰도록 generator와 분리한다.

3종 점수(모두 '클수록 좋음'):
  - sum        : 정규화 후 단순합            (균형 baseline)
  - chebyshev  : augmented Chebyshev         (최악 성분 보호)
  - owa        : bottom-k OWA(하위 k 평균)   (하위 목적 동반 향상)

정규화 방향: 최대화 목적은 (y-lo)/(hi-lo), 최소화 목적은 1-(y-lo)/(hi-lo)
→ 모든 목적이 [0,1], 1=best 로 정렬되어 합/Chebyshev/OWA가 일관됨.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# 정규화
# ---------------------------------------------------------------------------
class MinMaxNormalizer:
    """목적별 raw-y 범위로 [0,1] 정규화 (1=best). 방향(min/max)을 흡수한다."""

    def __init__(self, lo, hi, maximize_mask):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.maximize = np.asarray(maximize_mask, dtype=bool)

    @classmethod
    def from_samples(cls, Y, maximize_mask):
        """관측 표본의 목적별 min/max 로 정규화 범위 설정(실문제 적응용)."""
        Y = np.atleast_2d(np.asarray(Y, float))
        return cls(Y.min(axis=0), Y.max(axis=0), maximize_mask)

    def transform(self, Y):
        """raw Y(...,M) -> z(...,M) in [0,1], 1=best."""
        Y = np.atleast_2d(np.asarray(Y, float))
        span = np.maximum(self.hi - self.lo, 1e-12)
        t = (Y - self.lo) / span                 # raw 스케일 0..1
        z = np.where(self.maximize, t, 1.0 - t)  # 최소화 목적 반전
        return np.clip(z, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 순수 점수 함수 (정규화된 z 에 작용, 모두 클수록 좋음)
# ---------------------------------------------------------------------------
def score_sum(z):
    """정규화 후 단순합."""
    return np.asarray(z).sum(axis=-1)


def score_chebyshev(z, weights=None, rho=0.01):
    """Augmented Chebyshev (ideal=1). -(max_i w_i·gap_i + rho·Σ gap_i).

    최악 성분(gap=1-z 가 큰 목적)을 지배 항으로 삼아 '망한 지표'를 억제하고,
    증강항으로 weakly-dominated 해를 회피한다.
    """
    z = np.asarray(z, float)
    gap = 1.0 - z
    w = np.ones(z.shape[-1]) if weights is None else np.asarray(weights, float)
    return -((w * gap).max(axis=-1) + rho * gap.sum(axis=-1))


def score_owa(z, k=2):
    """Bottom-k OWA: 정규화 점수 하위 k개의 평균(하위 목적 동반 향상 강제)."""
    z = np.sort(np.asarray(z, float), axis=-1)   # 오름차순
    k = max(1, min(int(k), z.shape[-1]))
    return z[..., :k].mean(axis=-1)


# ---------------------------------------------------------------------------
# 점수 체계 묶음
# ---------------------------------------------------------------------------
@dataclass
class ScoreConfig:
    cheby_weights: object = None     # None=동일가중
    cheby_rho: float = 0.01
    owa_k: int = 2


class ScoreSystem:
    """정규화기 + 3종 점수를 하나로 묶어 raw Y -> 점수 제공."""

    KINDS = ("sum", "chebyshev", "owa")

    def __init__(self, normalizer: MinMaxNormalizer, cfg: ScoreConfig | None = None):
        self.norm = normalizer
        self.cfg = cfg or ScoreConfig()

    def z(self, Y):
        return self.norm.transform(Y)

    def score(self, Y, kind):
        z = self.z(Y)
        if kind == "sum":
            return score_sum(z)
        if kind == "chebyshev":
            return score_chebyshev(z, self.cfg.cheby_weights, self.cfg.cheby_rho)
        if kind == "owa":
            return score_owa(z, self.cfg.owa_k)
        raise ValueError(f"unknown score kind: {kind}")

    def all_scores(self, Y):
        z = self.z(Y)
        return {
            "sum": score_sum(z),
            "chebyshev": score_chebyshev(z, self.cfg.cheby_weights, self.cfg.cheby_rho),
            "owa": score_owa(z, self.cfg.owa_k),
        }
