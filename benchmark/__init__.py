"""합성 black-box 최적화 벤치마크."""
from .generator import (
    BlackBoxBenchmark,
    BMConfig,
    OBJECTIVES,
    MAXIMIZE,
    MINIMIZE,
    COMMON,
    SET1,
    SET2,
)
from .scoring import (
    MinMaxNormalizer,
    ScoreConfig,
    ScoreSystem,
    score_sum,
    score_chebyshev,
    score_owa,
)
from . import configs

__all__ = [
    "BlackBoxBenchmark",
    "BMConfig",
    "OBJECTIVES",
    "MAXIMIZE",
    "MINIMIZE",
    "COMMON",
    "SET1",
    "SET2",
    "MinMaxNormalizer",
    "ScoreConfig",
    "ScoreSystem",
    "score_sum",
    "score_chebyshev",
    "score_owa",
    "configs",
]
