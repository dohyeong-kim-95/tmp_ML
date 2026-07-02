"""marginal_balanced_design 의 marginal 균등성 테스트(C2)."""
import numpy as np
import pytest

from optim.design import marginal_balanced_design


@pytest.mark.parametrize("levels,n,seed", [
    ([3, 5, 2], 20, 0),
    ([3, 5, 2], 7, 1),      # n이 어떤 L로도 안 나눠떨어짐
    ([10, 4, 6, 2], 50, 42),
    ([2], 100, 7),
])
def test_marginal_counts_are_balanced(levels, n, seed):
    rng = np.random.default_rng(seed)
    X = marginal_balanced_design(levels, n, rng)
    assert X.shape == (n, len(levels))
    for j, L in enumerate(levels):
        counts = np.bincount(X[:, j], minlength=L)
        assert counts.shape[0] == L
        lo, hi = n // L, -(-n // L)     # floor(n/L), ceil(n/L)
        assert counts.min() >= lo
        assert counts.max() <= hi


def test_values_within_level_range():
    rng = np.random.default_rng(0)
    levels = [4, 7, 3]
    X = marginal_balanced_design(levels, 30, rng)
    for j, L in enumerate(levels):
        assert X[:, j].min() >= 0
        assert X[:, j].max() <= L - 1


def test_prefix_is_also_balanced():
    """임의 prefix(앞 k개)도 거의 균등해야(예산 체크포인트마다 공정, 설계 의도)."""
    rng = np.random.default_rng(3)
    levels = [5]
    X = marginal_balanced_design(levels, 25, rng)
    prefix = X[:5, 0]
    # 레벨 5개 중 첫 5개 prefix는 순열 1블록이므로 각 레벨이 정확히 1번씩.
    assert sorted(prefix.tolist()) == list(range(5))
