"""scoring.py 3종 점수 단위테스트 — 수작업 계산값과 대조(C2).

AGENTS.md의 "benchmark 목적함수·노이즈·scoring 변경 금지" 제약을 지키기 위한
회귀 테스트. 이 값들이 바뀌면 scoring 정의 자체가 바뀐 것이므로 실패해야 한다.
"""
import numpy as np
import pytest

from benchmark.scoring import (
    MinMaxNormalizer, ScoreConfig, ScoreSystem,
    score_sum, score_chebyshev, score_owa,
)


@pytest.fixture
def normalizer():
    # obj0: maximize, range [0, 10]. obj1: minimize, range [0, 10].
    return MinMaxNormalizer(lo=[0.0, 0.0], hi=[10.0, 10.0],
                            maximize_mask=[True, False])


def test_minmax_normalize_direction(normalizer):
    # y = [5, 3] -> z = [(5-0)/10, 1-(3-0)/10] = [0.5, 0.7]
    z = normalizer.transform([[5.0, 3.0]])
    assert z.shape == (1, 2)
    assert z[0, 0] == pytest.approx(0.5)
    assert z[0, 1] == pytest.approx(0.7)


def test_minmax_clip_out_of_range(normalizer):
    z = normalizer.transform([[20.0, -20.0]])   # 범위 밖 -> [0,1]로 clip
    assert z[0, 0] == pytest.approx(1.0)
    assert z[0, 1] == pytest.approx(1.0)


def test_score_sum():
    z = np.array([[0.5, 0.7]])
    assert score_sum(z)[0] == pytest.approx(1.2)


def test_score_chebyshev():
    z = np.array([[0.5, 0.7]])
    # gap = [0.5, 0.3], rho=0.01 -> -(max(gap) + rho*sum(gap)) = -(0.5+0.008)
    val = score_chebyshev(z, weights=None, rho=0.01)[0]
    assert val == pytest.approx(-0.508)


def test_score_owa_k1_is_min():
    z = np.array([[0.5, 0.7, 0.2]])
    assert score_owa(z, k=1)[0] == pytest.approx(0.2)


def test_score_owa_k2_is_mean_of_two_smallest():
    z = np.array([[0.5, 0.7, 0.2]])
    # 오름차순 정렬 [0.2, 0.5, 0.7], bottom-2 평균 = (0.2+0.5)/2 = 0.35
    assert score_owa(z, k=2)[0] == pytest.approx(0.35)


def test_score_system_all_scores_consistent(normalizer):
    sys_ = ScoreSystem(normalizer, ScoreConfig(cheby_rho=0.01, owa_k=2))
    Y = np.array([[5.0, 3.0]])
    all_s = sys_.all_scores(Y)
    assert all_s["sum"][0] == pytest.approx(1.2)
    assert all_s["chebyshev"][0] == pytest.approx(-0.508)
    # k=2, 목적이 2개뿐이므로 bottom-2 평균 = sum/2
    assert all_s["owa"][0] == pytest.approx(0.6)
    for kind in ScoreSystem.KINDS:
        assert sys_.score(Y, kind)[0] == pytest.approx(all_s[kind][0])


def test_score_unknown_kind_raises(normalizer):
    sys_ = ScoreSystem(normalizer)
    with pytest.raises(ValueError):
        sys_.score(np.array([[5.0, 3.0]]), "unknown")
