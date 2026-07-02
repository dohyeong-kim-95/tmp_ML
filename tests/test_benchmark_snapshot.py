"""BM1~BM4 raw Y 스냅샷 테스트 — 벤치마크 불변 보증(C2, A4 관련 회귀 안전망).

AGENTS.md 제약("benchmark 목적함수·노이즈·scoring 변경 금지")을 지키기 위한
회귀 테스트. generator.py/configs.py 를 의도치 않게 건드리면(리팩토링,
numpy 버전업 등) 이 테스트가 값이 바뀐 걸 잡아낸다. 값은 numpy==2.4.6 환경에서
1회 실행해 고정한 스냅샷(정답이 아니라 "지금과 같은가"를 확인).
"""
import numpy as np
import pytest

from benchmark import configs
from benchmark.generator import BlackBoxBenchmark, N_VARS

# {BM: {label: (x, expected raw Y)}}. x는 levels 에 의존하므로 build_x()로 생성.
EXPECTED = {
    "BM1": {
        "zeros": [-4.19083, -2.594662, -3.809659, 1.560189, -2.113472, 0.08016],
        "mid": [2.368381, -1.218255, 0.286799, -0.524379, 0.140687, -2.429093],
        "max": [5.795135, 4.586139, 6.122763, -4.023135, 2.775127, 0.513644],
    },
    "BM2": {
        "zeros": [-3.100188, 6.949578, -2.702143, -1.013415, 2.75372, 4.617825],
        "mid": [-2.361654, 1.725551, -4.01059, -5.554947, -2.669307, 4.300782],
        "max": [3.829025, -4.80802, 2.562272, 3.032343, 9.456035, -2.358401],
    },
    "BM3": {
        "zeros": [-1.787668, 13.302378, 1.404546, -11.610969, 4.085464, 9.149922],
        "mid": [15.759671, 10.892098, 2.42396, 16.719117, 7.687269, -7.200834],
        "max": [0.907241, -0.517393, 2.93101, -4.567198, 12.249435, 18.291919],
    },
    "BM4": {
        "zeros": [-6.532066, 7.510798, -18.509417, -22.266763, 11.126386, -0.952741],
        "mid": [5.542461, -5.234658, -10.138048, 1.637182, -7.02602, 10.209203],
        "max": [-10.933725, -8.983956, 11.77499, 21.345022, 12.78935, -3.529264],
    },
}


def _build_x(levels, label):
    L = np.asarray(levels)
    if label == "zeros":
        return np.zeros(N_VARS, dtype=int)
    if label == "mid":
        return (L // 2).astype(int)
    if label == "max":
        return (L - 1).astype(int)
    raise ValueError(label)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_raw_y_snapshot(name):
    bm = BlackBoxBenchmark(configs.ALL[name])
    for label, expected in EXPECTED[name].items():
        x = _build_x(bm.levels, label)
        y = bm.raw(x[None, :])[0]
        assert y == pytest.approx(expected, abs=1e-5), (
            f"{name}/{label}: raw Y 가 스냅샷과 다름 — "
            f"generator.py/configs.py 가 의도치 않게 바뀌었을 수 있음"
        )


def test_raw_y_deterministic_across_instances():
    """같은 config 로 다시 만든 인스턴스도 동일한 raw Y (seed 기반 결정성)."""
    bm_a = BlackBoxBenchmark(configs.BM1)
    bm_b = BlackBoxBenchmark(configs.BM1)
    x = np.zeros(N_VARS, dtype=int)
    assert bm_a.raw(x[None, :])[0] == pytest.approx(bm_b.raw(x[None, :])[0])
