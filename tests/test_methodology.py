"""A-tier 방법론 수정(Fable_feedback A1~A4) 회귀 테스트.

  A1: reference_ceiling — 앙상블 max ≥ 각 탐색기, 메타데이터 포함
  A2: noise_scale = noise_frac × 주효과 스프레드(닫힌형)
  A3: MinMaxNormalizer clip knob + saturation_fraction
  A4: integrity — config 해시/fingerprint 검증이 변조를 잡는지
"""
import dataclasses

import numpy as np
import pytest

from benchmark import configs
from benchmark.generator import BlackBoxBenchmark, OBJECTIVES
from benchmark.integrity import config_hash, integrity_record, verify_artifact
from benchmark.scoring import MinMaxNormalizer


@pytest.fixture(scope="module")
def bm1():
    return BlackBoxBenchmark(configs.BM1)


@pytest.fixture(scope="module")
def bm3():
    return BlackBoxBenchmark(configs.BM3)


# ---------------------------------------------------------------------------
# A2: 노이즈 = 주효과 스프레드 기준
# ---------------------------------------------------------------------------
def test_noise_scale_is_main_effect_spread(bm3):
    """noise_scale 이 주효과 분산 합의 제곱근 × noise_frac 과 일치(닫힌형)."""
    for mi, m in enumerate(OBJECTIVES):
        main_var = sum(float(tab.var()) for tab in bm3.main[m].values())
        expected = bm3.cfg.noise_frac * np.sqrt(main_var)
        assert bm3.noise_scale[mi] == pytest.approx(expected, rel=1e-12)


def test_noise_scale_excludes_interactions(bm3):
    """교호가 강한 BM3에서 주효과 기준 노이즈 < 전체 스프레드 기준 노이즈."""
    assert np.all(bm3.main_effect_spread < bm3.total_spread)
    assert np.all(bm3.noise_scale < bm3.cfg.noise_frac * bm3.total_spread)


def test_noise_scale_bm1_nearly_total_spread(bm1):
    """교호가 없는 BM1은 주효과 스프레드 ≈ 전체 스프레드(가법성 검산)."""
    ratio = bm1.main_effect_spread / bm1.total_spread
    assert np.all(ratio > 0.9) and np.all(ratio < 1.1)


# ---------------------------------------------------------------------------
# A3: clip knob + saturation_fraction
# ---------------------------------------------------------------------------
def test_clip_knob():
    norm = MinMaxNormalizer([0.0], [10.0], [True], clip=False)
    z = norm.transform([[20.0]])
    assert z[0, 0] == pytest.approx(2.0)          # clip 안 함
    norm_c = MinMaxNormalizer([0.0], [10.0], [True], clip=True)
    assert norm_c.transform([[20.0]])[0, 0] == pytest.approx(1.0)


def test_saturation_fraction():
    norm = MinMaxNormalizer([0.0, 0.0], [10.0, 10.0], [True, False])
    Y = np.array([[5.0, 5.0],     # 둘 다 범위 안
                  [20.0, -5.0],   # obj0 상단 포화, obj1(min) 상단 포화(z>1)
                  [-5.0, 20.0]])  # obj0 하단, obj1 하단
    low, high = norm.saturation_fraction(Y)
    assert low == pytest.approx([1 / 3, 1 / 3])
    assert high == pytest.approx([1 / 3, 1 / 3])


# ---------------------------------------------------------------------------
# A1: reference_ceiling (작은 예산으로 구조만 검증 — 값 자체는 build 가 담당)
# ---------------------------------------------------------------------------
def test_reference_ceiling_is_ensemble_max(bm1):
    d = bm1.reference_ceiling("sum", seed=0, n_restart=2, n_sweep=2,
                              global_budget=800, sa_budget=800, ga_budget=800)
    assert set(d["by_searcher"]) == {"coord_multistart", "block_coord", "sa", "ga"}
    assert d["utility"] == pytest.approx(max(d["by_searcher"].values()))
    assert d["winner"] in d["by_searcher"]
    assert d["spread"] >= 0.0
    assert d["x"].shape == (30,)
    # 하위호환 래퍼도 동일 값
    x, u = bm1.reference_optimum("sum", seed=0, n_restart=2, n_sweep=2,
                                 global_budget=800, sa_budget=800, ga_budget=800)
    assert u == pytest.approx(d["utility"])


# ---------------------------------------------------------------------------
# A4: integrity 검증
# ---------------------------------------------------------------------------
def test_integrity_roundtrip(bm1):
    art = {"integrity": integrity_record(configs.BM1, bm1)}
    verify_artifact(configs.BM1, bm1, art)        # 통과해야 함


def test_integrity_detects_config_change(bm1):
    art = {"integrity": integrity_record(configs.BM1, bm1)}
    tampered = dataclasses.replace(configs.BM1, conflict_rho=0.99)
    with pytest.raises(RuntimeError, match="config"):
        verify_artifact(tampered, bm1, art)


def test_integrity_detects_function_change(bm1, bm3):
    """다른 BM 인스턴스(=다른 함수)를 들이밀면 fingerprint 가 잡는다."""
    art = {"integrity": integrity_record(configs.BM1, bm1)}
    art["integrity"]["config_sha256"] = config_hash(configs.BM3)  # 해시는 통과시키고
    with pytest.raises(RuntimeError):
        verify_artifact(configs.BM3, bm3, art)


def test_integrity_rejects_legacy_artifact(bm1):
    with pytest.raises(RuntimeError, match="integrity"):
        verify_artifact(configs.BM1, bm1, {"name": "BM1"})


def test_config_hash_deterministic():
    assert config_hash(configs.BM1) == config_hash(configs.BM1)
    assert config_hash(configs.BM1) != config_hash(configs.BM2)
