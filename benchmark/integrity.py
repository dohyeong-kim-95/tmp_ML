"""Artifact 정합성 검증 (A4) — stale artifact 를 조용히 쓰는 사고 방지.

artifacts/<BM>.json 의 ref_opt/정규화범위는 '그 시점의 configs.py + generator.py
+ numpy 버전'으로 만든 BM 인스턴스에 대한 값이다. 이후 configs 를 고치고 build 를
안 돌리거나, numpy 버전이 바뀌어 Generator 비트스트림이 달라지면, run.py 가
**다른 함수에 대한 천장**으로 closure 를 계산하게 된다(조용한 오염).

방어 2단:
  - config_sha256 : BMConfig 전체 필드(레이아웃 포함)의 해시 → configs.py 변경 감지
  - fingerprint   : 고정 seed 로 뽑은 X 10개의 raw Y + noise_scale
                    → generator 코드/NumPy 스트림 변경 감지
build.py 가 integrity 레코드를 artifact 에 저장하고, run.py 가 로드 시
verify_artifact() 로 대조해 불일치면 명확한 에러로 중단한다.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

FP_SEED = 20260702   # fingerprint 용 고정 X 를 뽑는 seed (임의 고정값)
FP_N = 10            # fingerprint X 개수


def config_hash(cfg) -> str:
    """BMConfig 의 결정적 sha256 (레벨/타입 레이아웃 포함)."""
    payload = {
        "name": cfg.name,
        "seed": cfg.seed,
        "n_harmonics": cfg.n_harmonics,
        "interaction_density": cfg.interaction_density,
        "interaction_strength": cfg.interaction_strength,
        "n_three_way": cfg.n_three_way,
        "conflict_rho": cfg.conflict_rho,
        "noise_frac": cfg.noise_frac,
        "n_strong": cfg.n_strong,
        "weak_ratio": cfg.weak_ratio,
        "cheby_rho": cfg.cheby_rho,
        "owa_k": cfg.owa_k,
        "levels": [int(v) for v in cfg.levels],
        "is_cat": [bool(v) for v in cfg.is_cat],
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def fingerprint(bm) -> dict:
    """BM 인스턴스의 함수 지문: 고정 X 의 raw Y + noise_scale."""
    rng = np.random.default_rng(FP_SEED)
    X = bm.random_X(rng, FP_N)
    Y = bm.raw(X)
    return {
        "raw_y": [[round(float(v), 9) for v in row] for row in Y],
        "noise_scale": [round(float(v), 9) for v in bm.noise_scale],
    }


def integrity_record(cfg, bm) -> dict:
    """build.py 가 artifact 에 저장할 integrity 레코드."""
    return {
        "config_sha256": config_hash(cfg),
        "numpy_version": np.__version__,
        "fingerprint": fingerprint(bm),
    }


def verify_artifact(cfg, bm, artifact: dict, path: str = "<artifact>") -> None:
    """현재 config/BM 인스턴스와 artifact 의 정합성 검증. 불일치 시 RuntimeError."""
    rec = artifact.get("integrity")
    if rec is None:
        raise RuntimeError(
            f"{path}: integrity 레코드가 없는 구버전 artifact 입니다. "
            f"`python -m benchmark.build` 로 재생성하세요."
        )
    if rec["config_sha256"] != config_hash(cfg):
        raise RuntimeError(
            f"{path}: configs.py 의 {cfg.name} 설정이 artifact 생성 시점과 다릅니다 "
            f"(config_sha256 불일치). ref_opt/정규화범위가 stale 하므로 "
            f"`python -m benchmark.build` 로 재생성하세요."
        )
    fp_now = fingerprint(bm)
    fp_old = rec["fingerprint"]
    if not np.allclose(np.asarray(fp_old["raw_y"], dtype=float),
                       np.asarray(fp_now["raw_y"], dtype=float), atol=1e-6):
        raise RuntimeError(
            f"{path}: BM 함수 지문(raw Y) 불일치 — generator.py 코드 또는 NumPy "
            f"버전(artifact: numpy=={rec.get('numpy_version', '?')}, 현재: "
            f"numpy=={np.__version__}) 차이로 벤치마크 함수 자체가 달라졌습니다. "
            f"`python -m benchmark.build` 로 재생성 후 기존 결과와의 비교 가능성을 "
            f"재검토하세요."
        )
    if not np.allclose(np.asarray(fp_old["noise_scale"], dtype=float),
                       np.asarray(fp_now["noise_scale"], dtype=float), atol=1e-9):
        raise RuntimeError(
            f"{path}: noise_scale 불일치 — 노이즈 산정 방식이 artifact 생성 시점과 "
            f"다릅니다. `python -m benchmark.build` 로 재생성하세요."
        )
