"""BM1/2/3 인스턴스를 만들고 참조 최적/정규화 범위를 산출해 저장한다.

산출물(benchmark/artifacts/<BM>.json):
  - config, cardinality, 블록 분할
  - 목적별 정규화 범위(goodness lo/hi)
  - 3종 scalarization 의 참조 최적 효용/해
  - 난이도 점검: 랜덤서치가 예산 내 닫는 gap

실행: python -m benchmark.build
"""
from __future__ import annotations

import json
import os

import numpy as np

from .generator import BlackBoxBenchmark, OBJECTIVES, COMMON, SET1, SET2
from . import configs

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
BUDGETS = (180, 780, 2400)


def random_search_best_utility(bm, kind, budget, seed):
    rng = np.random.default_rng(seed)
    X = bm.random_X(rng, budget)
    return float(bm.utility(X, kind).max())


def build_one(cfg):
    bm = BlackBoxBenchmark(cfg)
    rec = {
        "name": cfg.name,
        "config": {
            k: getattr(cfg, k)
            for k in ("seed", "n_harmonics", "interaction_density",
                      "interaction_strength", "n_three_way", "conflict_rho",
                      "noise_frac", "n_strong", "weak_ratio", "cheby_rho", "owa_k")
        },
        "layout": {
            "common_cols": COMMON, "set1_cols": SET1, "set2_cols": SET2,
            "levels": [int(x) for x in bm.levels],
            "is_cat": [bool(x) for x in bm.is_cat],
            "space_size_log10": float(np.log10(bm.levels.astype(float)).sum()),
        },
        "noise_scale": [float(x) for x in bm.noise_scale],
        "norm_lo": [float(x) for x in bm._lo],
        "norm_hi": [float(x) for x in bm._hi],
        "reference_optimum": {},
        "difficulty_random_search": {},
    }
    for kind in BlackBoxBenchmark.SCALARIZATIONS:
        x_star, u_star = bm.reference_optimum(kind, seed=100)
        rec["reference_optimum"][kind] = {
            "utility": u_star,
            "x": [int(v) for v in x_star],
        }
        # 난이도 점검: 랜덤서치 best 가 참조최적 대비 닫는 비율(여러 seed 평균)
        rec["difficulty_random_search"][kind] = {}
        for b in BUDGETS:
            vals = [random_search_best_utility(bm, kind, b, s) for s in range(5)]
            rec["difficulty_random_search"][kind][str(b)] = {
                "rs_best_mean": float(np.mean(vals)),
                "ref_optimum": u_star,
            }
    return bm, rec


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    summary = []
    for name, cfg in configs.ALL.items():
        bm, rec = build_one(cfg)
        path = os.path.join(ART_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        # equal 기준 난이도 한눈에
        rs = rec["difficulty_random_search"]["equal"]["180"]
        summary.append((name, rec["layout"]["space_size_log10"],
                        rs["rs_best_mean"], rs["ref_optimum"]))
        print(f"[{name}] saved -> {path}")
        print(f"    space 10^{rec['layout']['space_size_log10']:.2f}, "
              f"noise_scale~{np.mean(rec['noise_scale']):.3f}")
        for kind in BlackBoxBenchmark.SCALARIZATIONS:
            d = rec["difficulty_random_search"][kind]
            ref = rec["reference_optimum"][kind]["utility"]
            gaps = {b: ref - d[str(b)]["rs_best_mean"] for b in BUDGETS}
            print(f"    [{kind:9s}] ref_opt={ref:.3f}  "
                  f"RS180={d['180']['rs_best_mean']:.3f} "
                  f"RS2400={d['2400']['rs_best_mean']:.3f}  "
                  f"gap180={gaps[180]:.3f}")

    print("\n=== 난이도 ladder 점검 (equal, budget=180) ===")
    print(f"{'BM':4s} {'space':8s} {'RS_best':8s} {'ref_opt':8s} {'gap':8s}")
    for name, sp, rs, ref in summary:
        print(f"{name:4s} 10^{sp:5.2f} {rs:8.3f} {ref:8.3f} {ref-rs:8.3f}")


if __name__ == "__main__":
    main()
