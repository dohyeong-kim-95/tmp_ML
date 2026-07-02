"""BM1/2/3 인스턴스를 만들고 참조 최적/정규화 범위를 산출해 저장한다.

산출물(benchmark/artifacts/<BM>.json):
  - config, cardinality, 블록 분할
  - 목적별 정규화 범위(raw-y lo/hi)
  - 3종 점수(sum/chebyshev/owa)의 참조 최적/해
  - 난이도 점검: 랜덤서치 / 예산제한 local-search 의 gap-closure

실행: python -m benchmark.build
"""
from __future__ import annotations

import json
import os

import numpy as np

from .generator import BlackBoxBenchmark, OBJECTIVES, COMMON, SET1, SET2, N_VARS
from . import configs

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
BUDGETS = (180, 780, 2400)


def random_search_best_utility(bm, kind, budget, seed):
    rng = np.random.default_rng(seed)
    X = bm.random_X(rng, budget)
    return float(bm.score(X, kind).max())


def local_search_best_utility(bm, kind, budget, seed):
    """예산 제한 random-restart 좌표 hill-climbing.

    랜덤서치보다 변별력 높은 난이도 probe: 분리가능/단봉이면 거의 풀고(BM1),
    교호작용/다봉이 강하면 좌표이동이 막혀 gap이 커진다(BM3).
    (참조최적과 동일하게 노이즈 없는 효용으로 측정 → 탐색 난이도만 평가)
    """
    rng = np.random.default_rng(seed)
    used, best = 0, -np.inf
    while used < budget:
        x = bm.random_X(rng, 1)[0]
        used += 1
        v = float(bm.score(x[None, :], kind)[0])
        best = max(best, v)
        improved = True
        while improved and used < budget:
            improved = False
            for j in rng.permutation(N_VARS):
                L = int(bm.levels[j])
                if used + L > budget:
                    continue
                cand = np.tile(x, (L, 1))
                cand[:, j] = np.arange(L)
                vals = bm.score(cand, kind)
                used += L
                bi = int(np.argmax(vals))
                if vals[bi] > v + 1e-12:
                    x, v = cand[bi].copy(), float(vals[bi])
                    improved = True
            best = max(best, v)
    return best


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
        "y_lo": [float(x) for x in bm.y_lo],
        "y_hi": [float(x) for x in bm.y_hi],
        "reference_optimum": {},
        "difficulty_random_search": {},
        "difficulty_local_search": {},
    }
    for kind in BlackBoxBenchmark.SCALARIZATIONS:
        x_star, u_star = bm.reference_optimum(kind, seed=100)
        rec["reference_optimum"][kind] = {
            "utility": u_star,
            "x": [int(v) for v in x_star],
        }
        # 난이도 점검: 랜덤서치 / 예산제한 local-search 가 참조최적 대비 닫는 정도
        rec["difficulty_random_search"][kind] = {}
        rec["difficulty_local_search"][kind] = {}
        for b in BUDGETS:
            rs = [random_search_best_utility(bm, kind, b, s) for s in range(5)]
            ls = [local_search_best_utility(bm, kind, b, s) for s in range(5)]
            rec["difficulty_random_search"][kind][str(b)] = {
                "rs_best_mean": float(np.mean(rs)), "ref_optimum": u_star,
            }
            rec["difficulty_local_search"][kind][str(b)] = {
                "ls_best_mean": float(np.mean(ls)), "ref_optimum": u_star,
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

        # 난이도 변별: 예산제한 local-search 의 gap-closure(참조최적 대비)
        ls = rec["difficulty_local_search"]["sum"]
        ref_eq = rec["reference_optimum"]["sum"]["utility"]
        closure = {b: ls[str(b)]["ls_best_mean"] / ref_eq for b in BUDGETS}
        summary.append((name, rec["layout"]["space_size_log10"],
                        closure[180], closure[2400], ref_eq))
        print(f"[{name}] saved -> {path}")
        print(f"    space 10^{rec['layout']['space_size_log10']:.2f}, "
              f"noise_scale~{np.mean(rec['noise_scale']):.3f}")
        for kind in BlackBoxBenchmark.SCALARIZATIONS:
            r = rec["difficulty_random_search"][kind]
            l = rec["difficulty_local_search"][kind]
            ref = rec["reference_optimum"][kind]["utility"]
            print(f"    [{kind:9s}] ref_opt={ref:.3f}  "
                  f"RS180={r['180']['rs_best_mean']:.3f}  "
                  f"LS180={l['180']['ls_best_mean']:.3f} "
                  f"LS2400={l['2400']['ls_best_mean']:.3f}")

    print("\n=== 난이도 ladder (sum): local-search gap-closure = LS_best/ref_opt ===")
    print(f"{'BM':4s} {'space':8s} {'closure@180':12s} {'closure@2400':12s} {'ref_opt':8s}")
    for name, sp, c180, c2400, ref in summary:
        print(f"{name:4s} 10^{sp:5.2f} {c180:11.2%} {c2400:11.2%} {ref:8.3f}")


if __name__ == "__main__":
    main()
