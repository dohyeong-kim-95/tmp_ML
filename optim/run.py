"""포트폴리오 벤치마킹 실행기.

각 (algo, BM, kind, seed) 에 대해 max_budget 까지 1회 실행하고 곡선에서
예산별 best_true 를 추출한다. 성능은 정규화 closure 로 보고:

  closure = (best_true - floor) / (ref_opt - floor)

  floor   = 무작위 단일추출 점수 평균(해당 BM/kind)
  ref_opt = 참조 최적(artifacts/<BM>.json)
  → 0=무작위 수준, 1=참조최적. sum/owa/chebyshev 모두 부호 무관하게 동작.

실행 예:
  python -m optim.run --algos random,sobol,sa,ga,tpe --max-budget 780 --seeds 3
  python -m optim.run --algos smac --max-budget 780 --seeds 2 --append
  python -m optim.run --algos botorch --max-budget 780 --seeds 1 --append
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from benchmark.generator import BlackBoxBenchmark
from benchmark import configs
from benchmark.scoring import ScoreSystem
from .problem import Problem
from .algos import REGISTRY

ART_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "benchmark", "artifacts")
OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "results.json")


def floor_score(bm, kind, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X = bm.random_X(rng, n)
    return float(bm.score(X, kind).mean())


def load_ref(name, kind):
    with open(os.path.join(ART_DIR, f"{name}.json")) as f:
        return json.load(f)["reference_optimum"][kind]["utility"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="random,sobol,sa,ga,tpe,smac,botorch")
    ap.add_argument("--bms", default="BM1,BM2,BM3")
    ap.add_argument("--kinds", default=",".join(ScoreSystem.KINDS))
    ap.add_argument("--max-budget", type=int, default=780)
    ap.add_argument("--budgets", default="180,780")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    algos = args.algos.split(",")
    bms = args.bms.split(",")
    kinds = args.kinds.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]
    checkpoints = [b for b in budgets if b <= args.max_budget]

    res = {"meta": {}, "floor": {}, "ref_opt": {}, "runs": {}}
    if args.append and os.path.exists(args.out):
        with open(args.out) as f:
            res = json.load(f)
    res["meta"] = {"budgets": budgets, "max_budget": args.max_budget,
                   "seeds": args.seeds}

    # BM 인스턴스 캐시(결정적; 노이즈는 problem 단계에서)
    bm_cache = {name: BlackBoxBenchmark(configs.ALL[name]) for name in bms}
    for name in bms:
        bm = bm_cache[name]
        res["floor"].setdefault(name, {})
        res["ref_opt"].setdefault(name, {})
        for kind in kinds:
            if kind not in res["floor"][name]:
                res["floor"][name][kind] = floor_score(bm, kind)
            if kind not in res["ref_opt"][name]:
                res["ref_opt"][name][kind] = load_ref(name, kind)

    for algo in algos:
        run_fn = REGISTRY[algo]
        res["runs"].setdefault(algo, {})
        for name in bms:
            bm = bm_cache[name]
            res["runs"][algo].setdefault(name, {})
            for kind in kinds:
                cells = {str(b): [] for b in checkpoints}
                t0 = time.time()
                for s in range(args.seeds):
                    prob = Problem(bm, kind, seed=1000 * s + 7)
                    run_fn(prob, args.max_budget, seed=s)
                    cp = prob.checkpoints(checkpoints)
                    for b in checkpoints:
                        cells[str(b)].append(cp[b])
                dt = time.time() - t0
                floor = res["floor"][name][kind]
                ref = res["ref_opt"][name][kind]
                denom = max(ref - floor, 1e-9)
                closure = {b: (float(np.mean(cells[str(b)])) - floor) / denom
                           for b in checkpoints}
                res["runs"][algo][name][kind] = {
                    "best_true": cells,
                    "closure": {str(b): closure[b] for b in checkpoints},
                    "sec": dt,
                }
                print(f"[{algo:8s}] {name} {kind:9s} "
                      + " ".join(f"clo@{b}={closure[b]:.2%}" for b in checkpoints)
                      + f"  ({dt:.1f}s)")

        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
