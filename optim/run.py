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
    ap.add_argument("--seed-list", default=None,
                    help="명시적 seed 인덱스(csv). 설정 시 --seeds 무시")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--merge-extend", action="store_true",
                    help="기존 셀의 seed 결과에 새 seed를 '추가'(덮어쓰지 않음)")
    args = ap.parse_args()

    algos = args.algos.split(",")
    bms = args.bms.split(",")
    kinds = args.kinds.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]
    checkpoints = [b for b in budgets if b <= args.max_budget]
    seed_indices = ([int(x) for x in args.seed_list.split(",")]
                    if args.seed_list else list(range(args.seeds)))

    res = {"meta": {}, "floor": {}, "ref_opt": {}, "runs": {}}
    if (args.append or args.merge_extend) and os.path.exists(args.out):
        with open(args.out) as f:
            res = json.load(f)
    # meta는 덮어쓰지 않고 budgets 를 합집합으로 병합(이전에 기록된 예산,
    # 예: 2400 을 180/780짜리 append 가 조용히 지우지 않도록).
    prev_meta = res.get("meta", {})
    res["meta"] = {
        "budgets": sorted(set(prev_meta.get("budgets", [])) | set(budgets)),
        "max_budget": max(prev_meta.get("max_budget", 0), args.max_budget),
        "seeds": max(prev_meta.get("seeds", 0), args.seeds),
    }

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
                # merge-extend: 기존 seed 결과를 보존하고 새 seed를 이어붙임.
                #  (a) prev 의 모든 budget 키를 보존(현재 checkpoints 에 없어도 유실 금지),
                #  (b) prev 에 없는 budget 은 빈 리스트로 시작(KeyError 금지),
                #  (c) 같은 seed 를 다시 돌리면 경고 후 스킵.
                prev = res["runs"][algo][name].get(kind) if args.merge_extend else None
                prev_best = (prev or {}).get("best_true", {})
                cells = {b: list(v) for b, v in prev_best.items()}
                for b in checkpoints:
                    cells.setdefault(str(b), [])
                # seed 이력: 신규 포맷은 "seeds" 를 명시 저장. 구 포맷(이력 없음)은
                # seed_indices 가 항상 0..n-1 순으로 쌓였다고 가정해 길이로부터 추정.
                if prev is not None and "seeds" in prev:
                    seeds_done = list(prev["seeds"])
                elif prev_best:
                    seeds_done = list(range(max(len(v) for v in prev_best.values())))
                else:
                    seeds_done = []
                t0 = time.time()
                for s in seed_indices:
                    if args.merge_extend and s in seeds_done:
                        print(f"[{algo:8s}] {name} {kind:9s} seed {s} 는 이미 "
                              f"포함돼 있음 — 중복 방지로 스킵")
                        continue
                    prob = Problem(bm, kind, seed=1000 * s + 7)
                    run_fn(prob, args.max_budget, seed=s)
                    cp = prob.checkpoints(checkpoints)
                    for b in checkpoints:
                        cells[str(b)].append(cp[b])
                    seeds_done.append(s)
                dt = time.time() - t0 + (prev["sec"] if prev else 0.0)
                floor = res["floor"][name][kind]
                ref = res["ref_opt"][name][kind]
                denom = max(ref - floor, 1e-9)
                closure = {b: (float(np.mean(v)) - floor) / denom
                           for b, v in cells.items() if v}
                res["runs"][algo][name][kind] = {
                    "best_true": cells,
                    "closure": closure,
                    "sec": dt,
                    "seeds": seeds_done,
                }
                print(f"[{algo:8s}] {name} {kind:9s} "
                      + " ".join(f"clo@{b}={closure[str(b)]:.2%}"
                                for b in checkpoints if str(b) in closure)
                      + f"  ({dt:.1f}s)")

        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
