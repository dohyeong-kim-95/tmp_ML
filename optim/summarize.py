"""results*.json 들을 병합해 알고리즘 비교표를 출력/저장한다.

closure(%) = (best_true - floor)/(ref_opt - floor) 의 seed 평균.
kind별로 (BM × budget) 표를 만들고 각 칼럼 승자를 표시한다.

실행: python -m optim.summarize            # 콘솔 표
      python -m optim.summarize --md optim/RESULTS.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from benchmark.scoring import ScoreSystem

ALGO_ORDER = ["random", "sobol", "mlhs", "block_coord_local",
              "sa", "ga", "tpe", "smac", "botorch",
              "random_blk", "sobol_blk", "mlhs_blk", "sa_blk",
              "ga_blk", "tpe_blk", "smac_blk", "botorch_blk"]
BMS = ["BM1", "BM2", "BM3"]


def merge(files):
    res = {"runs": {}, "floor": {}, "ref_opt": {}, "meta": {}}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        res["meta"].update(d.get("meta", {}))
        for k in ("floor", "ref_opt"):
            for name, sub in d.get(k, {}).items():
                res[k].setdefault(name, {}).update(sub)
        for algo, byb in d.get("runs", {}).items():
            for name, byk in byb.items():
                for kind, cell in byk.items():
                    res["runs"].setdefault(algo, {}).setdefault(name, {})[kind] = cell
    return res


def closure(res, algo, bm, kind, budget):
    try:
        return res["runs"][algo][bm][kind]["closure"][str(budget)]
    except KeyError:
        return None


def per_seed_closures(res, algo, bm, kind, budget):
    """seed별 closure 배열 = (best_true_s − floor)/(ref_opt − floor).

    run.py 가 셀에 저장한 per-seed best_true 리스트를 그대로 정규화한다.
    랭킹은 seed 평균 하나가 아니라 평균±표준편차·worst(최소 seed)로 본다
    (AGENTS.md 결론 5: seed 편차가 알고리즘 격차보다 클 수 있음)."""
    try:
        cell = res["runs"][algo][bm][kind]
        bt = cell["best_true"][str(budget)]
    except KeyError:
        return None
    if not bt:
        return None
    floor = res["floor"][bm][kind]
    ref = res["ref_opt"][bm][kind]
    denom = max(ref - floor, 1e-9)
    return (np.asarray(bt, dtype=float) - floor) / denom


def stats(res, algo, bm, kind, budget):
    """(mean, std, worst, n_seed) 또는 None."""
    arr = per_seed_closures(res, algo, bm, kind, budget)
    if arr is None:
        return None
    return float(arr.mean()), float(arr.std()), float(arr.min()), int(arr.size)


def render_robust(res, budgets):
    """평균±표준편차 + worst-case(최소 seed) 표.

    각 칼럼에서 평균 1등(★)과 worst 1등(◆)을 표시. seed 수도 함께 보여
    어느 셀이 적은 seed로만 뒷받침되는지(예: smac=3) 투명하게 드러낸다."""
    algos = [a for a in ALGO_ORDER if a in res["runs"]] + \
            [a for a in res["runs"] if a not in ALGO_ORDER]
    lines = []
    for kind in ScoreSystem.KINDS:
        cols = [(bm, b) for bm in BMS for b in budgets]
        head = f"### kind = {kind}  (mean±std · worst, n=seed)\n"
        head += "| algo | " + " | ".join(f"{bm}@{b}" for bm, b in cols) + " |\n"
        head += "|" + "---|" * (len(cols) + 1) + "\n"
        best_mean, best_worst = {}, {}
        for c in cols:
            vals = [(a, stats(res, a, c[0], kind, c[1])) for a in algos]
            vals = [(a, s) for a, s in vals if s is not None]
            if vals:
                best_mean[c] = max(vals, key=lambda t: t[1][0])[0]
                best_worst[c] = max(vals, key=lambda t: t[1][2])[0]
        rows = ""
        for a in algos:
            cells = []
            for c in cols:
                s = stats(res, a, c[0], kind, c[1])
                if s is None:
                    cells.append("·")
                    continue
                mean, std, worst, n = s
                txt = f"{mean:.0%}±{std:.0%} · {worst:.0%} (n{n})"
                mark = ("★" if best_mean.get(c) == a else "") + \
                       ("◆" if best_worst.get(c) == a else "")
                cells.append((mark + " " + txt).strip())
            rows += f"| {a} | " + " | ".join(cells) + " |\n"
        lines.append(head + rows)
    return "\n".join(lines)


def render_rank(res, budgets):
    """평균/worst 두 기준 단일 랭킹(모든 BM×budget×kind 평균). 둘 다 보여
    'worst까지 안전하게 1등인가'를 한눈에."""
    algos = [a for a in ALGO_ORDER if a in res["runs"]] + \
            [a for a in res["runs"] if a not in ALGO_ORDER]
    agg = {}
    for a in algos:
        means, worsts = [], []
        for kind in ScoreSystem.KINDS:
            for bm in BMS:
                for b in budgets:
                    s = stats(res, a, bm, kind, b)
                    if s is not None:
                        means.append(s[0]); worsts.append(s[2])
        if means:
            agg[a] = (float(np.mean(means)), float(np.mean(worsts)), len(means))
    out = "### 종합 랭킹 (전 BM×budget×kind 평균)\n"
    out += "| rank | algo | mean-closure | worst-closure | cells |\n|---|---|---|---|---|\n"
    for i, (a, (m, w, n)) in enumerate(
            sorted(agg.items(), key=lambda t: -t[1][0]), 1):
        out += f"| {i} | {a} | {m:.1%} | {w:.1%} | {n} |\n"
    return out


def render(res, budgets):
    algos = [a for a in ALGO_ORDER if a in res["runs"]] + \
            [a for a in res["runs"] if a not in ALGO_ORDER]
    lines = []
    for kind in ScoreSystem.KINDS:
        cols = [(bm, b) for bm in BMS for b in budgets]
        head = f"### kind = {kind}\n"
        head += "| algo | " + " | ".join(f"{bm}@{b}" for bm, b in cols) + " |\n"
        head += "|" + "---|" * (len(cols) + 1) + "\n"
        # 칼럼별 승자
        best = {}
        for c in cols:
            vals = [(a, closure(res, a, c[0], kind, c[1])) for a in algos]
            vals = [(a, v) for a, v in vals if v is not None]
            best[c] = max(vals, key=lambda t: t[1])[0] if vals else None
        rows = ""
        for a in algos:
            cells = []
            for c in cols:
                v = closure(res, a, c[0], kind, c[1])
                if v is None:
                    cells.append("·")
                else:
                    s = f"{v:.0%}"
                    cells.append(f"**{s}**" if best[c] == a else s)
            rows += f"| {a} | " + " | ".join(cells) + " |\n"
        lines.append(head + rows)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(os.path.dirname(__file__), "results*.json"))
    ap.add_argument("--md", default=None)
    ap.add_argument("--robust", action="store_true",
                    help="평균±표준편차·worst-case·종합랭킹(다중 seed 신뢰성) 출력")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    res = merge(files)
    budgets = res["meta"].get("budgets", [180, 780])
    if args.robust:
        table = render_rank(res, budgets) + "\n\n" + render_robust(res, budgets)
    else:
        table = render(res, budgets)
    print(f"merged {len(files)} files: {[os.path.basename(f) for f in files]}\n")
    print(table)
    if args.md:
        with open(args.md, "w") as f:
            if args.robust:
                f.write("# 알고리즘 랭킹 — 다중 seed 신뢰성\n\n")
                f.write("closure(%) = (best_true − floor)/(ref_opt − floor). "
                        "셀=`평균±표준편차 · worst(최소 seed) (n=seed수)`. "
                        "★=칼럼 평균 1등, ◆=칼럼 worst 1등. "
                        "n이 작은 행(예: smac=3)은 그만큼 추정이 거칠다.\n\n")
            else:
                f.write("# Phase 3 — 알고리즘 포트폴리오 결과\n\n")
                f.write("closure(%) = (best_true − floor)/(ref_opt − floor), seed 평균. "
                        "**굵게**=해당 칼럼 승자.\n\n")
            f.write(table)
        print(f"\nsaved -> {args.md}")


if __name__ == "__main__":
    main()
