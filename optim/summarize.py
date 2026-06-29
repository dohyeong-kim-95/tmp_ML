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
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    res = merge(files)
    budgets = res["meta"].get("budgets", [180, 780])
    table = render(res, budgets)
    print(f"merged {len(files)} files: {[os.path.basename(f) for f in files]}\n")
    print(table)
    if args.md:
        with open(args.md, "w") as f:
            f.write("# Phase 3 — 알고리즘 포트폴리오 결과\n\n")
            f.write("closure(%) = (best_true − floor)/(ref_opt − floor), seed 평균. "
                    "**굵게**=해당 칼럼 승자.\n\n")
            f.write(table)
        print(f"\nsaved -> {args.md}")


if __name__ == "__main__":
    main()
