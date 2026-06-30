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
              "sa", "ga", "pso", "aco", "tpe", "smac", "botorch",
              "random_blk", "sobol_blk", "mlhs_blk", "sa_blk",
              "ga_blk", "pso_blk", "aco_blk", "tpe_blk", "smac_blk", "botorch_blk"]
BMS = ["BM1", "BM2", "BM3", "BM4"]

# 축소 풀(plot용): 각 항목 = (라벨, [후보 algo들]).
#  - SF       = space-filling 중 best (random/sobol/mlhs)
#  - 메타휴리스틱은 flat/blk 중 좋은 쪽만 (per-cell best)
#  - block_coord_local 은 항상 포함(구조-활용 챔피언)
POOL = [
    ("SF", ["random", "sobol", "mlhs"]),
    ("block_coord_local", ["block_coord_local"]),
    ("SA", ["sa", "sa_blk"]),
    ("GA", ["ga", "ga_blk"]),
    ("PSO", ["pso", "pso_blk"]),
    ("ACO", ["aco", "aco_blk"]),
    ("TPE", ["tpe", "tpe_blk"]),
]


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


def pool_pick(res, members, bm, kind, budget):
    """후보 algo들 중 평균 closure가 가장 높은 것을 골라 그 통계 반환.

    반환 (mean, std, worst, n, winner) 또는 None.  flat vs blk 중 '좋은 쪽만' /
    SF의 'best of random/sobol/mlhs'를 per-cell로 구현."""
    cands = []
    for a in members:
        s = stats(res, a, bm, kind, budget)
        if s is not None:
            cands.append((a, s))
    if not cands:
        return None
    winner, (mean, std, worst, n) = max(cands, key=lambda t: t[1][0])
    return mean, std, worst, n, winner


def render_pool(res, budgets):
    """축소 풀(6항목) robust 표 + 칸별 선택된 변형(winner) 표기."""
    lines = []
    for kind in ScoreSystem.KINDS:
        cols = [(bm, b) for bm in BMS for b in budgets]
        cols = [c for c in cols if any(pool_pick(res, m, c[0], kind, c[1])
                                       for _, m in POOL)]
        if not cols:
            continue
        head = f"### kind = {kind}  (mean±std · worst, [선택변형])\n"
        head += "| pool | " + " | ".join(f"{bm}@{b}" for bm, b in cols) + " |\n"
        head += "|" + "---|" * (len(cols) + 1) + "\n"
        best_mean, best_worst = {}, {}
        for c in cols:
            picks = [(lbl, pool_pick(res, m, c[0], kind, c[1])) for lbl, m in POOL]
            picks = [(lbl, p) for lbl, p in picks if p is not None]
            if picks:
                best_mean[c] = max(picks, key=lambda t: t[1][0])[0]
                best_worst[c] = max(picks, key=lambda t: t[1][2])[0]
        rows = ""
        for lbl, members in POOL:
            cells = []
            for c in cols:
                p = pool_pick(res, members, c[0], kind, c[1])
                if p is None:
                    cells.append("·"); continue
                mean, std, worst, n, win = p
                tag = "" if (len(members) == 1 or win == lbl.lower()) else f" [{win}]"
                mark = ("★" if best_mean.get(c) == lbl else "") + \
                       ("◆" if best_worst.get(c) == lbl else "")
                cells.append((mark + f" {mean:.0%}±{std:.0%}·{worst:.0%}{tag}").strip())
            rows += f"| {lbl} | " + " | ".join(cells) + " |\n"
        lines.append(head + rows)
    return "\n".join(lines)


def render_pool_rank(res, budgets):
    """축소 풀 종합 랭킹(전 BM×budget×kind 평균, mean·worst)."""
    agg = {}
    for lbl, members in POOL:
        means, worsts = [], []
        for kind in ScoreSystem.KINDS:
            for bm in BMS:
                for b in budgets:
                    p = pool_pick(res, members, bm, kind, b)
                    if p is not None:
                        means.append(p[0]); worsts.append(p[2])
        if means:
            agg[lbl] = (float(np.mean(means)), float(np.mean(worsts)), len(means))
    out = "### 축소 풀 종합 랭킹 (전 BM×budget×kind 평균)\n"
    out += "| rank | pool | mean | worst | cells |\n|---|---|---|---|---|\n"
    for i, (a, (m, w, n)) in enumerate(
            sorted(agg.items(), key=lambda t: -t[1][0]), 1):
        out += f"| {i} | {a} | {m:.1%} | {w:.1%} | {n} |\n"
    return out


def render_pool_by_budget(res, budgets):
    """예산별 풀 랭킹 — 2400에서 순위 역전(crossover) 가시화."""
    out = "### 예산별 풀 랭킹 (전 BM×kind 평균; budget별로 분리 → crossover 확인)\n"
    for b in budgets:
        agg = {}
        for lbl, members in POOL:
            ms = [pool_pick(res, members, bm, kind, b)
                  for kind in ScoreSystem.KINDS for bm in BMS]
            ms = [p[0] for p in ms if p is not None]
            if ms:
                agg[lbl] = float(np.mean(ms))
        if not agg:
            continue
        rank = " > ".join(f"{a}({m:.0%})"
                          for a, m in sorted(agg.items(), key=lambda t: -t[1]))
        out += f"- **@{b}**: {rank}\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(os.path.dirname(__file__), "results*.json"))
    ap.add_argument("--md", default=None)
    ap.add_argument("--robust", action="store_true",
                    help="평균±표준편차·worst-case·종합랭킹(다중 seed 신뢰성) 출력")
    ap.add_argument("--pool", action="store_true",
                    help="축소 풀(SF/block_coord_local/SA/GA/PSO/ACO, blk중 best) 표")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    res = merge(files)
    budgets = res["meta"].get("budgets", [180, 780])
    if args.pool:
        table = (render_pool_rank(res, budgets) + "\n\n"
                 + render_pool_by_budget(res, budgets) + "\n\n"
                 + render_pool(res, budgets))
    elif args.robust:
        table = render_rank(res, budgets) + "\n\n" + render_robust(res, budgets)
    else:
        table = render(res, budgets)
    print(f"merged {len(files)} files: {[os.path.basename(f) for f in files]}\n")
    print(table)
    if args.md:
        with open(args.md, "w") as f:
            if args.pool:
                f.write("# 축소 풀 랭킹 — SF / block_coord_local / SA / GA / PSO / ACO\n\n")
                f.write("메타휴리스틱은 flat/blk 중 **per-cell best**만, SF=random/sobol/mlhs 중 best. "
                        "셀=`평균±표준편차·worst [선택변형]`. ★=칼럼 평균 1등, ◆=worst 1등. "
                        "closure는 affine 변환이라 **칸 안 순위는 reference와 무관하게 유효**; "
                        "BM4에서 비좌표 방법이 1위면 그게 'coordinate 한계'의 증거(closure>1 가능).\n\n")
            elif args.robust:
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
