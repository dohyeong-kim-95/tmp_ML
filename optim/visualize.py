"""results*.json 병합 결과를 시각화한다.

예산별로 kind(sum/chebyshev/owa) 3개 서브플롯, 각 서브플롯은 algo×BM 그룹막대.
막대=closure seed 평균, 오차막대=seed 분포(min~max) → 단일 실전의 다운사이드 표시.

실행: python -m optim.visualize           # optim/figs/closure_{180,780}.png
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.scoring import ScoreSystem
from .summarize import merge, ALGO_ORDER, BMS, POOL, pool_pick, per_seed_closures

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")
BM_COLORS = {"BM1": "#4C9F70", "BM2": "#E1A53F", "BM3": "#C0504D", "BM4": "#6A4C9C"}


def _cell(res, algo, bm, kind):
    try:
        return res["runs"][algo][bm][kind]
    except KeyError:
        return None


def plot_budget(res, budget, path):
    algos = [a for a in ALGO_ORDER if a in res["runs"]]
    kinds = ScoreSystem.KINDS
    fig, axes = plt.subplots(len(kinds), 1, figsize=(max(8, 1.2 * len(algos)), 11))
    fig.suptitle("closure = (best_true - floor) / (ref_opt - floor)  "
                 f"@ budget={budget}\n"
                 "bar = seed mean,  error bar = seed min~max "
                 "(single-run spread / downside)", fontsize=12)
    x = np.arange(len(algos))
    w = 0.25
    for ax, kind in zip(axes, kinds):
        floor_map = res["floor"]; ref_map = res["ref_opt"]
        for bi, bm in enumerate(BMS):
            means, lo, hi = [], [], []
            fl = floor_map[bm][kind]; rf = ref_map[bm][kind]; den = max(rf - fl, 1e-9)
            for a in algos:
                c = _cell(res, a, bm, kind)
                if c is None or str(budget) not in c["best_true"]:
                    means.append(np.nan); lo.append(0); hi.append(0); continue
                vals = np.array(c["best_true"][str(budget)])
                clo = (vals - fl) / den
                m = clo.mean()
                means.append(m)
                lo.append(m - clo.min()); hi.append(clo.max() - m)
            ax.bar(x + (bi - 1) * w, means, w, yerr=[lo, hi], capsize=3,
                   color=BM_COLORS[bm], label=bm, error_kw=dict(alpha=0.5))
        ax.axhline(1.0, ls="--", c="gray", lw=0.8)
        ax.set_title(f"kind = {kind}")
        ax.set_xticks(x); ax.set_xticklabels(algos)
        ax.set_ylabel("closure"); ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(loc="upper left", ncol=3, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_by_kind(res, budget, path):
    """전치 레이아웃: 서브플롯=algo, x축=kind, legend=BM."""
    import math
    algos = [a for a in ALGO_ORDER if a in res["runs"]]
    kinds = list(ScoreSystem.KINDS)
    ncol = 3
    nrow = math.ceil(len(algos) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    x = np.arange(len(kinds))
    w = 0.25
    for idx, a in enumerate(algos):
        ax = axes[idx // ncol][idx % ncol]
        for bi, bm in enumerate(BMS):
            vals = []
            for kind in kinds:
                c = _cell(res, a, bm, kind)
                fl = res["floor"][bm][kind]; rf = res["ref_opt"][bm][kind]
                den = max(rf - fl, 1e-9)
                if c is None or str(budget) not in c["best_true"]:
                    vals.append(np.nan)
                else:
                    arr = np.array(c["best_true"][str(budget)])
                    vals.append(float(((arr - fl) / den).mean()))
            ax.bar(x + (bi - 1) * w, vals, w, color=BM_COLORS[bm], label=bm)
        ax.set_title(a)
        ax.set_xticks(x); ax.set_xticklabels(kinds)
        ax.set_ylim(0, 1.15); ax.axhline(1.0, ls="--", c="gray", lw=0.8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("closure")
        if idx == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
    for j in range(len(algos), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"closure by kind  @ budget={budget}  "
                 f"(subplot=algo, xtick=kind, legend=BM)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_block_lift(res, kind, budget, path):
    """공정 비교: 각 base 에 블록을 줬을 때(flat -> +block) 와 block_coord_local 기준선.

    base 별 flat vs *_blk 막대를 BM별 서브플롯으로, block_coord_local 을 점선으로.
    → '우위가 블록 덕인지(=모두 상승) base 덕인지(=blk끼리 차이)'를 분리해 보여줌.
    """
    bases = ["random", "sobol", "mlhs", "sa", "ga", "tpe"]
    bases = [b for b in bases if b in res["runs"] and f"{b}_blk" in res["runs"]]
    fig, axes = plt.subplots(1, len(BMS), figsize=(5 * len(BMS), 4.6))
    x = np.arange(len(bases)); w = 0.38

    def clo(a, bm):
        try:
            return res["runs"][a][bm][kind]["closure"][str(budget)]
        except KeyError:
            return np.nan

    for ax, bm in zip(axes, BMS):
        ax.bar(x - w / 2, [clo(b, bm) for b in bases], w,
               label="flat (no block)", color="#9bbcc4", edgecolor="k", lw=0.4)
        ax.bar(x + w / 2, [clo(b + "_blk", bm) for b in bases], w,
               label="+block (=blk)", color="#E1A53F", edgecolor="k", lw=0.4)
        bc = clo("block_coord_local", bm)
        if np.isfinite(bc):
            ax.axhline(bc, ls="--", c="#C0504D", lw=1.6,
                       label=f"block_coord_local ({bc:.0%})")
        ax.set_title(f"{bm} | {kind} @{budget}")
        ax.set_xticks(x); ax.set_xticklabels(bases, rotation=30)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", alpha=0.25)
        if bm == BMS[0]:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Fair comparison ({kind}@{budget}): block structure given to "
                 f"every base.  flat -> +block, vs block_coord_local", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_vs_global(res, path, ref_algo="block_coord_local", ref_budget=20000):
    """global maxima(=ref_algo@ref_budget) 기준 bar graph.

    빨간 점선 = ref_algo@ref_budget 가 찾은 반응표면 global maxima(seed별 max).
    막대 = block_coord_local@180, @780 이 그 global max의 몇 %에 도달했는지
    (closure_G = (best_true-floor)/(global_max-floor)).
    """
    runs = res["runs"]
    if ref_algo not in runs:
        return None
    fig, axes = plt.subplots(1, len(BMS), figsize=(5 * len(BMS), 4.6))
    kinds = list(ScoreSystem.KINDS)
    x = np.arange(len(kinds)); w = 0.38
    for ax, bm in zip(axes, BMS):
        c180, c780, ok = [], [], []
        for kind in kinds:
            try:
                bt = runs[ref_algo][bm][kind]["best_true"]
                fl = res["floor"][bm][kind]
                G = max(bt[str(ref_budget)])               # global max(=seed별 best)
                den = max(G - fl, 1e-9)
                c180.append((np.mean(bt["180"]) - fl) / den)
                c780.append((np.mean(bt["780"]) - fl) / den)
                ok.append(True)
            except (KeyError, ValueError):
                c180.append(np.nan); c780.append(np.nan); ok.append(False)
        ax.bar(x - w / 2, c180, w, label="block_coord_local@180",
               color="#9bbcc4", edgecolor="k", lw=0.4)
        ax.bar(x + w / 2, c780, w, label="block_coord_local@780",
               color="#4C9F70", edgecolor="k", lw=0.4)
        ax.axhline(1.0, ls="--", c="#C0504D", lw=1.8,
                   label=f"global max ({ref_algo}@{ref_budget})")
        ax.set_title(f"{bm}")
        ax.set_xticks(x); ax.set_xticklabels(kinds)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", alpha=0.25)
        if bm == BMS[0]:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"block_coord_local @180/@780 vs assumed global maxima "
                 f"(= {ref_algo}@{ref_budget}, red dashed = 100%)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pool(res, budget, path):
    """축소 풀(6항목) × BM 그룹막대. 메타휴리스틱은 flat/blk 중 best,
    SF=space-filling 중 best. 오차막대=seed min~max. 막대 위에 선택변형 표기.

    BM4(비분리)에서 block_coord_local이 1위를 잃는지(coordinate 한계)를 한눈에.
    """
    labels = [lbl for lbl, _ in POOL]
    kinds = ScoreSystem.KINDS
    bms = [b for b in BMS if any(
        pool_pick(res, m, b, k, budget) for _, m in POOL for k in kinds)]
    fig, axes = plt.subplots(len(kinds), 1,
                             figsize=(max(8, 1.6 * len(labels)), 11))
    fig.suptitle(f"축소 풀 closure @ budget={budget}   "
                 "(메타휴리스틱=flat/blk중 best, SF=space-filling중 best)\n"
                 "bar=seed mean, error=seed min~max, 막대위=선택변형", fontsize=12)
    x = np.arange(len(labels)); w = 0.8 / max(len(bms), 1)
    for ax, kind in zip(axes, kinds):
        for bi, bm in enumerate(bms):
            means, lo, hi, wins = [], [], [], []
            for lbl, members in POOL:
                p = pool_pick(res, members, bm, kind, budget)
                if p is None:
                    means.append(np.nan); lo.append(0); hi.append(0); wins.append("")
                    continue
                win = p[4]
                arr = per_seed_closures(res, win, bm, kind, budget)
                m = float(arr.mean())
                means.append(m); lo.append(m - arr.min()); hi.append(arr.max() - m)
                wins.append("blk" if win.endswith("_blk") else
                            ("" if len(members) == 1 else "flat"))
            off = (bi - (len(bms) - 1) / 2) * w
            bars = ax.bar(x + off, means, w, yerr=[lo, hi], capsize=2,
                          color=BM_COLORS[bm], label=bm, error_kw=dict(alpha=0.4))
            for xi, (m, tag) in enumerate(zip(means, wins)):
                if tag and np.isfinite(m):
                    ax.text(x[xi] + off, m + 0.01, tag, ha="center", va="bottom",
                            fontsize=6, rotation=90, color="#333")
        ax.axhline(1.0, ls="--", c="gray", lw=0.8)
        ax.set_title(f"kind = {kind}")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("closure"); ax.set_ylim(0, 1.2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(loc="upper right", ncol=len(bms), fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results*.json")))
    res = merge(files)
    budgets = res["meta"].get("budgets", [180, 780])
    out = []
    for b in budgets:
        out.append(plot_budget(res, b, os.path.join(FIG_DIR, f"closure_{b}.png")))
        out.append(plot_by_kind(res, b, os.path.join(FIG_DIR, f"by_kind_{b}.png")))
        out.append(plot_pool(res, b, os.path.join(FIG_DIR, f"pool_{b}.png")))
    # 공정 비교(블록 주입) — *_blk 결과가 있으면 생성
    if any(a.endswith("_blk") for a in res["runs"]):
        for kind in ("sum", "owa"):
            out.append(plot_block_lift(res, kind, max(budgets),
                                       os.path.join(FIG_DIR, f"block_lift_{kind}.png")))
    # global maxima(=block_coord_local@20000) 기준 그림 — 데이터 있으면
    try:
        if "20000" in res["runs"]["block_coord_local"]["BM1"]["sum"]["best_true"]:
            p = plot_vs_global(res, os.path.join(FIG_DIR, "vs_global_max.png"))
            if p:
                out.append(p)
    except KeyError:
        pass
    print("merged:", [os.path.basename(f) for f in files])
    for p in out:
        print("saved ->", p)


if __name__ == "__main__":
    main()
