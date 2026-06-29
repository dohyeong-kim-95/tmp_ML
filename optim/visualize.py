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
from .summarize import merge, ALGO_ORDER, BMS

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")
BM_COLORS = {"BM1": "#4C9F70", "BM2": "#E1A53F", "BM3": "#C0504D"}


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


def main():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results*.json")))
    res = merge(files)
    budgets = res["meta"].get("budgets", [180, 780])
    out = []
    for b in budgets:
        out.append(plot_budget(res, b, os.path.join(FIG_DIR, f"closure_{b}.png")))
    print("merged:", [os.path.basename(f) for f in files])
    for p in out:
        print("saved ->", p)


if __name__ == "__main__":
    main()
