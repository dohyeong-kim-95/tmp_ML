# bm3_visualize.py
# repo root 에서 실행:
#   python -m optim.bm3_visualize
#
# 저장 파일:
#   optim/figs/bm3_explain.png

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import configs
from benchmark.generator import BlackBoxBenchmark, OBJECTIVES

ART_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "benchmark", "artifacts")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")


def load_reference_optimum(name, kind):
    """artifacts/<name>.json 에 저장된 참조 최적 x/utility 를 읽는다.

    이전엔 매 실행마다 bm.reference_optimum()(다중시작 좌표상승 40-restart +
    block_coord@20k evals)을 재계산해 느렸다. build.py 가 이미 이 값을 산출해
    저장해 두므로(run.py의 load_ref와 동일 소스) 그걸 그대로 읽는다.
    """
    with open(os.path.join(ART_DIR, f"{name}.json")) as f:
        d = json.load(f)
    ref = d["reference_optimum"][kind]
    return np.array(ref["x"], dtype=int), float(ref["utility"])


def pick_strongest_pair(bm):
    """bm.inter 중 Frobenius norm이 가장 큰 2차 교호작용 선택"""
    best = None
    best_norm = -np.inf
    for obj_name, terms in bm.inter.items():
        for (j, k, Mjk) in terms:
            nrm = np.linalg.norm(Mjk)
            if nrm > best_norm:
                best_norm = nrm
                best = (obj_name, j, k, Mjk)
    return best


def pick_strongest_threeway(bm):
    """bm.three 중 norm product가 가장 큰 3차 교호작용 선택"""
    best = None
    best_norm = -np.inf
    for obj_name, terms in bm.three.items():
        for (j, k, l, sj, sk, sl) in terms:
            nrm = np.linalg.norm(sj) * np.linalg.norm(sk) * np.linalg.norm(sl)
            if nrm > best_norm:
                best_norm = nrm
                best = (obj_name, j, k, l, sj, sk, sl)
    return best


def sweep_one_var_score(bm, x_base, j, kind="sum"):
    """x_base에서 변수 j만 sweep 했을 때 점수"""
    xs = np.arange(int(bm.levels[j]))
    ys = []
    for v in xs:
        x = x_base.copy()
        x[j] = v
        ys.append(float(bm.score(x[None, :], kind)[0]))
    return xs, np.array(ys)


def objective_grid(bm, x_base, obj_name, j, k, l_fixed=None):
    """변수 j,k를 sweep한 2D grid. l_fixed=(l_idx, l_val) 가능."""
    obj_idx = OBJECTIVES.index(obj_name)
    Lj = int(bm.levels[j])
    Lk = int(bm.levels[k])
    grid = np.zeros((Lk, Lj), dtype=float)

    for a in range(Lj):
        for b in range(Lk):
            x = x_base.copy()
            x[j] = a
            x[k] = b
            if l_fixed is not None:
                l_idx, l_val = l_fixed
                x[l_idx] = l_val
            grid[b, a] = float(bm.raw(x[None, :])[0, obj_idx])
    return grid


def sample_tradeoff_points(bm, n=2000, seed=123):
    rng = np.random.default_rng(seed)
    X = bm.random_X(rng, n)
    Y = bm.raw(X)
    Z = bm.scorer.z(Y)
    S = bm.scorer.score(Y, "sum")
    return X, Y, Z, S


def repeated_noisy_scores(bm, X_candidates, kind="sum", n_rep=40, seed=777):
    rng = np.random.default_rng(seed)
    all_obs = []
    for x in X_candidates:
        vals = []
        for _ in range(n_rep):
            # 같은 X를 noisy evaluate
            y_noisy = bm.evaluate(x[None, :], rng=rng)
            s_obs = float(bm.scorer.score(y_noisy, kind)[0])
            vals.append(s_obs)
        all_obs.append(vals)
    return all_obs


def main():
    bm1 = BlackBoxBenchmark(configs.BM1)
    bm3 = BlackBoxBenchmark(configs.BM3)

    # 기준점: BM3 sum reference incumbent (artifacts/BM3.json 에서 읽음, B8)
    x_ref, ref_score = load_reference_optimum("BM3", "sum")

    # -----------------------------
    # Panel A: BM1 vs BM3 local sweep (multimodality)
    # 가장 level 수가 큰 ordinal 변수 선택
    # DEFAULT 기준으로 j=2가 ordinal & 5 levels
    # -----------------------------
    j_ord = 2
    xs1, ys_bm1 = sweep_one_var_score(bm1, x_ref.copy(), j_ord, kind="sum")
    xs3, ys_bm3 = sweep_one_var_score(bm3, x_ref.copy(), j_ord, kind="sum")

    # -----------------------------
    # Panel B: strongest 2-way interaction
    # -----------------------------
    obj2, j2, k2, _ = pick_strongest_pair(bm3)
    grid2 = objective_grid(bm3, x_ref.copy(), obj2, j2, k2)

    # -----------------------------
    # Panel C/D: strongest 3-way interaction slices
    # -----------------------------
    obj3, j3, k3, l3, *_ = pick_strongest_threeway(bm3)
    L3 = int(bm3.levels[l3])
    # low / high slice
    low_level = 0
    high_level = L3 - 1
    grid3_low = objective_grid(bm3, x_ref.copy(), obj3, j3, k3, l_fixed=(l3, low_level))
    grid3_high = objective_grid(bm3, x_ref.copy(), obj3, j3, k3, l_fixed=(l3, high_level))

    # -----------------------------
    # Panel E: trade-off scatter
    # -----------------------------
    Xs, Ys, Zs, Ss = sample_tradeoff_points(bm3, n=2500, seed=1234)
    # 예시: set1 안의 maximize vs minimize 목적
    # y11 (maximize) vs y13 (minimize)
    idx_y11 = OBJECTIVES.index("y11")
    idx_y13 = OBJECTIVES.index("y13")

    # -----------------------------
    # Panel F: noisy repeated-eval boxplot
    # -----------------------------
    true_scores = np.array([float(bm3.score(x[None, :], "sum")[0]) for x in Xs])
    qs = np.quantile(true_scores, [0.10, 0.30, 0.50, 0.70, 0.90, 0.98])
    picked_idx = []
    for q in qs:
        picked_idx.append(int(np.argmin(np.abs(true_scores - q))))
    picked_idx = list(dict.fromkeys(picked_idx))  # 중복 제거
    X_pick = Xs[picked_idx]
    T_pick = true_scores[picked_idx]
    obs_lists = repeated_noisy_scores(bm3, X_pick, kind="sum", n_rep=40, seed=2026)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axA, axB = axes[0]
    axC, axD = axes[1]
    axE, axF = axes[2]

    # A. BM1 vs BM3 local sweep
    axA.plot(xs1, ys_bm1, marker="o", label="BM1")
    axA.plot(xs3, ys_bm3, marker="o", label="BM3")
    axA.set_title(f"A. Local sweep on ordinal variable x[{j_ord}]")
    axA.set_xlabel(f"level of x[{j_ord}]")
    axA.set_ylabel("sum score (true)")
    axA.legend()
    axA.grid(alpha=0.3)

    # B. 2-way interaction heatmap
    imB = axB.imshow(grid2, aspect="auto", origin="lower")
    axB.set_title(f"B. Strongest 2-way interaction ({obj2})\n"
                  f"x[{j2}] vs x[{k2}]")
    axB.set_xlabel(f"x[{j2}] level")
    axB.set_ylabel(f"x[{k2}] level")
    fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

    # C. 3-way interaction low slice
    imC = axC.imshow(grid3_low, aspect="auto", origin="lower")
    axC.set_title(f"C. 3-way slice ({obj3})\n"
                  f"x[{j3}] vs x[{k3}] | x[{l3}]={low_level}")
    axC.set_xlabel(f"x[{j3}] level")
    axC.set_ylabel(f"x[{k3}] level")
    fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)

    # D. 3-way interaction high slice
    imD = axD.imshow(grid3_high, aspect="auto", origin="lower")
    axD.set_title(f"D. 3-way slice ({obj3})\n"
                  f"x[{j3}] vs x[{k3}] | x[{l3}]={high_level}")
    axD.set_xlabel(f"x[{j3}] level")
    axD.set_ylabel(f"x[{k3}] level")
    fig.colorbar(imD, ax=axD, fraction=0.046, pad=0.04)

    # E. Trade-off scatter
    sc = axE.scatter(
        Zs[:, idx_y11],
        Zs[:, idx_y13],
        c=Ss,
        s=14,
        alpha=0.65
    )
    axE.set_title("E. Trade-off scatter (normalized objectives)")
    axE.set_xlabel("z(y11)  [1 = best]")
    axE.set_ylabel("z(y13)  [1 = best]")
    axE.grid(alpha=0.3)
    fig.colorbar(sc, ax=axE, fraction=0.046, pad=0.04, label="sum score")

    # F. Repeated noisy evaluation
    labels = [f"P{i+1}\ntrue={t:.2f}" for i, t in enumerate(T_pick)]
    axF.boxplot(obs_lists, tick_labels=labels, showfliers=False)  # mpl>=3.9: labels→tick_labels
    axF.set_title("F. Noise visualization (same X, repeated noisy eval)")
    axF.set_xlabel("candidate point")
    axF.set_ylabel("observed sum score")
    axF.grid(alpha=0.3)

    fig.suptitle(
        f"BM3 visualization — deceptive multimodality, interactions, trade-off, noise\n"
        f"(reference incumbent sum score = {ref_score:.3f})",
        fontsize=14,
        y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "bm3_explain.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
