"""
캐시된 곡선 데이터(optimize/curves.pkl)로 수렴 그래프(3col×2row)를 그린다.
run_cases.py 가 데이터를 만들어 pkl 로 저장하고, 이 스크립트로 언제든 재플롯.
제목/범례 스타일만 바꾸려면 재계산 없이 이 파일만 실행하면 된다.

범례 표기:
  EVAL budget : ALGO(t=wall time)
  TIME budget : ALGO(n=iterations)
"""
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plots(data, out_eval="optimize/convergence_eval.png",
               out_time="optimize/convergence_time.png"):
    cases = data["cases"]
    algos = data["algos"]
    J_star = data["J_star"]
    cx_idx = data["cx_idx"]
    T = data["time_budget"]

    def plot(mode, fname):
        curves = data["curves"][mode]
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        for ax, case in zip(axes.ravel(), cases):
            for name in algos:
                xaxis, y, wt, ne = curves[case][name]
                label = f"{name} (t={wt:.1f}s)" if mode == "eval" else f"{name} (n={ne})"
                ax.plot(xaxis, y, label=label, lw=1.7)
            ax.axhline(J_star[case], ls="--", c="k", lw=1)
            ax.set_title(f"{case}  (J*={J_star[case]:.1f}, complexity={cx_idx[case]})")
            ax.set_xlabel("evaluations" if mode == "eval" else "wall-clock time (s)")
            ax.set_ylabel("best J(X) = sum Y")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=16)   # 기존 7의 약 2.25배
        ttl = "EVAL budget" if mode == "eval" else f"TIME budget (T={T}s)"
        plt.suptitle(f"SA vs PSO(bin) vs GA vs BO vs TPE vs SMAC  -  {ttl}", fontsize=14)
        plt.tight_layout()
        plt.savefig(fname, dpi=110)
        print(f"saved: {fname}")

    plot("eval", out_eval)
    plot("time", out_time)


if __name__ == "__main__":
    with open("optimize/curves.pkl", "rb") as f:
        data = pickle.load(f)
    make_plots(data)
