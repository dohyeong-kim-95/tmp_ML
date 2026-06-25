"""warm-BO 타당성 평가 결과 요약 막대그래프 (experiment_warmbo.py 출력값)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cases = ["Case1", "Case3", "Case6"]
data = {  # gap% (낮을수록 좋음)
    "EVAL": {"cold-BO": [0.70, 3.37, 0.14], "warm-BO": [0.01, 0.06, 29.06],
             "GA": [0.73, 0.84, 0.13], "ACO": [0.23, 2.22, 1.08]},
    "TIME": {"cold-BO": [17.98, 26.50, 21.86], "warm-BO": [13.51, 9.43, 48.51],
             "GA": [0.00, 0.00, 0.00], "ACO": [0.00, 0.34, 0.00]},
}
algos = ["cold-BO", "warm-BO", "GA", "ACO"]
colors = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
for ax, mode in zip(axes, ["EVAL", "TIME"]):
    x = np.arange(len(cases)); w = 0.2
    for i, a in enumerate(algos):
        ax.bar(x + (i - 1.5) * w, data[mode][a], w, label=a, color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_ylabel("gap to global optimum (%)  - lower is better")
    bud = "EVAL budget (BO=500, GA/ACO=2000)" if mode == "EVAL" else "TIME budget (T=3s)"
    ax.set_title(f"{mode}  -  {bud}")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=11)
plt.suptitle("Warm-started BO feasibility:  does pretraining the surrogate let BO overtake GA?",
             fontsize=13)
plt.tight_layout()
plt.savefig("optimize/warmbo_summary.png", dpi=120)
print("saved: optimize/warmbo_summary.png")
