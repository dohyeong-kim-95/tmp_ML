"""prob1 iter 예산 sweep: 180/780/2400. score=NMSE(Case별 Var정규화 후 3 Case 평균)."""
import sys, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from expensive_common import run_cap, random_search
from problem import Problem
from simulated_annealing import simulated_annealing
from binary_pso import binary_pso
from genetic_algorithm import genetic_algorithm
from memetic_ga import memetic_ga
from chc import chc
from gomea import gomea
from aco import aco
from bayesian_optimization import bayesian_optimization
from tpe import tpe
from smac import smac

CASES = {"Case1":"prob1/data/ground_truth.json","Case3":"prob1/data/case3/ground_truth.json",
         "Case6":"prob1/data/case6/ground_truth.json"}
BUDGETS=[180,780,2400]; SEEDS=list(range(2))
J={}; VAR={}
for c,p in CASES.items():
    pr=Problem(gt_path=p)
    _,J[c]=pr.coordinate_ascent(np.random.default_rng(0),restarts=50)
    rng=np.random.default_rng(7)
    VAR[c]=float(np.var([pr.objective(pr.random_solution(rng)) for _ in range(3000)]))
print("prob1 J*:",{c:round(v,1) for c,v in J.items()})
print("prob1 Var:",{c:round(v,1) for c,v in VAR.items()})
def rs(p,s,cap): return random_search(p,s)
ALGOS={"Random":rs,
 "SA":lambda p,s,c:simulated_annealing(p,max_eval=c,seed=s),
 "PSO":lambda p,s,c:binary_pso(p,max_eval=c,seed=s),
 "GA":lambda p,s,c:genetic_algorithm(p,max_eval=c,seed=s),
 "MemGA":lambda p,s,c:memetic_ga(p,max_eval=c,seed=s),
 "CHC":lambda p,s,c:chc(p,max_eval=c,seed=s),
 "GOMEA":lambda p,s,c:gomea(p,max_eval=c,seed=s),
 "ACO":lambda p,s,c:aco(p,max_eval=c,seed=s),
 "BO":lambda p,s,c:bayesian_optimization(p,max_eval=c,seed=s),
 "TPE":lambda p,s,c:tpe(p,max_eval=c,seed=s),
 "SMAC":lambda p,s,c:smac(p,max_eval=c,seed=s)}
data={n:[] for n in ALGOS}
for n,fn in ALGOS.items():
    for B in BUDGETS:
        # Case별 NMSE = mean_seeds[(J*-bt)^2]/Var_case, 그 평균
        per=[]
        for c,p in CASES.items():
            se=[(J[c]-run_cap(Problem(gt_path=p),fn,s,B))**2 for s in SEEDS]
            per.append(np.mean(se)/VAR[c])
        data[n].append(float(np.mean(per)))
    print(f"  {n:8} "+" ".join(f"B={B}:{data[n][i]:8.4f}" for i,B in enumerate(BUDGETS)))
plt.figure(figsize=(9,5.5))
for n in ALGOS: plt.plot(BUDGETS,[max(x,1e-6) for x in data[n]],marker="o",label=n,lw=1.7)
plt.xscale("log"); plt.yscale("log"); plt.xticks(BUDGETS,[str(b) for b in BUDGETS])
plt.xlabel("iter (evaluation) budget"); plt.ylabel("normalized MSE (mean 3 cases)")
plt.title("prob1 — iter budget sweep (NMSE, lower=better)")
plt.grid(alpha=0.3,which="both"); plt.legend(fontsize=9,ncol=2)
plt.tight_layout(); plt.savefig("prob1/itersweep.png",dpi=120)
print("saved: prob1/itersweep.png")
