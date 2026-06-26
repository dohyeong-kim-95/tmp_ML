"""Chebyshev(균형) 목적함수에서 알고리즘 iter 예산 sweep(180/780/2400). prob2 base."""
import sys, numpy as np
sys.path.insert(0,".")
from expensive_common import sweep, random_search
from cheby import Cheb
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

base=Cheb(seed=0)                          # 정규화 1회 계산 후 재사용
NORM=(base.ideal,base.nadir,base.rng_)
xo,Jstar=base.coordinate_ascent(np.random.default_rng(0),restarts=80)
rng=np.random.default_rng(7)
BV=float(np.var([base.true_objective(base.random_solution(rng)) for _ in range(4000)]))
print(f"J*_cheb={Jstar:.4f}, Var={BV:.4f}")
def nmse(bt): return (Jstar-bt)**2/BV
def rs(p,s,c): return random_search(p,s)
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
sweep(lambda s: Cheb(seed=s, norm=NORM), ALGOS, nmse, list(range(3)),
      [180,780,2400], "prob2 Chebyshev(balance) objective",
      "prob2/cheby_sweep.png", "normalized MSE (to balance-optimum)")
