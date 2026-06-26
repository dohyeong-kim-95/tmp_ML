"""prob3(trap) iter 예산 sweep: 180/780/2400 (1 iter=1분 → 3h/13h/40h)."""
import sys, numpy as np
sys.path.insert(0, ".")
from expensive_common import sweep, random_search
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

ref = Problem(); _, J = ref.global_reference(seed=0)
rng = np.random.default_rng(7)
BV = float(np.var([ref.true_objective(ref.random_solution(rng)) for _ in range(4000)]))
def nmse(bt): return (J - bt) ** 2 / BV
def rs(p, s, cap): return random_search(p, s)
ALGOS = {
 "Random": rs,
 "SA": lambda p,s,c: simulated_annealing(p,max_eval=c,seed=s),
 "PSO": lambda p,s,c: binary_pso(p,max_eval=c,seed=s),
 "GA": lambda p,s,c: genetic_algorithm(p,max_eval=c,seed=s),
 "MemGA": lambda p,s,c: memetic_ga(p,max_eval=c,seed=s),
 "CHC": lambda p,s,c: chc(p,max_eval=c,seed=s),
 "GOMEA": lambda p,s,c: gomea(p,max_eval=c,seed=s),
 "ACO": lambda p,s,c: aco(p,max_eval=c,seed=s),
 "BO": lambda p,s,c: bayesian_optimization(p,max_eval=c,seed=s),
 "TPE": lambda p,s,c: tpe(p,max_eval=c,seed=s),
 "SMAC": lambda p,s,c: smac(p,max_eval=c,seed=s),
}
print(f"prob3 J*={J:.2f}")
sweep(lambda s: Problem(seed=s), ALGOS, nmse, list(range(3)), [180,780,2400],
      "prob3 TRAP — iter budget", "prob3/itersweep.png", "normalized MSE")
