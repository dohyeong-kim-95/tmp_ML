"""prob3(trap)을 '비싼 목적함수' 영역(10분 @20s/eval = 30 evals)에서 재시도."""
import sys, numpy as np
sys.path.insert(0, ".")
from expensive_common import benchmark, CAP
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
from expensive_common import random_search

ref = Problem()
_, J_star = ref.global_reference(seed=0)
rng = np.random.default_rng(7)
BASE_VAR = float(np.var([ref.true_objective(ref.random_solution(rng)) for _ in range(4000)]))
print(f"prob3 J*={J_star:.2f}, Var_random={BASE_VAR:.1f}")


def nmse(best_true):
    return (J_star - best_true) ** 2 / BASE_VAR


E = CAP
ALGOS = {
    "Random": random_search,
    "SA":   lambda p, s: simulated_annealing(p, max_eval=E, seed=s),
    "PSO":  lambda p, s: binary_pso(p, max_eval=E, n_particles=8, seed=s),
    "GA":   lambda p, s: genetic_algorithm(p, max_eval=E, pop_size=10, seed=s),
    "MemGA": lambda p, s: memetic_ga(p, max_eval=E, pop_size=6, n_ls=1, seed=s),
    "CHC":  lambda p, s: chc(p, max_eval=E, pop_size=10, seed=s),
    "GOMEA": lambda p, s: gomea(p, max_eval=E, pop_size=10, seed=s),
    "ACO":  lambda p, s: aco(p, max_eval=E, n_ants=10, seed=s),
    "BO":   lambda p, s: bayesian_optimization(p, max_eval=E, n_init=10, seed=s),
    "TPE":  lambda p, s: tpe(p, max_eval=E, n_startup=10, seed=s),
    "SMAC": lambda p, s: smac(p, max_eval=E, n_init=10, seed=s),
}

benchmark(lambda s: Problem(seed=s), ALGOS, nmse, list(range(10)),
          "prob3 TRAP — expensive objective", "prob3/expensive.png",
          ylabel="normalized MSE (lower=better)")
