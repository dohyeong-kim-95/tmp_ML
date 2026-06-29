"""알고리즘 어댑터 — 모두 problem.evaluate(관측 점수)를 최대화한다.

각 어댑터: run(problem, budget, seed). 라이브러리는 lazy import(미설치 시 해당
어댑터만 비활성).  포트폴리오:
  random / sobol  : 정직한 하한선
  sa              : 단일목적 이산공간 Simulated Annealing
  ga              : pymoo 단일목적 정수 GA (model-free 대표)
  tpe             : Optuna TPESampler (이산 강함, 경량)
  smac            : SMAC3 RF-SMBO (혼합/이산 native, anchor)
  botorch         : BoTorch GP-BO qLogEI (연속완화, GP 상한선 reference)
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 하한선
# --------------------------------------------------------------------------
def run_random(problem, budget, seed):
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        x = np.array([rng.integers(0, L) for L in problem.levels])
        problem.evaluate(x)


def run_sobol(problem, budget, seed):
    from scipy.stats import qmc
    eng = qmc.Sobol(d=problem.dim, scramble=True, seed=seed)
    U = eng.random(budget)
    L = problem.levels
    X = np.floor(U * L).astype(int)
    X = np.minimum(X, L - 1)
    for i in range(budget):
        problem.evaluate(X[i])


# --------------------------------------------------------------------------
# Simulated Annealing (단일목적 이산공간)
# --------------------------------------------------------------------------
def run_sa(problem, budget, seed):
    rng = np.random.default_rng(seed)
    L = problem.levels
    x = np.array([rng.integers(0, l) for l in L])
    cur = problem.evaluate(x)
    # 온도 스케일: 초기 무작위 점수 표준편차
    warm = [problem.evaluate(np.array([rng.integers(0, l) for l in L]))
            for _ in range(min(20, max(2, budget // 20)))]
    T0 = max(np.std(warm), 1e-6)
    used = 1 + len(warm)
    for t in range(used, budget):
        frac = t / budget
        T = T0 * (1.0 - frac) + 1e-6
        j = rng.integers(0, problem.dim)
        cand = x.copy()
        if L[j] > 1:
            nv = rng.integers(0, L[j] - 1)
            if nv >= x[j]:
                nv += 1
            cand[j] = nv
        s = problem.evaluate(cand)
        if s > cur or rng.random() < np.exp((s - cur) / T):
            x, cur = cand, s


# --------------------------------------------------------------------------
# pymoo 단일목적 정수 GA
# --------------------------------------------------------------------------
def run_ga(problem, budget, seed):
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.termination.max_eval import MaximumFunctionCallTermination
    from pymoo.optimize import minimize

    L = problem.levels

    class _P(PymooProblem):
        def __init__(self):
            super().__init__(n_var=problem.dim, n_obj=1,
                             xl=np.zeros(problem.dim), xu=(L - 1).astype(float),
                             vtype=int)

        def _evaluate(self, X, out, *a, **k):
            Xi = np.clip(np.round(X).astype(int), 0, L - 1)
            f = np.array([-problem.evaluate(Xi[i]) for i in range(Xi.shape[0])])
            out["F"] = f.reshape(-1, 1)

    algo = GA(
        pop_size=20,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(prob=0.9, eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    minimize(_P(), algo, MaximumFunctionCallTermination(budget),
             seed=seed, verbose=False)


# --------------------------------------------------------------------------
# Optuna TPESampler
# --------------------------------------------------------------------------
def run_tpe(problem, budget, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    L = problem.levels

    def obj(trial):
        x = np.empty(problem.dim, dtype=int)
        for j in range(problem.dim):
            if problem.is_cat[j]:
                x[j] = trial.suggest_categorical(f"x{j}", list(range(int(L[j]))))
            else:
                x[j] = trial.suggest_int(f"x{j}", 0, int(L[j]) - 1)
        return problem.evaluate(x)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=budget, show_progress_bar=False)


# --------------------------------------------------------------------------
# SMAC3 RF-SMBO (anchor)
# --------------------------------------------------------------------------
def run_smac(problem, budget, seed):
    import tempfile
    from ConfigSpace import ConfigurationSpace, Categorical, Integer
    from smac import HyperparameterOptimizationFacade, Scenario

    L = problem.levels
    cs = ConfigurationSpace(seed=seed)
    for j in range(problem.dim):
        if problem.is_cat[j]:
            cs.add(Categorical(f"x{j}", list(range(int(L[j])))))
        else:
            cs.add(Integer(f"x{j}", (0, int(L[j]) - 1)))

    def target(config, seed=0):
        x = np.array([config[f"x{j}"] for j in range(problem.dim)], dtype=int)
        return -problem.evaluate(x)        # SMAC는 최소화

    with tempfile.TemporaryDirectory() as tmp:
        scenario = Scenario(cs, deterministic=False, n_trials=budget,
                            output_directory=tmp, seed=seed)
        smac = HyperparameterOptimizationFacade(
            scenario, target,
            overwrite=True,
            logging_level=False,
        )
        smac.optimize()


# --------------------------------------------------------------------------
# BoTorch GP-BO (qLogEI, 연속완화) — GP 상한선 reference
# --------------------------------------------------------------------------
def run_botorch(problem, budget, seed, n_init=None, refit_every=15, max_train=256):
    """연속완화 GP-BO. GP cubic 비용을 막기 위해 학습점 상한(max_train, best+recent)
    과 주기적 hyperparameter refit(refit_every)으로 큰 예산(≤780)까지 현실화."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from scipy.stats import qmc

    torch.set_num_threads(1)        # 코어 독식 방지(다른 잡과 공존)
    torch.manual_seed(seed)
    dtype = torch.double
    dim = problem.dim
    L = problem.levels.astype(float)
    span = np.maximum(L - 1, 1)
    n_init = min(n_init or 2 * dim, budget)

    def encode(Xi):
        return Xi / span

    def decode(U):
        Xi = np.rint(U * span).astype(int)
        return np.minimum(np.maximum(Xi, 0), problem.levels - 1)

    # 초기 Sobol 설계 (예산 내)
    eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
    U0 = eng.random(n_init)
    X_all, Y_all = [], []
    for u in U0:
        if problem.n >= budget:
            break
        xi = decode(u)
        Y_all.append(problem.evaluate(xi))
        X_all.append(encode(xi))

    bounds = torch.stack([torch.zeros(dim, dtype=dtype), torch.ones(dim, dtype=dtype)])
    model = None
    while problem.n < budget:
        Xa = np.array(X_all)
        Ya = np.array(Y_all)
        # 학습점 상한: 최고점 + 최근점 (cubic 비용 억제)
        if len(Ya) > max_train:
            top = np.argsort(-Ya)[: max_train // 2]
            recent = np.arange(len(Ya))[-(max_train // 2):]
            idx = np.unique(np.concatenate([top, recent]))
        else:
            idx = np.arange(len(Ya))
        Xt = torch.tensor(Xa[idx], dtype=dtype)
        Yt = torch.tensor(Ya[idx], dtype=dtype).unsqueeze(-1)
        Ys = (Yt - Yt.mean()) / (Yt.std() + 1e-9)
        if model is None or (problem.n % refit_every == 0):
            model = SingleTaskGP(Xt, Ys)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass
        else:
            model.set_train_data(Xt, Ys.squeeze(-1), strict=False)
        acqf = qLogExpectedImprovement(model, best_f=Ys.max())
        try:
            cand, _ = optimize_acqf(acqf, bounds=bounds, q=1,
                                    num_restarts=3, raw_samples=64)
            u = cand.detach().cpu().numpy().reshape(-1)
        except Exception:
            u = np.random.default_rng(problem.n).random(dim)
        xi = decode(u)
        Y_all.append(problem.evaluate(xi))
        X_all.append(encode(xi))


REGISTRY = {
    "random": run_random,
    "sobol": run_sobol,
    "sa": run_sa,
    "ga": run_ga,
    "tpe": run_tpe,
    "smac": run_smac,
    "botorch": run_botorch,
}
