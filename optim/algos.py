"""알고리즘 어댑터 — 모두 problem.evaluate(관측 점수)를 최대화한다.

각 어댑터: run(problem, budget, seed). 라이브러리는 lazy import(미설치 시 해당
어댑터만 비활성).  포트폴리오:
  random / sobol      : 정직한 하한선
  mlhs                : 혼합변수 marginal-balanced 초기설계 (Sobol floor 대안)
  block_coord_local   : 블록-인지 좌표 local search (구조 활용 baseline)
  sa                  : 단일목적 이산공간 Simulated Annealing
  ga                  : pymoo 단일목적 정수 GA (model-free 대표)
  tpe                 : Optuna TPESampler (이산 강함, 경량)
  smac                : SMAC3 RF-SMBO (혼합/이산 native, anchor)
  botorch             : BoTorch GP-BO qLogEI (연속완화, GP 상한선 reference)
"""
from __future__ import annotations

import numpy as np

from benchmark.generator import COMMON, SET1, SET2
from .design import marginal_balanced_design


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
# 혼합변수 marginal-balanced 초기설계 (Sobol floor 매핑의 대안)
# --------------------------------------------------------------------------
def run_mlhs(problem, budget, seed):
    """변수별 level marginal이 균등하도록 설계한 점들을 평가(비적응 baseline).

    Sobol floor 매핑의 (categorical 가짜순서, cardinality별 coverage 불균형)을
    피한다. prefix도 균등하므로 180/780 어느 체크포인트에서도 공정.
    """
    rng = np.random.default_rng(seed)
    X = marginal_balanced_design(problem.levels, budget, rng)
    for i in range(budget):
        problem.evaluate(X[i])


# --------------------------------------------------------------------------
# 블록-인지 좌표 local search (구조 활용 baseline)
# --------------------------------------------------------------------------
def run_block_coord_local(problem, budget, seed,
                          block_order=("common", "set2", "set1"), n_init=None):
    """블록-인지 좌표 local search.

    설계:
      - 초기점: marginal-balanced 설계의 관측 best 를 incumbent 로.
      - 라운드마다 block_order(기본 common→set2→set1)로 각 변수를 1-hop 스윕하며
        변수별 best-improvement 채택. common 을 매 라운드 재방문해 블록 간 결합 흡수.
      - 수렴(라운드 내 개선 없음) 시 marginal-balanced 새 점으로 random-restart →
        남은 예산을 다른 basin 탐색에 사용(random-restart hill climbing).
      - 노이즈 관측 점수로 탐색하고, 같은 X 재평가는 캐시로 회피(예산 절약).
        참 점수 anytime 평가는 Problem 이 그대로 담당.

    블록 순서 근거: set1 ⫫ set2 | common 이라 common 을 외부 좌표로 두고 매 라운드
    재방문(=반복 block-coordinate)하면 사용자가 제안한 common→set2→common→set1
    1-패스보다 결합을 더 안정적으로 흡수한다. set2(25차원, 어려움)를 set1 앞에 둬
    예산을 먼저 투입한다.
    """
    rng = np.random.default_rng(seed)
    L = problem.levels
    blocks = {"common": list(COMMON), "set1": list(SET1), "set2": list(SET2)}

    cache = {}

    def ev(x):
        t = tuple(int(v) for v in x)
        if t in cache:
            return cache[t]
        if problem.n >= budget:
            return -np.inf
        s = problem.evaluate(x)
        cache[t] = s
        return s

    n_init = n_init or max(problem.dim, min(2 * problem.dim, budget // 5))
    init = marginal_balanced_design(L, min(n_init, budget), rng)
    x, cur = None, -np.inf
    for i in range(init.shape[0]):
        if problem.n >= budget:
            break
        s = ev(init[i])
        if s > cur:
            cur, x = s, init[i].copy()
    if x is None:
        x = init[0].copy()
        cur = cache.get(tuple(int(v) for v in x), -np.inf)

    while problem.n < budget:
        progressed = False
        for bname in block_order:
            if problem.n >= budget:
                break
            for j in rng.permutation(blocks[bname]):
                if problem.n >= budget:
                    break
                bv, bs = int(x[j]), cur
                for v in range(int(L[j])):
                    if v == x[j] or problem.n >= budget:
                        continue
                    cand = x.copy()
                    cand[j] = v
                    s = ev(cand)
                    if s > bs:
                        bs, bv = s, v
                if bv != x[j]:
                    x[j], cur = bv, bs
                    progressed = True
        if problem.n >= budget:
            break
        if not progressed:                       # 수렴 → random-restart
            nx = marginal_balanced_design(L, 1, rng)[0]
            x, cur = nx.copy(), ev(nx)


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


# --------------------------------------------------------------------------
# Particle Swarm Optimization (연속완화 + 정수 반올림)
# --------------------------------------------------------------------------
def run_pso(problem, budget, seed, n_part=20, w=0.7, c1=1.5, c2=1.5):
    """이산공간 PSO. 위치는 연속 [0,L-1], 평가 시 정수 반올림.

    swarm 기반 전역탐색의 대표. ordinal엔 무난하지만 categorical엔 가짜순서를
    강제(PSO의 알려진 약점) → ACO와의 대비점. budget = n_part × iterations.
    """
    rng = np.random.default_rng(seed)
    L = problem.levels.astype(float)
    span = np.maximum(L - 1, 1e-9)
    dim = problem.dim

    def ev(pos):
        xi = np.clip(np.rint(pos), 0, problem.levels - 1).astype(int)
        return problem.evaluate(xi)

    n_part = min(n_part, max(2, budget))
    X = rng.uniform(0, 1, size=(n_part, dim)) * span
    V = rng.uniform(-1, 1, size=(n_part, dim)) * span * 0.1
    pbest = X.copy()
    pbest_s = np.full(n_part, -np.inf)
    gbest, gbest_s = X[0].copy(), -np.inf
    for i in range(n_part):
        if problem.n >= budget:
            break
        s = ev(X[i])
        pbest_s[i] = s
        if s > gbest_s:
            gbest_s, gbest = s, X[i].copy()
    while problem.n < budget:
        for i in range(n_part):
            if problem.n >= budget:
                break
            r1, r2 = rng.random(dim), rng.random(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = np.clip(X[i] + V[i], 0, span)
            s = ev(X[i])
            if s > pbest_s[i]:
                pbest_s[i], pbest[i] = s, X[i].copy()
            if s > gbest_s:
                gbest_s, gbest = s, X[i].copy()


# --------------------------------------------------------------------------
# Ant Colony Optimization (이산 변수용 Ant System, 레벨별 페로몬)
# --------------------------------------------------------------------------
def run_aco(problem, budget, seed, n_ants=20, rho=0.1, top_k=3, alpha=1.0):
    """범주형/이산 친화 ACO. 변수 j·레벨 v 마다 페로몬 tau[j][v].

    각 개미는 변수별로 tau^alpha 에 비례해 레벨을 샘플(black-box라 heuristic η 없음).
    rank-기반 deposit(상위 top_k 개미)로 점수 스케일/부호(cheby<0)에 무관.
    categorical에 순서를 강제하지 않음 → PSO와 대비. budget = n_ants × iters.
    """
    rng = np.random.default_rng(seed)
    Lv = [int(v) for v in problem.levels]
    tau = [np.ones(L) for L in Lv]                      # 변수별 페로몬
    n_ants = min(n_ants, max(2, budget))
    while problem.n < budget:
        sols, scores = [], []
        for _ in range(n_ants):
            if problem.n >= budget:
                break
            x = np.empty(problem.dim, dtype=int)
            for j in range(problem.dim):
                p = tau[j] ** alpha
                p = p / p.sum()
                x[j] = rng.choice(Lv[j], p=p)
            sols.append(x)
            scores.append(problem.evaluate(x))
        if not sols:
            break
        for j in range(problem.dim):
            tau[j] *= (1.0 - rho)                       # 증발
        order = np.argsort(scores)[::-1][:top_k]        # 상위 개미만 deposit
        for rank, idx in enumerate(order):
            x = sols[idx]
            amount = (top_k - rank) / top_k             # rank-기반(스케일 무관)
            for j in range(problem.dim):
                tau[j][x[j]] += amount
        for j in range(problem.dim):
            tau[j] = np.clip(tau[j], 1e-6, None)


REGISTRY = {
    "random": run_random,
    "sobol": run_sobol,
    "mlhs": run_mlhs,
    "block_coord_local": run_block_coord_local,
    "sa": run_sa,
    "ga": run_ga,
    "pso": run_pso,
    "aco": run_aco,
    "tpe": run_tpe,
    "smac": run_smac,
    "botorch": run_botorch,
}

# 공정 비교용: 각 base 에 동일한 블록-분해 구조를 주입한 '*_blk' 버전.
# (block_coord_local 은 사실상 block_decomp(coordinate-descent) 에 해당)
from .blockwrap import make_block_decomp  # noqa: E402

for _base in ("random", "sobol", "mlhs", "sa", "ga", "pso", "aco", "tpe", "smac", "botorch"):
    REGISTRY[f"{_base}_blk"] = make_block_decomp(REGISTRY[_base])
