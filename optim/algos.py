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
from .design import marginal_balanced_design, default_n_init


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

    n_init = n_init or default_n_init(problem.dim, budget)
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
    """이산공간 SA. (공정화)
      - warm-up 점들의 best 를 시작 incumbent 로 사용(이전엔 첫 랜덤점에서 시작해
        warm-up 정보를 T0 계산에만 썼다 → 낭비).
      - 이동: ordinal 은 ±1 local step(순서 활용), categorical 은 random-reset
        (TPE/SMAC 의 cat/ord 구분과 동등한 취급).
    """
    rng = np.random.default_rng(seed)
    L = problem.levels
    # warm-up: 무작위점 평가 → best 는 incumbent, 점수 분포는 T0
    # sub_budget이 작은(blockwrap 후반 블록) 경우 warm-up만으로 budget을 넘겨
    # 다음 블록 예산을 잠식하지 않도록 budget 이내로 클램프(B5).
    n_warm = min(20, max(2, budget // 20), budget)
    warm_x = [np.array([rng.integers(0, l) for l in L]) for _ in range(n_warm)]
    warm_s = [problem.evaluate(wx) for wx in warm_x]
    T0 = max(np.std(warm_s), 1e-6)
    bi = int(np.argmax(warm_s))
    x, cur = warm_x[bi].copy(), warm_s[bi]
    for t in range(n_warm, budget):
        frac = t / budget
        T = T0 * (1.0 - frac) + 1e-6
        j = rng.integers(0, problem.dim)
        cand = x.copy()
        if L[j] > 1:
            if problem.is_cat[j]:                 # categorical: random reset
                nv = rng.integers(0, L[j] - 1)
                if nv >= x[j]:
                    nv += 1
            else:                                 # ordinal: ±1 local step
                step = 1 if rng.random() < 0.5 else -1
                nv = int(np.clip(x[j] + step, 0, L[j] - 1))
                if nv == x[j]:
                    nv = int(np.clip(x[j] - step, 0, L[j] - 1))
            cand[j] = nv
        s = problem.evaluate(cand)
        if s > cur or rng.random() < np.exp((s - cur) / T):
            x, cur = cand, s


# --------------------------------------------------------------------------
# pymoo 단일목적 정수 GA
# --------------------------------------------------------------------------
def run_ga(problem, budget, seed):
    """혼합변수 GA. (공정화)

    이전엔 SBX(교배)+PM(돌연변이) — *순서·거리 가정* 연산자라 categorical 에 가짜
    보간을 강제(불리). MixedVariableGA 로 교체: categorical(Choice)엔 이산 연산자
    (uniform 교배 + random reset), ordinal(Integer)엔 정수 연산자 → 타입별 적정 처리
    (TPE/SMAC 의 native cat/ord 구분과 동등).
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.variable import Choice, Integer
    from pymoo.core.mixed import MixedVariableGA
    from pymoo.termination.max_eval import MaximumFunctionCallTermination
    from pymoo.optimize import minimize

    L = problem.levels
    dim = problem.dim

    class _P(ElementwiseProblem):
        def __init__(s):
            vars = {}
            for j in range(dim):
                if problem.is_cat[j]:
                    vars[f"x{j}"] = Choice(options=list(range(int(L[j]))))
                else:
                    vars[f"x{j}"] = Integer(bounds=(0, int(L[j]) - 1))
            super().__init__(vars=vars, n_obj=1)

        def _evaluate(s, X, out, *a, **k):
            x = np.array([X[f"x{j}"] for j in range(dim)], dtype=int)
            out["F"] = -problem.evaluate(x)

    algo = MixedVariableGA(pop_size=20)
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


def run_pso_mixed(problem, budget, seed, n_part=20, w=0.7, c1=1.5, c2=1.5):
    """혼합변수 PSO (공정화) — 변수 타입별 적정 업데이트.

      - binary (L==2)         : sigmoid velocity (BPSO) → P(bit=1)=σ(v)
      - ordinal (L>2, 순서O)  : 연속 velocity → round/clamp
      - categorical (L>2)     : 레벨별 logit + velocity, softmax sampling (가짜순서 없음)

    연속완화 단일 PSO(run_pso)가 categorical 에 강제하던 가짜순서를 제거 → GA 공정화와
    동일 취지. pbest/gbest 는 디코딩된 정수해를 attractor 로 사용.
    """
    rng = np.random.default_rng(seed)
    L = problem.levels.astype(int)
    dim = problem.dim
    typ = ["bin" if L[j] <= 2 else ("cat" if problem.is_cat[j] else "ord")
           for j in range(dim)]
    n_part = min(n_part, max(2, budget))

    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def smax(v):
        e = np.exp(v - v.max())
        return e / e.sum()

    pos = np.array([[rng.uniform(0, max(L[j] - 1, 0)) if typ[j] == "ord" else 0.0
                     for j in range(dim)] for _ in range(n_part)])
    vel = np.zeros((n_part, dim))                                  # ord & bin
    logit = [[rng.standard_normal(L[j]) if typ[j] == "cat" else None
              for j in range(dim)] for _ in range(n_part)]
    vlog = [[np.zeros(L[j]) if typ[j] == "cat" else None
             for j in range(dim)] for _ in range(n_part)]
    last = [None] * n_part

    def decode(i):
        x = np.empty(dim, dtype=int)
        for j in range(dim):
            if typ[j] == "ord":
                x[j] = int(np.clip(round(pos[i][j]), 0, L[j] - 1))
            elif typ[j] == "bin":
                x[j] = 1 if rng.random() < sig(vel[i][j]) else 0
            else:
                x[j] = int(rng.choice(L[j], p=smax(logit[i][j])))
        return x

    pbest_x = [None] * n_part
    pbest_s = [-np.inf] * n_part
    gbest_x, gbest_s = None, -np.inf
    for i in range(n_part):
        if problem.n >= budget:
            break
        xi = decode(i); last[i] = xi
        s = problem.evaluate(xi)
        pbest_x[i], pbest_s[i] = xi.copy(), s
        if s > gbest_s:
            gbest_s, gbest_x = s, xi.copy()
    while problem.n < budget:
        for i in range(n_part):
            if problem.n >= budget:
                break
            pb, gb, cur = pbest_x[i], gbest_x, last[i]
            for j in range(dim):
                r1, r2 = rng.random(), rng.random()
                if typ[j] == "ord":
                    vel[i][j] = (w * vel[i][j] + c1 * r1 * (pb[j] - pos[i][j])
                                 + c2 * r2 * (gb[j] - pos[i][j]))
                    pos[i][j] = np.clip(pos[i][j] + vel[i][j], 0, L[j] - 1)
                elif typ[j] == "bin":
                    vel[i][j] = (w * vel[i][j] + c1 * r1 * (pb[j] - cur[j])
                                 + c2 * r2 * (gb[j] - cur[j]))
                else:
                    p = smax(logit[i][j])
                    oh_pb = np.zeros(L[j]); oh_pb[pb[j]] = 1.0
                    oh_gb = np.zeros(L[j]); oh_gb[gb[j]] = 1.0
                    vlog[i][j] = (w * vlog[i][j] + c1 * r1 * (oh_pb - p)
                                  + c2 * r2 * (oh_gb - p))
                    logit[i][j] = logit[i][j] + vlog[i][j]
            xi = decode(i); last[i] = xi
            s = problem.evaluate(xi)
            if s > pbest_s[i]:
                pbest_s[i], pbest_x[i] = s, xi.copy()
            if s > gbest_s:
                gbest_s, gbest_x = s, xi.copy()


REGISTRY = {
    "random": run_random,
    "sobol": run_sobol,
    "mlhs": run_mlhs,
    "block_coord_local": run_block_coord_local,
    "sa": run_sa,
    "ga": run_ga,
    "pso": run_pso,
    "pso_mixed": run_pso_mixed,
    "aco": run_aco,
    "tpe": run_tpe,
    "smac": run_smac,
    "botorch": run_botorch,
}

# 공정 비교용: 각 base 에 동일한 블록-분해 구조를 주입한 '*_blk' 버전.
# (block_coord_local 은 사실상 block_decomp(coordinate-descent) 에 해당)
from .blockwrap import make_block_decomp  # noqa: E402

for _base in ("random", "sobol", "mlhs", "sa", "ga", "pso", "pso_mixed", "aco",
              "tpe", "smac", "botorch"):
    REGISTRY[f"{_base}_blk"] = make_block_decomp(REGISTRY[_base])
