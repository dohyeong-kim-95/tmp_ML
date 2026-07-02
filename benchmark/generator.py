"""합성 black-box 벤치마크 생성기.

실제 calculator(1 eval = 1분)를 흉내 내되 즉시 평가되고 참조 최적이 알려지는
합성 함수를 생성한다. functional-ANOVA 분해(주효과 + 희소 교호작용)로 구성하며,
난이도 knob(다봉성/교호작용/충돌/노이즈)으로 BM1<BM2<BM3 ladder를 만든다.

X 구조(BM 공통):
  - 공통(common) 10열   : 6목적 전부에 영향          (idx 0..9,  변량 ~1e6)
  - set1 전용 5열        : y11,y12,y13 에 영향        (idx 10..14, 변량 ~1e3)
  - set2 전용 15열       : y21,y22,y23 에 영향        (idx 15..29, 변량 ~1e6)
  → set1 유효차원 15(~1e9, 쉬움), set2 유효차원 25(~1e12, 어려움)

Y: y11,y12,y21,y22 최대화 / y13,y23 최소화.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .scoring import MinMaxNormalizer, ScoreConfig, ScoreSystem

# ----------------------------------------------------------------------------
# 고정 레이아웃 (BM1/2/3 공통)
# ----------------------------------------------------------------------------
N_VARS = 30
COMMON = list(range(0, 10))
SET1 = list(range(10, 15))
SET2 = list(range(15, 30))

OBJECTIVES = ["y11", "y12", "y13", "y21", "y22", "y23"]
MAXIMIZE = {"y11", "y12", "y21", "y22"}
MINIMIZE = {"y13", "y23"}
OBJ_OWN_BLOCK = {
    "y11": SET1, "y12": SET1, "y13": SET1,
    "y21": SET2, "y22": SET2, "y23": SET2,
}

# 변량 ~1e6 x 1e3 x 1e6 = 1e15 (목표 범위 상단)
DEFAULT_LEVELS = (
    [30, 10, 5, 4, 3, 3, 2, 2, 2, 2]                          # 공통(10)  ~1e6
    + [10, 6, 3, 3, 2]                                        # set1(5)   ~1e3
    + [6, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]          # set2(15)  ~1e6
)
# True=categorical(순서 없음), False=ordinal(순서 있음). categorical 10 / ordinal 20
DEFAULT_IS_CAT = (
    [True, True, False, True, False, False, True, False, False, False]   # 공통: cat 4
    + [True, True, False, False, False]                                  # set1: cat 2
    + [True, True, True, True, False, False, False, False, False,        # set2: cat 4
       False, False, False, False, False, False]
)


@dataclass
class BMConfig:
    """벤치마크 한 인스턴스의 난이도/구조 설정."""
    name: str
    seed: int
    n_harmonics: int            # ordinal 주효과의 봉우리(=multimodality) 성분 수
    interaction_density: float  # 영향 변수쌍 중 교호작용이 걸리는 비율
    interaction_strength: float # 교호작용 항의 상대 크기
    n_three_way: int            # 3차 교호항 개수 (BM3용)
    conflict_rho: float         # 공통블록을 통한 max/min 상충 강도 [0,1]
    noise_frac: float = 0.05    # 관측 노이즈 = 주효과 스프레드의 비율
    n_strong: int = 4           # 목적당 strong driver 수
    weak_ratio: float = 0.15    # weak driver의 상대 가중
    cheby_rho: float = 0.01     # augmented Chebyshev 증강항 계수
    owa_k: int = 2              # k-bottom OWA 의 k
    levels: list = field(default_factory=lambda: list(DEFAULT_LEVELS))
    is_cat: list = field(default_factory=lambda: list(DEFAULT_IS_CAT))


def _unit(v: np.ndarray) -> np.ndarray:
    """평균0/표준편차1 로 표준화 (상수면 0 반환)."""
    v = v - v.mean()
    s = v.std()
    return v / s if s > 1e-12 else v


def _ordinal_shape(rng, L, n_harm) -> np.ndarray:
    """순서 있는 변수의 매끄러운 주효과 곡선 (성분수=봉우리 수 ↔ 다봉성)."""
    u = np.linspace(0.0, 1.0, L)
    s = np.zeros(L)
    for r in range(n_harm):
        omega = rng.uniform(0.5, 2.5) * (r + 1)
        phase = rng.uniform(0.0, 2 * np.pi)
        coef = rng.normal() / (r + 1)
        s += coef * np.sin(np.pi * omega * u + phase)
    return _unit(s)


def _categorical_shape(rng, L) -> np.ndarray:
    """순서 없는 변수의 레벨별 랜덤 효과."""
    return _unit(rng.normal(size=L))


class BlackBoxBenchmark:
    """X(정수 레벨 30개) -> 6개 Y. 노이즈/참조최적/정규화/3종 효용 제공."""

    def __init__(self, cfg: BMConfig):
        self.cfg = cfg
        self.levels = np.asarray(cfg.levels, dtype=int)
        self.is_cat = list(cfg.is_cat)
        rng = np.random.default_rng(cfg.seed)

        self.affected = {m: COMMON + OBJ_OWN_BLOCK[m] for m in OBJECTIVES}
        self.maximize_mask = np.array([m in MAXIMIZE for m in OBJECTIVES], dtype=bool)

        # 1) 공통블록 base shape (충돌 인코딩의 토대)
        base = {j: self._shape(rng, j) for j in COMMON}

        # 2) 목적별 주효과 테이블 + 가중
        self.main = {m: {} for m in OBJECTIVES}   # m -> {j: ndarray(L_j)}
        for m in OBJECTIVES:
            aff = self.affected[m]
            strong = self._pick_strong(rng, aff)
            for j in aff:
                if j in COMMON:
                    # 공통변수: base 와 rho 만큼 상관 → max/min 좋은방향 상충
                    indep = self._shape(rng, j)
                    rho = cfg.conflict_rho
                    shape = _unit(rho * base[j] + np.sqrt(1 - rho ** 2) * indep)
                else:
                    shape = self._shape(rng, j)
                w = 1.0 if j in strong else cfg.weak_ratio
                self.main[m][j] = w * shape

        # 3) 교호작용 (rank-1 행렬) + 3차 항
        self.inter = {m: [] for m in OBJECTIVES}      # m -> [(j,k,Mjk)]
        self.three = {m: [] for m in OBJECTIVES}      # m -> [(j,k,l,sj,sk,sl)]
        for m in OBJECTIVES:
            aff = self.affected[m]
            pairs = [(a, b) for ii, a in enumerate(aff) for b in aff[ii + 1:]]
            rng.shuffle(pairs)
            n_pick = int(round(cfg.interaction_density * len(pairs)))
            for (j, k) in pairs[:n_pick]:
                sj = self._shape(rng, j)
                sk = self._shape(rng, k)
                Mjk = cfg.interaction_strength * np.outer(sj, sk)
                self.inter[m].append((j, k, Mjk))
            for _ in range(cfg.n_three_way):
                j, k, l = rng.choice(aff, size=3, replace=False)
                self.three[m].append(
                    (int(j), int(k), int(l),
                     cfg.interaction_strength * self._shape(rng, j),
                     self._shape(rng, k), self._shape(rng, l))
                )

        # 4) 정규화 범위(목적별 raw-y min/max)·노이즈 산정 + 점수체계 구성
        self._calibrate(rng)
        self.scorer = ScoreSystem(
            MinMaxNormalizer(self._y_lo, self._y_hi, self.maximize_mask),
            ScoreConfig(cheby_rho=cfg.cheby_rho, owa_k=cfg.owa_k),
        )

    # ---- shape / strong 선택 -------------------------------------------------
    def _shape(self, rng, j):
        L = int(self.levels[j])
        if self.is_cat[j]:
            return _categorical_shape(rng, L)
        return _ordinal_shape(rng, L, self.cfg.n_harmonics)

    def _pick_strong(self, rng, aff):
        # 공통에서 최소 2개 포함(충돌이 보이도록) + 나머지는 영향변수 전체에서
        common_aff = [j for j in aff if j in COMMON]
        forced = list(rng.choice(common_aff, size=min(2, len(common_aff)), replace=False))
        remaining = [j for j in aff if j not in forced]
        extra = self.cfg.n_strong - len(forced)
        if extra > 0 and remaining:
            forced += list(rng.choice(remaining, size=min(extra, len(remaining)), replace=False))
        return set(int(x) for x in forced)

    # ---- 평가 ----------------------------------------------------------------
    def raw(self, X):
        """노이즈 없는 원시 Y. X: (30,) 또는 (N,30) int. 반환 (...,6)."""
        X = np.atleast_2d(np.asarray(X, dtype=int))
        out = np.zeros((X.shape[0], len(OBJECTIVES)))
        for mi, m in enumerate(OBJECTIVES):
            tot = np.zeros(X.shape[0])
            for j, tab in self.main[m].items():
                tot += tab[X[:, j]]
            for (j, k, Mjk) in self.inter[m]:
                tot += Mjk[X[:, j], X[:, k]]
            for (j, k, l, sj, sk, sl) in self.three[m]:
                tot += sj[X[:, j]] * sk[X[:, k]] * sl[X[:, l]]
            out[:, mi] = tot
        return out

    def evaluate(self, X, rng=None):
        """관측값(노이즈 포함). 실제 calculator 대응."""
        y = self.raw(X)
        rng = rng or np.random.default_rng()
        y = y + rng.normal(scale=self.noise_scale, size=y.shape)
        return y

    # ---- 정규화 범위(공개 프로퍼티) -------------------------------------------
    @property
    def y_lo(self):
        """목적별 raw-y 도달 가능 최솟값(정규화 하한). 읽기 전용."""
        return self._y_lo

    @property
    def y_hi(self):
        """목적별 raw-y 도달 가능 최댓값(정규화 상한). 읽기 전용."""
        return self._y_hi

    # ---- 점수(정규화/3종 scalarization) — scoring 모듈에 위임 ----------------
    SCALARIZATIONS = ScoreSystem.KINDS  # ("sum", "chebyshev", "owa")

    def z(self, X):
        """목적별 정규화 점수 [0,1], 1=best (scoring 모듈)."""
        return self.scorer.z(self.raw(X))

    def score(self, X, kind):
        """raw(X) -> 단일 점수 (kind ∈ {'sum','chebyshev','owa'})."""
        return self.scorer.score(self.raw(X), kind)

    def all_scores(self, X):
        return self.scorer.all_scores(self.raw(X))

    # ---- 캘리브레이션 & 참조 최적 -------------------------------------------
    def _calibrate(self, rng):
        """목적별 raw-y 의 도달 가능 min/max(정규화 범위)와 노이즈 스케일 산정."""
        n_obj = len(OBJECTIVES)
        sample = self.random_X(rng, 20000)
        ys = self.raw(sample)
        y_lo = ys.min(axis=0)
        y_hi = ys.max(axis=0)
        for mi in range(n_obj):
            _, vmax = self._coord_opt(rng, lambda X, mi=mi: self.raw(X)[:, mi], maximize=True)
            _, vmin = self._coord_opt(rng, lambda X, mi=mi: self.raw(X)[:, mi], maximize=False)
            y_hi[mi] = max(y_hi[mi], vmax)
            y_lo[mi] = min(y_lo[mi], vmin)
        self._y_lo, self._y_hi = y_lo, y_hi
        # 노이즈: '주효과' 스프레드(표준편차)의 noise_frac (A2 — 문서 정의와 일치).
        # 이전엔 교호·3차항까지 포함한 전체 raw-Y 표준편차를 썼는데, 교호가 강한
        # BM일수록 실효 노이즈가 설정 의도(noise_frac)보다 커지는 문제가 있었다.
        # 주효과는 변수별 가법항이라 균등 레벨 분포에서 분산 = Σ_j Var(tab_j) 로
        # 닫힌형 계산이 가능(표본 불필요, 결정적).
        main_var = np.array([
            sum(float(tab.var()) for tab in self.main[m].values())
            for m in OBJECTIVES
        ])
        self.main_effect_spread = np.sqrt(main_var)
        self.total_spread = ys.std(axis=0)          # 참고/리포트용(교호 포함)
        self.noise_scale = self.cfg.noise_frac * self.main_effect_spread

    def random_X(self, rng, n):
        X = np.empty((n, N_VARS), dtype=int)
        for j in range(N_VARS):
            X[:, j] = rng.integers(0, self.levels[j], size=n)
        return X

    def _coord_opt(self, rng, score_fn, maximize=True, n_restart=25, n_sweep=4):
        """좌표상승(coordinate ascent) 다중시작 최적화 (참조용, 노이즈 없음)."""
        sign = 1.0 if maximize else -1.0
        best_x, best_v = None, -np.inf
        for _ in range(n_restart):
            x = self.random_X(rng, 1)[0]
            for _ in range(n_sweep):
                for j in range(N_VARS):
                    L = int(self.levels[j])
                    cand = np.tile(x, (L, 1))
                    cand[:, j] = np.arange(L)
                    vals = sign * score_fn(cand)
                    x[j] = int(np.argmax(vals))
            v = sign * score_fn(x[None, :])[0]
            if v > best_v:
                best_v, best_x = v, x.copy()
        return best_x, best_v

    def _block_coord_opt(self, kind, budget=20000, seed=0):
        """노이즈 없는 block-coordinate 탐색(common→set2→set1 반복 + random-restart).

        AGENTS.md의 'global maxima(= block_coord_local@20000)'를 reference로 산출한다.
        측정상 다중시작 좌표상승보다, 그리고 GA-200k보다 더 높은 천장을 준다(가장 강한
        탐색기). 변수별 L후보를 한 번에 벡터평가(빠름), 전역 score-eval 예산 budget.
        """
        rng = np.random.default_rng(seed)
        L = self.levels
        blocks = [list(COMMON), list(SET2), list(SET1)]
        sf = lambda X: self.score(X, kind)

        def fresh():
            x = self.random_X(rng, 1)[0]
            return x, float(sf(x[None, :])[0])

        used = 1
        x, cur = fresh()
        bx, bv = x.copy(), cur
        while used < budget:
            progressed = False
            for blk in blocks:
                for j in rng.permutation(blk):
                    Lj = int(L[j])
                    if used + Lj > budget:
                        continue
                    cand = np.tile(x, (Lj, 1))
                    cand[:, j] = np.arange(Lj)
                    vals = sf(cand)
                    used += Lj
                    bi = int(np.argmax(vals))
                    if vals[bi] > cur + 1e-12:
                        x, cur = cand[bi].copy(), float(vals[bi])
                        progressed = True
            if cur > bv:
                bx, bv = x.copy(), cur
            if not progressed:                       # 수렴 → random-restart
                x, cur = fresh()
                used += 1
        return bx, bv

    def _sa_opt(self, kind, budget=30000, seed=0, n_restart=6):
        """노이즈 없는 mixed-move SA (참조 천장용, A1).

        좌표법과 다른 inductive bias: 한 번에 1~3개 변수를 함께 움직이고
        (ordinal ±1~2 / categorical random-reset) 내리막도 확률 수용 →
        좌표 스윕이 못 넘는 결합/능선을 넘을 수 있다. geometric cooling,
        n_restart 회 재시작.
        """
        rng = np.random.default_rng(seed)
        sf = lambda x: float(self.score(np.asarray(x)[None, :], kind)[0])
        T0 = float(np.std(self.score(self.random_X(rng, 512), kind)))
        T0 = max(T0, 1e-6)
        best_x, best_v = None, -np.inf
        per = max(budget // n_restart, 1)
        for _ in range(n_restart):
            x = self.random_X(rng, 1)[0]
            cur = sf(x)
            if cur > best_v:
                best_v, best_x = cur, x.copy()
            for t in range(per):
                T = T0 * (0.001 ** (t / per))       # T0 → T0/1000
                k = 1 + (rng.random() < 0.35) + (rng.random() < 0.10)
                cand = x.copy()
                for j in rng.choice(N_VARS, size=int(k), replace=False):
                    L = int(self.levels[j])
                    if L <= 1:
                        continue
                    if self.is_cat[j]:               # categorical: random reset
                        nv = int(rng.integers(0, L - 1))
                        nv += (nv >= cand[j])
                    else:                            # ordinal: ±1~2 local step
                        step = int(rng.integers(1, 3))
                        step = step if rng.random() < 0.5 else -step
                        nv = int(np.clip(cand[j] + step, 0, L - 1))
                    cand[j] = nv
                s = sf(cand)
                if s >= cur or rng.random() < np.exp((s - cur) / max(T, 1e-9)):
                    x, cur = cand, s
                    if cur > best_v:
                        best_v, best_x = cur, x.copy()
        return best_x, float(best_v)

    def _ga_opt(self, kind, budget=30000, seed=0, pop=64):
        """노이즈 없는 GA-lite (참조 천장용, A1).

        좌표법과 다른 inductive bias: uniform crossover 로 여러 변수를 동시에
        재조합(빌딩블록 결합) + 혼합 돌연변이(ordinal ±1 / categorical reset),
        (μ+λ) 생존선택. 세대 단위 벡터평가라 빠르다.
        """
        rng = np.random.default_rng(seed)
        P = self.random_X(rng, pop)
        F = self.score(P, kind)
        used = pop
        bi = int(np.argmax(F))
        best_x, best_v = P[bi].copy(), float(F[bi])
        while used < budget:
            n_child = min(pop, budget - used)
            t1 = rng.integers(0, pop, size=(n_child, 2))
            t2 = rng.integers(0, pop, size=(n_child, 2))
            p1 = np.where(F[t1[:, 0]] >= F[t1[:, 1]], t1[:, 0], t1[:, 1])
            p2 = np.where(F[t2[:, 0]] >= F[t2[:, 1]], t2[:, 0], t2[:, 1])
            mask = rng.random((n_child, N_VARS)) < 0.5
            C = np.where(mask, P[p1], P[p2])
            mut = rng.random((n_child, N_VARS)) < (1.5 / N_VARS)
            for j in range(N_VARS):
                idx = np.nonzero(mut[:, j])[0]
                if idx.size == 0:
                    continue
                L = int(self.levels[j])
                if self.is_cat[j] or L <= 2:         # categorical: random reset
                    C[idx, j] = rng.integers(0, L, size=idx.size)
                else:                                # ordinal: ±1 local step
                    C[idx, j] = np.clip(
                        C[idx, j] + rng.choice([-1, 1], size=idx.size), 0, L - 1)
            Fc = self.score(C, kind)
            used += n_child
            allP = np.vstack([P, C])
            allF = np.concatenate([F, Fc])
            top = np.argsort(-allF)[:pop]
            P, F = allP[top], allF[top]
            if float(F[0]) > best_v:
                best_v, best_x = float(F[0]), P[0].copy()
        return best_x, float(best_v)

    #: reference_ceiling 앙상블 구성(탐색기 이름 → 사용 여부/예산 기본값)
    CEILING_SEARCHERS = ("coord_multistart", "block_coord", "sa", "ga")

    def reference_ceiling(self, kind, seed=0, n_restart=40, n_sweep=5,
                          global_budget=20000, sa_budget=30000, ga_budget=30000):
        """scalarization 의 참조 천장 = **서로 다른 inductive bias 탐색기 앙상블의 max** (A1).

        기존에는 좌표상승 + block-coordinate(둘 다 좌표 계열)만으로 천장을 정해,
        비분리 BM에서 천장이 과소평가되고 그 편향이 챔피언(block_coord_local)과
        같은 방향이라 closure 비교가 순환적이라는 문제가 있었다(Fable_feedback A1).
        여기에 비좌표 계열(SA: 다변수 확률이동, GA: 재조합)을 추가하고, 탐색기별
        값(by_searcher)과 그 편차(spread)를 함께 반환해 천장의 불확실성을 드러낸다.
        어떤 탐색기가 이기든 max 를 취하므로 천장은 후퇴하지 않는다.
        """
        rng = np.random.default_rng(seed)
        by, xs = {}, {}
        xc, vc = self._coord_opt(
            rng, lambda X: self.score(X, kind),
            maximize=True, n_restart=n_restart, n_sweep=n_sweep,
        )
        by["coord_multistart"], xs["coord_multistart"] = float(vc), xc
        if global_budget and global_budget > 0:
            xb, vb = self._block_coord_opt(kind, budget=global_budget, seed=seed)
            by["block_coord"], xs["block_coord"] = float(vb), xb
        if sa_budget and sa_budget > 0:
            xa, va = self._sa_opt(kind, budget=sa_budget, seed=seed)
            by["sa"], xs["sa"] = float(va), xa
        if ga_budget and ga_budget > 0:
            xg, vg = self._ga_opt(kind, budget=ga_budget, seed=seed)
            by["ga"], xs["ga"] = float(vg), xg
        winner = max(by, key=by.get)
        return {
            "x": xs[winner],
            "utility": by[winner],
            "winner": winner,
            "by_searcher": by,
            "spread": float(max(by.values()) - min(by.values())),
        }

    def reference_optimum(self, kind, seed=0, n_restart=40, n_sweep=5,
                          global_budget=20000, **ceiling_kw):
        """(하위호환 래퍼) reference_ceiling 의 (x, utility)만 반환."""
        d = self.reference_ceiling(kind, seed=seed, n_restart=n_restart,
                                   n_sweep=n_sweep, global_budget=global_budget,
                                   **ceiling_kw)
        return d["x"], d["utility"]
