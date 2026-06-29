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

        # 4) 정규화 범위(goodness 의 min/max)와 노이즈 스케일 산정
        self._calibrate(rng)

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

    # ---- goodness / 정규화 / 효용 -------------------------------------------
    def goodness(self, X):
        """higher=better 로 통일 (최소화 목적은 부호 반전)."""
        y = self.raw(X)
        g = y.copy()
        for mi, m in enumerate(OBJECTIVES):
            if m in MINIMIZE:
                g[:, mi] = -g[:, mi]
        return g

    def z(self, X):
        """목적별 goodness 를 [0,1] 정규화 (1=best)."""
        g = self.goodness(X)
        z = (g - self._lo) / np.maximum(self._hi - self._lo, 1e-12)
        return np.clip(z, 0.0, 1.0)

    def utility_equal(self, X):
        return self.z(X).mean(axis=1)

    def utility_chebyshev(self, X):
        z = self.z(X)
        gap = 1.0 - z                       # ideal=1 까지의 거리
        return -(gap.max(axis=1) + self.cfg.cheby_rho * gap.sum(axis=1))

    def utility_owa(self, X):
        z = np.sort(self.z(X), axis=1)      # 오름차순
        return z[:, : self.cfg.owa_k].mean(axis=1)

    SCALARIZATIONS = ("equal", "chebyshev", "owa")

    def utility(self, X, kind):
        return {
            "equal": self.utility_equal,
            "chebyshev": self.utility_chebyshev,
            "owa": self.utility_owa,
        }[kind](X)

    # ---- 캘리브레이션 & 참조 최적 -------------------------------------------
    def _calibrate(self, rng):
        # 각 목적 goodness 의 극값을 coordinate ascent + 랜덤표본으로 추정
        n_obj = len(OBJECTIVES)
        lo = np.full(n_obj, np.inf)
        hi = np.full(n_obj, -np.inf)
        sample = self.random_X(rng, 20000)
        g = self.goodness(sample)
        lo = np.minimum(lo, g.min(axis=0))
        hi = np.maximum(hi, g.max(axis=0))
        for mi in range(n_obj):
            _, vmax = self._coord_opt(rng, lambda X, mi=mi: self.goodness(X)[:, mi], maximize=True)
            _, vmin = self._coord_opt(rng, lambda X, mi=mi: self.goodness(X)[:, mi], maximize=False)
            hi[mi] = max(hi[mi], vmax)
            lo[mi] = min(lo[mi], vmin)
        self._lo, self._hi = lo, hi
        # 노이즈: 원시 Y 스프레드(표준편차)의 noise_frac
        self.noise_scale = self.cfg.noise_frac * self.raw(sample).std(axis=0)

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

    def reference_optimum(self, kind, seed=0, n_restart=40, n_sweep=5):
        """주어진 scalarization 의 참조 최적해/효용 (대량 다중시작 좌표상승)."""
        rng = np.random.default_rng(seed)
        x, v = self._coord_opt(
            rng, lambda X: self.utility(X, kind),
            maximize=True, n_restart=n_restart, n_sweep=n_sweep,
        )
        return x, float(v)
