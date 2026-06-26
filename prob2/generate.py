"""
prob2 벤치마크 생성기 (model-free 최적화용, A 패러다임 → dataset CSV 불필요).

prob1 대비 변경:
  - 40열 → 50열 (binary 40 / ordinal 5 / categorical 5, 비율 80/10/10 유지)
  - Y 4개 → 6개 : y11 y12 y13 y21 y22 y23 (family1 / family2 각 3개)
  - '난이도 상승': 목적함수 평가에 소폭 노이즈 추가 (meta.objective_noise_sd)
    → 최적화기는 노이즈 낀 값을 보고, 성능은 노이즈 없는 진짜값으로 평가.

ground_truth.json (비밀식)만 출력한다.
"""
import json
import numpy as np

SEED = 202
CAT_LEVELS = ["A", "B", "C", "D"]
N_BINARY, N_ORDINAL, N_CATEG = 40, 5, 5
NOISE_FRAC = 0.04                 # 목적함수 노이즈 = 주효과(평균 strong 계수)의 4%

rng = np.random.default_rng(SEED)

bin_cols = [f"xb{i:02d}" for i in range(1, N_BINARY + 1)]
ord_cols = [f"xo{i}" for i in range(1, N_ORDINAL + 1)]
cat_cols = [f"xc{i}" for i in range(1, N_CATEG + 1)]
all_x = bin_cols + ord_cols + cat_cols
ord_levels = {c: int(rng.integers(4, 17)) for c in ord_cols}
num_pool = bin_cols + ord_cols

Y_COLS = ["y11", "y12", "y13", "y21", "y22", "y23"]
# family별 공통 강인자(구조)
fam_shared = {"1": list(rng.choice(num_pool, 2, replace=False)),
              "2": list(rng.choice(num_pool, 2, replace=False))}

P = dict(n_strong=5, n_weak=12, n_inter2=4, n_quad=1, n_inter3=1, cat_scale=2.5)


def build_response(fam):
    shared = fam_shared[fam]
    extra = [c for c in num_pool if c not in shared]
    strong_vars = shared + list(rng.choice(extra, P["n_strong"] - 2, replace=False))
    strong = {v: float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0)) for v in strong_vars}
    remain = [c for c in num_pool if c not in strong_vars]
    weak_vars = list(rng.choice(remain, P["n_weak"], replace=False))
    weak = {v: float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.6)) for v in weak_vars}

    inters = []
    pool = strong_vars + list(rng.choice(remain, 6, replace=False))
    for _ in range(P["n_inter2"]):
        a, b = rng.choice(pool, 2, replace=False)
        inters.append({"vars": [str(a), str(b)],
                       "coef": float(rng.choice([-1, 1]) * rng.uniform(1.5, 3.5)),
                       "kind": "2way"})
    for _ in range(P["n_quad"]):
        a = rng.choice(pool)
        inters.append({"vars": [str(a), str(a)],
                       "coef": float(rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)),
                       "kind": "quad"})
    for _ in range(P["n_inter3"]):
        a, b, c = rng.choice(pool, 3, replace=False)
        inters.append({"vars": [str(a), str(b), str(c)],
                       "coef": float(rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)),
                       "kind": "3way"})

    cat_eff = {}
    for c in cat_cols:
        vals = rng.normal(0, P["cat_scale"], size=4); vals -= vals.mean()
        cat_eff[c] = {lv: float(v) for lv, v in zip(CAT_LEVELS, vals)}

    return {"intercept": float(rng.uniform(-2, 2)), "strong": strong, "weak": weak,
            "interactions": inters, "categorical": cat_eff}


spec = {y: build_response(y[1]) for y in Y_COLS}   # y[1] = family '1' or '2'

# 주효과 크기 = 모든 Y의 |strong 계수| 평균. 노이즈는 그 NOISE_FRAC 배.
strong_coefs = [abs(c) for s in spec.values() for c in s["strong"].values()]
main_effect = float(np.mean(strong_coefs))
OBJECTIVE_NOISE_SD = round(NOISE_FRAC * main_effect, 4)

gt = {"meta": {"seed": SEED, "objective_noise_sd": OBJECTIVE_NOISE_SD,
               "noise_frac_of_main_effect": NOISE_FRAC,
               "main_effect_ref": round(main_effect, 4),
               "x_columns": {"binary": bin_cols,
                             "ordinal": {c: ord_levels[c] for c in ord_cols},
                             "categorical": {c: CAT_LEVELS for c in cat_cols}},
               "y_columns": Y_COLS},
      "responses": spec}

with open("prob2/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

n_terms = sum(len(s["strong"]) + len(s["weak"]) + len(s["interactions"])
              for s in spec.values())
print(f"saved prob2/ground_truth.json")
print(f"X: {len(all_x)}열 (binary {N_BINARY}/ordinal {N_ORDINAL}/categorical {N_CATEG})")
print(f"ordinal levels: {ord_levels}")
print(f"Y: {Y_COLS}")
print(f"총 항수: {n_terms}")
print(f"주효과(평균 strong 계수) = {main_effect:.2f}")
print(f"목적함수 노이즈 sd = {OBJECTIVE_NOISE_SD}  (= 주효과의 {NOISE_FRAC*100:.0f}%)")
