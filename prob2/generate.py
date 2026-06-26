"""
prob2 벤치마크 생성기 v2 (model-free 최적화용, A 패러다임).

변경(더 어렵게):
  - 50열 → 70열 (binary 56 / ordinal 7 / categorical 7, 비율 80/10/10 유지)
  - 각 열에 전역 위치(0~69) 부여:
      front-50 = 위치 0~49,  back-50 = 위치 20~69  (중앙 30열은 공유)
  - y1_(y11,y12,y13) 는 front-50 열만, y2_(y21,y22,y23) 는 back-50 열만 영향.
  - 같은 family 의 3개 response 는 '서로 다른 열'을 참조하도록 그 절반 열을
    3등분(chunk)해 분배 → 관련 변수 수↑, 공유구조↓ 로 목적함수가 어려워짐.
  - 노이즈 = 주효과(평균 strong 계수)의 4%.
"""
import json
import numpy as np

SEED = 202
CAT_LEVELS = ["A", "B", "C", "D"]
N_BINARY, N_ORDINAL, N_CATEG = 56, 7, 7
N_TOTAL = N_BINARY + N_ORDINAL + N_CATEG       # 70
NOISE_FRAC = 0.04

rng = np.random.default_rng(SEED)

bin_cols = [f"xb{i:02d}" for i in range(1, N_BINARY + 1)]
ord_cols = [f"xo{i}" for i in range(1, N_ORDINAL + 1)]
cat_cols = [f"xc{i}" for i in range(1, N_CATEG + 1)]
all_x = bin_cols + ord_cols + cat_cols
ord_levels = {c: int(rng.integers(4, 17)) for c in ord_cols}

# 전역 위치(0~69) 무작위 배정 → front/back 분할이 타입 균형을 갖도록
positions = rng.permutation(N_TOTAL)
pos = {c: int(positions[i]) for i, c in enumerate(all_x)}
front = [c for c in all_x if pos[c] < 50]          # y1_ 영향
back = [c for c in all_x if pos[c] >= 20]           # y2_ 영향 (중앙 20~49 공유)

numeric = set(bin_cols + ord_cols)
P = dict(n_strong=5, n_weak=8, n_inter2=4, n_quad=1, n_inter3=1, cat_scale=2.5)
Y_COLS = ["y11", "y12", "y13", "y21", "y22", "y23"]


def split3(cols):
    cols = list(cols)
    rng.shuffle(cols)
    k = len(cols) // 3
    return [cols[:k], cols[k:2 * k], cols[2 * k:]]


def build_response(chunk_numeric, half_numeric, half_cats):
    """chunk_numeric = 이 response 전용 열(서로 다른 열 참조). half_* = family 절반."""
    strong_vars = list(rng.choice(chunk_numeric, P["n_strong"], replace=False))
    strong = {v: float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0)) for v in strong_vars}
    rem = [c for c in chunk_numeric if c not in strong_vars]
    weak_vars = list(rng.choice(rem, min(P["n_weak"], len(rem)), replace=False))
    weak = {v: float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.6)) for v in weak_vars}

    # 교호작용: 자기 chunk + family 절반의 다른 chunk 일부(=response 간 결합 → 더 어려움)
    inter_pool = strong_vars + list(rng.choice(half_numeric,
                                               min(5, len(half_numeric)), replace=False))
    inters = []
    for _ in range(P["n_inter2"]):
        a, b = rng.choice(inter_pool, 2, replace=False)
        inters.append({"vars": [str(a), str(b)],
                       "coef": float(rng.choice([-1, 1]) * rng.uniform(1.5, 3.5)), "kind": "2way"})
    for _ in range(P["n_quad"]):
        a = rng.choice(inter_pool)
        inters.append({"vars": [str(a), str(a)],
                       "coef": float(rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)), "kind": "quad"})
    for _ in range(P["n_inter3"]):
        a, b, c = rng.choice(inter_pool, 3, replace=False)
        inters.append({"vars": [str(a), str(b), str(c)],
                       "coef": float(rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)), "kind": "3way"})

    # categorical: 이 family 절반의 cat 만 효과, 나머지는 0 (영향 없음)
    cat_eff = {}
    for c in cat_cols:
        if c in half_cats:
            vals = rng.normal(0, P["cat_scale"], size=4); vals -= vals.mean()
            cat_eff[c] = {lv: float(v) for lv, v in zip(CAT_LEVELS, vals)}
        else:
            cat_eff[c] = {lv: 0.0 for lv in CAT_LEVELS}

    return {"intercept": float(rng.uniform(-2, 2)), "strong": strong, "weak": weak,
            "interactions": inters, "categorical": cat_eff}


spec = {}
for fam, ys in [("1", ["y11", "y12", "y13"]), ("2", ["y21", "y22", "y23"])]:
    half = front if fam == "1" else back
    half_numeric = [c for c in half if c in numeric]
    half_cats = [c for c in half if c in set(cat_cols)]
    chunks = split3(half_numeric)              # 3개 response 가 서로 다른 열 참조
    for y, chunk in zip(ys, chunks):
        spec[y] = build_response(chunk, half_numeric, half_cats)

# 주효과 = 모든 Y의 |strong 계수| 평균. 노이즈 = 그 4%.
strong_coefs = [abs(c) for s in spec.values() for c in s["strong"].values()]
main_effect = float(np.mean(strong_coefs))
OBJECTIVE_NOISE_SD = round(NOISE_FRAC * main_effect, 4)

gt = {"meta": {"seed": SEED, "objective_noise_sd": OBJECTIVE_NOISE_SD,
               "noise_frac_of_main_effect": NOISE_FRAC, "main_effect_ref": round(main_effect, 4),
               "column_position": pos,
               "family_columns": {"y1_(front50)": front, "y2_(back50)": back},
               "x_columns": {"binary": bin_cols,
                             "ordinal": {c: ord_levels[c] for c in ord_cols},
                             "categorical": {c: CAT_LEVELS for c in cat_cols}},
               "y_columns": Y_COLS},
      "responses": spec}

with open("prob2/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

# 관련 변수 수(서로 다른 열을 얼마나 참조하는지) 요약
relevant = {}
for y, s in spec.items():
    rv = set(s["strong"]) | set(s["weak"])
    for it in s["interactions"]:
        rv |= set(it["vars"])
    rv |= {c for c in cat_cols if any(v != 0 for v in s["categorical"][c].values())}
    relevant[y] = len(rv)
all_relevant = set()
for y, s in spec.items():
    all_relevant |= set(s["strong"]) | set(s["weak"])

print("saved prob2/ground_truth.json")
print(f"X: {N_TOTAL}열 (binary {N_BINARY}/ordinal {N_ORDINAL}/categorical {N_CATEG})")
print(f"front-50(y1_) {len(front)}열, back-50(y2_) {len(back)}열, 공유 {len(set(front)&set(back))}열")
print(f"response별 관련 변수 수: {relevant}")
print(f"주효과(평균 strong계수)={main_effect:.2f}, 노이즈 sd={OBJECTIVE_NOISE_SD} (=4%)")
