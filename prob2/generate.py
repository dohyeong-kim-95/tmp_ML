"""
prob2 벤치마크 생성기 v3 — 블록 구조로 난이도 상승.

구조:
  - 70열, 위치 0~69. 10열 단위 7블록(block b = 위치 [10b,10b+10)).
  - 블록 '내부' 상호작용: 강함(계수 3~6, main effect 동급) + 2way/3way
      → binary AND-결합이라 "블록 변수를 함께 맞춰야" 이득 → greedy 단일이동 교란.
  - 블록 '간' 상호작용: 약함(계수 0.1~0.5).
  - family: y1_(y11,y12,y13)=front-50(블록0~4), y2_(y21,y22,y23)=back-50(블록2~6).
  - 같은 family 3개 response 는 서로 다른 블록 묶음 참조(다른 열 참조 유지).
  - 노이즈 = 주효과(평균 strong계수)의 4%.
"""
import json
import numpy as np

SEED = 202
CAT_LEVELS = ["A", "B", "C", "D"]
N_BINARY, N_ORDINAL, N_CATEG = 56, 7, 7
N_TOTAL = N_BINARY + N_ORDINAL + N_CATEG       # 70
BLOCK = 10
N_BLOCKS = N_TOTAL // BLOCK                     # 7
NOISE_FRAC = 0.04

rng = np.random.default_rng(SEED)

bin_cols = [f"xb{i:02d}" for i in range(1, N_BINARY + 1)]
ord_cols = [f"xo{i}" for i in range(1, N_ORDINAL + 1)]
cat_cols = [f"xc{i}" for i in range(1, N_CATEG + 1)]
all_x = bin_cols + ord_cols + cat_cols
ord_levels = {c: int(rng.integers(4, 17)) for c in ord_cols}
numeric = set(bin_cols + ord_cols)

positions = rng.permutation(N_TOTAL)
pos = {c: int(positions[i]) for i, c in enumerate(all_x)}
block_of = {c: pos[c] // BLOCK for c in all_x}
blocks_numeric = {b: [c for c in all_x if block_of[c] == b and c in numeric]
                  for b in range(N_BLOCKS)}
front = [c for c in all_x if pos[c] < 50]
back = [c for c in all_x if pos[c] >= 20]

Y_COLS = ["y11", "y12", "y13", "y21", "y22", "y23"]
# 각 response 가 참조할 블록 묶음(서로 다른 열) — family1=블록0~4, family2=블록2~6
RESP_BLOCKS = {"y11": [0, 1, 2], "y12": [1, 2, 3], "y13": [2, 3, 4],
               "y21": [2, 3, 4], "y22": [3, 4, 5], "y23": [4, 5, 6]}


def sgn():
    return float(rng.choice([-1, 1]))


def build_response(resp_blocks, half_cats):
    strong, weak, inters = {}, {}, []
    chosen = {}
    for b in resp_blocks:
        pool = blocks_numeric[b]
        k = min(4, len(pool))
        vs = list(rng.choice(pool, k, replace=False))
        chosen[b] = vs
        for i, v in enumerate(vs):
            if i < 2:
                strong[v] = sgn() * rng.uniform(3.0, 6.0)      # 강한 main
            else:
                weak[v] = sgn() * rng.uniform(0.1, 0.6)        # 약한 main
        # 블록 '내부' 강한 상호작용
        if len(vs) >= 2:
            for _ in range(2):
                a, c = rng.choice(vs, 2, replace=False)
                inters.append({"vars": [str(a), str(c)],
                               "coef": sgn() * rng.uniform(3.0, 6.0), "kind": "2way_intra"})
        if len(vs) >= 3:
            a, c, d = rng.choice(vs, 3, replace=False)
            inters.append({"vars": [str(a), str(c), str(d)],
                           "coef": sgn() * rng.uniform(3.0, 6.0), "kind": "3way_intra"})
    # 블록 '간' 약한 상호작용
    flat = [(b, v) for b, vs in chosen.items() for v in vs]
    for _ in range(4):
        (b1, v1), (b2, v2) = (flat[i] for i in rng.choice(len(flat), 2, replace=False))
        if b1 == b2:
            continue
        inters.append({"vars": [str(v1), str(v2)],
                       "coef": sgn() * rng.uniform(0.1, 0.5), "kind": "2way_cross"})

    cat_eff = {}
    for c in cat_cols:
        if c in half_cats:
            vals = rng.normal(0, 2.5, size=4); vals -= vals.mean()
            cat_eff[c] = {lv: float(v) for lv, v in zip(CAT_LEVELS, vals)}
        else:
            cat_eff[c] = {lv: 0.0 for lv in CAT_LEVELS}

    return {"intercept": float(rng.uniform(-2, 2)), "strong": strong, "weak": weak,
            "interactions": inters, "categorical": cat_eff}


spec = {}
for y in Y_COLS:
    half = front if y[1] == "1" else back
    half_cats = [c for c in half if c in set(cat_cols)]
    spec[y] = build_response(RESP_BLOCKS[y], half_cats)

strong_coefs = [abs(c) for s in spec.values() for c in s["strong"].values()]
main_effect = float(np.mean(strong_coefs))
OBJECTIVE_NOISE_SD = round(NOISE_FRAC * main_effect, 4)

gt = {"meta": {"seed": SEED, "objective_noise_sd": OBJECTIVE_NOISE_SD,
               "noise_frac_of_main_effect": NOISE_FRAC, "main_effect_ref": round(main_effect, 4),
               "block_size": BLOCK, "n_blocks": N_BLOCKS,
               "block_of": block_of, "resp_blocks": RESP_BLOCKS,
               "column_position": pos,
               "family_columns": {"y1_(front50)": front, "y2_(back50)": back},
               "x_columns": {"binary": bin_cols,
                             "ordinal": {c: ord_levels[c] for c in ord_cols},
                             "categorical": {c: CAT_LEVELS for c in cat_cols}},
               "y_columns": Y_COLS},
      "responses": spec}

with open("prob2/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

from collections import Counter
kc = Counter(it["kind"] for s in spec.values() for it in s["interactions"])
print("saved prob2/ground_truth.json")
print(f"X: {N_TOTAL}열, {N_BLOCKS}블록×{BLOCK}열")
print(f"교호작용 종류: {dict(kc)}")
print(f"주효과(평균 strong계수)={main_effect:.2f}, 노이즈 sd={OBJECTIVE_NOISE_SD} (=4%)")
