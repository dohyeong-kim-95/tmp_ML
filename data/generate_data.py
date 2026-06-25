"""
최적화 문제용 더미 데이터 생성기.

X 인자 (40 columns)
  - 32 columns : binary           {0, 1}
  -  4 columns : ordinal          정수 0..(L-1), L은 4~16에서 column마다 무작위
  -  4 columns : categorical      4 level {A, B, C, D}

Y 인자 : [y11, y12, y21, y22]
  - 각 Y는 X의 '소수 인자'에 크게(strong) 영향받고,
    '다수 인자'에 작게(weak) 영향받음.
  - X 인자끼리 교호작용(interaction)이 일부 존재.

생성과 동시에 ground_truth.json 에 '어떤 X가 어떤 Y를 얼마나 움직이는가'를
모두 기록하므로, 분석 결과(변수선택/회귀계수)와 정답을 비교할 수 있다.
"""

import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
SEED = 42
N_ROWS = 2000

N_BINARY = 32
N_ORDINAL = 4
N_CATEG = 4
CAT_LEVELS = ["A", "B", "C", "D"]

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# 컬럼 이름
# ----------------------------------------------------------------------
bin_cols = [f"xb{i:02d}" for i in range(1, N_BINARY + 1)]
ord_cols = [f"xo{i}" for i in range(1, N_ORDINAL + 1)]
cat_cols = [f"xc{i}" for i in range(1, N_CATEG + 1)]
all_x_cols = bin_cols + ord_cols + cat_cols

# ordinal 별 level 수 (4~16)
ord_levels = {c: int(rng.integers(4, 17)) for c in ord_cols}

# ----------------------------------------------------------------------
# X 데이터 생성
# ----------------------------------------------------------------------
data = {}

# binary : 각 컬럼마다 약간씩 다른 1의 비율 (0.3~0.7)
for c in bin_cols:
    p = rng.uniform(0.3, 0.7)
    data[c] = rng.integers(0, 2, size=N_ROWS) if False else (rng.random(N_ROWS) < p).astype(int)

# ordinal : 0..(L-1) 균등
for c in ord_cols:
    L = ord_levels[c]
    data[c] = rng.integers(0, L, size=N_ROWS)

# categorical : 4 level 균등(약간 불균형)
for c in cat_cols:
    probs = rng.dirichlet(np.ones(4) * 3.0)  # 적당히 균형
    data[c] = rng.choice(CAT_LEVELS, size=N_ROWS, p=probs)

df = pd.DataFrame(data)[all_x_cols]


# ----------------------------------------------------------------------
# 효과 계산용 정규화(scaled) X  -> 모두 대략 [0,1] 또는 {0,1}
# ----------------------------------------------------------------------
def scaled(col):
    if col in bin_cols:
        return df[col].to_numpy(dtype=float)
    if col in ord_cols:
        L = ord_levels[col]
        return df[col].to_numpy(dtype=float) / (L - 1)
    raise ValueError(col)  # categorical은 별도 처리


# categorical level별 효과(랜덤). cat_effect[col][level] = 값
def make_cat_effect(scale):
    eff = {}
    for c in cat_cols:
        vals = rng.normal(0, scale, size=4)
        vals = vals - vals.mean()  # 합 0 (식별성)
        eff[c] = {lv: float(v) for lv, v in zip(CAT_LEVELS, vals)}
    return eff


# ----------------------------------------------------------------------
# 각 Y의 ground-truth 구조 정의
#   strong : 큰 계수 (소수 인자)
#   weak   : 작은 계수 (다수 인자)
#   inter  : 교호작용 (xA * xB)
# ----------------------------------------------------------------------
def pick(cols, k):
    return list(rng.choice(cols, size=k, replace=False))


def build_response(name, n_strong, n_weak, n_inter, noise_sd,
                   strong_pool, use_cat):
    # strong main effects : binary/ordinal 풀에서 선택
    strong_vars = pick(strong_pool, n_strong)
    strong = {v: float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0))
              for v in strong_vars}

    # weak main effects : 나머지 binary 중 다수
    remaining = [c for c in bin_cols if c not in strong_vars]
    weak_vars = pick(remaining, n_weak)
    weak = {v: float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.5))
            for v in weak_vars}

    # interactions : strong 변수들 사이 + 일부 랜덤
    inter = []
    inter_pool = strong_vars + pick(remaining, min(4, len(remaining)))
    for _ in range(n_inter):
        a, b = rng.choice(inter_pool, size=2, replace=False)
        inter.append({
            "a": str(a), "b": str(b),
            "coef": float(rng.choice([-1, 1]) * rng.uniform(1.5, 3.5)),
        })

    cat_effect = make_cat_effect(2.5 if use_cat else 0.0)

    intercept = float(rng.uniform(-2, 2))

    return {
        "intercept": intercept,
        "strong": strong,
        "weak": weak,
        "interactions": inter,
        "categorical": cat_effect,
        "noise_sd": noise_sd,
    }


# strong effect 후보 풀 = binary + ordinal
strong_pool = bin_cols + ord_cols

# y11, y12 는 '계열1' (공통 강한 인자 일부 공유) / y21, y22 는 '계열2'
spec = {}
spec["y11"] = build_response("y11", n_strong=4, n_weak=10, n_inter=2,
                             noise_sd=2.0, strong_pool=strong_pool, use_cat=True)
spec["y12"] = build_response("y12", n_strong=5, n_weak=12, n_inter=3,
                             noise_sd=2.5, strong_pool=strong_pool, use_cat=True)
spec["y21"] = build_response("y21", n_strong=3, n_weak=8, n_inter=1,
                             noise_sd=1.5, strong_pool=strong_pool, use_cat=False)
spec["y22"] = build_response("y22", n_strong=6, n_weak=14, n_inter=4,
                             noise_sd=3.0, strong_pool=strong_pool, use_cat=True)


# ----------------------------------------------------------------------
# Y 계산
# ----------------------------------------------------------------------
def compute_y(s):
    y = np.full(N_ROWS, s["intercept"], dtype=float)
    for v, c in s["strong"].items():
        y += c * scaled(v)
    for v, c in s["weak"].items():
        y += c * scaled(v)
    for it in s["interactions"]:
        y += it["coef"] * scaled(it["a"]) * scaled(it["b"])
    for c in cat_cols:
        eff = s["categorical"][c]
        y += df[c].map(eff).to_numpy(dtype=float)
    y += rng.normal(0, s["noise_sd"], size=N_ROWS)
    return y


for name, s in spec.items():
    df[name] = compute_y(s)

# ----------------------------------------------------------------------
# 저장
# ----------------------------------------------------------------------
df.to_csv("data/dummy_data.csv", index=False)

ground_truth = {
    "meta": {
        "seed": SEED,
        "n_rows": N_ROWS,
        "x_columns": {
            "binary": bin_cols,
            "ordinal": {c: ord_levels[c] for c in ord_cols},
            "categorical": {c: CAT_LEVELS for c in cat_cols},
        },
        "y_columns": list(spec.keys()),
    },
    "responses": spec,
}
with open("data/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=2)

# 요약 출력
print(f"rows={N_ROWS}, X={len(all_x_cols)} cols, Y={list(spec.keys())}")
print("ordinal levels:", {c: ord_levels[c] for c in ord_cols})
for name, s in spec.items():
    print(f"\n[{name}] noise_sd={s['noise_sd']}  intercept={s['intercept']:.2f}")
    print("  strong:", {k: round(v, 2) for k, v in s["strong"].items()})
    print(f"  weak  : {len(s['weak'])} vars (|coef| 0.1~0.5)")
    print(f"  inter : {[(it['a'], it['b'], round(it['coef'],2)) for it in s['interactions']]}")
print("\nsaved: data/dummy_data.csv, data/ground_truth.json")
