"""
난이도 단계별 더미 DB 생성기 (DB2 ~ DB5).

DB1 = 기존 data/dummy_data.csv (가장 단순). 이 스크립트는 DB2~DB5를 만든다.
레벨 d가 커질수록 점점 더 '복잡하고 노이즈가 큰' 데이터:

  - n_strong  : 강한 main effect 개수 ↑
  - n_weak    : 약한 main effect 개수 ↑ (다수의 작은 영향)
  - n_inter2  : 2차 교호작용 개수 ↑
  - n_quad    : 자기 비선형(quadratic, scaled^2) 항 ↑  (d>=2)
  - n_inter3  : 3차 교호작용 개수 ↑  (d>=3, 비선형/고차 결합)
  - noise_sd  : 잡음 표준편차 ↑
  - outlier   : d>=4 에서 일부 행에 두꺼운-꼬리(outlier) 잡음 추가

X 구조는 DB1과 동일: binary 32 / ordinal 4(4~16 level) / categorical 4(4 level).
ground_truth.json 스키마도 DB1과 호환(strong/weak/interactions/categorical),
단 interactions 는 vars 리스트로 2·3차·quadratic 을 일반 표현한다.
"""
import json
import os
import numpy as np
import pandas as pd

CAT_LEVELS = ["A", "B", "C", "D"]
N_ROWS = 2000
N_BINARY, N_ORDINAL, N_CATEG = 32, 4, 4

bin_cols = [f"xb{i:02d}" for i in range(1, N_BINARY + 1)]
ord_cols = [f"xo{i}" for i in range(1, N_ORDINAL + 1)]
cat_cols = [f"xc{i}" for i in range(1, N_CATEG + 1)]
all_x = bin_cols + ord_cols + cat_cols


def level_params(d):
    """난이도 d(2~5) -> 생성 파라미터."""
    return dict(
        n_strong=3 + d,                 # d2=5  ... d5=8
        n_weak=8 + 5 * (d - 1),         # d2=13 ... d5=28
        n_inter2=1 + 2 * (d - 1),       # d2=3  ... d5=9
        n_quad=(d - 1),                 # d2=1  ... d5=4
        n_inter3=max(0, d - 2),         # d3=1  ... d5=3
        noise_mult=1.0 + 0.8 * (d - 1), # d2=1.8 ... d5=4.4
        cat_scale=2.0 + 0.7 * (d - 1),  # d2=2.7 ... d5=4.8
        outlier_frac=0.0 if d < 4 else 0.04 * (d - 3),  # d4=4%, d5=8%
    )


def make_dataset(d, seed):
    rng = np.random.default_rng(seed)
    P = level_params(d)
    ord_levels = {c: int(rng.integers(4, 17)) for c in ord_cols}

    # ---- X ----
    data = {}
    for c in bin_cols:
        p = rng.uniform(0.3, 0.7)
        data[c] = (rng.random(N_ROWS) < p).astype(int)
    for c in ord_cols:
        data[c] = rng.integers(0, ord_levels[c], size=N_ROWS)
    for c in cat_cols:
        probs = rng.dirichlet(np.ones(4) * 3.0)
        data[c] = rng.choice(CAT_LEVELS, size=N_ROWS, p=probs)
    df = pd.DataFrame(data)[all_x]

    def scaled(col):
        if col in bin_cols:
            return df[col].to_numpy(float)
        return df[col].to_numpy(float) / (ord_levels[col] - 1)

    num_pool = bin_cols + ord_cols

    def build_response(base_noise, use_cat=True):
        strong_vars = list(rng.choice(num_pool, P["n_strong"], replace=False))
        strong = {v: float(rng.choice([-1, 1]) * rng.uniform(3.0, 6.0))
                  for v in strong_vars}
        remain = [c for c in num_pool if c not in strong_vars]
        weak_vars = list(rng.choice(remain, P["n_weak"], replace=False))
        weak = {v: float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.6))
                for v in weak_vars}

        inters = []
        pool = strong_vars + list(rng.choice(remain, min(6, len(remain)), replace=False))
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
            vals = rng.normal(0, P["cat_scale"] if use_cat else 0.0, size=4)
            vals -= vals.mean()
            cat_eff[c] = {lv: float(v) for lv, v in zip(CAT_LEVELS, vals)}

        return {"intercept": float(rng.uniform(-2, 2)),
                "strong": strong, "weak": weak, "interactions": inters,
                "categorical": cat_eff, "noise_sd": float(base_noise)}

    # 4개 반응: 기준 noise를 다르게 주고 레벨 배수 적용
    base_noises = np.array([1.5, 2.0, 2.5, 3.0]) * P["noise_mult"]
    spec = {}
    for y, bn, uc in zip(["y11", "y12", "y21", "y22"], base_noises,
                         [True, True, False, True]):
        spec[y] = build_response(bn, use_cat=uc)

    # ---- Y 계산 ----
    def scaled_named(name):
        return scaled(name)

    for y, s in spec.items():
        val = np.full(N_ROWS, s["intercept"])
        for v, c in {**s["strong"], **s["weak"]}.items():
            val += c * scaled_named(v)
        for it in s["interactions"]:
            prod = np.ones(N_ROWS)
            for v in it["vars"]:
                prod = prod * scaled_named(v)
            val += it["coef"] * prod
        for c in cat_cols:
            val += df[c].map(s["categorical"][c]).to_numpy(float)
        # 잡음
        noise = rng.normal(0, s["noise_sd"], size=N_ROWS)
        if P["outlier_frac"] > 0:
            m = rng.random(N_ROWS) < P["outlier_frac"]
            noise[m] += rng.normal(0, s["noise_sd"] * 5, size=m.sum())
        df[y] = val + noise

    gt = {"meta": {"db_level": d, "seed": seed, "n_rows": N_ROWS,
                   "difficulty_params": P,
                   "x_columns": {"binary": bin_cols,
                                 "ordinal": {c: ord_levels[c] for c in ord_cols},
                                 "categorical": {c: CAT_LEVELS for c in cat_cols}},
                   "y_columns": list(spec.keys())},
          "responses": spec}
    return df, gt, P, ord_levels


def summarize(d, df, gt, P):
    y_cols = gt["meta"]["y_columns"]
    sumY = df[y_cols].sum(axis=1)
    n_terms = sum(len(s["strong"]) + len(s["weak"]) + len(s["interactions"])
                  for s in gt["responses"].values())
    return {"case": f"case{d}", "n_strong/Y": P["n_strong"], "n_weak/Y": P["n_weak"],
            "n_inter2/Y": P["n_inter2"], "n_quad/Y": P["n_quad"],
            "n_inter3/Y": P["n_inter3"], "noise_mult": round(P["noise_mult"], 2),
            "outlier%": round(P["outlier_frac"] * 100, 1),
            "total_terms": n_terms, "ΣY_std": round(float(sumY.std()), 2)}


if __name__ == "__main__":
    rows = []
    for d in [2, 3, 4, 5]:
        df, gt, P, ordl = make_dataset(d, seed=42 + d)
        os.makedirs(f"data/case{d}", exist_ok=True)
        df.to_csv(f"data/case{d}/dummy_data.csv", index=False)
        json.dump(gt, open(f"data/case{d}/ground_truth.json", "w"),
                  ensure_ascii=False, indent=2)
        rows.append(summarize(d, df, gt, P))
        print(f"case{d}: saved  (ordinal levels {ordl})")

    print("\n난이도 요약 (Case1=기존 data/dummy_data.csv 가 최저난이도):")
    print(pd.DataFrame(rows).to_string(index=False))
