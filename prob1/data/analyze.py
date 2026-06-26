"""
더미 데이터 분석: 각 Y에 대해
  1) main effect 변수 중요도 (LassoCV, 표준화)
  2) 비선형/상호작용 포함 중요도 (GradientBoosting permutation importance)
  3) 교호작용(pairwise) 탐지 (top 변수 product 항 LassoCV)
을 수행하고 ground_truth.json 과 대조해 recall/precision 을 출력.
"""

import json
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

df = pd.read_csv("data/dummy_data.csv")
gt = json.load(open("data/ground_truth.json"))

Y_COLS = gt["meta"]["y_columns"]
bin_cols = gt["meta"]["x_columns"]["binary"]
ord_cols = list(gt["meta"]["x_columns"]["ordinal"].keys())
cat_cols = list(gt["meta"]["x_columns"]["categorical"].keys())

# ---- 설계행렬: binary/ordinal 그대로, categorical one-hot ----
X_num = df[bin_cols + ord_cols].astype(float)
X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False).astype(float)
X = pd.concat([X_num, X_cat], axis=1)
feat_names = list(X.columns)

# 원본 X 변수명(상호작용 평가용): binary+ordinal+categorical(대표)
base_vars = bin_cols + ord_cols + cat_cols


def gt_strong(y):
    return set(gt["responses"][y]["strong"].keys())


def gt_inter(y):
    return {frozenset([it["a"], it["b"]]) for it in gt["responses"][y]["interactions"]}


def base_of(feat):
    # one-hot 컬럼을 원본 categorical 변수명으로 환원
    for c in cat_cols:
        if feat.startswith(c + "_"):
            return c
    return feat


print("=" * 70)
print(f"데이터: {df.shape[0]} 행, X 설계행렬 {X.shape[1]} 열 (one-hot 후)")
print("=" * 70)

summary = []

for y in Y_COLS:
    yv = df[y].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.3, random_state=1)

    # ---------- 1) Lasso main effect ----------
    scaler = StandardScaler().fit(Xtr)
    lasso = LassoCV(cv=5, n_jobs=-1, max_iter=20000).fit(scaler.transform(Xtr), ytr)
    coef = pd.Series(lasso.coef_, index=feat_names)
    # 원본 변수 단위로 |coef| 집계 (categorical one-hot 합산)
    imp_lasso = {}
    for f, c in coef.items():
        b = base_of(f)
        imp_lasso[b] = imp_lasso.get(b, 0.0) + abs(c)
    imp_lasso = pd.Series(imp_lasso).sort_values(ascending=False)

    # ---------- 2) GBM + permutation importance ----------
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=2
    ).fit(Xtr, ytr)
    r2 = gbm.score(Xte, yte)
    perm = permutation_importance(gbm, Xte, yte, n_repeats=10,
                                  random_state=3, n_jobs=-1)
    imp_perm = {}
    for f, m in zip(feat_names, perm.importances_mean):
        b = base_of(f)
        imp_perm[b] = imp_perm.get(b, 0.0) + max(m, 0.0)
    imp_perm = pd.Series(imp_perm).sort_values(ascending=False)

    # ---------- 평가: strong 변수 회수율 ----------
    strong = gt_strong(y)
    k = len(strong)
    top_lasso = set(imp_lasso.head(k).index)
    top_perm = set(imp_perm.head(k).index)
    rec_lasso = len(top_lasso & strong) / k
    rec_perm = len(top_perm & strong) / k

    # ---------- 3) 교호작용 탐지 ----------
    # 후보: lasso+perm 상위 8개 변수의 pairwise product (binary/ordinal만; cat 제외 단순화)
    cand = [v for v in (list(imp_perm.head(8).index) + list(imp_lasso.head(8).index))
            if v in bin_cols + ord_cols]
    cand = list(dict.fromkeys(cand))[:8]
    inter_feats, inter_pairs = [], []
    Xi = X[bin_cols + ord_cols].copy()
    # 정규화(생성 로직과 동일하게 ordinal 0~1)
    for c in ord_cols:
        L = gt["meta"]["x_columns"]["ordinal"][c]
        Xi[c] = Xi[c] / (L - 1)
    base_design = []
    base_cols = []
    for v in cand:
        base_design.append(Xi[v].to_numpy())
        base_cols.append(v)
    for a, b in combinations(cand, 2):
        base_design.append((Xi[a] * Xi[b]).to_numpy())
        base_cols.append(f"{a}*{b}")
        inter_pairs.append((a, b))
    M = np.column_stack(base_design)
    Mtr, Mte, ytr2, yte2 = train_test_split(M, yv, test_size=0.3, random_state=1)
    sc2 = StandardScaler().fit(Mtr)
    la2 = LassoCV(cv=5, n_jobs=-1, max_iter=20000).fit(sc2.transform(Mtr), ytr2)
    cc = pd.Series(la2.coef_, index=base_cols)
    detected_inter = {frozenset(p.split("*")) for p, v in cc.items()
                      if "*" in p and abs(v) > 0.05}
    true_inter = gt_inter(y)
    inter_hit = len(detected_inter & true_inter)

    print(f"\n■ {y}  (GBM test R²={r2:.3f}, noise_sd={gt['responses'][y]['noise_sd']})")
    print(f"  strong(정답, {k}개): {sorted(strong)}")
    print(f"  Lasso top{k}      : {sorted(top_lasso)}   recall={rec_lasso:.0%}")
    print(f"  Perm  top{k}      : {sorted(top_perm)}   recall={rec_perm:.0%}")
    print(f"  교호작용 정답      : {[tuple(s) for s in true_inter]}")
    print(f"  교호작용 탐지      : {[tuple(s) for s in detected_inter]}  "
          f"({inter_hit}/{len(true_inter)} hit)")

    summary.append({
        "y": y, "gbm_r2": round(r2, 3),
        "strong_recall_lasso": rec_lasso, "strong_recall_perm": rec_perm,
        "inter_recall": inter_hit / max(len(true_inter), 1),
        "inter_detected": len(detected_inter),
    })

print("\n" + "=" * 70)
print("요약")
print("=" * 70)
sm = pd.DataFrame(summary)
print(sm.to_string(index=False))
print(f"\n평균 strong recall (perm): {sm['strong_recall_perm'].mean():.0%}")
print(f"평균 교호작용 recall      : {sm['inter_recall'].mean():.0%}")
sm.to_csv("data/analysis_summary.csv", index=False)
print("\nsaved: data/analysis_summary.csv")
