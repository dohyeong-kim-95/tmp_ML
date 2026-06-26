"""
prob3 벤치마크 생성기 — deceptive TRAP 블록으로 난이도 최대화.

핵심(난이도 두 축):
  ① local maxima 다수  + ② local↔global 거리 최대 (deception)
구현:
  - 70열, 위치 0~69, 10열 7블록. family: y1_=front50(블록0~4), y2_=back50(블록2~6).
  - 블록 '내부' = unitation TRAP: 블록의 binary k=5비트에 trap-k 적용.
      trap(u) = k (u==k=all-ones, 전역최적 스파이크) else (k-1-u) (u=0=all-zeros가 가짜최적)
      → 한 비트씩 greedy는 all-zeros(가짜)로 수렴, 진짜 정답은 정반대 all-ones.
  - 선형 main effect 는 ordinal + (trap에 안 쓰인) binary 에만 → trap 비트는 'trap 전용'(순수 기만).
  - 블록 '간' 약한 상호작용.
  - 노이즈 = 주효과(평균 strong 선형계수)의 4%.
전역최적은 해석적으로 알려짐(trap비트=1) → problem.global_reference 로 안정 계산.
"""
import json
import numpy as np

SEED = 303
CAT_LEVELS = ["A", "B", "C", "D"]
N_BINARY, N_ORDINAL, N_CATEG = 56, 7, 7
N_TOTAL = N_BINARY + N_ORDINAL + N_CATEG
BLOCK, N_BLOCKS = 10, 7
TRAP_K = 5
NOISE_FRAC = 0.04

rng = np.random.default_rng(SEED)

bin_cols = [f"xb{i:02d}" for i in range(1, N_BINARY + 1)]
ord_cols = [f"xo{i}" for i in range(1, N_ORDINAL + 1)]
cat_cols = [f"xc{i}" for i in range(1, N_CATEG + 1)]
all_x = bin_cols + ord_cols + cat_cols
ord_levels = {c: int(rng.integers(4, 17)) for c in ord_cols}

positions = rng.permutation(N_TOTAL)
pos = {c: int(positions[i]) for i, c in enumerate(all_x)}
block_of = {c: pos[c] // BLOCK for c in all_x}
block_bins = {b: [c for c in bin_cols if block_of[c] == b] for b in range(N_BLOCKS)}
front = [c for c in all_x if pos[c] < 50]
back = [c for c in all_x if pos[c] >= 20]

Y_COLS = ["y11", "y12", "y13", "y21", "y22", "y23"]
RESP_BLOCKS = {"y11": [0, 1, 2], "y12": [1, 2, 3], "y13": [2, 3, 4],
               "y21": [2, 3, 4], "y22": [3, 4, 5], "y23": [4, 5, 6]}


def sgn():
    return float(rng.choice([-1, 1]))


# --- 1단계: 각 response/블록 trap 비트 배정 (trap 전용 비트 집합 구축) ---
resp_traps = {y: [] for y in Y_COLS}
trap_bits = set()
for y in Y_COLS:
    half = set(front if y[1] == "1" else back)
    for b in RESP_BLOCKS[y]:
        cand = [c for c in block_bins[b] if c in half]
        if len(cand) < TRAP_K:
            continue
        bits = list(rng.choice(cand, TRAP_K, replace=False))
        amp = rng.uniform(1.5, 3.0)          # trap 강도(스파이크 ~ amp*k)
        resp_traps[y].append({"vars": [str(v) for v in bits], "k": TRAP_K,
                              "coef": float(amp), "kind": "trap"})
        trap_bits.update(bits)


# --- 2단계: 선형 main effect (ordinal + trap에 안 쓰인 binary) + 블록간 약한 상호작용 ---
def build_response(y):
    half = set(front if y[1] == "1" else back)
    half_cats = [c for c in half if c in set(cat_cols)]
    lin_pool = [c for c in half if (c in ord_cols) or
                (c in bin_cols and c not in trap_bits)]
    strong_vars = list(rng.choice(lin_pool, min(4, len(lin_pool)), replace=False))
    strong = {v: sgn() * rng.uniform(3.0, 6.0) for v in strong_vars}
    rem = [c for c in lin_pool if c not in strong_vars]
    weak_vars = list(rng.choice(rem, min(6, len(rem)), replace=False))
    weak = {v: sgn() * rng.uniform(0.1, 0.6) for v in weak_vars}

    # 블록 간 약한 상호작용 (trap 비트끼리 약하게 결합 → 분해 더 어렵게)
    inters = []
    tb = [v for tr in resp_traps[y] for v in tr["vars"]]
    if len(tb) >= 2:
        for _ in range(4):
            a, c = (tb[i] for i in rng.choice(len(tb), 2, replace=False))
            if block_of[a] != block_of[c]:
                inters.append({"vars": [a, c], "coef": sgn() * rng.uniform(0.1, 0.5),
                               "kind": "2way_cross"})

    cat_eff = {}
    for c in cat_cols:
        if c in half_cats:
            vals = rng.normal(0, 2.5, size=4); vals -= vals.mean()
            cat_eff[c] = {lv: float(v) for lv, v in zip(CAT_LEVELS, vals)}
        else:
            cat_eff[c] = {lv: 0.0 for lv in CAT_LEVELS}

    return {"intercept": float(rng.uniform(-2, 2)), "strong": strong, "weak": weak,
            "interactions": inters, "traps": resp_traps[y], "categorical": cat_eff}


spec = {y: build_response(y) for y in Y_COLS}

strong_coefs = [abs(c) for s in spec.values() for c in s["strong"].values()]
main_effect = float(np.mean(strong_coefs))
OBJECTIVE_NOISE_SD = round(NOISE_FRAC * main_effect, 4)

gt = {"meta": {"seed": SEED, "objective_noise_sd": OBJECTIVE_NOISE_SD,
               "noise_frac_of_main_effect": NOISE_FRAC, "main_effect_ref": round(main_effect, 4),
               "block_size": BLOCK, "n_blocks": N_BLOCKS, "trap_k": TRAP_K,
               "block_of": block_of, "resp_blocks": RESP_BLOCKS,
               "trap_bits": sorted(trap_bits), "column_position": pos,
               "family_columns": {"y1_(front50)": front, "y2_(back50)": back},
               "x_columns": {"binary": bin_cols,
                             "ordinal": {c: ord_levels[c] for c in ord_cols},
                             "categorical": {c: CAT_LEVELS for c in cat_cols}},
               "y_columns": Y_COLS},
      "responses": spec}

with open("prob3/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

n_traps = sum(len(s["traps"]) for s in spec.values())
print("saved prob3/ground_truth.json")
print(f"X: {N_TOTAL}열, {N_BLOCKS}블록, trap-{TRAP_K}")
print(f"trap 항 수: {n_traps} (전용 trap 비트 {len(trap_bits)}개)")
print(f"주효과(평균 strong선형계수)={main_effect:.2f}, 노이즈 sd={OBJECTIVE_NOISE_SD} (=4%)")
