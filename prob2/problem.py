"""
prob2 Problem — 목적함수에 '소폭 노이즈'를 추가한 혼합 이산 최적화.

핵심:
  - objective(x)      : 최적화기가 보는 값 = 진짜 ΣY + N(0, noise_sd)  (n_eval 증가)
  - true_objective(x) : 노이즈 없는 진짜 ΣY (성능 측정/참조용, n_eval 미증가)
  최적화기는 노이즈에 속을 수 있고, 최종 성능은 '추천한 X의 진짜값'으로 평가한다.

model-free 최적화기들이 쓰는 인터페이스(vars/meta/objective/random_solution/
coordinate_ascent)를 그대로 제공한다.
"""
import json
import numpy as np

CAT_LEVELS = ["A", "B", "C", "D"]


class Problem:
    def __init__(self, gt_path="prob2/ground_truth.json", weights=None, seed=0):
        gt = json.load(open(gt_path))
        self.gt = gt
        self.y_cols = gt["meta"]["y_columns"]
        self.noise_sd = gt["meta"]["objective_noise_sd"]
        self.bin_cols = gt["meta"]["x_columns"]["binary"]
        self.ord_levels = gt["meta"]["x_columns"]["ordinal"]
        self.ord_cols = list(self.ord_levels.keys())
        self.cat_cols = list(gt["meta"]["x_columns"]["categorical"].keys())
        self.vars = self.bin_cols + self.ord_cols + self.cat_cols
        self.weights = weights or {y: 1.0 for y in self.y_cols}
        self.n_eval = 0
        self._rng = np.random.default_rng(seed)   # 평가 노이즈용

        self.meta = {}
        for c in self.bin_cols:
            self.meta[c] = ("bin", [0, 1])
        for c in self.ord_cols:
            self.meta[c] = ("ord", list(range(self.ord_levels[c])))
        for c in self.cat_cols:
            self.meta[c] = ("cat", list(CAT_LEVELS))

    def _scaled(self, x, col):
        t = self.meta[col][0]
        if t == "bin":
            return float(x[col])
        if t == "ord":
            return float(x[col]) / (self.ord_levels[col] - 1)
        raise ValueError("categorical")

    def response(self, x, y):
        s = self.gt["responses"][y]
        val = s["intercept"]
        for v, c in s["strong"].items():
            val += c * self._scaled(x, v)
        for v, c in s["weak"].items():
            val += c * self._scaled(x, v)
        for it in s["interactions"]:
            vs = it["vars"] if "vars" in it else [it["a"], it["b"]]
            prod = 1.0
            for v in vs:
                prod *= self._scaled(x, v)
            val += it["coef"] * prod
        for c in self.cat_cols:
            val += s["categorical"][c][x[c]]
        return val

    def _true(self, x):
        return sum(self.weights[y] * self.response(x, y) for y in self.y_cols)

    def objective(self, x):
        """최적화기가 보는 노이즈 낀 값."""
        self.n_eval += 1
        return self._true(x) + self._rng.normal(0, self.noise_sd)

    def true_objective(self, x):
        """노이즈 없는 진짜값 (측정/참조, 평가수 미포함)."""
        return self._true(x)

    def random_solution(self, rng):
        x = {}
        for c, (t, dom) in self.meta.items():
            x[c] = dom[rng.integers(len(dom))]
        return x

    # 참조: 노이즈 없는 진짜 목적함수 기준 전역최적
    def coordinate_ascent(self, rng, restarts=20):
        best_x, best_v = None, -1e18
        for _ in range(restarts):
            x = self.random_solution(rng)
            improved = True
            while improved:
                improved = False
                for col, (t, dom) in self.meta.items():
                    cur, bv, bestv = x[col], self._true(x), x[col]
                    for cand in dom:
                        if cand == cur:
                            continue
                        x[col] = cand
                        v = self._true(x)
                        if v > bv:
                            bv, bestv = v, cand
                    x[col] = bestv
                    if bestv != cur:
                        improved = True
            v = self._true(x)
            if v > best_v:
                best_v, best_x = v, dict(x)
        return best_x, best_v

    def linear_optimum(self):
        x = {}
        for col in self.bin_cols + self.ord_cols:
            cs = 0.0
            for y in self.y_cols:
                s = self.gt["responses"][y]
                cs += self.weights[y] * (s["strong"].get(col, 0.0) + s["weak"].get(col, 0.0))
            x[col] = (1 if cs > 0 else 0) if self.meta[col][0] == "bin" else \
                     ((self.ord_levels[col] - 1) if cs > 0 else 0)
        for col in self.cat_cols:
            best_lv, best_v = None, -1e18
            for lv in CAT_LEVELS:
                v = sum(self.weights[y] * self.gt["responses"][y]["categorical"][col][lv]
                        for y in self.y_cols)
                if v > best_v:
                    best_v, best_lv = v, lv
            x[col] = best_lv
        return x
