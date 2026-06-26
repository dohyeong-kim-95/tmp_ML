"""
prob3 Problem — deceptive TRAP 블록 + 노이즈 목적함수.

objective(x)      = 진짜 ΣY + N(0, sd)   (최적화기가 봄, n_eval 증가)
true_objective(x) = 노이즈 없는 진짜 ΣY  (측정/참조)
global_reference()= 해석적 전역최적(trap비트=1, 선형 best, cat best) + 국소폴리시
   ※ trap은 coordinate ascent 를 무력화하므로 J*는 해석적으로 구한다.
"""
import json
import numpy as np

CAT_LEVELS = ["A", "B", "C", "D"]


class Problem:
    def __init__(self, gt_path="prob3/ground_truth.json", weights=None, seed=0):
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
        self._rng = np.random.default_rng(seed)

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
        for tr in s.get("traps", []):
            k = tr["k"]
            u = sum(int(x[v]) for v in tr["vars"])          # unitation (binary 0/1)
            val += tr["coef"] * (k if u == k else (k - 1 - u))
        for c in self.cat_cols:
            val += s["categorical"][c][x[c]]
        return val

    def _true(self, x):
        return sum(self.weights[y] * self.response(x, y) for y in self.y_cols)

    def objective(self, x):
        self.n_eval += 1
        return self._true(x) + self._rng.normal(0, self.noise_sd)

    def true_objective(self, x):
        return self._true(x)

    def random_solution(self, rng):
        return {c: dom[rng.integers(len(dom))] for c, (t, dom) in self.meta.items()}

    # ---- 해석적 전역최적 + 국소 폴리시 (trap 때문에 무작위 좌표상승은 못 찾음) ----
    def _linear_total(self, col):
        tot = 0.0
        for y in self.y_cols:
            s = self.gt["responses"][y]
            tot += self.weights[y] * (s["strong"].get(col, 0.0) + s["weak"].get(col, 0.0))
        return tot

    def _analytic_opt(self):
        trap_bits = set(self.gt["meta"]["trap_bits"])
        x = {}
        for c in self.bin_cols:
            x[c] = 1 if (c in trap_bits or self._linear_total(c) > 0) else 0
        for c in self.ord_cols:
            x[c] = (self.ord_levels[c] - 1) if self._linear_total(c) > 0 else 0
        for c in self.cat_cols:
            best_lv, best_v = "A", -1e18
            for lv in CAT_LEVELS:
                v = sum(self.weights[y] * self.gt["responses"][y]["categorical"][c][lv]
                        for y in self.y_cols)
                if v > best_v:
                    best_v, best_lv = v, lv
            x[c] = best_lv
        return x

    def _ascend(self, x):
        x = dict(x); improved = True
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
        return x

    def global_reference(self, polish_restarts=8, seed=0):
        x = self._ascend(self._analytic_opt())     # 해석적 최적 + 약한 상호작용 폴리시
        best_x, best_v = dict(x), self._true(x)
        rng = np.random.default_rng(seed)
        for _ in range(polish_restarts):            # 무작위 재시작도 비교(보수적)
            xx = self._ascend(self.random_solution(rng))
            v = self._true(xx)
            if v > best_v:
                best_v, best_x = v, dict(xx)
        return best_x, best_v
