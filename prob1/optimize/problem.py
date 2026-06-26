"""
최적화 문제 정의 (공통 모듈).

결정변수 X (40개):
  - binary 32개      : {0, 1}
  - ordinal 4개      : 0..(L-1)
  - categorical 4개  : {A, B, C, D}

목적함수:
  ground_truth.json 으로부터 '노이즈 없는' 진짜 반응함수 f_y(X) 를 재구성하고,
  네 반응의 합  J(X) = y11 + y12 + y21 + y22  를 '최대화'한다.
  (세 메타휴리스틱이 모두 이 동일한 결정적 black-box 를 최적화 → 공정 비교)

평가 횟수(eval) 를 카운트해 알고리즘별 수렴을 같은 예산에서 비교한다.
"""
import json
import numpy as np

CAT_LEVELS = ["A", "B", "C", "D"]


class Problem:
    def __init__(self, gt_path="data/ground_truth.json", weights=None):
        gt = json.load(open(gt_path))
        self.gt = gt
        self.y_cols = gt["meta"]["y_columns"]
        self.bin_cols = gt["meta"]["x_columns"]["binary"]
        self.ord_levels = gt["meta"]["x_columns"]["ordinal"]   # {col: L}
        self.ord_cols = list(self.ord_levels.keys())
        self.cat_cols = list(gt["meta"]["x_columns"]["categorical"].keys())
        self.vars = self.bin_cols + self.ord_cols + self.cat_cols
        self.weights = weights or {y: 1.0 for y in self.y_cols}
        self.n_eval = 0

        # 변수 메타: type, domain
        self.meta = {}
        for c in self.bin_cols:
            self.meta[c] = ("bin", [0, 1])
        for c in self.ord_cols:
            L = self.ord_levels[c]
            self.meta[c] = ("ord", list(range(L)))
        for c in self.cat_cols:
            self.meta[c] = ("cat", list(CAT_LEVELS))

    # --- native 값 -> 효과 계산용 scaled 값 (생성기와 동일 규칙) ---
    def _scaled(self, x, col):
        t = self.meta[col][0]
        if t == "bin":
            return float(x[col])
        if t == "ord":
            L = self.ord_levels[col]
            return float(x[col]) / (L - 1)
        raise ValueError("categorical은 _scaled 대상 아님")

    def response(self, x, y):
        """노이즈 없는 단일 반응 f_y(x). x는 {col: native_value} dict."""
        s = self.gt["responses"][y]
        val = s["intercept"]
        for v, c in s["strong"].items():
            val += c * self._scaled(x, v)
        for v, c in s["weak"].items():
            val += c * self._scaled(x, v)
        for it in s["interactions"]:
            # 스키마 호환: DB1은 {a,b}, DB2~5는 {vars:[...]} (2·3차·quadratic)
            vs = it["vars"] if "vars" in it else [it["a"], it["b"]]
            prod = 1.0
            for v in vs:
                prod *= self._scaled(x, v)
            val += it["coef"] * prod
        for c in self.cat_cols:
            val += s["categorical"][c][x[c]]
        return val

    def objective(self, x):
        """J(x) = Σ w_y · f_y(x)  (최대화 대상). eval 카운트 증가."""
        self.n_eval += 1
        return sum(self.weights[y] * self.response(x, y) for y in self.y_cols)

    # --- 무작위 해 생성 ---
    def random_solution(self, rng):
        x = {}
        for c, (t, dom) in self.meta.items():
            x[c] = dom[rng.integers(len(dom))] if t != "cat" else dom[rng.integers(4)]
        return x

    # ============================================================
    # 참조 기준선
    # ============================================================
    def linear_optimum(self):
        """교호작용을 무시한 분리형 최적해 (변수별 독립 최적). 강한 baseline."""
        x = {}
        # binary/ordinal: 모든 Y에 대한 (strong+weak) 선형계수 합
        for col in self.bin_cols + self.ord_cols:
            coef_sum = 0.0
            for y in self.y_cols:
                s = self.gt["responses"][y]
                coef_sum += self.weights[y] * (s["strong"].get(col, 0.0)
                                               + s["weak"].get(col, 0.0))
            t = self.meta[col][0]
            if t == "bin":
                x[col] = 1 if coef_sum > 0 else 0
            else:  # ord: scaled∈[0,1] → 계수 양수면 최대레벨, 아니면 0
                x[col] = (self.ord_levels[col] - 1) if coef_sum > 0 else 0
        # categorical: level별 효과 합 최대
        for col in self.cat_cols:
            best_lv, best_v = None, -1e18
            for lv in CAT_LEVELS:
                v = sum(self.weights[y] * self.gt["responses"][y]["categorical"][col][lv]
                        for y in self.y_cols)
                if v > best_v:
                    best_v, best_lv = v, lv
            x[col] = best_lv
        return x

    def coordinate_ascent(self, rng, restarts=20):
        """랜덤 재시작 좌표상승 → best-known (사실상 전역최적 근사)."""
        best_x, best_v = None, -1e18
        for _ in range(restarts):
            x = self.random_solution(rng)
            improved = True
            while improved:
                improved = False
                for col, (t, dom) in self.meta.items():
                    cur = x[col]
                    best_local, bv = cur, self.objective(x)
                    for cand in dom:
                        if cand == cur:
                            continue
                        x[col] = cand
                        v = self.objective(x)
                        if v > bv:
                            bv, best_local = v, cand
                    x[col] = best_local
                    if best_local != cur:
                        improved = True
            v = self.objective(x)
            if v > best_v:
                best_v, best_x = v, dict(x)
        return best_x, best_v
