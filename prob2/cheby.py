"""prob2 위에 augmented Chebyshev(균형) 목적함수 wrapper. 두 family에 다른 스케일."""
import numpy as np
from problem import Problem

SCALE = {"y11":1.0,"y12":1.0,"y13":1.0,"y21":5.0,"y22":5.0,"y23":5.0}  # family2 ×5
EPS = 0.05

def _extreme(base, scorefn, rng, restarts, maximize=True):
    """scorefn의 실제 최대(또는 최소)값 추정 (좌표상승 재시작)."""
    sense = 1 if maximize else -1
    best=-1e18
    for _ in range(restarts):
        x=base.random_solution(rng); imp=True
        while imp:
            imp=False
            for col,(t,dom) in base.meta.items():
                cur,bv,b2=x[col],sense*scorefn(x),x[col]
                for cand in dom:
                    if cand==cur: continue
                    x[col]=cand; v=sense*scorefn(x)
                    if v>bv: bv,b2=v,cand
                x[col]=b2
                if b2!=cur: imp=True
        best=max(best, sense*scorefn(x))
    return sense*best          # 최소면 sense=-1 곱해 실제 극값 복원

class Cheb:
    """optimizer 인터페이스 제공: vars/meta/objective/true_objective/random_solution/..."""
    def __init__(self, seed=0, norm=None):
        b=Problem(seed=seed); self.b=b
        for a in ["vars","meta","bin_cols","ord_cols","cat_cols","ord_levels","y_cols"]:
            setattr(self,a,getattr(b,a))
        self.noise_sd=b.noise_sd; self.n_eval=0
        self._rng=np.random.default_rng(seed)
        if norm is not None:               # 미리 계산한 정규화 재사용(속도)
            self.ideal,self.nadir,self.rng_=norm
        else:
            rng=np.random.default_rng(0)
            self.ideal={c:_extreme(b, lambda x,c=c: SCALE[c]*b.response(x,c), rng, 30, True) for c in self.y_cols}
            self.nadir={c:_extreme(b, lambda x,c=c: SCALE[c]*b.response(x,c), rng, 30, False) for c in self.y_cols}
            self.rng_={c:max(self.ideal[c]-self.nadir[c],1e-6) for c in self.y_cols}
    def shortfalls(self,x):
        return [(self.ideal[c]-SCALE[c]*self.b.response(x,c))/self.rng_[c] for c in self.y_cols]
    def _true(self,x):
        sf=self.shortfalls(x); return -(max(sf)+EPS*sum(sf))   # 높을수록(0에 가까울수록) 좋음
    def true_objective(self,x): return self._true(x)
    def objective(self,x):
        self.n_eval+=1; return self._true(x)+self._rng.normal(0,self.noise_sd)
    def random_solution(self,rng): return self.b.random_solution(rng)
    def coordinate_ascent(self,rng,restarts=20):
        bx,bv=None,-1e18
        for _ in range(restarts):
            x=self.random_solution(rng); imp=True
            while imp:
                imp=False
                for col,(t,dom) in self.meta.items():
                    cur,bvv,b2=x[col],self._true(x),x[col]
                    for cand in dom:
                        if cand==cur: continue
                        x[col]=cand; v=self._true(x)
                        if v>bvv: bvv,b2=v,cand
                    x[col]=b2
                    if b2!=cur: imp=True
            v=self._true(x)
            if v>bv: bv,bx=v,dict(x)
        return bx,bv
    def minz(self,x):  # 균형 지표: 가장 뒤처진 Y의 정규화 달성도
        return 1-max(self.shortfalls(x))
