"""
CHC (Eshelman) — 혼합 이산 적응.

구성:
  - HUX 교차: 두 부모가 '다른' 위치의 절반을 교환(최대 다양성 교차)
  - incest prevention: Hamming 거리 > 임계 d 인 쌍만 교배, 진전 없으면 d 감소
  - elitist 병합: (부모+자식) 중 상위 N 생존
  - cataclysmic restart: d<0 이면 best만 남기고 나머지를 best의 강한 변이로 재생성
  - 돌연변이 없음(다양성은 restart로 공급)
평가당 비용이 싸고 작은 예산·다봉 문제에 견고 → TIME 예산에 적합. deadline 지원.
"""
import time
import numpy as np


def _rand_val(prob, col, rng):
    dom = prob.meta[col][1]
    return dom[rng.integers(len(dom))]


def _to_dict(prob, ch):
    return {c: ch[i] for i, c in enumerate(prob.vars)}


def _hamming(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _hux(p1, p2, rng):
    diff = [i for i in range(len(p1)) if p1[i] != p2[i]]
    rng.shuffle(diff)
    swap = diff[:len(diff) // 2]
    c1, c2 = list(p1), list(p2)
    for i in swap:
        c1[i], c2[i] = p2[i], p1[i]
    return c1, c2


def chc(prob, max_eval=2000, pop_size=40, div=0.35, restart_mut=0.35,
        seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    cols = prob.vars
    n = len(cols)

    pop = [[_rand_val(prob, c, rng) for c in cols] for _ in range(pop_size)]
    fit = np.array([prob.objective(_to_dict(prob, ch)) for ch in pop])
    bi = int(np.argmax(fit))
    best_chrom, best_f = list(pop[bi]), fit[bi]
    history = [best_f]

    d_thresh = int(div * n)   # incest 임계 초기값

    while prob.n_eval < max_eval:
        # 무작위 짝짓기 → incest 통과쌍만 HUX
        order = rng.permutation(pop_size)
        children = []
        for a, b in zip(order[::2], order[1::2]):
            if _hamming(pop[a], pop[b]) > d_thresh:
                c1, c2 = _hux(pop[a], pop[b], rng)
                children.extend([c1, c2])

        if not children:
            d_thresh -= 1   # 진전 없음 → 임계 완화
        else:
            cfit = np.array([prob.objective(_to_dict(prob, ch)) for ch in children])
            # elitist 병합: (부모+자식) 상위 N
            allc = pop + children
            allf = np.concatenate([fit, cfit])
            keep = np.argsort(allf)[::-1][:pop_size]
            new_pop = [allc[i] for i in keep]
            new_fit = allf[keep]
            # 세대 진전 여부 = 개체군 구성이 바뀌었나
            if np.array_equal(np.sort(new_fit), np.sort(fit)):
                d_thresh -= 1
            pop, fit = new_pop, new_fit

        gi = int(np.argmax(fit))
        if fit[gi] > best_f:
            best_f, best_chrom = fit[gi], list(pop[gi])

        # cataclysmic restart
        if d_thresh < 0:
            pop = [list(best_chrom)]
            while len(pop) < pop_size:
                ch = list(best_chrom)
                for i, c in enumerate(cols):
                    if rng.random() < restart_mut:
                        ch[i] = _rand_val(prob, c, rng)
                pop.append(ch)
            fit = np.array([prob.objective(_to_dict(prob, ch)) for ch in pop])
            d_thresh = int(div * n)

        history.append(best_f)
        if deadline and time.time() > deadline:
            break
    return _to_dict(prob, best_chrom), best_f, history
