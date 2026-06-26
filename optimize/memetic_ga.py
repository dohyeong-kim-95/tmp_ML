"""
Memetic GA = GA(tournament + uniform crossover + elitism) + 국소탐색(Lamarckian).

매 세대 상위 개체에 좌표상승 국소탐색을 적용해 '바구니는 GA가 탐색, 바닥은 LS가 정밀화'.
평가당 비용이 싸므로 TIME 예산에서 특히 강함. deadline(시간예산) 지원.
"""
import time
import numpy as np


def _rand_val(prob, col, rng):
    dom = prob.meta[col][1]
    return dom[rng.integers(len(dom))]


def _to_dict(prob, ch):
    return {c: ch[i] for i, c in enumerate(prob.vars)}


def _local_search(prob, x, rng, max_passes=2):
    """best-improvement 좌표상승(부분), Lamarckian. x를 제자리 개선하고 fitness 반환."""
    cols = list(prob.vars)
    f = prob.objective(x)
    for _ in range(max_passes):
        improved = False
        rng.shuffle(cols)
        for c in cols:
            dom = prob.meta[c][1]
            cur, bestv, bestf = x[c], x[c], f
            for cand in dom:
                if cand == cur:
                    continue
                x[c] = cand
                nf = prob.objective(x)
                if nf > bestf:
                    bestf, bestv = nf, cand
            x[c] = bestv
            if bestv != cur:
                f, improved = bestf, True
        if not improved:
            break
    return f


def _tournament(fit, rng, k=3):
    idx = rng.integers(0, len(fit), size=k)
    return idx[np.argmax(fit[idx])]


def memetic_ga(prob, max_eval=2000, pop_size=40, pm=None, n_ls=2,
               seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    cols = prob.vars
    n_genes = len(cols)
    pm = pm if pm is not None else 1.0 / n_genes

    pop = [[_rand_val(prob, c, rng) for c in cols] for _ in range(pop_size)]
    fit = np.array([prob.objective(_to_dict(prob, ch)) for ch in pop])
    bi = int(np.argmax(fit))
    best_chrom, best_f = list(pop[bi]), fit[bi]
    history = [best_f]

    while prob.n_eval < max_eval:
        new_pop = [list(best_chrom)]
        while len(new_pop) < pop_size:
            p1 = pop[_tournament(fit, rng)]
            p2 = pop[_tournament(fit, rng)]
            mask = rng.random(n_genes) < 0.5
            child = [p1[i] if mask[i] else p2[i] for i in range(n_genes)]
            for i, c in enumerate(cols):
                if rng.random() < pm:
                    child[i] = _rand_val(prob, c, rng)
            new_pop.append(child)
        pop = new_pop
        fit = np.array([prob.objective(_to_dict(prob, ch)) for ch in pop])

        # 상위 n_ls 개체에 국소탐색(Lamarckian)
        for j in np.argsort(fit)[::-1][:n_ls]:
            xd = _to_dict(prob, pop[j])
            f = _local_search(prob, xd, rng)
            pop[j] = [xd[c] for c in cols]
            fit[j] = f
            if prob.n_eval >= max_eval or (deadline and time.time() > deadline):
                break

        gi = int(np.argmax(fit))
        if fit[gi] > best_f:
            best_f, best_chrom = fit[gi], list(pop[gi])
        history.append(best_f)
        if deadline and time.time() > deadline:
            break
    return _to_dict(prob, best_chrom), best_f, history
