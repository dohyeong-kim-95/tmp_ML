"""
Vanilla Genetic Algorithm (혼합 이산공간).

- 염색체  : 결정변수의 native 값 리스트 (binary/ordinal 정수, categorical 문자)
- 선택    : 토너먼트 (size 3)
- 교차    : uniform crossover (유전자별 50% 교환)
- 돌연변이: 유전자별 확률 pm 로 도메인 내 무작위 재샘플
- 엘리트  : 최상 1개 보존
"""
import time
import numpy as np


def random_chrom(prob, rng):
    return [prob.random_solution_value(col, rng) for col in prob.vars] \
        if hasattr(prob, "random_solution_value") else \
        [_rand_val(prob, col, rng) for col in prob.vars]


def _rand_val(prob, col, rng):
    t, dom = prob.meta[col]
    return dom[rng.integers(len(dom))]


def to_dict(prob, chrom):
    return {col: chrom[i] for i, col in enumerate(prob.vars)}


def tournament(pop_f, rng, k=3):
    idx = rng.integers(0, len(pop_f), size=k)
    return idx[np.argmax(pop_f[idx])]


def genetic_algorithm(prob, max_eval=20000, pop_size=80, pm=None, seed=0,
                      deadline=None):
    rng = np.random.default_rng(seed)
    n_genes = len(prob.vars)
    pm = pm if pm is not None else 1.0 / n_genes  # 유전자당 기대 1회

    pop = [[_rand_val(prob, c, rng) for c in prob.vars] for _ in range(pop_size)]
    fit = np.array([prob.objective(to_dict(prob, ch)) for ch in pop])

    best_i = int(np.argmax(fit))
    best_chrom, best_f = list(pop[best_i]), fit[best_i]
    history = [best_f]

    n_gen = (max_eval - pop_size) // pop_size
    for _ in range(n_gen):
        new_pop = [list(best_chrom)]  # 엘리트
        while len(new_pop) < pop_size:
            p1 = pop[tournament(fit, rng)]
            p2 = pop[tournament(fit, rng)]
            # uniform crossover
            mask = rng.random(n_genes) < 0.5
            child = [p1[i] if mask[i] else p2[i] for i in range(n_genes)]
            # mutation
            for i, col in enumerate(prob.vars):
                if rng.random() < pm:
                    child[i] = _rand_val(prob, col, rng)
            new_pop.append(child)

        pop = new_pop
        fit = np.array([prob.objective(to_dict(prob, ch)) for ch in pop])
        gi = int(np.argmax(fit))
        if fit[gi] > best_f:
            best_f, best_chrom = fit[gi], list(pop[gi])
        history.append(best_f)
        if deadline and time.time() > deadline:
            break

    return to_dict(prob, best_chrom), best_f, history
