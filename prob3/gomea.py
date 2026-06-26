"""
LT-GOMEA (Linkage Tree Gene-pool Optimal Mixing EA) — 혼합 이산.

핵심: 변수간 '연관구조(linkage)'를 개체군에서 학습해 building block 단위로 교환.
  1) 매 세대 변수쌍 정규화 상호정보(NMI)로 유사도 → UPGMA 계층군집 = Linkage Tree(FOS)
  2) Gene-pool Optimal Mixing: 각 해마다 FOS 부분집합을 무작위 donor로 통째로 덮어쓰고
     평가 → 나빠지지 않으면(>=) 수용. (교호작용을 깨지 않고 통째로 옮김)
교호작용·중첩 구조에 강함. 평가당 비용이 싸 TIME 예산에도 적합. deadline 지원.
"""
import time
import numpy as np


def _rand_val(prob, col, rng):
    dom = prob.meta[col][1]
    return dom[rng.integers(len(dom))]


def _to_dict(prob, ch):
    return {c: ch[i] for i, c in enumerate(prob.vars)}


def _learn_fos(pop_idx, L, rng):
    """개체군(정수인덱스 행렬)에서 NMI 기반 UPGMA 링키지 트리 → FOS(부분집합 리스트)."""
    N = pop_idx.shape[0]
    # 변수쌍 정규화 상호정보
    sim = np.zeros((L, L))
    ent = np.zeros(L)
    probs = []
    for i in range(L):
        vals, cnt = np.unique(pop_idx[:, i], return_counts=True)
        p = cnt / N
        ent[i] = max(0.0, -np.sum(p * np.log(p + 1e-12)))   # 음수(부동소수) 방지
        probs.append(dict(zip(vals.tolist(), p)))
    for i in range(L):
        for j in range(i + 1, L):
            # joint
            pair = pop_idx[:, i] * 1000 + pop_idx[:, j]
            _, cnt = np.unique(pair, return_counts=True)
            pj = cnt / N
            Hij = -np.sum(pj * np.log(pj + 1e-12))
            mi = max(0.0, ent[i] + ent[j] - Hij)
            denom = np.sqrt(ent[i] * ent[j])
            nmi = mi / denom if denom > 1e-9 else 0.0
            sim[i, j] = sim[j, i] = nmi

    # UPGMA(평균연결) 벡터화: 군집-유사도 행렬을 Lance-Williams로 갱신
    fos = [[i] for i in range(L)]      # 단일 변수 포함
    C = sim.copy()
    np.fill_diagonal(C, -1e9)
    members = {i: [i] for i in range(L)}
    size = {i: 1 for i in range(L)}
    active = list(range(L))
    next_id = L
    while len(active) > 1:
        idx = np.array(active)
        sub = C[np.ix_(idx, idx)]
        flat = int(np.argmax(sub))
        ai, bi2 = divmod(flat, len(idx))
        a, b = int(idx[ai]), int(idx[bi2])
        merged = members[a] + members[b]
        if len(merged) < L:
            fos.append(merged)
        # 새 군집 행/열 = 평균연결(가중)
        sa, sb = size[a], size[b]
        new_row = np.full(next_id + 1, -1e9)
        for c in active:
            if c in (a, b):
                continue
            new_row[c] = (sa * C[a, c] + sb * C[b, c]) / (sa + sb)
        # 행렬 확장
        C2 = np.full((next_id + 1, next_id + 1), -1e9)
        C2[:next_id, :next_id] = C
        C2[next_id, :] = new_row
        C2[:, next_id] = new_row
        C = C2
        members[next_id] = merged; size[next_id] = sa + sb
        active.remove(a); active.remove(b); active.append(next_id)
        next_id += 1
    return fos


def gomea(prob, max_eval=2000, pop_size=30, seed=0, deadline=None):
    rng = np.random.default_rng(seed)
    cols = prob.vars
    L = len(cols)
    doms = [list(prob.meta[c][1]) for c in cols]
    idx_of = [{v: k for k, v in enumerate(doms[i])} for i in range(L)]

    # 개체군(값 그대로 저장) + fitness
    pop = [[_rand_val(prob, c, rng) for c in cols] for _ in range(pop_size)]
    fit = [prob.objective(_to_dict(prob, ch)) for ch in pop]
    best_chrom, best_f, history = None, -1e18, []

    def stop():
        return prob.n_eval >= max_eval or (deadline and time.time() > deadline)

    # 재시작(IMS): 수렴하면 새 개체군으로 다시 시작, 전역 best 유지, 예산까지 반복
    cur_pop = pop_size
    while not stop():
        pop = [[_rand_val(prob, c, rng) for c in cols] for _ in range(cur_pop)]
        fit = [prob.objective(_to_dict(prob, ch)) for ch in pop]
        bi = int(np.argmax(fit))
        if fit[bi] > best_f:
            best_f, best_chrom = fit[bi], list(pop[bi])
        history.append(best_f)

        while not stop():
            evals_before = prob.n_eval
            pop_idx = np.array([[idx_of[i][pop[s][i]] for i in range(L)]
                                for s in range(cur_pop)])
            if all(len(np.unique(pop_idx[:, i])) == 1 for i in range(L)):
                break                                 # 수렴 → 재시작
            fos = _learn_fos(pop_idx, L, rng)

            for s in range(cur_pop):
                rng.shuffle(fos)
                for subset in fos:
                    donor = rng.integers(cur_pop)
                    if donor == s:
                        continue
                    backup = {i: pop[s][i] for i in subset}
                    changed = False
                    for i in subset:
                        if pop[s][i] != pop[donor][i]:
                            pop[s][i] = pop[donor][i]; changed = True
                    if not changed:
                        continue
                    nf = prob.objective(_to_dict(prob, pop[s]))
                    if nf >= fit[s]:
                        fit[s] = nf
                        if nf > best_f:
                            best_f, best_chrom = nf, list(pop[s])
                    else:
                        for i in subset:
                            pop[s][i] = backup[i]
                    if stop():
                        break
                history.append(best_f)
                if stop():
                    break
            if prob.n_eval == evals_before:           # 평가 0 → 수렴, 재시작
                break
        cur_pop = min(cur_pop * 2, 200)               # IMS: 재시작마다 개체군 2배
    return _to_dict(prob, best_chrom), best_f, history
