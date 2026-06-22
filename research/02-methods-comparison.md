# 02 — 후보 방법 비교 (고차원 binary/조합 BO)

> 평가 축: **표본효율 / 배치(q≈40) 지원 / 제약·구조 처리 / N>60 확장성 / 반복 wall-clock / 해석성 / 구현 성숙도**

## 요약 표

| 방법 | 대리모델 | 표본효율 | 배치 | N>60 | 반복 wall-clock | 해석성(상호작용) | 구현 |
|------|----------|---------|------|------|-----------------|------------------|------|
| **BOCS** | 희소 베이지안 **선형회귀 + 2차 상호작용**(horseshoe) | 높음 | △(직접 구현) | △ (pairwise O(N²)≈1800항, SA로 가능; 대형은 PSR 필요) | 중 (선형적합 가벼움, SA 획득 중간) | ★★★ (계수=상호작용) | MATLAB+CVX, Python(BOCSpy) |
| **COMBO** | GP + **그래프 확산(diffusion) 커널** | 높음 | △ | ○ (그래프 Cartesian product로 GFT 선형화) | 중~높음 (GP) | ★ | 논문 공개 구현 |
| **Casmopolitan** | GP(범주 커널) + **로컬 trust region** | 높음 | ✕(명시 없음) | ○ (ackley53 등 ~50d) | 중~높음 (GP) | ★ | GitHub(xingchenwan) |
| **SMAC3** | **랜덤포레스트** | 중 | ○ | ◎ (범주·조건부·고차원 강건) | **낮음(빠름, 데이터에 선형)** | ★★ (변수중요도) | 성숙(pip) |
| **TPE (Optuna)** | 트리 Parzen 추정 | 중(낮음~) | ○(ask/tell) | ◎ (선형 확장) | **낮음(매우 빠름)** | ★ | 매우 성숙(pip) |
| **SAASBO** | GP + **희소 축정렬 부분공간**(half-Cauchy, HMC) | **매우 높음** | △(Ax) | ◎ (수백 차원, 무관변수 多) | **높음(HMC 비쌈)** | ★★ (중요 축 식별) | Ax/BoTorch |

(◎>○>△>✕, ★ 많을수록 우수)

## 방법별 메모

### BOCS — Bayesian Optimization of Combinatorial Structures (Baptista & Poloczek, ICML 2018)
- 대리모델: **binary 입력의 베이지안 선형회귀 + pairwise(2차) 상호작용 항**. 희소화 위해 **horseshoe** 사전분포(구현 옵션: `horseshoe`/`bayes`/`mle`).
- 획득함수 최적화: **BOCS-SDP**(준정부호계획) 또는 **BOCS-SA**(시뮬레이티드 어닐링). 대형 차원에서는 SDP 병목→SA가 대안.
- 예시 벤치: Ising 9-노드(12변수), 예산 100 평가.
- **장점**: 계수 자체가 **주효과/상호작용 → 해석성 최고**, 추후 도메인지식화의 다리. 선형이라 적합이 가벼움.
- **약점**: binary 수가 매우 커지면 확장성 저하(알려진 한계) → **PSR(parametrized submodular relaxation)** 으로 획득 최적화 확장성·정확도 개선 연구 존재(아래).
- 본 문제 적합도: **N≈60~100, 예산 O(10³)** 범위면 pairwise 항(~수천 개) + 희소 사전분포로 충분히 시도 가능.

### Scalable Combinatorial BO (PSR, Deshwal et al. 2020)
- BOCS의 **획득함수 최적화 확장성** 한계를 submodular relaxation으로 개선. 대형 binary/categorical에 BOCS형을 밀어붙일 때 참고.
- ⚠️ 본문 PDF 직접 접근 차단 → 초록/2차 출처 기반.

### COMBO — Graph Cartesian Product (Oh et al., NeurIPS 2019)
- GP에 **ARD diffusion(heat) 커널**을 그래프 위에 정의 → **고차 상호작용** 모델링. 그래프 Cartesian product로 Graph Fourier Transform이 **지수→선형** 스케일.
- MaxSAT, NAS 등에서 SOTA, 통계·계산 효율 양호 보고.

### Casmopolitan (Wan et al., ICML 2021) — "Think Global and Act Local"
- **로컬 trust region** + 범주/혼합 특화 커널. ~50차원(ackley53) 문제. 배치 언급 없음.
- 참고: Casmopolitan·COMBO 커널은 **Hamming 그래프 위 heat kernel과 동치**라는 후속 분석 존재(즉 one-hot 후 RBF로도 근사 구현 가능).

### SMAC3 (랜덤포레스트 대리모델)
- RF는 **고차원·범주·조건부 입력에 강건**, **데이터에 선형 확장**, wall-clock 빠름(수 초/스터디).
- 단점: GP 대비 표본효율은 다소 낮음, 불확실성 추정이 거칠다.
- **조건부/제약·구조가 많은 본 문제에 매우 실용적 baseline.**

### TPE (Optuna)
- 범주형 처리·**선형 확장·매우 빠른 반복**. GP보다 표본효율은 낮지만 가장 구현·운영이 쉬움. ask/tell로 배치 가능.
- "가장 빠르게 굴려보는 baseline"으로 적합.

### SAASBO (Eriksson & Jankowiak 2021)
- GP 역길이척도에 **half-Cauchy 계층 희소 사전분포** + HMC 추론 → **수백 차원·소수 평가**에 강력, 무관 변수 자동 솎음.
- ⚠️ **HMC가 비싸 반복 wall-clock 부담**(데이터 누적 시 가중). 본 문제의 "반복시간 제약"과 충돌 가능 → 후보지만 주력 신중.

## 본 문제 관점 1차 선별
- **주력 후보**: BOCS형(해석·표본효율) + SMAC3/TPE(강건·빠름)의 **2-트랙 비교**.
- **관찰 후보**: COMBO/Casmopolitan(표본효율 우수하나 GP 반복비용·배치 미지원 보완 필요), SAASBO(반복시간 부담).

## Sources
- [BOCS paper (PMLR v80)](http://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf) · [BOCS arXiv](https://arxiv.org/pdf/1806.08838) · [BOCS code](https://github.com/baptistar/BOCS)
- [Scalable Combinatorial BO with Tractable Statistical Models (PSR)](https://arxiv.org/pdf/2008.08177)
- [COMBO — Graph Cartesian Product (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html) · [PDF](https://arxiv.org/pdf/1902.00448)
- [Casmopolitan (ICML 2021) code](https://github.com/xingchenwan/Casmopolitan) · [paper](https://arxiv.org/pdf/2102.07188)
- [Heat Kernels in Combinatorial BO (kernel equivalence)](https://arxiv.org/pdf/2510.26633)
- [SMAC3 paper](https://www.researchgate.net/publication/354766153_SMAC3_A_Versatile_Bayesian_Optimization_Package_for_Hyperparameter_Optimization)
- [Optuna paper](https://arxiv.org/pdf/1907.10902)
- [SAASBO (arXiv 2103.00349)](https://arxiv.org/abs/2103.00349) · [Ax tutorial](https://ax.dev/docs/0.5.0/tutorials/saasbo/) · [BoTorch tutorial](https://botorch.org/docs/tutorials/saasbo)
- [Dictionary-based Embeddings for High-Dim Combinatorial BO](https://arxiv.org/pdf/2303.01774)
