# 09 — Phase 2 준비: "DSMGA-II/LT-GOMEA + GBM이 최선인가?" 검증 프레임 & 설계 질문

> 이 문서는 **방향성 구체화**용이다. **초기 데이터 없이 단정할 수 없는 항목은 "보류(데이터 필요)"** 로 표시한다.
> 목적: Phase 2에서 "현재 가설(MAIN=연관학습 EA+GBM)이 정말 최선인가"에 **데이터로 답할 준비**를 갖춘다.

## 0. 핵심 태도
현 §5의 MAIN 권고는 **확정이 아니라 가설**이다. Phase 2의 일은 이 가설을
**(a) 시뮬레이터/조기 데이터 기반 벤치마크**와 **(b) ablation**으로 검증·반증하는 것.

---

## A. 방법 선정 벤치마크 프레임 (Phase 2 핵심 산출)
- **후보군(optimizer)**: ① 연관학습 EA(DSMGA-II, LT-GOMEA) ② 단순 GA(uniform/2-point) ③ SUB의 BO(BOCS형/SMAC/TPE)
  ④ SAEA(EA+GBM 선별) ⑤ 순수 대리모델+local search ⑥ 랜덤/DoE 하한선.
- **테스트베드(데이터 0건 단계)**: 알려진 구조의 **합성 함수**(NK-landscape, trap, Ising, MAX-SAT)에
  본 문제 특성(그룹/배타/큰 상호작용/노이즈/배타제약)을 모사해 주입 → 실제 설비 전에 상대비교.
- **지표**: ① **target threshold 도달까지 평가 수**(§7 1차 기준) ② best-so-far 곡선 ③ 배치당 wall-clock
  ④ 추론 지연 ⑤ 노이즈 견딤 ⑥ 구조 추정 정확도(아래 C).
- **ablation**: linkage 학습 on/off, GBM 선별 on/off, 배치 다양성 on/off → "각 부품이 정말 기여하나" 분리 측정.
- ⚠️ **최종 우열 판정은 실제 데이터 특성에 의존 → 보류.** 본 단계는 "판정 도구"를 만든다.

---

## B. Q3 — 수천 표본으로 N>60 binary 상호작용을 어디까지 학습하나
**부분 답변(원리)** + **보류(식별 가능성은 데이터 SNR 의존)**.
- 자유도(파라미터 수)로 본 상한:
  - 주효과: N=60 → **60개**.
  - 2차(pairwise): C(60,2)=**1,770개**.
  - 3차: C(60,3)=**34,220개**, 4차: ~487,635개 → **희소성 없이는 수천~수만 표본으로 불가**.
- 따라서:
  - **수천 표본**: 주효과 + (희소 가정 하) **2차 상호작용**의 신뢰 추정이 현실적 상한.
  - **수만 표본(MAIN)**: 2차는 비교적 안정, **3차 이상은 effect heredity 등 강한 희소 가정에서만** 부분적.
  - **트리 앙상블(GBM/RF)**: 고차 상호작용을 *암묵적*으로 포착하나, **명시적 식별·신뢰도는 데이터·노이즈에 좌우** → 과적합 위험, 정규화·교차검증 필수.
- ⚠️ "실제로 몇 차까지 보이는가"는 **신호 대 잡음비와 구조에 의존 → 초기 데이터 후 측정**(학습곡선·교차검증으로 판별). **보류.**

---

## C. Q5 — 학습된 linkage가 진짜 구조인가, 허위 패턴인가
**검증 프로토콜(방향)** 제시, **판정은 데이터 필요 → 보류.**
- linkage를 **확정이 아닌 가설**로 취급하고 다음으로 검증:
  1. **안정성(stability)**: 부트스트랩/서로 다른 시드·세대에서 **반복적으로 같은 묶음**이 나오는가(재현성).
  2. **통계적 유의성**: DSM의 쌍별 의존(상호정보 등)을 **귀무가설(독립) 대비** 검정, 다중비교 보정.
  3. **예측 기여**: 그 linkage를 **존중한 모델/연산자**가 무시한 경우보다 **예측·탐색이 실제로 향상**되는가(held-out).
  4. **개입(intervention) 검증**: 후속 단계에서 해당 변수쌍을 **표적 실험**으로 흔들어 상호작용 확인.
  5. **도메인 대조**: 추정 linkage를 전문가 규칙과 대조(추후 도메인지식 단계와 양방향).
- 운영: linkage 추정치에 **신뢰도 라벨**을 붙여 "강/약/허위의심"으로 관리.

---

## D. Q6 — GBM 사전선별 시 배치 다양성 보장
**답변(방향) 확정**: 순수 top-k(예측 평균 상위 320)는 **다양성 붕괴** → 안 됨.
- 권장: **품질 + 다양성**을 함께 고르기.
  - **DPP(Determinantal Point Process) 기반 배치 선택**: 척력으로 다양·고품질 부분집합 선택(DPP-MAX/SAMPLE),
    품질점수를 커널에 결합 → batch BO의 표준 다양화 기법.
  - **클러스터링 후 대표 선발**, 또는 **거리/해밍 반경 제약**으로 근접 후보 제거.
  - **앙상블/사후 Thompson sampling**으로 후보별 다른 시나리오 반영 → 자연 다양성.
  - **ε-비율 탐색 슬롯**: 배치의 일부(예 5~10%)를 불확실성 높은/미탐색 영역에 강제 배정.
- 즉 GBM은 **품질 점수원**으로 쓰되 **선택 규칙에 다양성**을 명시적으로 넣는다.

---

## E. Q7 — SUB q=40을 어떻게 뽑나 (단순 ask×40의 함정)
**답변(방향) 확정**: 독립적으로 ask 40번 → **중복/근접 후보** 발생(맞음). 배치-aware로 뽑아야 함.
- 권장:
  - **배치 Thompson sampling**: 매 추출마다 **다른 사후 표본**의 최적점 → q=40에 가장 매끄럽고 자연 다양.
  - **Local Penalization**: 고른 점 주변 acquisition을 순차 페널티 → 다양성 확보.
  - **Kriging Believer / Constant Liar**: pending 후보에 가상값을 채워 다음 후보를 조건부로 선택(Optuna의 `constant_liar` 등).
  - **DPP 다양화**(위 D와 동일 도구).
- 핵심: "한 모델에서 argmax 40번" ❌ → "**조건부/표본/페널티/척력**으로 서로 다른 40점" ✅. (research/05 참조)

---

## F. Q11 — 왜 GBM/RF인가, 고차 binary 상호작용에서 캘리브레이션은?
**왜 GBM/RF (baseline 대리모델로):**
- binary/범주 **native 처리**, 트리 분기로 **상호작용 자동 포착**, 무관변수에 강건, **학습·추론 빠름(ms)**,
  데이터에 선형 확장, 약한 튜닝으로도 좋은 정확도, **SHAP**으로 변수중요도·상호작용 해석 → 표 형식의 강한 기본기.
**캘리브레이션 우려(정확한 지적):**
- 트리 앙상블의 **점추정은 쓸만하나 *불확실성*은 기본적으로 잘 캘리브레이션 안 됨.**
  - RF는 트리 분산으로 대략의 분산 추정 가능하나 **GBM은 그 방식이 잘 안 통함**.
- 대리모델을 **acquisition 안에서 불확실성으로 쓰려면 보정 필요**:
  - **Conformalized Quantile Regression(CQR)**: 분위수회귀+컨포멀 → **엄밀히 캘리브레이션된 예측구간**.
  - **NGBoost**: 분포 파라미터를 부스팅으로 추정(확률적 예측).
  - **분위수 부스팅**(LightGBM quantile), **앙상블/jackknife+**, **컨포멀 예측**.
- 운영 가이드:
  - GBM을 **순수 랭킹/triage**(상위 후보 선별)에 쓰면 캘리브레이션 민감도↓.
  - **불확실성 기반 탐색**(UCB/EI류)에 쓰려면 **CQR/NGBoost/컨포멀로 보정**하거나 BO 트랙(GP/BOCS)로.
- ⚠️ "이 문제에서 GBM 캘리브레이션이 충분한가"는 **데이터로 신뢰도 다이어그램 확인 필요 → 보류.**

---

## G. confirmatory measurement는 필요한가 (target threshold 성공 기준)
**답변(방향) 확정: 필요하다.** 단발 측정으로 threshold 통과를 "성공" 선언하면 안 됨.
- **이유 — winner's / optimizer's curse**: 노이즈가 있는 다수 후보에서 "임계 초과한 것"만 선택하면,
  그 선택집합은 **노이즈로 과대추정된 후보가 우선 포함** → 재측정 시 **평균 회귀(regression to the mean)** 로 값이 내려감.
  (선택/최적화 자체가 예측오차를 악용하는 현상; A/B·GWAS·의사결정분석에서 잘 알려짐.)
- **대응**:
  1. **2단계 판정(discovery → confirmation)**: 통과 의심 상위 후보를 **반복 측정(confirmation runs)** 해 참값 확정.
  2. **편향 보정**: 선택편향/winner's curse 보정 추정으로 "재현 확률"을 추정, 다중비교 보정.
  3. **예산 배정**: 매 배치의 일부를 **상위 후보 재측정**에 할당 → 노이즈(Q10) 추정과 confirmation을 겸함.
- 성공 기준(§7)은 따라서 **"confirmation으로 확정된 threshold 통과 X"** 로 정의해야 신뢰 가능.
- ⚠️ 필요한 **반복 횟수·임계**는 노이즈 크기에 의존 → **데이터 후 확정(보류)**.

---

## 종합
- **확정(데이터 불요)**: 배치 다양성(D)·SUB 배치추출(E)·GBM 캘리브레이션 보정 도구(F)·상호작용 자유도 상한(B 일부)·검증 프로토콜(C)·**confirmatory 2단계 판정(G)**.
- **보류(데이터 필요)**: 식별 가능 상호작용 차수의 실제값(B), linkage 진위 판정(C), 방법 우열 최종판정(A), GBM 캘리브레이션 충분성(F), confirmation 반복횟수(G).
- Phase 2의 산출은 **"판정 도구(벤치+ablation+검증 프로토콜)"** 이며, **우열 결론은 초기 데이터 확보 후**.

## Sources
- 자유도/상호작용: [DOE 스크리닝 원리](https://online.stat.psu.edu/stat503/lesson/8/8.4) (effect sparsity/heredity)
- winner's/optimizer's curse·confirmation: [The Optimizer's Curse (Smith & Winkler)](https://www.researchgate.net/publication/220534939_The_Optimizer's_Curse_Skepticism_and_Postdecision_Surprise_in_Decision_Analysis) · [Statistical correction of the Winner's Curse (PLOS Genetics)](https://journals.plos.org/plosgenetics/article?id=10.1371%2Fjournal.pgen.1006916) · [Winner's Curse in A/B Testing](https://atticusli.com/replication-crisis/ab-testing-winners-curse/) · [Beating the Winner's Curse via Inference-Aware Policy Optimization](https://www.arxiv.org/pdf/2510.18161)
- 배치 다양성(DPP): [Diversified Sampling for Batched BO with DPPs](https://arxiv.org/abs/2110.11665) · [Batched GP Bandit via DPP](https://arxiv.org/pdf/1611.04088) · [Enhancing Batch Diversity in Surrogate Optimization (DPP)](https://dl.acm.org/doi/10.1145/3721296)
- GBM/RF 불확실성: [Instance-Based Uncertainty for Gradient-Boosted Trees](https://arxiv.org/pdf/2205.11412) · [Conformalized Quantile Regression](https://valeman.medium.com/conformalized-quantile-regression-smarter-uncertainty-prediction-for-data-scientists-6389bea7a7c4) · [sklearn GBM quantile intervals](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
- 배치 BO 추출/연관학습: research/05, research/02, research/07 및 그 출처(Batch Thompson, DSMGA-II 등)
