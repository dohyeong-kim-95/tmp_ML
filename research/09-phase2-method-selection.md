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
- **교수 질문에 대한 한 줄 답**: *"우연한 패턴과 실제 구조는 ① 재표집(부트스트랩/시드)에서 **재현되는가**, ② **귀무모형(변수 독립·순열검정) 대비 유의**한가, ③ **out-of-sample 예측·탐색을 실제로 개선**하는가 — 세 관문을 통과해야 실제 구조로 인정. 결정적으로는 ④ 표적 개입실험으로 확인."*
  - **순열검정(null model)**: 라벨/비트를 셔플해 만든 "구조 없는" 데이터에서 같은 강도의 linkage가 나오는 빈도 → 우연 가능성 정량화(다중비교 보정).
  - **학습곡선(consistency)**: 데이터가 늘수록 **진짜 linkage는 강화·안정, 허위는 사라짐** → budget에 따른 안정성 추이를 모니터.
- 운영: linkage 추정치에 **신뢰도 라벨**(강/약/허위의심)을 붙여 관리. EA는 **강 신뢰** 묶음만 강하게 활용하고 약/의심은 탐색적으로.

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

### D2. Q2 — "두 X가 얼마나 다른가": mixed-type distance를 어떻게 정의하나
다양성/DPP/클러스터링은 **거리(또는 유사도)** 가 있어야 한다. X는 binary/categorical/ordinal/conditional **혼합형** → 유클리드 부적합.
- **1순위: Gower distance (혼합형 표준).** 컬럼별로 dissimilarity를 정의해 가중평균(→[0,1]):
  - binary/categorical → **일치=0, 불일치=1**(비대칭 binary는 Dice/Jaccard).
  - ordinal → **정규화 순위차**(|rank_i−rank_j| / range).
  - numeric → 정규화 |차이|.
- **conditional(계층) 처리 — 핵심 주의점**: 비활성(NA) 컬럼이 생김 → ① 활성집합이 다르면 **활성 불일치 페널티**를 주고 ② 거리는 **공통 활성 차원에서만** 계산하는 식으로 정의(HPO 혼합·계층 거리 관행).
- **(선택) target-aware**: 단순 일치/불일치 대신 **VDM(Value Difference Metric, HVDM 계열)** 로 "그 차이가 Y를 얼마나 바꾸나"를 반영 → **Y-관련 방향의 다양성**. 또는 Gower 가중치를 **surrogate 변수중요도(SHAP)** 로 설정.
- **DPP/커널 변환**: 거리 d → 유사도 k=exp(−d²/2ℓ²) 로 DPP 커널 구성(품질점수를 곱해 quality×diversity).
- ⚠️ **가중치·길이척도(ℓ)·VDM 통계는 데이터로 튜닝 → 보류.** baseline은 **Gower(균등 가중) + 활성집합 페널티**로 시작.

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

### F2. Q3 — surrogate가 "좋은 후보를 잘 고른다"를 무슨 지표로 확인하나
**핵심(맞는 지적)**: surrogate의 목적은 **선별(triage/ranking)** 이지 정밀 회귀가 아니다 → **RMSE는 2차 지표**.
"최적화에 좋은 모델"은 **순위·선별·임계분류**를 잘하는 모델. 다음을 본다:
- **순위 일치도(point-error 아님)**: 예측 vs 실측의 **Spearman / Kendall τ**, 특히 **상위 구간에서**.
- **선별 품질**: **precision@k**(surrogate 상위 k 중 실제 우수 비율), **NDCG**, **top-k regret / simple regret**(상위 k의 실측 best가 진짜 best와 얼마나 가까운가).
- **임계(성공기준) 지향**: 성공이 Y_target 통과이므로 surrogate를 **pass/fail 분류기**로 평가 — **PR-AUC·recall@threshold·precision@k(threshold-hit)**. "surrogate 상위 선택이 랜덤/DoE보다 confirmed-hit를 더 많이 내는가".
- **운영(online) 검증**: 매일 320을 실측하므로 **선택된 배치의 실측 Y로 그날그날 순위상관·precision@k를 추적** → 시간에 따른 추이.
- ⚠️ **선택 편향 주의**: 측정은 surrogate가 고른 것만 → **recall(놓친 좋은 후보)** 은 직접 못 봄. **배치의 ε-비율을 무작위/탐색에 배정**(D의 ε-슬롯)해 **편향 없는 ground truth**로 recall·캘리브레이션을 추정.
- 한 줄: **"RMSE 낮음"이 아니라 "상위 순위·선별·임계분류가 좋은가 + 랜덤 대비 selection이 실익을 주는가"** 로 판단. 실제 수치는 데이터 후 측정(보류).

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

### G2. Q5 — "어떤 통계 기준이면 성공으로 판정하나" (구체 결정규칙)
> 원칙: **"관측 평균이 Y_target을 넘었다"가 아니라 "참값이 Y_target을 넘었다고 정해진 신뢰수준으로 말할 수 있다"** 를 성공으로.
후보를 n회 반복측정한 뒤, 아래 중 하나의 **결정규칙**을 사용(모두 같은 목표 — false-pass 위험 통제):

- **① one-sided lower confidence bound(LCB) ≥ Y_target**: 평균의 **한쪽 하한신뢰한계**가 target 이상이면 성공.
  - LCB = ȳ − z₁₋α · (s/√n). **acceptance ⟺ ȳ ≥ Y_target + z₁₋α·(s/√n)**.
- **② guardband(=①의 동치 표현)**: **ȳ ≥ Y_target + guardband**, guardband = z₁₋α·SE. **반도체 테스트의 표준 기법** → shmoo 맥락에 자연스러움(거짓합격↓ 대신 거짓불합격↑ 트레이드오프).
- **③ bootstrap(비모수)**: 반복측정치를 부트스트랩 → **P(mean ≥ Y_target) ≥ 1−α** 또는 하위 분위수(예 5%) ≥ target.
- **④ Bayesian/Monte Carlo**: 사후 **P(참 Y ≥ Y_target | 반복) ≥ 1−α**(예 0.95).

추가 필수 고려:
- **반복 횟수 n**: 추정 노이즈 σ와 목표 검정력으로 산정(σ 클수록 n↑). 초기 DoE의 반복측정으로 σ 선추정.
- **다중성(multiplicity)**: 여러 후보를 동시에 confirm → **family-wise/FDR 보정**(Bonferroni/BH)으로 전체 false-pass 통제.
- **권고 baseline**: **① LCB(=② guardband) at 95% one-sided** + n은 σ 기반 산정 + 다중성 보정. (도메인이 이미 guardband를 쓰면 그 관행에 정렬)
- ⚠️ **α·n·guardband 폭의 구체값은 노이즈 σ·허용 위험에 의존 → 데이터/정책 확정 후(보류).**

---

## 종합
- **확정(데이터 불요·방법론 방향)**: 배치 다양성(D)·**mixed-type 거리=Gower+활성페널티(D2)**·SUB 배치추출(E)·GBM 캘리브레이션 보정 도구(F)·**surrogate 평가=순위/선별/임계지표(F2)**·검증 프로토콜(C)·invalid 처리 계층(research/04 §4.3)·**confirmatory 결정규칙=LCB/guardband(G·G2)**.
- **보류(데이터 필요)**: 식별 가능 상호작용 차수 실제값(B), linkage 진위 판정(C), 방법 우열 최종판정(A), GBM 캘리브레이션 충분성(F), 거리 가중치·ℓ(D2), confirmation의 α·n·guardband(G2).
- Phase 2의 산출은 **"판정 도구(벤치+ablation+검증 프로토콜+지표 정의)"** 이며, **우열·수치 결론은 초기 데이터 확보 후**.

## Sources
- 자유도/상호작용: [DOE 스크리닝 원리](https://online.stat.psu.edu/stat503/lesson/8/8.4) (effect sparsity/heredity)
- mixed-type 거리: [Gower's distance (Wikipedia)](https://en.wikipedia.org/wiki/Gower%27s_distance) · [Distances with Mixed-Type Variables (Modified Gower)](https://arxiv.org/pdf/2101.02481) · [Unbiased mixed variables distance](https://arxiv.org/pdf/2411.00429)
- surrogate 순위/선별 평가: [Impact of Surrogate Model Accuracy in SAEAs (Kendall/Spearman vs RMSE)](https://arxiv.org/pdf/2503.00844) · [Surrogate Functions for Maximizing Precision at the Top (prec@k)](http://proceedings.mlr.press/v37/kar15.pdf)
- confirmation 결정규칙(LCB/guardband): [Evaluation of Guardbanding (Sandia)](https://www.osti.gov/servlets/purl/1855029) · [Risk-Calibrated Process Capability Approval (LCB=Cpk−z·SE)](https://arxiv.org/pdf/2603.14479) · [Guard Banding in Calibration (Tektronix)](https://www.tek.com/en/blog/understanding-guard-banding-in-calibration-and-why-it-matters)
- winner's/optimizer's curse·confirmation: [The Optimizer's Curse (Smith & Winkler)](https://www.researchgate.net/publication/220534939_The_Optimizer's_Curse_Skepticism_and_Postdecision_Surprise_in_Decision_Analysis) · [Statistical correction of the Winner's Curse (PLOS Genetics)](https://journals.plos.org/plosgenetics/article?id=10.1371%2Fjournal.pgen.1006916) · [Winner's Curse in A/B Testing](https://atticusli.com/replication-crisis/ab-testing-winners-curse/) · [Beating the Winner's Curse via Inference-Aware Policy Optimization](https://www.arxiv.org/pdf/2510.18161)
- 배치 다양성(DPP): [Diversified Sampling for Batched BO with DPPs](https://arxiv.org/abs/2110.11665) · [Batched GP Bandit via DPP](https://arxiv.org/pdf/1611.04088) · [Enhancing Batch Diversity in Surrogate Optimization (DPP)](https://dl.acm.org/doi/10.1145/3721296)
- GBM/RF 불확실성: [Instance-Based Uncertainty for Gradient-Boosted Trees](https://arxiv.org/pdf/2205.11412) · [Conformalized Quantile Regression](https://valeman.medium.com/conformalized-quantile-regression-smarter-uncertainty-prediction-for-data-scientists-6389bea7a7c4) · [sklearn GBM quantile intervals](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
- 배치 BO 추출/연관학습: research/05, research/02, research/07 및 그 출처(Batch Thompson, DSMGA-II 등)
