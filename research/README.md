# research/ — Shmoo Score 최대화 Baseline 리서치 모음

> 작성일: 2026-06-22 · 작성: 직접 웹 리서치
> 목적: `01_Problem_def.md`에 정의된 문제(N>60 binary → metric 최대화, 데이터 0건,
> 하루 ~40개 비싼 배치 평가, 15주, 반복 시간 제약, 구조 추후 주입)에 대한
> baseline 방법론 근거 수집.

## 문제 한 줄 재정의
**고차원(N>60) binary 조합 + 비싼 black-box + 배치(q≈40) + 저예산(누적 O(10³)) + 반복시간 제약**
하의 전역 최적화 → **순차적 실험설계(DoE) + 대리모델 기반 배치 베이지안 최적화(BO)** 가 정답 계열.

## 문서 구성
| 파일 | 내용 |
|------|------|
| [01-problem-framing.md](01-problem-framing.md) | 왜 BO 계열인가 (vs 그리드/랜덤/GA/순수 지도학습) |
| [02-methods-comparison.md](02-methods-comparison.md) | 후보 방법 비교표 (BOCS / COMBO / Casmopolitan / SMAC3 / SAASBO / TPE) |
| [03-initial-design-doe.md](03-initial-design-doe.md) | cold start 초기 설계 — screening DoE |
| [04-constraints-encoding.md](04-constraints-encoding.md) | 그룹/배타/조건부 제약 인코딩 |
| [05-batch-acquisition.md](05-batch-acquisition.md) | 배치(q≈40) 제안 전략 |
| [06-recommendation.md](06-recommendation.md) | **최종 baseline 권고 + 15주 계획** (저예산 O(10³) 기준) |
| [07-budget-regimes.md](07-budget-regimes.md) | **비용 인하(320/일) 시나리오 → 모델기반 진화알고리즘으로 권고 전환** |
| [08-inference-latency.md](08-inference-latency.md) | **추론 지연 ≤3s(목표≤500ms) 제약 → 학습/추론 분리(후보 큐·amortized·anytime)** |
| [09-phase2-method-selection.md](09-phase2-method-selection.md) | **"EA+GBM이 최선인가" 검증 프레임 + 설계질문(상호작용차수/linkage진위/배치다양성/배치추출/GBM캘리브레이션)** |
| [10-representation-pipeline.md](10-representation-pipeline.md) | **도메인지식 점증 주입용 표현 파이프라인(X0→…→latent→…→metric) 타당성 평가** |

## TL;DR 결론 (자세한 근거는 06 참조)
1. **첫 1~2 배치**는 무작위 대신 **screening DoE**(Plackett–Burman / D-optimal)로 주효과+저차 상호작용 신호 확보.
2. **대리모델 2-트랙 비교**:
   - (강건·빠름) **SMAC3(랜덤포레스트)** 또는 **Optuna(TPE)** — 데이터 증가에 선형, 범주/조건부/제약 native, 반복시간 빠름.
   - (표본효율·해석) **BOCS형 희소 pairwise 베이지안 회귀** — 상호작용을 *발견*해 추후 도메인지식화의 다리.
3. **배치 제안(q≈40)** 은 q-EI(q>4 비현실적) 대신 **배치 Thompson sampling** 또는 **local penalization**.
4. ⚠️ **표준 GP-BO/SAASBO는 N>60·데이터 누적 시 반복시간(HMC, O(n³))이 부담** → baseline 주력으로는 신중.

## 주의
- 모든 출처는 각 문서 하단 "Sources"에 링크.
- arxiv PDF 일부는 직접 접근이 차단되어 초록/2차 출처/공식 구현 README 기반으로 정리한 항목이 있음(해당 문서에 표시).
- 구체 수식·대외비 구조는 포함하지 않음.
