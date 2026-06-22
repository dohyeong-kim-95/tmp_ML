# 06 — 최종 baseline 권고 + 15주 계획

## 0. 제약 재확인
- N>60 binary, 데이터 0건, 하루 ~40 배치, 15주(누적 O(10³)), 반복시간 제약, 구조 추후 주입, 데이터 누적.

## 1. 권고 아키텍처 — "DoE 시드 → 2-트랙 배치 BO"

```
[Phase 0] 인코딩/제약 슬롯 정의 (범주화·조건부·feasibility 경계 분리)
        │
[Phase 1] Screening DoE 시드 배치 (D-optimal 또는 PB/Res III) ── 1~2 배치
        │   → 중요 변수 솎기, 대리모델 초기 데이터 확보
[Phase 2] 배치 BO 루프 (매일 40점), 데이터 누적하며 갱신
        │   ├─ Track A(강건·빠름): SMAC3(RF) 또는 Optuna(TPE) + batch ask/tell
        │   └─ Track B(표본효율·해석): BOCS형 희소 pairwise 베이지안회귀 + batch Thompson sampling
        │   → 두 트랙을 같은 데이터로 병행, best-so-far 곡선으로 우열 판단
[Phase 3] 도메인지식 주입 (prior/제약/feature/커널) — baseline 졸업
```

### 왜 이 구성인가 (설득 논리)
1. **비싼 평가 → BO 계열**(01): 표본효율이 지배 지표.
2. **N>60 binary → 표준 GP-BO 밖**(02): 트리(SMAC) 또는 조합 선형(BOCS)이 현실적.
3. **반복시간 제약 → 경량 대리모델 선호**: RF/TPE는 데이터에 선형·빠름; BOCS 선형회귀도 가벼움. (SAASBO/GP-HMC·SDP는 wall-clock 부담 → 주력 제외, 관찰만.)
4. **배치 q≈40 → Thompson sampling**(05): q-EI(q>4) 비현실적, TS는 q 커도 매끄럽고 병렬화 용이.
5. **구조 추후 주입 → BOCS형 해석성**(04): 상호작용 계수를 *발견*해 도메인지식화의 다리. 동시에 SMAC/TPE로 **강건한 하한선** 확보.
6. **2-트랙 비교 자체가 baseline의 가치**: "쉽고 강건" vs "효율·해석" 둘을 같은 데이터에서 견줘 의사결정 근거 생성.

## 2. 도구 스택
| 단계 | 1순위(빠른 baseline) | 본격/대안 |
|------|----------------------|-----------|
| DoE 시드 | `pyDOE2`(PB/factorial) | D-optimal(`dexpy`/직접/JMP) |
| Track A | **Optuna(TPE)** ask/tell 배치 | **SMAC3**(RF, 조건부 강건) |
| Track B | BOCS Python(`BOCSpy`) 또는 희소회귀(`scikit-learn`/`numpyro` horseshoe)+SA | BoTorch/Ax(GP, LP/q-batch), COMBO |
| 배치 | Batch Thompson sampling(직접) | Local Penalization |

## 3. 15주 로드맵(잠정 — 가동일수 확정 후 조정)
| 주차 | 활동 | 산출 |
|------|------|------|
| W1 | 인코딩/제약 슬롯 설계, 평가 파이프라인·데이터 스키마 확정, 시뮬레이터(가짜 score)로 코드 검증 | repo 골격, 데이터 스키마 |
| W2 | Screening DoE 시드 배치 실행·분석, 중요 변수 솎기 | 초기 데이터셋, 변수 중요도 |
| W3–W4 | Track A/B 둘 다 가동, batch TS 루프 안정화, best-so-far 모니터링 대시보드 | 작동하는 BO 루프 |
| W5–W11 | 일일 배치 루프 운영, 두 트랙 비교, 주효과/상호작용 리포트 축적 | 수렴 곡선, 상호작용 후보 |
| W12–W13 | **도메인지식 주입**(prior/제약/feature)로 고도화, baseline 대비 개선 측정 | 개선된 모델 |
| W14 | 결과 정리, ablation(랜덤/DoE-only/단일트랙 대비) | 성능 비교표 |
| W15 | 방법론 문서 + HTML 프리젠테이션 마감 | 최종 산출물 |

## 4. 성공 기준(제안)
- **1차(필수)**: 동일 예산에서 **랜덤 서치 대비 best-so-far Y 유의하게 우수**.
- **2차**: best-Y 수렴 곡선 우상향, 안정적 수렴.
- **3차**: 도메인지식 주입 시 추가 개선 확인(확장성 입증).
- **운영**: 한 배치(40점) 제안 wall-clock 목표 **수 분 내**.

## 5. 리스크 & 완화
| 리스크 | 완화 |
|--------|------|
| pairwise 항 폭발(N²) | effect heredity로 후보 제한, horseshoe 희소화, 중요변수 솎기 후 모델링 |
| 데이터 누적 시 GP 반복시간↑ | 주력은 RF/선형(데이터 선형 확장), GP는 보조 |
| 측정 노이즈 | 대리모델에 noise 항, 일부 배치 반복측정으로 추정 |
| 배타·불가능 조합 | 범주형 재파라미터화/feasibility 모델로 원천 차단 |
| screening 교락(PB) | 상호작용 의심 시 Res IV/V·D-optimal로 보강 |

## 6. baseline에서 일부러 **하지 않는** 것
- 구체 도메인 규칙 하드코딩(Phase 3로 이연), GP-HMC/SDP 등 무거운 방법 주력화, 실시간 서빙.

## Sources
- 본 권고는 [01](01-problem-framing.md)~[05](05-batch-acquisition.md)의 출처를 종합. 핵심:
  [BOCS](http://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf),
  [COMBO](https://arxiv.org/pdf/1902.00448),
  [SMAC3](https://www.researchgate.net/publication/354766153_SMAC3_A_Versatile_Bayesian_Optimization_Package_for_Hyperparameter_Optimization),
  [Optuna](https://arxiv.org/pdf/1907.10902),
  [SAASBO](https://arxiv.org/abs/2103.00349),
  [Batch TS](https://arxiv.org/pdf/1706.01825),
  [Plackett–Burman](https://www.jmp.com/en/statistics-knowledge-portal/design-of-experiments/screening-designs/plackett-burman-designs).
