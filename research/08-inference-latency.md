# 08 — 추론 지연 제약: 학습/추론 분리 설계

> 제약: **다음 iteration(60-bit on/off 신호 1건) 생성 = 추론**은 **≤ 3초, 목표 ≤ 500ms**.
> **학습(training)은 시간 무제한.** → "무거운 탐색은 학습/오프라인, 후보 방출은 즉시"가 설계 원칙.

## 1. 왜 이 분리가 가능한가 (개념: amortized optimization)
- **Amortized 최적화/추론**: 초기 계산예산을 **오프라인 학습**에 크게 투입해 전역 모델/정책을 만들고,
  이후 개별 추론은 **단일 forward pass**로 처리 → MCMC·수치최적화 같은 반복 절차를 신경망 추론으로 대체.
- 보고 사례: 진화탐색으로 **며칠** 걸리던 설계 스윕을 학습된 정책의 **밀리초** 추론으로 대체;
  amortized neural optimizer가 **CPU 1ms / GPU 4ms**로 near-optimal 파라미터 예측.
- 즉 본 제약은 알고리즘을 배제하는 게 아니라 **계산을 어느 단계에 둘지**를 규정.

## 2. 세 가지 구현 패턴

### 패턴 A — 후보 버퍼/큐 (1순위·방법 무관)
- 오프라인에서 옵티마이저(EA/BO/SAEA)를 **충분히** 돌려 다음 평가 후보 다수를 생성·정렬 → **큐 적재**.
- 서빙은 **dequeue만** → 사실상 0ms(메모리/DB 조회). 3s·500ms 한도 **자동 충족**.
- MAIN(하루 320 배치)과 천연 궁합: 하루치 배치를 미리 만들어 두고 하나씩 방출.
- 데이터 누적 시 백그라운드에서 큐 **재생성**(stale 방지). 큐 소진율 < 생성율이면 안전.
- 장점: 구현 단순, 어떤 옵티마이저와도 호환, 추론 한도와 옵티마이저 무게를 **완전 분리**.
- 단점: 큐 생성 시점의 모델 기준 → 아주 최신 1~2건 결과를 즉시 반영하진 못함(배치 운영엔 무방).

### 패턴 B — amortized 추론 모델 (상태반영 즉시생성이 필요할 때)
- **정책/대리모델**(예: transformer/GNN/MLP)을 오프라인 학습 → 현재 history를 입력받아
  **단일 forward pass로 다음 후보** 생성. 추론 1~4ms 보고.
- "다음에 무엇을 측정할지"를 학습하는 amortized active learning/experimental design 계열과 동형.
- 장점: 최신 상태를 반영하면서도 ms 추론. 단점: 학습 파이프라인·데이터량 요구, 구현 복잡.

### 패턴 C — anytime / 예산제한 지역탐색
- GBM/RF **대리모델 추론은 ms 단위** → 500ms 예산 안에서 **비트플립/타부서치/소규모 EA**를 돌려
  시간초과 시 **best-so-far** 반환(anytime). 예산을 늘리면 품질↑(3s까지 허용).
- 장점: 대리모델만 있으면 즉시 적용, 최신 모델 반영. 단점: 500ms 내 탐색량 한계.

## 3. 방법별 영향 (이 제약 관점)
| 방법 | 요청시점 직접 추론 | 권장 운영 |
|------|---------------------|-----------|
| 연관학습 EA(DSMGA-II 등) | 세대 진행은 무거울 수 있음 | **패턴 A**(오프라인 진화 → 큐) |
| BO(BOCS/SMAC/TPE) | acquisition 최적화가 느릴 수 있음 | **패턴 A**(배치 미리 생성) |
| GBM/RF 대리모델 | 점수 추론은 ms | **패턴 C**(예산제한 탐색) 또는 A |
| amortized 정책망 | **단일 forward pass(ms)** | **패턴 B** |

→ 결론: **공통으로 패턴 A(후보 큐)** 를 기본 채택하면 어떤 주력 옵티마이저든 추론 한도를 보장.
상태반영 즉시생성이 필요해지면 B/C를 추가.

## 4. 검증 체크리스트
- [ ] 서빙 경로에서 "다음 60-bit 생성" p99 지연 측정 → **≤3s(목표≤500ms)** 확인.
- [ ] 큐 생성율 ≥ 소비율(하루 320) 모니터링, 소진 시 fallback(랜덤/패턴 C) 준비.
- [ ] 학습(무거운 탐색)과 추론(방출) 코드 경로를 **명확히 분리**(서로 블로킹하지 않게).

## Sources
- [Tutorial on Amortized Optimization](https://www.researchgate.net/publication/372065237_Tutorial_on_Amortized_Optimization)
- [Forward-Pass Amortization in Neural Models](https://www.emergentmind.com/topics/forward-pass-amortization)
- [Amortized Neural Optimization … Differentiable Surrogates (CPU 1ms/GPU 4ms)](https://arxiv.org/html/2606.07463)
- [ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](https://arxiv.org/html/2506.07259v1)
- [Amortized Safe Active Learning for Real-Time Data Acquisition](https://arxiv.org/html/2501.15458)
- [Amortized Bayesian Experimental Design for Decision-Making (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)
- [Neural Methods for Amortized Inference (Annual Reviews)](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-112723-034123)
