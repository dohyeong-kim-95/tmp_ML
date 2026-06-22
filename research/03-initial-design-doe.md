# 03 — 초기 설계 (cold start): screening DoE

## 왜 첫 배치는 "무작위"가 아니라 "설계"인가
- 데이터 0건 + 점당 비용 큼 → **첫 40~80점을 어디에 쓰느냐**가 이후 대리모델 품질을 좌우.
- 목표: 적은 실험으로 **주효과(main effect)** 와 가능하면 **저차(2-factor) 상호작용** 신호를 최대한 확보.
- 전제 원리: **effect sparsity(소수 변수만 크게 작용)** 와 **effect heredity(상호작용은 주효과가 있는 변수들 사이에서 주로 발생)**.

## 후보 설계

### Plackett–Burman (PB)
- **N개 실험으로 최대 N−1개 요인** 스크리닝 (N은 4의 배수: 4,8,12,16,20,24,…).
- Resolution III: **주효과끼리는 분리**되나 **2-요인 상호작용과 교락(confounding)**.
- 용도: "어떤 변수가 중요한가"를 **최소 실험**으로. 상호작용이 작다고 가정될 때.
- 본 문제: N>60이면 **64런 PB**로 ~63요인 주효과 스크리닝 가능 → 첫 1.5~2일치 배치와 부합.

### Fractional Factorial (2^(k−p))
- 런 수가 **2의 거듭제곱**(4,8,16,32,64,…). Resolution(III/IV/V)로 교락 구조 제어.
  - Res IV: 주효과 ⟂ 2-요인 상호작용(상호작용끼리는 교락).
  - Res V: 2-요인 상호작용까지 분리(런 수↑).
- 상호작용을 일부라도 보고 싶으면 PB보다 유리하나 런 수 증가.

### D-optimal (컴퓨터 생성)
- 원하는 모델(주효과+선택된 2차항)과 **제약**, 임의 런 수에 맞춰 정보행렬 최적화.
- **배타/그룹 제약(불가능 조합 제외)** 을 직접 반영 가능 → 04 문서의 제약과 결합하기 좋음.
- 본 문제처럼 **제약 + 특정 상호작용 관심 + 런 수 고정(40의 배수)** 이면 가장 유연.

## 본 문제 권고
1. **첫 배치(=하루 40, 또는 64런)**: 제약을 반영한 **D-optimal**(주효과 + 의심되는 소수 2차항) 우선.
   - 제약 표현이 아직 없으면 **PB/Res III**로 시작 → 제약 확보 후 D-optimal로 전환.
2. 첫 배치 분석으로 **중요 변수 솎기** → 이후 BO 루프는 축소된/가중된 공간에서 진행(표본효율↑, 반복시간↓).
3. 구현: `pyDOE2`(factorial/PB), D-optimal은 `dexpy`/직접 구현/`scikit-optimize`/상용(JMP) 참고.

## 주의
- PB의 주효과–상호작용 교락 때문에, **상호작용이 큰 변수**가 의심되면 PB 단독은 위험 → Res IV/V 또는 D-optimal로 보강.
- screening은 BO를 대체하지 않고 **BO의 사전 데이터(prior batch)** 를 만드는 단계.

## Sources
- [Plackett–Burman Designs (JMP)](https://www.jmp.com/en/statistics-knowledge-portal/design-of-experiments/screening-designs/plackett-burman-designs)
- [Plackett–Burman (PSU STAT 503)](https://online.stat.psu.edu/stat503/lesson/8/8.4)
- [When and How to Use Plackett–Burman (iSixSigma)](https://www.isixsigma.com/design-of-experiments-doe/when-and-how-to-use-plackett-burman-experimental-design/)
- [2^k-p Fractional Factorial vs Plackett–Burman (JMP Community)](https://community.jmp.com/t5/Discussions/2k-p-fractional-factorial-designs-vs-Plackett-Burman-designs/td-p/238587)
- [MOODE: Multi-Objective Optimal Design of Experiments (R)](https://arxiv.org/pdf/2412.17158)
