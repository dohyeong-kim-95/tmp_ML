# prob3 — deceptive TRAP 벤치마크 (난이도 최대)

prob2(블록 강결합)에서 한 단계 더 — **deceptive trap**으로 난이도의 두 축을 모두 올림:
① local maxima 다수 + ② **local↔global 거리 최대**(기만성). prob2와 동일하게
model-free 7종 / 70열 / 6 Y / family front·back / 노이즈=주효과 4% / score=NMSE / TIME 예산.

## trap 구조
- 70열, 10열 7블록. 각 블록의 binary **k=5비트에 unitation trap** 적용:
  - `trap(u) = k (u=k, all-ones=전역최적 스파이크)  else (k-1-u)` (u=0=all-zeros가 가짜최적)
  - 한 비트씩 greedy → all-zeros(가짜최적)로 수렴, 진짜 정답은 정반대 all-ones(해밍거리 k)
- 선형 main effect 는 ordinal + (trap 미사용) binary 에만 → **trap 비트는 순수 기만 전용**
- 블록 간 약한 상호작용(trap 비트끼리) → 분해 난이도↑

## 난이도 확인 (단일출발 좌표상승)
| | prob2(블록 강결합) | prob3(trap) |
|---|---|---|
| 전역최적 도달률 | 10% | **0%** |
| 평균 gap | 1.1% | **12.3%** ← local이 global과 멂(기만) |

→ greedy/단일이동 계열은 가짜최적에 갇힘. 블록을 통째로 뒤집는 능력(linkage 인식
재조합, 큰 변이, 충분한 탐색)이 있어야 탈출. 1초 예산에선 전 알고리즘이 고전(NMSE 2+).

## 전역최적(J*)
trap 때문에 coordinate ascent로는 못 찾으므로 **해석적으로 계산**:
trap 비트=1(all-ones), 선형변수=계수부호 best, categorical=best level → 약한 상호작용 폴리시.
(`problem.global_reference`)

## 실행
```bash
python3 prob3/generate.py
PYTHONPATH=prob3 python3 prob3/run.py   # TIME sweep, NMSE (단독 실행 권장)
```
