# 발표용 요약 — 저예산 혼합이산 최적화, 알고리즘 리서치 (model-free 포트폴리오, 10 seed)

> 범위: 합성 벤치마크 BM1<BM2<BM3 × 예산 {180, 780} × scalarization {sum, chebyshev, owa},
> **각 셀 10 seed**(smac=3 별도). closure = (best_true − floor)/(ref_opt − floor),
> 0=무작위 평균, 1=참조최적. 성능은 방문점의 **참(노이즈 없는) 점수 누적최댓값**(추천정책 무관).
> 본 표는 **model-free 포트폴리오**(random/sobol/mlhs/block_coord_local/sa/ga + 각 블록주입 `*_blk`).
> BO 계열(tpe/smac)은 시간상 본 라운드에서 보류 — 단, 헤드라인 결론은 model-free만으로 완결.

---

## 한 장 요약 (말할 것)

1. **블록-인지 좌표탐색(block_coord_local)이 저예산에서 압도하며, 그것이 운이 아니다.**
   전 18셀(3 BM×2 예산×3 kind) **평균·worst-case 모두에서 1위(★◆)**. 종합 평균 closure **80.4%**,
   worst-seed 평균 **69.2%** — 2위(sa_blk 64.6%/52.4%)와 **약 16pp** 차. 즉 "잘 나온 seed"가 아니라
   **최악 seed로 평가해도 여전히 1위**. (10 seed라 이 주장이 통계적으로 선다.)

2. **성능의 가장 큰 레버는 알고리즘이 아니라 "블록 구조(도메인 지식)"다.**
   동일 base에 블록 분해만 주입(`*_blk`)했을 때 평균 closure 상승:
   | base | flat | +block | Δ |
   |---|---|---|---|
   | random | 48.3% | 57.9% | **+9.6pp** |
   | sobol | 46.7% | 57.9% | **+11.2pp** |
   | mlhs | 46.4% | 56.9% | **+10.5pp** |
   | sa | 60.3% | 64.6% | +4.3pp |
   | ga | 61.1% | 62.1% | +1.0pp |
   → **무지성 샘플러도 블록만 주면 +10pp 점프.** 우리 문제는 블록을 *안다*(common/set1/set2)이므로
   이 레버를 쓰는 것이 정당하고, 그 위에 좌표탐색을 얹은 block_coord_local이 천장(80.4%).

3. **초기 space-filling 그 자체는 거의 무의미하다 (= '초기설계' 항목의 결론).**
   순수 샘플러 random ≈ sobol ≈ mlhs **(46~48%, 전부 최하위)**. 혼합변수 marginal-balanced 설계(mlhs)도
   **단독으로는 random을 못 이김.** 이득은 초기설계가 아니라 **탐색 연산자 + 블록 구조**에서 나온다.
   → 결론: **화려한 space-filling에 예산 쓰지 말고, 블록-인지 local search에 쓸 것.**

4. **위험 구간 = 고난도 + 저예산.** block_coord_local의 최악 칸은 BM3@180 (73%±13%, worst 55%).
   여기서 seed 편차가 가장 큼 → 실전에선 **random-restart + 끝단 confirmation(top-3 재측정)**으로 다운사이드 방어.

---

## 권장(실문제 적용안)
- **1순위 baseline = block_coord_local** (블록-인지 좌표탐색: common→set2→set1 라운드 반복 + best-improvement 1-hop + random-restart, 평가 캐시).
- 배포 점수는 한 지표 폭락을 막는 **owa 또는 chebyshev** 권장(둘 다에서 block_coord_local이 1위).
- 예산 분할: 탐색 + 끝단 **confirmation**(top-3 각 4~5회 재측정).

## 그림 (optim/figs/)
- `closure_780.png`, `closure_180.png` — algo×BM 그룹막대, **오차막대=seed min~max**(단일 실전 다운사이드).
- `by_kind_{180,780}.png` — algo별 서브플롯(x=kind).
- `block_lift_sum.png`, `block_lift_owa.png` — **공정비교**: flat→+block, block_coord_local 천장선.
- (전체 수치표: `optim/RESULTS_robust.md` — 평균±표준편차·worst·n.)

## 한계 / 다음
- 본 라운드는 **model-free만** + **BO(tpe/smac) 보류**(발표 일정상). BO는 "GP/RF로도 못 이긴다" 보강용 → 다음 라운드.
- BM은 전 알고리즘 3 BM 유지(난이도 ladder). tpe/smac 재개 시엔 BM3 중심으로 축소 실행 예정.
- `vs_global_max.png`는 @20000 global-max 데이터 미생성으로 **이번 갱신 제외(옛 버전)** — 발표 사용 주의.
