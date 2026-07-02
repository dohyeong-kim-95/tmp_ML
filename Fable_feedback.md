# 리포지토리 문제점 정리 (Fable feedback)

> 검토 범위: `benchmark/`(generator/configs/scoring/build), `optim/`(problem/algos/blockwrap/design/run/summarize/visualize), 루트 스크립트·문서.
> 심각도 순서: **A. 방법론(결론 신뢰성)** → **B. 버그/동작 결함** → **C. 재현성·인프라** → **D. 코드 품질(경미)**.

---

## A. 방법론 — 결론의 신뢰성에 영향을 주는 문제

### A1. 참조최적(ref_opt)의 순환성 — "block_coord_local 1위" 결론과 측정 잣대가 같은 계열
- `generator.py:291-308` `reference_optimum`은 **다중시작 좌표상승 + block-coordinate@20k**의 max로 정의됨. 즉 closure의 천장(분자·분모의 기준)을 **챔피언 알고리즘과 같은 inductive bias(좌표법)** 를 가진 탐색기가 정한다.
- 비분리 BM(BM3/BM4)에서 좌표법이 못 찾는 더 좋은 영역이 있으면 ref_opt가 과소평가되고, closure는 좌표법에 유리하게 부풀려진다. RESULTS_pool.md 스스로 "BM4에서 closure>1 가능"을 인정하는데, 이는 곧 천장이 새고 있다는 뜻.
- **00_Plan.md와의 괴리**: Plan(83-84행)은 "참 최적을 **설계상 심는다(plant the optimum)** + 탐색으로 교차검증"이라 했지만, 실제 generator는 심지 않고 **탐색 추정만** 한다. 계획 문서와 구현이 다름.
- 제안: (a) 최적을 실제로 plant하도록 generator 수정(주효과·교호 부호를 특정 x*에 정렬), 또는 (b) 서로 다른 계열(GA 대예산, SA, TPE 대예산 등) 앙상블의 max로 천장을 정의하고 천장의 불확실성을 명시.

### A2. 노이즈 정의가 문서와 구현이 다름
- 문서/설정(`generator.py:63`, 00_Plan, AGENTS): "노이즈 = **주효과 스프레드**의 5%".
- 구현(`generator.py:223-224`): `noise_scale = noise_frac * ys.std(axis=0)` — 랜덤 표본의 **전체 raw Y(주효과+교호+3차 포함) 표준편차** 기준.
- 교호가 강한 BM3/BM4일수록 실제 노이즈가 문서상 정의보다 커진다 → BM 간 노이즈 난이도가 의도(설정값 3%/5%/8%)와 다르게 스케일됨. 정의를 하나로 통일하고 문서를 맞출 것.

### A3. 정규화 clip이 천장 근방 정보를 죽일 수 있음
- `scoring.py:44` `np.clip(z, 0, 1)`: 캘리브레이션 범위(`_calibrate`: 랜덤 2만 + 목적별 좌표상승)를 벗어나는 더 좋은 y는 z=1로 **포화**된다. A1과 결합하면, 진짜 최적이 y_hi 바깥일 때 점수 함수 자체가 그 이득을 구분 못 함(sum에서 특히). 최소한 clip 여부를 knob로 두고 포화 빈도를 로깅할 것.

### A4. artifacts(ref_opt)와 런타임 BM의 정합성 검증 없음
- `run.py:42-44` `load_ref`는 `artifacts/<BM>.json`의 숫자만 읽고, **현재 `configs.py`/generator 코드가 artifact 생성 시점과 동일한지 확인하지 않는다**. configs를 고치고 build를 안 돌리면 stale 천장으로 조용히 closure가 계산됨.
- 또한 BM 함수 자체(주효과 테이블 등)는 저장되지 않고 seed에서 매번 재생성되는데, NumPy `Generator`의 비트스트림/분포 메서드는 **NumPy 버전 간 호환이 보장되지 않는다** → 버전이 바뀌면 artifact와 런타임 BM이 다른 함수가 될 수 있다. artifact에 config 해시 + numpy 버전 + 함수 fingerprint(예: 고정 X 몇 개의 raw Y)를 저장하고 run.py에서 검증할 것.

---

## B. 버그 / 동작 결함

### B1. `pso_mixed`가 그림에서 조용히 누락됨
- `summarize.py:20-23` `ALGO_ORDER`에 `pso_mixed`, `pso_mixed_blk`가 없다. summarize 표는 "ALGO_ORDER 외 알고리즘을 뒤에 붙이는" 로직(96-97행)으로 살아남지만, `visualize.py:34,77`의 `plot_budget`/`plot_by_kind`는 `[a for a in ALGO_ORDER if a in res["runs"]]`만 그려서 **pso_mixed 계열이 그림에서 제외**된다. `plot_block_lift`(126행)의 bases 목록에도 없음. 표와 그림이 서로 다른 알고리즘 집합을 보여주는 상태.

### B2. `--merge-extend`의 데이터 유실/중복/크래시
- `run.py:98-100`:
  - prev 셀에 **현재 checkpoints에 없는 budget**(예: 이전 2400 결과)이 있으면 병합 시 **조용히 소실**된다(새 cells dict에 그 키가 없음).
  - 반대로 prev에 없는 budget을 현재 checkpoints가 요구하면 `prev["best_true"][str(b)]`에서 **KeyError로 크래시**.
  - 같은 seed를 두 번 돌려도 방지/경고 없이 **중복 append** → seed 평균이 조용히 편향됨.

### B3. `meta` 덮어쓰기로 이전 budgets 정보 소실
- `run.py:75-76`: append/merge 시에도 `res["meta"]`를 현재 인자로 통째로 덮어쓴다. 2400 포함 실행 후 180/780짜리 append를 하면 meta.budgets가 [180,780]이 되어 `summarize`/`visualize`(meta.budgets 기본값 사용, summarize.py:287, visualize.py:319)가 **저장돼 있는 2400 데이터를 무시**한다.

### B4. `Problem`이 예산을 강제하지 않음 — 초과/미달이 조용히 통과
- `problem.py`에는 budget cap이 없고 어댑터의 선의에 의존한다. pymoo GA는 세대 단위 종료라 최대 pop_size−1회 초과 평가가 curve에 기록된다(체크포인트 추출로 공정성은 유지되지만 무결성 검증이 없음).
- 반대로 어댑터가 조기 종료·예외로 **b회 미만**만 평가해도 `checkpoints`(problem.py:47-53)가 `min(b, len(curve))-1`로 **마지막 값을 그 예산의 성적처럼** 반환 — 경고 없음. `len(curve) < b`이면 최소한 경고/실패 표시가 필요.

### B5. blockwrap 하의 SA warm-up이 서브예산을 초과
- `algos.py:149` `n_warm = min(20, max(2, budget//20))` — `sa_blk`에서 sub_budget이 1~2인 후반 블록에 들어가면 warm-up만으로 sub_budget을 초과한다. `SubProblem`은 전역 캡만 보장(blockwrap.py:44)하므로 **블록 간 예산 배분이 설계와 달라짐**(뒤 블록 예산 잠식).

### B6. 예산 소진 후 SubProblem이 가짜 상수를 반환
- `blockwrap.py:44-45`: 전역 예산 소진 시 `best_obs`를 평가값인 척 반환한다. base 옵티마이저(TPE/GA 등)는 남은 반복을 **모든 점이 동점인 가짜 피드백**으로 돌며 내부 상태를 오염시키고 wall-time을 낭비한다(무한루프는 아니지만 낭비 + 로그 왜곡). 예외를 던져 base 루프를 끊는 쪽이 깨끗함.

### B7. 그림 y축 상한이 closure>1 케이스를 자름
- `visualize.py:63` `set_ylim(0,1.15)`, pool 그림들은 1.2 — 스스로 문서화한 "BM4에서 closure>1 가능"(RESULTS_pool.md 헤더) 상황에서 **막대가 잘려** 역전의 크기가 안 보인다. 데이터 max 기준 동적 ylim 필요.

### B8. `bm3_visualize.py`의 환경 의존성
- `matplotlib.use("Agg")` 없이 `plt.show()`(220행) 호출 → headless에서 경고/블로킹. `tick_labels`(206행)는 matplotlib≥3.9 전용인데 의존성 명시가 없음. 실행마다 `reference_optimum`(좌표상승 40-restart + block_coord 20k evals)을 재계산해 불필요하게 느림 — artifacts JSON의 x를 읽으면 됨.

### B9. 소소한 결함
- `run.py --append`: 같은 algo/BM/kind를 다시 돌리면 기존 seed 결과를 **경고 없이 통째로 교체**(merge-extend와 의미가 헷갈리기 쉬움).
- `algos.py:34-37` Sobol: budget이 2^k가 아니면 scipy UserWarning + balance 저하(180/780 모두 해당). 최소한 경고 억제 근거나 next-power-of-2 후 절단을 고려.
- `generator.py:191` `evaluate`의 기본 rng가 비시드(`default_rng()`) → 기본 경로 재현 불가. (하니스는 `Problem`이 자체 노이즈를 쓰므로 실험엔 영향 없으나 API로서 위험.)
- `summarize.py:41-54` `merge`: 여러 results 파일에 같은 셀이 있으면 **정렬상 나중 파일이 조용히 승리** — 충돌 감지/경고 없음.

---

## C. 재현성 · 인프라

### C1. 의존성 명세 파일이 없음
- `requirements.txt`/`pyproject.toml`이 없고 의존성이 AGENTS.md 산문에만 있다(numpy, scipy, optuna, pymoo, smac, botorch/torch, matplotlib). 특히 **"smac은 scikit-learn==1.6.1 핀 필요"** 같은 핵심 제약이 코드화되어 있지 않아 새 환경 재현이 운에 맡겨짐. matplotlib≥3.9(B8)도 마찬가지.

### C2. 테스트가 0개
- 리포 전체에 테스트가 하나도 없다. 특히 AGENTS.md의 **"benchmark 목적함수·노이즈·scoring 변경 금지"** 제약을 지켜줄 회귀 테스트(예: scoring 3종 단위테스트, 고정 X에 대한 BM raw Y 스냅샷, artifact 값 일치 검증)가 없어서, 리팩토링/버전업 시 벤치마크가 조용히 변해도 알 수 없다(A4와 직결). CI도 없음.

### C3. 결과 원데이터 보존 정책
- `results*.json`은 gitignore(`.gitignore:3`)인데 수치 결론은 RESULTS*.md/FINDINGS.md 표에만 남는다. 컨테이너/로컬이 사라지면 **10-seed 원데이터를 재실행 없이는 복구 불가**(2400 예산 셀은 수 시간). 최종 라운드 results JSON은 커밋하거나 별도 보관 경로를 정할 것.

### C4. 문서/구조 정리
- **루트 README 없음** — 진입 문서가 AGENTS.md(에이전트용)와 00_Plan.md(계획)로 흩어져 있다.
- 루트에 `bm3_visualize.py`, `bm3_explain.png`가 놓여 있어 컨벤션(`optim/figs/`에 그림, 모듈은 패키지 안)과 어긋남.
- `.gitignore`가 최소한(venv, .DS_Store, *.egg-info 등 부재).
- AGENTS.md는 "main에 직접 커밋"이라 하지만 현재 작업은 feature 브랜치에서 진행 중 — 관례 문서가 현재 워크플로와 불일치.

---

## D. 코드 품질 (경미)

- **상수 중복 정의**: `ALGO_ORDER`/`POOL`/`BMS`가 summarize에 있고 visualize가 import하지만, `plot_block_lift`의 bases(visualize.py:126), `plot_pool_budgets`의 색상 budget 키(273행, "180"/"780"/"2400" 하드코딩), `plot_vs_global`의 "180"/"780" 키(187-188행) 등 **예산·알고리즘 목록이 여러 곳에 하드코딩** — 새 알고리즘/예산 추가 시 누락되기 쉬움(B1이 실제 사례).
- **n_init 공식 중복**: `algos.py:95`와 `blockwrap.py:64`에 동일한 `max(dim, min(2*dim, budget//5))` 공식이 복붙되어 있음 — 한쪽만 바꾸면 초기설계 이점의 공정성이 깨짐.
- `build.py:81-82`가 `bm._y_lo/_y_hi` 프라이빗 속성에 직접 접근 — 공개 프로퍼티로 노출하는 게 맞음.
- `summarize.py:226` pool 태그 판정 `win == lbl.lower()`가 라벨-키 네이밍 규칙에 암묵 의존(현재는 우연히 동작).
- `blockwrap.make_block_decomp`의 잔여 예산 루프(91-101행)는 **common만** 재최적화 — set1/set2는 rounds=3 이후 다시 못 본다. 의도라면 docstring에 명시 필요.
- `Problem.evaluate`가 매 호출 `bm.scorer.score`를 2회(관측/참) 호출 — raw 1회 계산 최적화는 했지만 정규화·정렬(owa의 sort)은 여전히 2회. 성능상 사소하나 `all_scores`류로 묶을 수 있음.

---

## 요약 — 우선순위 제안

| 순위 | 항목 | 이유 |
|---|---|---|
| 1 | A1+A3 (ref_opt 순환성 / clip 포화) | 핵심 결론("block_coord_local 1위")의 잣대 자체가 챔피언과 같은 편향을 가짐. plant-the-optimum(원래 계획)으로 복귀 검토 |
| 2 | A4+C2 (artifact 정합성 검증 + 회귀 테스트) | "벤치마크 변경 금지" 제약을 지킬 수단이 현재 없음 |
| 3 | B2+B3 (merge-extend 유실/중복, meta 덮어쓰기) | 수 시간짜리 실험 데이터가 조용히 사라지거나 편향되는 경로 |
| 4 | A2 (노이즈 정의 통일) | BM 난이도 ladder의 노이즈 축이 문서와 다르게 동작 |
| 5 | B1+B7 (pso_mixed 그림 누락, ylim 잘림) | 보고 그림이 표와 다른 내용을 보여줌 |
| 6 | C1 (requirements 명세) | 환경 재현의 최소 조건 |
