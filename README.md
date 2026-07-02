# tmp_ML — 혼합변수 이산공간 최적화 벤치마크 & 알고리즘 포트폴리오

30개 컬럼(categorical 10 + ordinal 20, 조합 수 ≈ 10¹⁵)의 혼합 이산 변수공간에서,
평가 1회가 비싼 black-box(`X → calculator → 6개 Y`)를 **180 / 780 / 2400
iteration의 작은 예산**으로 최적화하는 알고리즘을 리서치하는 프로젝트다.

실제 calculator는 평가가 비싸 알고리즘 튜닝에 쓸 수 없으므로, 실제 문제의
구조(블록 분할, 교호작용, 노이즈, trade-off)를 본뜨되 즉시·노이즈 포함으로
평가되고 참조 최적을 알 수 있는 **합성 black-box 벤치마크(BM1<BM2<BM3<BM4)**
를 만들고, 그 위에서 여러 옵티마이저를 공정한 점수 체계로 겨루게 한다.

더 자세한 배경·문제 정의·설계 결정은 다음 문서를 참고할 것:

- [`AGENTS.md`](AGENTS.md) — 세션 인수인계용 프로젝트 맥락(구조/개념/지금까지의 결론)
- [`00_Plan.md`](00_Plan.md) — 원 계획 문서(문제 정의/설계 결정/로드맵)
- [`Fable_feedback.md`](Fable_feedback.md) — 코드 리뷰 피드백(방법론/버그/인프라/품질)
- [`optim/FINDINGS.md`](optim/FINDINGS.md), [`optim/RESULTS.md`](optim/RESULTS.md) — 실험 결과

## 레포 구조

```
benchmark/        합성 black-box (목적함수/노이즈/scoring — 변경 금지, 아래 참고)
  generator.py     BlackBoxBenchmark: X→6Y (functional-ANOVA: 주효과+희소교호)
  configs.py       BM1<BM2<BM3<BM4 난이도 ladder
  scoring.py       MinMaxNormalizer + 3종 점수(sum/chebyshev/owa)
  build.py         BM 인스턴스 + 참조최적/난이도 산출 → artifacts/<BM>.json
optim/             알고리즘 벤치마킹 하니스
  problem.py       Problem: 노이즈 관측점수 최대화 + 참(true)점수 anytime 곡선
  design.py        marginal_balanced_design(=mlhs) 초기설계 + n_init 공식
  algos.py         REGISTRY: random/sobol/mlhs/block_coord_local/sa/ga/tpe/
                   smac/botorch/pso/pso_mixed/aco (+ 각 base의 *_blk 블록판)
  blockwrap.py     make_block_decomp: 임의 base를 block-coordinate로 감싸는 래퍼
  run.py           실험 실행기 → results*.json (--append/--merge-extend 지원)
  summarize.py     결과 병합 + 비교표 RESULTS*.md
  visualize.py     그림: closure_*, by_kind_*, pool_*, block_lift_*, vs_global_max
  bm3_visualize.py BM3 구조 설명 그림(다봉성/교호작용/trade-off/노이즈)
tests/             pytest 회귀 테스트(scoring/benchmark snapshot/design/problem)
```

## 실행법

```bash
pip install -r requirements.txt

# 1) 벤치마크 인스턴스 + 참조최적/난이도 artifacts 생성
python -m benchmark.build

# 2) 알고리즘 포트폴리오 실행 (BM1~BM3, 180/780 예산, seed 3개)
python -m optim.run --algos random,sobol,mlhs,block_coord_local,sa,tpe \
    --max-budget 780 --budgets 180,780 --seeds 3

# 3) 결과 병합 → 비교표 / 그림
python -m optim.summarize --md optim/RESULTS.md
python -m optim.visualize

# BM3 구조 설명 그림
python -m optim.bm3_visualize

# 회귀 테스트
pytest
```

무거운 옵티마이저(smac/botorch)는 별도로, 작은 seed로 돌리는 걸 권장한다.
`--append`는 새 알고리즘/BM 셀을 추가하고, `--merge-extend`는 기존 셀의 seed
결과를 보존한 채 새 seed를 이어붙인다(자세한 의미는 `optim/run.py` docstring).

## 제약

- **`benchmark/`의 목적함수·노이즈·scoring 정의는 변경 금지.** 알고리즘/인프라
  코드는 자유롭게 고치되, 벤치마크가 조용히 바뀌지 않았는지는 `pytest`
  (특히 `tests/test_benchmark_snapshot.py`)로 확인한다.
- 개발 관례(브랜치/커밋 방식)는 `AGENTS.md`를 따른다.
