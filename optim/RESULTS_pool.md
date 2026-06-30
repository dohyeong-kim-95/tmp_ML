# 축소 풀 랭킹 — SF / block_coord_local / SA / GA / PSO / ACO

메타휴리스틱은 flat/blk 중 **per-cell best**만, SF=random/sobol/mlhs 중 best. 셀=`평균±표준편차·worst [선택변형]`. ★=칼럼 평균 1등, ◆=worst 1등. closure는 affine 변환이라 **칸 안 순위는 reference와 무관하게 유효**; BM4에서 비좌표 방법이 1위면 그게 'coordinate 한계'의 증거(closure>1 가능).

### 축소 풀 종합 랭킹 (전 BM×budget×kind 평균)
| rank | pool | mean | worst | cells |
|---|---|---|---|---|
| 1 | block_coord_local | 82.1% | 69.5% | 9 |
| 2 | ACO | 69.8% | 58.2% | 9 |
| 3 | SA | 68.3% | 55.1% | 9 |
| 4 | PSO | 68.2% | 54.2% | 9 |
| 5 | GA | 67.5% | 56.3% | 9 |
| 6 | SF | 48.3% | 40.6% | 9 |


### 예산별 풀 랭킹 (전 BM×kind 평균; budget별로 분리 → crossover 확인)
- **@180**: block_coord_local(71%) > PSO(54%) > GA(51%) > ACO(50%) > SA(49%) > SF(43%)
- **@780**: block_coord_local(85%) > ACO(74%) > SA(72%) > GA(71%) > PSO(71%) > SF(49%)
- **@2400**: block_coord_local(90%) > ACO(86%) > SA(83%) > GA(80%) > PSO(80%) > SF(53%)


### kind = sum  (mean±std · worst, [선택변형])
| pool | BM3@180 | BM3@780 | BM3@2400 |
|---|---|---|---|
| SF | 38%±6%·28% [random] | 45%±5%·36% [mlhs] | 49%±5%·44% [mlhs] |
| block_coord_local | ★◆ 72%±13%·55% | ★◆ 87%±6%·77% | ★◆ 92%±5%·86% |
| SA | 49%±13%·28% [sa_blk] | 74%±11%·60% [sa_blk] | 86%±7%·76% |
| GA | 46%±6%·35% | 69%±10%·48% [ga_blk] | 79%±5%·68% [ga_blk] |
| PSO | 54%±10%·39% | 69%±13%·49% [pso_blk] | 80%±4%·71% [pso_blk] |
| ACO | 49%±10%·34% [aco_blk] | 76%±9%·64% [aco_blk] | 84%±7%·75% [aco_blk] |

### kind = chebyshev  (mean±std · worst, [선택변형])
| pool | BM3@180 | BM3@780 | BM3@2400 |
|---|---|---|---|
| SF | 47%±9%·39% [mlhs] | 52%±7%·45% [mlhs] | 58%±5%·52% [mlhs] |
| block_coord_local | ★◆ 69%±12%·58% | ★◆ 84%±8%·70% | ★ 88%±8%·72% |
| SA | 48%±7%·34% [sa_blk] | 67%±7%·56% [sa_blk] | 81%±6%·68% |
| GA | 54%±6%·47% | 72%±6%·60% | 81%±7%·72% [ga_blk] |
| PSO | 55%±8%·42% | 72%±10%·52% | 80%±7%·66% [pso_blk] |
| ACO | 51%±9%·33% [aco_blk] | 74%±9%·63% | ◆ 86%±7%·78% [aco_blk] |

### kind = owa  (mean±std · worst, [선택변형])
| pool | BM3@180 | BM3@780 | BM3@2400 |
|---|---|---|---|
| SF | 43%±9%·33% [mlhs] | 49%±7%·41% [mlhs] | 54%±5%·47% [mlhs] |
| block_coord_local | ★◆ 71%±12%·55% | ★◆ 86%±9%·72% | ★◆ 91%±6%·80% |
| SA | 49%±10%·29% [sa_blk] | 76%±7%·66% [sa_blk] | 83%±3%·79% [sa_blk] |
| GA | 55%±4%·49% | 72%±9%·59% [ga_blk] | 81%±8%·66% [ga_blk] |
| PSO | 54%±6%·44% | 72%±8%·54% | 78%±7%·69% [pso_blk] |
| ACO | 50%±8%·36% [aco_blk] | 72%±6%·63% | 86%±7%·77% [aco_blk] |
