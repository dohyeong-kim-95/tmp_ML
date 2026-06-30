# 축소 풀 랭킹 — SF / block_coord_local / SA / GA / PSO / ACO

메타휴리스틱은 flat/blk 중 **per-cell best**만, SF=random/sobol/mlhs 중 best. 셀=`평균±표준편차·worst [선택변형]`. ★=칼럼 평균 1등, ◆=worst 1등. closure는 affine 변환이라 **칸 안 순위는 reference와 무관하게 유효**; BM4에서 비좌표 방법이 1위면 그게 'coordinate 한계'의 증거(closure>1 가능).

### 축소 풀 종합 랭킹 (전 BM×budget×kind 평균)
| rank | pool | mean | worst | cells |
|---|---|---|---|---|
| 1 | block_coord_local | 82.1% | 69.5% | 9 |
| 2 | ACO | 69.8% | 58.2% | 9 |
| 3 | PSO | 69.7% | 56.9% | 9 |
| 4 | TPE | 69.3% | 58.6% | 9 |
| 5 | GA | 67.9% | 58.1% | 9 |
| 6 | SA | 67.3% | 53.9% | 9 |
| 7 | SF | 48.3% | 40.6% | 9 |


### 예산별 풀 랭킹 (전 BM×kind 평균; budget별로 분리 → crossover 확인)
- **@180**: block_coord_local(71%) > PSO(55%) > TPE(54%) > ACO(50%) > GA(48%) > SA(48%) > SF(43%)
- **@780**: block_coord_local(85%) > ACO(74%) > TPE(73%) > PSO(72%) > GA(71%) > SA(70%) > SF(49%)
- **@2400**: block_coord_local(90%) > ACO(86%) > GA(84%) > SA(83%) > PSO(82%) > TPE(81%) > SF(53%)


### kind = sum  (mean±std · worst, [선택변형])
| pool | BM3@180 | BM3@780 | BM3@2400 |
|---|---|---|---|
| SF | 38%±6%·28% [random] | 45%±5%·36% [mlhs] | 49%±5%·44% [mlhs] |
| block_coord_local | ★◆ 72%±13%·55% | ★◆ 87%±6%·77% | ★◆ 92%±5%·86% |
| SA | 46%±10%·29% [sa_blk] | 74%±8%·59% [sa_blk] | 84%±6%·75% [sa_blk] |
| GA | 42%±3%·36% | 70%±7%·54% [ga_blk] | 82%±6%·75% [ga_blk] |
| PSO | 54%±10%·39% | 73%±6%·62% [pso_mixed_blk] | 84%±5%·76% [pso_mixed_blk] |
| ACO | 49%±10%·34% [aco_blk] | 76%±9%·64% [aco_blk] | 84%±7%·75% [aco_blk] |
| TPE | 49%±7%·39% | 70%±10%·53% [tpe_blk] | 80%±4%·76% [tpe_blk] |

### kind = chebyshev  (mean±std · worst, [선택변형])
| pool | BM3@180 | BM3@780 | BM3@2400 |
|---|---|---|---|
| SF | 47%±9%·39% [mlhs] | 52%±7%·45% [mlhs] | 58%±5%·52% [mlhs] |
| block_coord_local | ★◆ 69%±12%·58% | ★◆ 84%±8%·70% | ★ 88%±8%·72% |
| SA | 52%±9%·40% [sa_blk] | 69%±11%·47% [sa_blk] | 83%±7%·73% |
| GA | 53%±4%·47% | 74%±7%·62% | 85%±5%·78% [ga_blk] |
| PSO | 56%±7%·46% [pso_mixed_blk] | 72%±10%·52% | 80%±7%·66% [pso_blk] |
| ACO | 51%±9%·33% [aco_blk] | 74%±9%·63% | ◆ 86%±7%·78% [aco_blk] |
| TPE | 57%±7%·48% | 72%±7%·60% | 81%±9%·70% [tpe_blk] |

### kind = owa  (mean±std · worst, [선택변형])
| pool | BM3@180 | BM3@780 | BM3@2400 |
|---|---|---|---|
| SF | 43%±9%·33% [mlhs] | 49%±7%·41% [mlhs] | 54%±5%·47% [mlhs] |
| block_coord_local | ★◆ 71%±12%·55% | ★◆ 86%±9%·72% | ★◆ 91%±6%·80% |
| SA | 48%±7%·36% [sa_blk] | 68%±8%·55% [sa_blk] | 83%±7%·72% |
| GA | 51%±5%·42% | 70%±8%·57% [ga_blk] | 86%±6%·72% [ga_blk] |
| PSO | 54%±6%·44% | 72%±10%·58% [pso_mixed_blk] | 83%±8%·67% [pso_mixed_blk] |
| ACO | 50%±8%·36% [aco_blk] | 72%±6%·63% | 86%±7%·77% [aco_blk] |
| TPE | 58%±8%·44% | 76%±6%·64% | 81%±5%·72% |
