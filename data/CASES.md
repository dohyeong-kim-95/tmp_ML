# 최적화 Case 모음 (난이도 1~5)

동일한 X 구조(binary 32 / ordinal 4 / categorical 4)에, **난이도(복잡도·노이즈)를
단계적으로 높인** 5개 Case. 최적화 알고리즘의 난이도별 성능 비교용.

> **A 패러다임**: 최적화의 목적함수는 각 Case 의 `ground_truth.json` 비밀식 자체다.
> dataset CSV(`dummy_data.csv`)는 초기 데이터 분석용일 뿐, 최적화엔 불필요하다.

| Case | ground_truth 위치 | 비고 |
|---|---|---|
| **Case1** | `data/ground_truth.json` | 기존 (최저 난이도) |
| Case2 | `data/case2/ground_truth.json` | quadratic(자기 비선형) 등장 |
| Case3 | `data/case3/ground_truth.json` | 3차 교호작용 등장 |
| Case4 | `data/case4/ground_truth.json` | outlier(두꺼운 꼬리) 잡음 4% |
| Case5 | `data/case5/ground_truth.json` | 최고 난이도, outlier 8% |

## 난이도 스케일 (Y당 평균)

| Case | strong | weak | 2차교호 | quadratic | 3차교호 | noise 배수 | outlier% | 총 항수 | ΣY std |
|---|---|---|---|---|---|---|---|---|---|
| Case1 | 4~6 | 10~14 | 1~4 | 0 | 0 | 1.0 | 0 | ~80 | ~14 |
| Case2 | 5 | 13 | 3 | 1 | 0 | 1.8 | 0 | 88 | 16.3 |
| Case3 | 6 | 18 | 5 | 2 | 1 | 2.6 | 0 | 128 | 19.7 |
| Case4 | 7 | 23 | 7 | 3 | 2 | 3.4 | 4 | 168 | 29.4 |
| Case5 | 8 | 28 | 9 | 4 | 3 | 4.2 | 8 | 208 | 40.8 |

복잡도가 올라가는 축:
- **main effect 수↑** (강·약 인자 모두 증가)
- **고차 결합↑**: 2차 교호작용 → quadratic(자기 비선형) → 3차 교호작용
- **노이즈↑**: 표준편차 배수 1.8→4.2, Case4·5는 outlier(잡음×5) 추가

> 노이즈/outlier 는 **dataset CSV 생성**에만 영향(A 패러다임 최적화는 노이즈 없는
> 비밀식을 직접 다루므로 무관). 즉 최적화 난이도는 항 수·고차결합이 좌우한다.

## 스키마
`ground_truth.json` 의 `responses[y]`:
- `strong` / `weak` : `{변수: 계수}`
- `interactions` : `[{vars:[...], coef, kind}]` — kind ∈ {2way, quad, 3way}
- `categorical` : `{col: {level: 효과}}`
- `noise_sd`

Case1만 과거 스키마(`interactions` 에 `a`,`b` 키)를 쓰며, `optimize/problem.py` 는 둘 다 처리.

## 재생성 / 실행
```bash
python3 data/generate_datasets.py     # Case2~Case5 생성
python3 optimize/run_cases.py         # Case1~5 × (SA/PSO/GA/BO) 검증, budget=3000
```
