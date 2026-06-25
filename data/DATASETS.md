# 더미 데이터셋 모음 (난이도 1~5)

동일한 X 구조(binary 32 / ordinal 4 / categorical 4)에, **난이도(복잡도·노이즈)를
단계적으로 높인** 5개 DB. 분석·최적화 방법론의 난이도별 성능 비교용.

| DB | 위치 | 비고 |
|---|---|---|
| **DB1** | `data/dummy_data.csv`, `data/ground_truth.json` | 기존 데이터 (최저 난이도) |
| DB2 | `data/db2/` | |
| DB3 | `data/db3/` | 3차 교호작용 등장 |
| DB4 | `data/db4/` | outlier(두꺼운 꼬리) 잡음 4% |
| DB5 | `data/db5/` | 최고 난이도, outlier 8% |

## 난이도 스케일 (Y당 평균)

| DB | strong | weak | 2차교호 | quadratic | 3차교호 | noise 배수 | outlier% | 총 항수 | ΣY std |
|---|---|---|---|---|---|---|---|---|---|
| DB1 | 4~6 | 10~14 | 1~4 | 0 | 0 | 1.0 | 0 | ~80 | ~14 |
| DB2 | 5 | 13 | 3 | 1 | 0 | 1.8 | 0 | 88 | 16.3 |
| DB3 | 6 | 18 | 5 | 2 | 1 | 2.6 | 0 | 128 | 19.7 |
| DB4 | 7 | 23 | 7 | 3 | 2 | 3.4 | 4 | 168 | 29.4 |
| DB5 | 8 | 28 | 9 | 4 | 3 | 4.2 | 8 | 208 | 40.8 |

복잡도가 올라가는 축:
- **main effect 수↑** (강·약 인자 모두 증가)
- **고차 결합↑**: 2차 교호작용 → quadratic(자기 비선형) → 3차 교호작용
- **노이즈↑**: 표준편차 배수 1.8→4.2, 그리고 DB4·5는 outlier(잡음×5) 추가

## 스키마
`ground_truth.json` 의 `responses[y]`:
- `strong` / `weak` : `{변수: 계수}`
- `interactions` : `[{vars:[...], coef, kind}]` — kind ∈ {2way, quad, 3way}
- `categorical` : `{col: {level: 효과}}`
- `noise_sd`

DB1만 과거 스키마(`interactions` 에 `a`,`b` 키)를 쓰며, `optimize/problem.py` 는 둘 다 처리한다.

## 재생성
```bash
python3 data/generate_datasets.py   # DB2~DB5
```
