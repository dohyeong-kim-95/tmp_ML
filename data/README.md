# 더미 데이터 (최적화 문제 분석용)

`generate_data.py` 로 생성. 재현 가능(seed=42).

## 파일
| 파일 | 설명 |
|---|---|
| `dummy_data.csv` | 데이터 본체 (2000 행 × 44 열 = X 40 + Y 4) |
| `ground_truth.json` | 정답 구조: 어떤 X가 어떤 Y를 얼마나 움직이는지 |
| `generate_data.py` | 생성 스크립트 |

## X 인자 (40 columns)
| 그룹 | 컬럼 | 형식 | 비고 |
|---|---|---|---|
| binary (32) | `xb01`~`xb32` | {0, 1} | 컬럼별 1의 비율 0.3~0.7 |
| ordinal (4) | `xo1`~`xo4` | 정수 0..(L-1) | L(level)은 컬럼별 무작위: xo1=5, xo2=14, xo3=12, xo4=9 |
| categorical (4) | `xc1`~`xc4` | {A,B,C,D} | 4 level (순서 없음) |

## Y 인자 (4 responses)
`y11, y12, y21, y22` — 연속값.

각 Y의 생성식:
```
y = intercept
  + Σ strong_coef · x      (소수 인자, |계수| 3~6  → 크게 영향)
  + Σ weak_coef   · x      (다수 인자, |계수| 0.1~0.5 → 작게 영향)
  + Σ inter_coef  · xA·xB  (교호작용)
  + categorical level 효과
  + N(0, noise_sd)
```
- ordinal/binary 는 효과 계산 시 `0~1` 로 정규화(ordinal = 값/(L-1)).
- categorical 은 level별 효과(합=0).
- `y11,y12` 는 계열1, `y21,y22` 는 계열2 (일부 강한 인자를 공유하도록 구성).

## 강하게 영향받는 인자 (정답 요약)
| Y | strong 변수 | 교호작용 | noise_sd |
|---|---|---|---|
| y11 | xb23, xo3, xb10, xb18 | xb10·xb23, xb22·xb10 | 2.0 |
| y12 | xb26, xb29, xb04, xo4, xb15 | xb18·xb12, xb12·xb08, xo4·xb04 | 2.5 |
| y21 | xb15, xo1, xb10 | xb17·xb15 | 1.5 |
| y22 | xb16, xb18, xb13, xb22, xb03, xo3 | xb17·xb18, xb17·xo3, xb04·xb17, xb14·xb16 | 3.0 |

> weak 변수와 정확한 계수·intercept·categorical 효과는 `ground_truth.json` 참조.
> 변수선택/회귀 분석 결과를 이 정답과 대조해 방법론을 검증할 수 있습니다.

## 재생성
```bash
python3 data/generate_data.py
```
