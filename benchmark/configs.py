"""BM1 < BM2 < BM3 난이도 ladder 설정.

X 구조/크기(블록 분할, cardinality)는 셋이 동일하게 공유하고,
난이도는 다봉성(n_harmonics)·교호작용(density/strength/3차)·
목적충돌(conflict_rho)·노이즈로만 조절해 알고리즘 비교의 공정성을 확보한다.
"""
from .generator import BMConfig

BM1 = BMConfig(
    name="BM1",                 # easy: 매끈한 가법 주효과, 교호 없음, 약한 충돌
    seed=1,
    n_harmonics=1,              # 단조/단봉
    interaction_density=0.0,    # 교호작용 없음
    interaction_strength=0.0,
    n_three_way=0,
    conflict_rho=0.20,          # 약한 max/min 상충
    noise_frac=0.05,
)

BM2 = BMConfig(
    name="BM2",                 # medium: 2차 교호 + 중간 다봉 + 충돌 존재
    seed=2,
    n_harmonics=3,
    interaction_density=0.12,
    interaction_strength=0.5,
    n_three_way=0,
    conflict_rho=0.55,
    noise_frac=0.05,
)

BM3 = BMConfig(
    name="BM3",                 # hard: 고차 교호 + 기만적 다봉 + 강한 충돌
    seed=3,
    n_harmonics=5,
    interaction_density=0.30,
    interaction_strength=0.9,
    n_three_way=6,
    conflict_rho=0.85,
    noise_frac=0.05,
)

ALL = {"BM1": BM1, "BM2": BM2, "BM3": BM3}
