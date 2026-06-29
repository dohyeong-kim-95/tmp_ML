"""BM1 < BM2 < BM3 난이도 ladder 설정.

X 구조/크기(블록 분할, cardinality)는 셋이 동일하게 공유하고,
난이도는 다봉성(n_harmonics)·교호작용(density/strength/3차)·
목적충돌(conflict_rho)·노이즈·유효참여(weak_ratio)로만 조절해
알고리즘 비교의 공정성을 확보한다.

난이도 격차를 또렷이 벌리기 위해 BM2/BM3에서 다봉성과 교호작용을
크게 키웠다(검증: build.py 의 예산제한 local-search gap).
"""
from .generator import BMConfig

BM1 = BMConfig(
    name="BM1",                 # easy: 매끈한 가법 주효과, 교호 없음, 약한 충돌
    seed=1,
    n_harmonics=1,              # 단조/단봉 → local search가 거의 푼다
    interaction_density=0.0,    # 교호작용 없음(완전 분리가능)
    interaction_strength=0.0,
    n_three_way=0,
    conflict_rho=0.15,          # 약한 max/min 상충
    noise_frac=0.03,
    n_strong=5,
    weak_ratio=0.10,            # weak 인자 영향 작음 → 유효차원 낮음
)

BM2 = BMConfig(
    name="BM2",                 # medium: 다봉 + 2차 교호 다수 + 뚜렷한 충돌
    seed=2,
    n_harmonics=4,              # 다봉(여러 국소최적)
    interaction_density=0.22,   # 변수쌍의 22%에 교호 → 부분 비분리
    interaction_strength=0.8,
    n_three_way=0,
    conflict_rho=0.55,
    noise_frac=0.05,
    n_strong=4,
    weak_ratio=0.20,
)

BM3 = BMConfig(
    name="BM3",                 # hard: 강한 다봉 + 고밀도 교호 + 3차항 + 강한 충돌
    seed=3,
    n_harmonics=8,              # 기만적 다봉
    interaction_density=0.45,   # 고밀도 교호 → 좌표법 무력화
    interaction_strength=1.3,
    n_three_way=12,             # 3차(고차) 교호항
    conflict_rho=0.85,          # 강한 max/min 상충
    noise_frac=0.08,
    n_strong=4,
    weak_ratio=0.35,            # weak 인자도 무시 못함 → 유효차원 ↑
)

ALL = {"BM1": BM1, "BM2": BM2, "BM3": BM3}
