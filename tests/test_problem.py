"""Problem의 anytime 곡선/checkpoints 테스트(C2, B4).

numpy 만으로 동작(BlackBoxBenchmark 생성은 numpy만 필요, optuna/pymoo/smac/
torch 불필요).
"""
import numpy as np
import pytest

from benchmark import configs
from benchmark.generator import BlackBoxBenchmark
from optim.problem import Problem


@pytest.fixture(scope="module")
def bm():
    return BlackBoxBenchmark(configs.BM1)


def test_curve_is_anytime_running_max(bm):
    prob = Problem(bm, "sum", seed=1)
    for _ in range(15):
        prob.evaluate(prob.random_x())
    assert len(prob.curve) == 15
    # anytime 곡선은 절대 감소하지 않아야(누적 최댓값)
    assert all(b >= a for a, b in zip(prob.curve, prob.curve[1:]))
    assert prob.curve[-1] == prob.best_true
    assert prob.best_true == max(prob.curve)


def test_checkpoints_extract_correct_index(bm):
    prob = Problem(bm, "sum", seed=2)
    for _ in range(10):
        prob.evaluate(prob.random_x())
    cp = prob.checkpoints([3, 7, 10])
    assert cp[3] == prob.curve[2]
    assert cp[7] == prob.curve[6]
    assert cp[10] == prob.curve[9]


def test_checkpoints_incomplete_flag_when_undershoot(bm, capsys):
    prob = Problem(bm, "sum", seed=3, budget=10)
    for _ in range(5):
        prob.evaluate(prob.random_x())
    cp = prob.checkpoints([3, 5, 10])
    assert prob.last_incomplete == {3: False, 5: False, 10: True}
    # 미달 예산의 값은 마지막 curve 값으로 채워지되(하위호환), incomplete 로 표시
    assert cp[10] == prob.curve[-1]
    out = capsys.readouterr().out
    assert "incomplete" in out


def test_checkpoints_before_any_eval_returns_neg_inf(bm):
    prob = Problem(bm, "sum", seed=4)
    cp = prob.checkpoints([5])
    assert cp[5] == -np.inf
    assert prob.last_incomplete[5] is True


def test_evaluate_overshoot_warns_once(bm, capsys):
    prob = Problem(bm, "sum", seed=5, budget=3)
    for _ in range(6):
        prob.evaluate(prob.random_x())
    assert prob.n == 6                      # Problem 은 초과 호출을 막지 않음(경고만)
    out = capsys.readouterr().out
    assert out.count("budget=3") == 1       # 한 번만 경고(스팸 방지)


def test_no_budget_no_warning(bm, capsys):
    prob = Problem(bm, "sum", seed=6)       # budget=None
    for _ in range(5):
        prob.evaluate(prob.random_x())
    out = capsys.readouterr().out
    assert "warning" not in out
