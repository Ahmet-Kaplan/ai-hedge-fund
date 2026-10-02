from hedge_fund.crypto.rules import RULES, ma100, ma_cross, mom12w


def test_ma100():
    assert ma100([1.0] * 99) == 0                       # not enough history: out
    assert ma100([1.0] * 99 + [2.0]) == 1
    assert ma100([2.0] * 99 + [1.0]) == 0


def test_mom12w():
    assert mom12w([1.0] * 84) == 0                      # needs 85 closes (an 84-day return)
    assert mom12w([1.0] * 84 + [1.1]) == 1
    assert mom12w([1.1] + [1.0] * 84) == 0


def test_ma_cross():
    assert ma_cross([1.0] * 199) == 0
    assert ma_cross([1.0] * 150 + [2.0] * 50) == 1      # 50-day avg 2.0 > 200-day avg 1.25
    assert ma_cross([2.0] * 150 + [1.0] * 50) == 0


def test_registry():
    assert set(RULES) == {"ma100", "mom12w", "ma_cross"}
