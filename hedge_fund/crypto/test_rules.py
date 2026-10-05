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
    assert set(RULES) == {"ma100", "mom12w", "ma_cross", "td1", "td2"}


from hedge_fund.crypto.rules import td1, td2  # noqa: E402


def test_td1_breakout_trails_and_reenters():
    up = [100 + i for i in range(40)]
    highs, lows = [c + 1 for c in up], [c - 1 for c in up]
    assert td1.series(up, highs, lows)[-1] == 1                              # steady rise: long
    crash = up + [50.0] * 5
    hc, lc = highs + [51.0] * 5, lows + [49.0] * 5
    s = td1.series(crash, hc, lc)
    assert s[40] == 0                                                        # close fell below the trailing low
    rebound = crash + [200.0]
    s = td1.series(rebound, hc + [201.0], lc + [199.0])
    assert s[-1] == 1                                                        # broke above the level: back in


def test_td2_vote_and_volatility_size():
    calm = [100 * 1.001 ** i for i in range(200)]                            # steady rise, tiny vol
    assert td2.series(calm)[-1] == 1.0                                       # vol << 50% → capped at 1
    import math
    wild = [100 * math.exp(0.002 * i + (0.08 if i % 2 else -0.08)) for i in range(200)]
    size = td2.series(wild)[-1]
    assert 0 < size < 1                                                      # high vol → smaller position
    falling = [100 * 0.999 ** i for i in range(200)]
    assert td2.series(falling)[-1] == 0                                      # 3 of 3 down: cash (long-only)
    assert td2.series(calm[:100])[-1] == 0                                   # not enough history yet
