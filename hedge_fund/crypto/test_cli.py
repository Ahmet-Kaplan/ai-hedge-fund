from hedge_fund.crypto.cli import CORE_WEIGHTS, SPLIT, build_parser, evaluate
from hedge_fund.crypto.test_backtest import days


def test_core_weights_and_parser():
    assert CORE_WEIGHTS == {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}
    assert build_parser().parse_args(["backtest"]).fee_bps == 25.0


def test_evaluate_runs_core_and_every_rule():
    import math
    closes = {s: {d: 100 * (1 + 0.3 * math.sin(i / 40)) for i, d in enumerate(days("2021-01-01", 1200))}
              for s in CORE_WEIGHTS}
    table = evaluate(closes, fee_bps=25, end=days("2021-01-01", 1200)[-1], highs=closes, lows=closes)
    assert [row["variant"] for row in table] == ["core", "ma100", "mom12w", "ma_cross", "td1", "td2", "band10", "band20"]
    assert table[0]["passes"] is None and all(isinstance(r["passes"], bool) for r in table[1:])
    assert table[0]["start"] == "2021-07-20"                       # first day + 200
