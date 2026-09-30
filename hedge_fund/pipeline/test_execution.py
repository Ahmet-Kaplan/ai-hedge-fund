"""build_orders tests — pure diffing math."""

from hedge_fund.brokers.models import Position
from hedge_fund.pipeline.execution import build_orders


def _positions(**shares):
    return {t: Position(ticker=t, shares=s) for t, s in shares.items()}


def test_floor_sizing_never_overshoots():
    orders = build_orders({"AAPL": 0.25}, {}, {"AAPL": 300.0}, equity=10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].quantity == 8  # 2500 / 300 = 8.33 -> 8


def test_delta_against_existing_position():
    orders = build_orders(
        {"AAPL": 0.25}, _positions(AAPL=5), {"AAPL": 250.0}, equity=10_000.0,
    )
    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].quantity == 5  # target 10, held 5


def test_held_name_missing_from_targets_is_closed():
    orders = build_orders({}, _positions(AAPL=7), {"AAPL": 100.0}, equity=10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].quantity == 7


def test_subshare_delta_emits_nothing():
    orders = build_orders({"AAPL": 0.005}, {}, {"AAPL": 100.0}, equity=10_000.0)
    assert orders == []  # 50 dollars / 100 = 0.5 shares -> floor 0


def test_sells_before_buys_alphabetical():
    orders = build_orders(
        {"AAPL": 0.2, "MSFT": 0.0, "NVDA": 0.2, "AMZN": 0.0},
        _positions(MSFT=10, NVDA=1, AMZN=5),
        {"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0, "AMZN": 100.0},
        equity=10_000.0,
    )
    assert [(o.ticker, o.side) for o in orders] == [
        ("AMZN", "sell"), ("MSFT", "sell"),
        ("AAPL", "buy"), ("NVDA", "buy"),
    ]


def test_short_target_sells_past_zero():
    orders = build_orders({"AAPL": -0.2}, {}, {"AAPL": 100.0}, equity=10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].quantity == 20


def test_check_projected_book_enforces_caps():
    import pytest
    from hedge_fund.brokers.models import Order
    from hedge_fund.pipeline.run_cycle import check_projected_book
    from hedge_fund.risk.limits import RiskLimits

    limits = RiskLimits(max_position_pct=0.5, max_gross_exposure=1.0)
    marks = {"A": 100.0, "B": 100.0}
    ok = [Order(ticker="A", side="buy", quantity=500, price=100.0)]
    check_projected_book(ok, {}, marks, 100_000.0, limits)
    too_big = [Order(ticker="A", side="buy", quantity=600, price=100.0)]
    with pytest.raises(ValueError, match="max_position_pct"):
        check_projected_book(too_big, {}, marks, 100_000.0, limits)
    # Each name within 50%, but 50% + 50% + 10% gross breaches 100%.
    too_gross = [Order(ticker="A", side="buy", quantity=400, price=100.0),
                 Order(ticker="B", side="sell", quantity=500, price=100.0),
                 Order(ticker="C", side="buy", quantity=100, price=100.0)]
    with pytest.raises(ValueError, match="max_gross_exposure"):
        check_projected_book(too_gross, {"A": 100}, {**marks, "C": 100.0}, 100_000.0, limits)


def test_min_trade_skips_small_opening_and_adding_trades():
    from hedge_fund.brokers.models import Position
    marks = {"A": 100.0, "B": 100.0, "C": 100.0}
    held = {"B": Position(ticker="B", shares=10), "C": Position(ticker="C", shares=3)}
    # A: new 0.2% position → skipped. B: +0.2% add → skipped. C: target 0 → closed anyway.
    orders = build_orders({"A": 0.002, "B": 0.012}, held, marks, 100_000.0, min_trade_pct=0.005)
    assert [(o.ticker, o.side, o.quantity) for o in orders] == [("C", "sell", 3)]


def test_min_trade_never_blocks_reductions():
    from hedge_fund.brokers.models import Position
    held = {"A": Position(ticker="A", shares=110)}
    orders = build_orders({"A": 0.10}, held, {"A": 100.0}, 100_000.0, min_trade_pct=0.005)
    assert [(o.ticker, o.side, o.quantity) for o in orders] == [("A", "sell", 10)]
