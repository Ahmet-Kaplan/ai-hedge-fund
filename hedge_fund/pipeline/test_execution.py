"""build_orders tests — pure diffing math."""

from hedge_fund.brokers.models import Order, Position
from hedge_fund.pipeline.execution import build_orders, stamp_client_order_ids


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


# ---------------------------------------------------------------------------
# Deterministic client order ids — what makes a retry safe
# ---------------------------------------------------------------------------

def _orders():
    return [
        Order(ticker="AAPL", side="sell", quantity=3, price=100.0),
        Order(ticker="BRK.B", side="buy", quantity=1, price=400.0),
    ]


def test_stamp_is_deterministic_and_positional():
    """Same inputs -> same ids, so a replay cannot place a second order."""
    first = stamp_client_order_ids("desk", "2025-01-10", _orders())
    second = stamp_client_order_ids("desk", "2025-01-10", _orders())

    assert [o.client_order_id for o in first] == [o.client_order_id for o in second]
    assert first[0].quantity == 3 and first[1].ticker == "BRK.B"  # untouched otherwise


def test_stamp_distinguishes_fund_session_and_side():
    base = stamp_client_order_ids("desk", "2025-01-10", _orders())[0].client_order_id

    assert stamp_client_order_ids("other", "2025-01-10", _orders())[0].client_order_id != base
    assert stamp_client_order_ids("desk", "2025-01-11", _orders())[0].client_order_id != base
    # A reversed direction on the same name is a different instruction.
    flipped = [Order(ticker="AAPL", side="buy", quantity=3, price=100.0)]
    assert stamp_client_order_ids("desk", "2025-01-10", flipped)[0].client_order_id != base


def test_ids_are_venue_safe_and_bounded():
    """Alpaca caps client_order_id at 48 characters."""
    long_fund = "a-very-long-mandate-name-that-would-blow-the-limit"
    orders = [Order(ticker="BRK.B", side="buy", quantity=1, price=400.0)]

    for o in stamp_client_order_ids(long_fund, "2025-01-10", orders):
        cid = o.client_order_id
        assert cid is not None
        assert 0 < len(cid) <= 48
        assert all(c.isalnum() or c == "-" for c in cid), cid


def test_ids_are_unique_within_a_session():
    orders = [
        Order(ticker="AAPL", side="buy", quantity=1, price=10.0),
        Order(ticker="AAPL", side="sell", quantity=1, price=10.0),
        Order(ticker="MSFT", side="buy", quantity=1, price=10.0),
    ]
    ids = [o.client_order_id for o in stamp_client_order_ids("d", "2025-01-10", orders)]
    assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# Turnover control
# ---------------------------------------------------------------------------

def _diff(targets, held, marks, equity, **kwargs):
    """Named apart from the module's existing `_orders` helper."""
    return build_orders(targets, held, marks, equity, **kwargs)


def test_a_dust_trade_that_grows_a_position_is_skipped():
    """Ticket count is what costs money at this size, so the cheapest lever is
    not making the trade at all."""
    held = {"AAPL": Position(ticker="AAPL", shares=100)}
    orders = _diff({"AAPL": 0.101}, held, {"AAPL": 100.0}, 100_000.0,
                     min_trade_pct=0.005)

    assert orders == []          # 1 share, $100, 0.1% of equity — below the bar


def test_a_shrinking_trade_is_never_skipped():
    """A cost control must not be a reason to hold risk you meant to shed — and
    a small trim is exactly what a cap or a drawdown brake asks for."""
    held = {"AAPL": Position(ticker="AAPL", shares=100)}
    orders = _diff({"AAPL": 0.099}, held, {"AAPL": 100.0}, 100_000.0,
                     min_trade_pct=0.05)

    assert len(orders) == 1 and orders[0].side == "sell" and orders[0].quantity == 1


def test_the_bar_compares_notional_to_equity():
    """The same $10,000 trade clears the bar on a small book and not on a large
    one — which is the point: it is a fraction of capital, not of the position.
    """
    marks = {"AAPL": 100.0}

    # 100 shares = $10,000 = 10% of a $100k book
    assert _diff({"AAPL": 0.10}, {}, marks, 100_000.0, min_trade_pct=0.05)
    # the same $10,000 is 1% of a $1M book
    assert _diff({"AAPL": 0.01}, {}, marks, 1_000_000.0, min_trade_pct=0.05) == []


def test_zero_is_the_default_and_trades_everything():
    held = {"AAPL": Position(ticker="AAPL", shares=100)}

    assert _diff({"AAPL": 0.101}, held, {"AAPL": 100.0}, 100_000.0)


def test_the_bar_round_trips_through_the_mandate():
    from hedge_fund.fund.spec import FundSpec

    spec = FundSpec(
        schema_version=2, name="t",
        strategies=[{"name": "s", "models": [{"name": "pead"}], "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        min_trade_pct=0.004,
    )

    assert FundSpec.model_validate_json(spec.model_dump_json()).min_trade_pct == 0.004
