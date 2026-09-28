"""PositionBook tests — cost basis, realized P&L, commissions.

Hand-computed, because this is the arithmetic that decides what a backtest
says a strategy earned. Ported from PR #19's accounting work, extended with
the seeded-position case (a receipt knows share counts but not what they cost)
and with parity between the two offline venues.
"""

from __future__ import annotations

import pytest

from hedge_fund.brokers.book import PositionBook
from hedge_fund.brokers.models import Commission, Order
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker


def _order(ticker="AAPL", side="buy", quantity=10, price=100.0, cid=None):
    return Order(ticker=ticker, side=side, quantity=quantity, price=price,
                 client_order_id=cid)


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def test_buying_establishes_a_basis():
    book = PositionBook(cash=10_000.0)

    book.apply("AAPL", "buy", 10, 100.0)

    assert book.shares("AAPL") == 10
    assert book.positions()["AAPL"].cost_basis == pytest.approx(100.0)
    assert book.cash == pytest.approx(9_000.0)
    assert book.realized_pnl == 0.0


def test_adding_blends_the_average():
    book = PositionBook(cash=10_000.0)

    book.apply("AAPL", "buy", 10, 100.0)
    book.apply("AAPL", "buy", 10, 120.0)

    assert book.shares("AAPL") == 20
    assert book.positions()["AAPL"].cost_basis == pytest.approx(110.0)
    assert book.realized_pnl == 0.0      # nothing closed


def test_the_basis_stays_positive_on_a_short():
    book = PositionBook(cash=10_000.0)

    book.apply("AAPL", "sell", 10, 100.0)

    position = book.positions()["AAPL"]
    assert position.shares == -10
    assert position.cost_basis == pytest.approx(100.0)   # the sign lives in shares
    assert book.cash == pytest.approx(11_000.0)


# ---------------------------------------------------------------------------
# Realizing
# ---------------------------------------------------------------------------

def test_reducing_realizes_on_the_shares_that_left():
    book = PositionBook(cash=10_000.0)
    book.apply("AAPL", "buy", 10, 100.0)

    movement = book.apply("AAPL", "sell", 4, 110.0)

    assert movement.realized_pnl == pytest.approx(40.0)      # 4 * (110 - 100)
    assert book.shares("AAPL") == 6
    # The remaining shares keep the basis they were carried at.
    assert book.positions()["AAPL"].cost_basis == pytest.approx(100.0)


def test_closing_realizes_on_everything_and_clears_the_row():
    book = PositionBook(cash=10_000.0)
    book.apply("AAPL", "buy", 10, 100.0)

    movement = book.apply("AAPL", "sell", 10, 90.0)

    assert movement.realized_pnl == pytest.approx(-100.0)
    assert book.shares("AAPL") == 0
    assert "AAPL" not in book.positions()
    # Flat and commissionless: cash moved by exactly the realized P&L.
    assert book.cash == pytest.approx(9_900.0)


def test_a_short_earns_when_the_price_falls():
    book = PositionBook(cash=10_000.0)
    book.apply("AAPL", "sell", 10, 100.0)

    movement = book.apply("AAPL", "buy", 10, 90.0)

    assert movement.realized_pnl == pytest.approx(100.0)     # short: basis - price
    assert book.cash == pytest.approx(10_100.0)


def test_a_round_trip_equals_the_realized_pnl():
    """The invariant that makes realized P&L auditable: flat book, cash delta
    equals what the trades earned."""
    book = PositionBook(cash=10_000.0)

    book.apply("AAPL", "buy", 10, 100.0)
    book.apply("AAPL", "sell", 4, 110.0)
    book.apply("AAPL", "sell", 6, 90.0)

    assert book.realized_pnl == pytest.approx(-20.0)
    assert book.shares("AAPL") == 0
    assert book.cash - 10_000.0 == pytest.approx(book.realized_pnl)


def test_crossing_zero_closes_then_opens_at_the_new_price():
    """One order, two events. Carrying the old basis across would price a short
    off shares the book no longer holds."""
    book = PositionBook(cash=10_000.0)
    book.apply("AAPL", "buy", 5, 100.0)

    movement = book.apply("AAPL", "sell", 8, 120.0)

    assert movement.realized_pnl == pytest.approx(100.0)     # 5 * (120 - 100)
    position = book.positions()["AAPL"]
    assert position.shares == -3
    assert position.cost_basis == pytest.approx(120.0)       # the new side, not 100
    # cash 10000 - 500 + 960 = 10460; short 3 marked at 120 is -360 -> NAV 10100
    assert book.cash == pytest.approx(10_460.0)
    assert book.cash - 3 * 120.0 == pytest.approx(10_100.0)


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------

def test_commission_is_a_cash_expense_not_a_basis_adjustment():
    book = PositionBook(cash=10_000.0, commission=Commission(per_trade=1.0, per_share=0.01))

    movement = book.apply("AAPL", "buy", 10, 100.0)

    assert movement.commission == pytest.approx(1.0 + 0.10)
    assert book.cash == pytest.approx(10_000.0 - 1_000.0 - 1.10)
    assert book.positions()["AAPL"].cost_basis == pytest.approx(100.0)  # unaffected
    assert book.realized_pnl == 0.0                                    # and not netted


def test_commission_does_not_leak_into_realized_pnl():
    book = PositionBook(cash=10_000.0, commission=Commission(per_trade=5.0))
    book.apply("AAPL", "buy", 10, 100.0)

    movement = book.apply("AAPL", "sell", 10, 110.0)

    assert movement.realized_pnl == pytest.approx(100.0)   # gross
    assert book.realized_pnl == pytest.approx(100.0)
    assert book.cash == pytest.approx(10_000.0 + 100.0 - 10.0)   # both tickets


def test_a_zero_commission_book_is_bit_identical_to_no_commission():
    """The reason costs can be added at all without invalidating old backtests."""
    plain = PositionBook(cash=10_000.0)
    explicit = PositionBook(cash=10_000.0, commission=Commission())

    for book in (plain, explicit):
        book.apply("AAPL", "buy", 7, 101.37)
        book.apply("MSFT", "sell", 3, 250.11)
        book.apply("AAPL", "sell", 7, 99.99)

    assert plain.cash == explicit.cash
    assert plain.realized_pnl == explicit.realized_pnl
    assert plain.positions() == explicit.positions()


# ---------------------------------------------------------------------------
# Seeded books: an unknown basis is not a zero basis
# ---------------------------------------------------------------------------

def test_a_seeded_position_has_no_basis():
    book = PositionBook(cash=5_000.0, positions={"AAPL": 10})

    assert book.shares("AAPL") == 10
    assert book.positions()["AAPL"].cost_basis is None


def test_closing_a_seeded_position_reports_unknown_rather_than_inventing_one():
    """A receipt knows the share count, not what it cost. Assuming 0.0 would
    manufacture a profit on every exit."""
    book = PositionBook(cash=5_000.0, positions={"AAPL": 10})

    movement = book.apply("AAPL", "sell", 10, 120.0)

    assert movement.realized_pnl is None
    assert book.realized_pnl == 0.0
    assert book.cash == pytest.approx(6_200.0)      # cash is still exactly right


def test_a_fill_gives_a_seeded_position_a_basis():
    book = PositionBook(cash=5_000.0, positions={"AAPL": 10})

    book.apply("AAPL", "buy", 10, 100.0)            # adds, and so establishes one

    assert book.positions()["AAPL"].cost_basis == pytest.approx(100.0)
    assert book.shares("AAPL") == 20


def test_a_supplied_basis_is_used():
    book = PositionBook(cash=5_000.0, positions={"AAPL": 10}, cost_basis={"AAPL": 80.0})

    movement = book.apply("AAPL", "sell", 10, 100.0)

    assert movement.realized_pnl == pytest.approx(200.0)


def test_a_basis_for_a_position_that_does_not_exist_is_ignored():
    book = PositionBook(cash=5_000.0, positions={"AAPL": 10}, cost_basis={"MSFT": 50.0})
    assert "MSFT" not in book.positions()


# ---------------------------------------------------------------------------
# Both offline venues, one arithmetic
# ---------------------------------------------------------------------------

def _replay(broker):
    broker.place_order(_order(side="buy", quantity=10, price=100.0))
    broker.place_order(_order(side="sell", quantity=4, price=110.0))
    broker.place_order(_order(side="sell", quantity=8, price=95.0))   # crosses zero
    return broker


def test_sim_and_paper_agree_on_cost_basis_and_realized_pnl():
    """A backtest and a paper run of the same mandate must not measure two
    different strategies, which is what a commission in one and not the other
    would do."""
    commission = Commission(per_trade=1.0, per_share=0.02)
    sim = _replay(SimBroker(cash=10_000.0, commission=commission))
    paper = _replay(PaperBroker(cash=10_000.0, commission=commission))

    assert sim.cash() == pytest.approx(paper.cash())
    assert sim.realized_pnl() == pytest.approx(paper.realized_pnl())
    assert {t: (p.shares, p.cost_basis) for t, p in sim.positions().items()} == \
           {t: (p.shares, p.cost_basis) for t, p in paper.positions().items()}


def test_the_fill_carries_the_cost_and_the_result():
    broker = SimBroker(cash=10_000.0, commission=Commission(per_trade=2.0))
    broker.place_order(_order(side="buy", quantity=10, price=100.0))

    fill = broker.place_order(_order(side="sell", quantity=10, price=105.0))

    assert fill.commission == pytest.approx(2.0)
    assert fill.realized_pnl == pytest.approx(50.0)
    assert broker.realized_pnl() == pytest.approx(50.0)


def test_idempotency_still_holds_with_costs(tmp_path):
    """The receipts path must not double-charge a replayed instruction."""
    broker = SimBroker(cash=10_000.0, commission=Commission(per_trade=5.0))
    order = _order(side="buy", quantity=1, price=100.0, cid="cid-1")

    first = broker.place_order(order)
    cash_after = broker.cash()
    second = broker.place_order(order)

    assert second == first
    assert broker.cash() == cash_after
    assert broker.positions()["AAPL"].shares == 1
