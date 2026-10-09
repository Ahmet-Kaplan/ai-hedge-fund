"""PositionBook — cash, signed shares, and a weighted-average cost basis.

One implementation, used by both offline venues. `SimBroker` (backtests) and
`PaperBroker` (live-clock paper) must not disagree about what a fill cost or
what it earned: a backtest that charges no commission while the paper run
charges one is measuring two different strategies.

The accounting itself is the interesting part, and it is the part that is easy
to get subtly wrong:

**Adding** blends the fill into the average. Nothing closes, so nothing
realizes.

**Reducing or closing** realizes on the shares that left, against the basis
they were carried at. Whatever remains keeps that basis — an exit says nothing
about what the rest was bought for.

**Crossing zero** is two events in one order: it closes the whole position and
opens the opposite one. Only the closed shares realize, and the new position
starts from this fill's price. Carrying the old basis across would price a
short off shares the book no longer holds.

**Commission** is a cash expense rather than a basis adjustment, so "what the
position earned" and "what trading it cost" stay separable instead of arriving
pre-mixed. That also keeps a zero-commission book bit-identical to the one this
project had before costs existed.

**An unknown basis is not a zero basis.** A book seeded from a receipt knows
its share counts and its cash but not what the shares were bought for. Closing
such a position moves cash correctly and reports `realized_pnl = None`, because
inventing a basis would manufacture a profit on every exit.
"""

from __future__ import annotations

from dataclasses import dataclass

from hedge_fund.brokers.models import Commission, Position


@dataclass(frozen=True)
class BookFill:
    """What one movement of the book produced."""

    commission: float
    #: None when the shares that left had no known basis.
    realized_pnl: float | None


class PositionBook:
    """Cash, positions and their basis. Pure bookkeeping; no venue, no orders."""

    def __init__(
        self,
        cash: float,
        positions: dict[str, int] | None = None,
        commission: Commission | None = None,
        cost_basis: dict[str, float] | None = None,
    ) -> None:
        self._cash = float(cash)
        self._shares: dict[str, int] = {
            ticker: shares
            for ticker, shares in (positions or {}).items()
            if shares != 0
        }
        # Only where a caller supplied one. Absent means unknown, and stays
        # that way until a fill gives the position a price to be carried at.
        self._basis: dict[str, float] = {
            ticker: float(basis)
            for ticker, basis in (cost_basis or {}).items()
            if ticker in self._shares
        }
        self._realized = 0.0
        self._commission = commission or Commission()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def realized_pnl(self) -> float:
        """Cumulative realized P&L, gross of commission.

        Commission is booked as a cash expense, so it shows up in cash and in
        NAV rather than here — the two numbers answer different questions.
        """
        return self._realized

    @property
    def commission(self) -> Commission:
        return self._commission

    def shares(self, ticker: str) -> int:
        return self._shares.get(ticker, 0)

    def positions(self) -> dict[str, Position]:
        """Signed share counts with their basis, and no zero-share rows."""
        return {
            ticker: Position(
                ticker=ticker, shares=shares,
                cost_basis=self._basis.get(ticker),
            )
            for ticker, shares in self._shares.items()
            if shares != 0
        }

    # ------------------------------------------------------------------
    # The one write
    # ------------------------------------------------------------------

    def apply(self, ticker: str, side: str, quantity: int, price: float) -> BookFill:
        """Move the book for one fill. Returns what it cost and realized."""
        signed = quantity if side == "buy" else -quantity
        old = self._shares.get(ticker, 0)
        basis = self._basis.get(ticker)
        new = old + signed

        realized: float | None = 0.0
        if old == 0 or (old > 0) == (signed > 0):
            # Opening, or adding in the direction already held: blend.
            if basis is None:
                basis = price
            else:
                basis = (abs(old) * basis + quantity * price) / abs(new)
        elif quantity <= abs(old):
            # Reducing or closing exactly: the leavers realize, the rest keeps
            # the basis they were carried at.
            realized = self._realize(old, quantity, basis, price)
        else:
            # Crossing zero: close it all at this price, then open the other
            # side from this price.
            realized = self._realize(old, abs(old), basis, price)
            basis = price

        commission = self._commission.charge(quantity, price)
        self._cash += -signed * price - commission
        if realized is not None:
            self._realized += realized

        if new == 0:
            self._shares.pop(ticker, None)
            self._basis.pop(ticker, None)
        else:
            self._shares[ticker] = new
            self._basis[ticker] = basis if basis is not None else price

        return BookFill(commission=commission, realized_pnl=realized)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _realize(old: int, quantity: int, basis: float | None, price: float) -> float | None:
        """P&L on `quantity` shares leaving a position of signed size `old`.

        The one place the sign of the position changes the arithmetic rather
        than just the bookkeeping: a long earns the rise above its basis, a
        short the fall below it.
        """
        if basis is None:
            return None
        return quantity * (price - basis) if old > 0 else quantity * (basis - price)
