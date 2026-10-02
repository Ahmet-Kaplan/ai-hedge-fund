"""How the live account is doing, with deposits taken out of the return."""

from __future__ import annotations

from pydantic import BaseModel

from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.settings import LiveSettings


class LiveReport(BaseModel):
    start: str
    end: str
    deposited: float                 # deposits − withdrawals, all time
    equity: float
    gain: float                      # equity − deposited
    time_weighted_return: float      # unaffected by when money arrived
    core_return: float               # the core ETF alone over the same dates
    core_value: float
    satellite_value: float
    crypto_value: float
    agent_share: float
    satellite_halted: bool
    satellite_vs_core: float | None  # since the latest review


def build_live_report(ledger: LiveLedger, settings: LiveSettings) -> LiveReport | None:
    rows = ledger.live_nav_rows()
    if not rows:
        return None
    twr = 1.0
    for prev, cur in zip(rows, rows[1:]):
        if prev.equity > 0:
            twr *= (cur.equity - cur.net_flow) / prev.equity
    last = rows[-1]
    review = ledger.last_review_date()
    return LiveReport(
        start=rows[0].date, end=last.date, deposited=round(ledger.total_deposited(), 2), equity=last.equity,
        gain=round(last.equity - ledger.total_deposited(), 2), time_weighted_return=round(twr - 1, 6),
        core_return=round(last.core_close / rows[0].core_close - 1, 6), core_value=last.core_value,
        satellite_value=last.satellite_value, crypto_value=last.crypto_value, agent_share=settings.agent_share,
        satellite_halted=ledger.satellite_halted(),
        satellite_vs_core=ledger.satellite_excess(review) if review else None,
    )
