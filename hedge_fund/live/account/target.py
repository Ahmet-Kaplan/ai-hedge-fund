"""What the live account should hold: the core plus the agents' share of the paper fund's bets."""

from __future__ import annotations

from datetime import date

from hedge_fund.live.account.settings import LiveSettings
from hedge_fund.live.ledger import Ledger


def satellite_weights(
    paper_weights: dict[str, float], share: float, *, shorts_ok: bool, max_name: float, core_ticker: str,
    max_names: int | None = None,
) -> dict[str, float]:
    """The paper fund's active book rescaled so its gross equals `share` of the account.

    Without shorts_ok, short bets are dropped and the longs fill the share.
    Each name is capped at max_name; what a cap removes goes to the core,
    never to other names.
    """
    bets = {t: w for t, w in paper_weights.items() if w and t != core_ticker and (shorts_ok or w > 0)}
    if max_names is not None:   # only the strongest convictions
        bets = dict(sorted(bets.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:max(0, max_names)])
    gross = sum(abs(w) for w in bets.values())
    if share <= 0 or gross <= 0:
        return {}
    return {t: max(-max_name, min(max_name, share * w / gross)) for t, w in bets.items()}


def target_book(satellite: dict[str, float], core_ticker: str) -> dict[str, float]:
    """Satellite weights plus the core, which takes everything the satellite doesn't use."""
    return {core_ticker: 1.0 - sum(abs(w) for w in satellite.values()), **satellite}


def latest_paper_weights(paper: Ledger, today: str, stale_days: int) -> tuple[dict[str, float], str | None, bool]:
    """The newest real paper plan's final (pre-equitization) weights, its session, and whether it's fresh."""
    sessions = [s for s in paper.plan_sessions() if s <= today]
    if not sessions:
        return {}, None, False
    session = sessions[-1]
    decision = (paper.read_plan(session) or {}).get("plan", {}).get("decision") or {}
    weights = {t: w for t, w in decision.get("final_weights", {}).items() if w}
    fresh = (date.fromisoformat(today) - date.fromisoformat(session)).days <= stale_days
    return weights, session, fresh


def account_book(stock_book: dict[str, float], settings: LiveSettings) -> dict[str, float]:
    """The whole account: the stock book scaled to (1 − crypto_share), plus the crypto core."""
    share = settings.crypto_share
    if share <= 0:
        return stock_book
    book = {t: w * (1 - share) for t, w in stock_book.items()}
    for t, w in settings.crypto_core.items():
        book[t] = book.get(t, 0.0) + w * share
    return book


def affordable_satellite(
    paper_weights: dict[str, float], share: float, *, shorts_ok: bool, max_name: float, core_ticker: str,
    satellite_dollars: float, min_position_usd: float,
) -> dict[str, float]:
    """The satellite with as many of the strongest picks as the money allows.

    Weights follow conviction; the weakest picks are dropped until every
    pick is worth at least `min_position_usd`. A small account holds a few
    picks, and the number grows with deposits.
    """
    n = int(satellite_dollars // min_position_usd) if min_position_usd > 0 else len(paper_weights)
    while n > 0:
        sat = satellite_weights(paper_weights, share, shorts_ok=shorts_ok, max_name=max_name,
                                core_ticker=core_ticker, max_names=n)
        if sat and min(abs(w) / share * satellite_dollars for w in sat.values()) >= min_position_usd - 1e-9:
            return sat
        n -= 1
    return {}
