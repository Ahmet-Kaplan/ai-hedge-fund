"""DPX macro stability alpha model — a market-wide regime overlay.

Forms a view from DPX's Stability Oracle (https://untitledfinancial.com), a
live macro/climate/FX/geopolitical signal pipeline that gates institutional
stablecoin settlement. The oracle produces one global stability score, not a
per-ticker one — so this is a *regime overlay*: the same conviction is returned
for every ticker on a given day, meant to be combined with ticker-specific
alpha models (PEAD, Buffett, …) rather than used alone.

No API key required — the endpoint used here is free and unauthenticated.

Live-only caveat: the Stability Oracle exposes current conditions, not a
queryable history. `predict()` can only form a real view for the current day;
for any other `date` it abstains rather than silently reusing today's score as
a stand-in for a past date, which would violate the point-in-time contract this
interface requires.

Two deviations from the contributed version, both about the abstention
contract:

**Abstentions are marked as such.** Returning `value=0.0` is not enough here.
The pipeline reads `metadata["abstained"] is not True` (see `_dropped_outputs`
in `hedge_fund/pipeline/run_cycle.py`), so an unmarked 0.0 is not an omission —
it is a genuine neutral vote that dilutes every other model in the blend, and
it never appears in the audit trail as a view that failed to form. Both abstain
paths set `abstained` and an `abstain_reason`, the same contract `LLMAgent`
follows for a failed call.

**`supports_blind` is declared honestly.** The oracle knows nothing about the
ticker — the conviction is identical for every symbol — so handing it the
ticker leaks nothing and a blind backtest can staff it. That is worth stating
explicitly rather than leaving to the default, because the model is unusual:
it is the only registered model whose output does not depend on its ticker
argument, and it does its own live network I/O instead of reading the injected
`DataClient`.

It is also, for the same reason, useless in a historical backtest: every past
date abstains. It is a live regime gate, not a backtestable signal.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime

import requests

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

_RELIABILITY_URL = "https://stability.untitledfinancial.com/reliability"
_TIMEOUT_SECONDS = 10.0


class DPXMacroStabilityModel(QuantModel):
    """Market-wide macro regime overlay from DPX's live Stability Oracle.

    `predict(ticker, date)` ignores `ticker` (the signal is market-wide) and
    returns the same conviction for every symbol on a given call. Conviction is
    derived from the oracle's 0-100 stability score: STABLE conditions map
    toward a mildly bullish risk-on overlay, UNSTABLE conditions toward
    bearish, CAUTION sits in between.
    """

    # Declared on the class, not inherited: get_investment_approach reads
    # vars(cls), and the contributed version omitted this — which raises at
    # registration validation rather than at use.
    investment_approach = "long_short"

    # The oracle is handed no ticker at all, so a blind run cannot leak one.
    supports_blind = True

    def __init__(self, *, timeout_seconds: float = _TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds
        self._cache: dict[str, dict] = {}  # keyed by date string, one lookup per day

    @property
    def name(self) -> str:
        return "dpx_macro"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        if not _is_today(date):
            return self._abstain(
                ticker, date,
                "DPX Stability Oracle is live-only (no historical query) — "
                "abstaining rather than reusing today's score for a past date.",
            )

        try:
            reliability = self._fetch_reliability(date)
            stability = reliability["stability"]
            # Parsed strictly, not through `_safe_float`. Its 0.0 default would
            # turn an unreadable response into a score of zero — which this
            # scale reads as *maximally unstable*, so garbage would arrive as
            # the most bearish signal in the book instead of as no view at all.
            score = float(stability["currentScore"])
            status = str(stability["latestStatus"])
            caution_threshold = float(stability["threshold"]["caution"])
            stable_threshold = float(stability["threshold"]["stable"])
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            # Infrastructure failure or a response we cannot read — abstain with
            # the error visible rather than returning a false neutral.
            return self._abstain(
                ticker, date, f"DPX Stability Oracle unavailable or unreadable: {exc}"
            )

        # Center the sigmoid on the caution/stable midpoint so STABLE trends
        # toward +1, UNSTABLE toward -1, with CAUTION near zero.
        midpoint = (caution_threshold + stable_threshold) / 2
        conviction = self._sigmoid((score - midpoint) / 10, scale=1.0)

        peg_bps = self._safe_float(reliability.get("peg", {}).get("deviationBps"))

        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=self._normalize_to_signal(conviction),
            reasoning=(
                f"DPX Stability Oracle: {status} (score {score:.0f}/100), "
                f"peg deviation {peg_bps:.0f}bps"
            ),
            components={"stability_score": score, "peg_deviation_bps": peg_bps},
            metadata={
                "abstained": False,
                "source": _RELIABILITY_URL,
                "status": status,
                # The overlay does its own fetching, outside the DataClient
                # contract — say so, the way ta_board does.
                "data_source": "dpx_oracle",
            },
        )

    def _abstain(self, ticker: str, date: str, why: str) -> Signal:
        """No view. Marked, so the pipeline omits it instead of blending a 0.0."""
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=f"abstained: {why}",
            metadata={
                "abstained": True,
                "abstain_reason": why,
                "source": _RELIABILITY_URL,
            },
        )

    def _fetch_reliability(self, date: str) -> dict:
        if date in self._cache:
            return self._cache[date]
        resp = requests.get(_RELIABILITY_URL, timeout=self._timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        self._cache[date] = data
        return data


def _is_today(date_str: str) -> bool:
    try:
        requested = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return False
    return requested == date_cls.today()
