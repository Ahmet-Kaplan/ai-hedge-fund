"""Equal-weight benchmark — the same full long view on every name.

Not an alpha model anyone should trade for edge: it is the yardstick. Run
through the identical engine (same risk caps, costs and cadence), it answers
"did the analysts beat simply owning the universe?" — a sharper question for
a large-cap book than beating SPY.
"""

from __future__ import annotations

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel


class EqualWeightModel(QuantModel):
    investment_approach = "long_only"

    @property
    def name(self) -> str:
        return "equal_weight"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        return Signal(model_name=self.name, ticker=ticker, date=date, value=1.0,
                      reasoning="equal-weight benchmark")
