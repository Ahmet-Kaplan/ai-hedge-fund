"""Reference strategies: direction, determinism, config identity, and no look-ahead."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hedge_fund.core import Strategy
from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.strategies import (
    REGISTRY, Breakout, CrossSectionalMomentum, MeanReversion, TimeSeriesMomentum, strategy_from_spec,
)
from hedge_fund.systematic.testing import SyntheticMarket, weekdays

DAYS = weekdays("2022-01-03", "2023-12-29")
ASOF = "2023-06-30"


def market(days=DAYS) -> SyntheticMarket:
    m = SyntheticMarket()
    n = len(days)
    m.add_series("SPY", days, SyntheticMarket.random_walk(days, seed=42, vol=0.008))
    m.add_series("UP", days, list(100 * np.exp(np.linspace(0, 0.6, n)) * (1 + 0.003 * np.sin(np.arange(n)))))
    m.add_series("DOWN", days, list(100 * np.exp(np.linspace(0, -0.5, n)) * (1 + 0.003 * np.cos(np.arange(n)))))
    for i, s in enumerate(("R1", "R2", "R3")):
        m.add_series(s, days, SyntheticMarket.random_walk(days, seed=i + 1, vol=0.01))
    # DIP: stable, then a sharp 3-day drop just before ASOF
    base = SyntheticMarket.random_walk(days, seed=9, vol=0.004)
    k = days.index(ASOF)
    dip = [p * (0.9 if k - 2 <= i <= k else 1.0) for i, p in enumerate(base)]
    m.add_series("DIP", days, dip)
    # BRK: quiet range, then a volatile surge to a new high at ASOF
    quiet = [100 + 0.2 * np.sin(i / 3) for i in range(n)]
    for j in range(k - 9, k + 1):
        quiet[j] = 100 + 1.5 * (j - (k - 10)) + (2 if j % 2 else -1)
    m.add_series("BRK", days, quiet)
    return m


@pytest.fixture(scope="module")
def panel():
    return MarketPanel.build(market(), ["UP", "DOWN", "R1", "R2", "R3", "DIP", "BRK"], DAYS[0], DAYS[-1])


def by_ticker(signals):
    return {s.ticker: s for s in signals}


ALL = [TimeSeriesMomentum(), CrossSectionalMomentum(), MeanReversion(), Breakout()]


def test_every_strategy_satisfies_the_protocol_and_knows_no_broker():
    for s in ALL:
        assert isinstance(s, Strategy)
        assert not any(hasattr(s, a) for a in ("broker", "place_order", "size", "positions"))


def test_tsmom_direction(panel):
    sig = by_ticker(TimeSeriesMomentum().generate(panel.as_of(ASOF)))
    assert sig["UP"].value > 0.5 and sig["DOWN"].value < -0.5
    assert sig["UP"].metadata["data_as_of"] == ASOF and "tstat" in sig["UP"].components


def test_xsmom_ranks_the_universe(panel):
    sig = by_ticker(CrossSectionalMomentum().generate(panel.as_of(ASOF)))
    assert sig["UP"].value == pytest.approx(1.0)
    order = sorted(sig.values(), key=lambda s: s.components["trailing_log_return"])
    assert [s.value for s in order] == sorted(s.value for s in order)      # score follows return rank
    assert order[0].value == pytest.approx(-1.0) and sig["DOWN"].value < 0
    assert all(-1 <= s.value <= 1 for s in sig.values())


def test_meanrev_buys_the_dip(panel):
    sig = by_ticker(MeanReversion().generate(panel.as_of(ASOF)))
    assert sig["DIP"].value > 0.7 and sig["DIP"].components["zscore"] < -1


def test_breakout_needs_new_high_and_vol_expansion(panel):
    sig = by_ticker(Breakout().generate(panel.as_of(ASOF)))
    assert sig["BRK"].value == 1.0 and sig["BRK"].components["atr_ratio"] > 1
    quiet = by_ticker(Breakout().generate(panel.as_of("2023-03-01")))
    assert quiet["BRK"].value == 0.0


def test_insufficient_history_abstains(panel):
    early = panel.as_of("2022-02-01")
    for s in (TimeSeriesMomentum(), Breakout()):
        assert all(x.metadata["abstained"] and x.value == 0 for x in s.generate(early))


def test_deterministic_and_config_identity(panel):
    view = panel.as_of(ASOF)
    for s in ALL:
        a = [x.model_dump() for x in s.generate(view)]
        b = [x.model_dump() for x in s.generate(view)]
        assert a == b
        spec = json.loads(json.dumps(s.spec()))
        clone = strategy_from_spec(spec)
        assert clone.config_hash() == s.config_hash()
    assert TimeSeriesMomentum(lookback=126).config_hash() != TimeSeriesMomentum().config_hash()
    assert set(REGISTRY) == {"tsmom", "xsmom", "meanrev", "breakout"}


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        TimeSeriesMomentum(lookback=5)
    with pytest.raises(ValueError):
        MeanReversion(bogus=1)


@pytest.mark.parametrize("asof", ["2023-03-15", ASOF, "2023-09-29"])
def test_signals_do_not_depend_on_the_future(asof):
    """A panel whose vendor history ends at asof gives identical signals."""
    full = MarketPanel.build(market(), ["UP", "DOWN", "R1", "R2", "R3", "DIP", "BRK"], DAYS[0], DAYS[-1])
    cut_days = [d for d in DAYS if d <= asof]
    m = market()
    m.bars = {k: [b for b in v if b.date <= asof] for k, v in m.bars.items()}
    cut = MarketPanel.build(m, ["UP", "DOWN", "R1", "R2", "R3", "DIP", "BRK"], cut_days[0], cut_days[-1])
    for s in ALL:
        a = [x.model_dump() for x in s.generate(full.as_of(asof))]
        b = [x.model_dump() for x in s.generate(cut.as_of(asof))]
        assert a == b, s.name


def test_excluded_and_untradable_names_are_not_scored(panel):
    sig = TimeSeriesMomentum(exclude=("UP",)).generate(panel.as_of(ASOF))
    assert "UP" not in {s.ticker for s in sig}
