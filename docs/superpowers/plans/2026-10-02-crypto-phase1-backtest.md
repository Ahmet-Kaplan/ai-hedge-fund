# Crypto Phase 1 — Data, Rules, Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with no money and no LLM, whether any pre-registered trend rule beats buy-and-hold 60/30/10 BTC/ETH/SOL after fees (spec §1–§3).

**Architecture:** `hedge_fund/data/crypto_prices.py` fetches free Alpaca crypto daily bars and caches them in the existing `MarketStore.prices` table under pair names (`BTC/USD`). `hedge_fund/crypto/` holds the rules, the backtest, and an `aihf-crypto` CLI. Phases 2–3 (overlay, paper lab, live) get their own plan only if a rule passes.

**Tech Stack:** Python 3.12, pydantic v2, requests, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-crypto-sleeve-design.md`

**Test command:** `env FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest <path> -q`

**Decisions made while planning:**
- Crypto bars reuse `MarketStore.prices` / `price_sync` keyed by pair (`BTC/USD`); no new table (spec §1 said one; the existing one fits).
- Every variant starts on the same date: the first data day + 200 days (the longest rule's warm-up), so comparisons are fair.
- A rule with too little history says 0 (out).
- trader.dev candidates (`td1`, `td2`) are not in this phase: the MCP isn't connected in this session. They can be added later as rules, transcribed before testing.

---

### Task 1: Crypto price source and cache

**Files:** Create `hedge_fund/data/crypto_prices.py`; Test `hedge_fund/data/test_crypto_prices.py`

- [ ] **Step 1: Failing tests**

```python
"""Crypto bars: fake session, temp SQLite store, no network."""

import pytest

from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_closes, pair
from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.store import MarketStore


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.headers, self.calls, self._responses = {}, [], list(responses)

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        return self._responses.pop(0)


def bar(day, close):
    return {"t": f"{day}T00:00:00Z", "o": close, "h": close, "l": close, "c": close, "v": 1.5}


def test_pair_normalizes_both_alpaca_forms():
    assert pair("BTCUSD") == pair("btc/usd") == "BTC/USD"
    with pytest.raises(ValueError):
        pair("BTCEUR")


def test_fetch_follows_pages():
    session = FakeSession(
        FakeResponse({"bars": {"BTC/USD": [bar("2021-01-01", 29000.0)]}, "next_page_token": "p2"}),
        FakeResponse({"bars": {"BTC/USD": [bar("2021-01-02", 32000.0)], "ETH/USD": [bar("2021-01-02", 700.0)]},
                      "next_page_token": None}),
    )
    out = CryptoPriceSource(session=session).fetch(["BTC/USD", "ETH/USD"], "2021-01-01", "2021-01-02")
    assert [p.close for p in out["BTC/USD"]] == [29000.0, 32000.0]
    assert out["ETH/USD"][0].time.startswith("2021-01-02")
    assert session.calls[1]["page_token"] == "p2"
    assert session.calls[0]["timeframe"] == "1Day"


def test_http_error_raises():
    with pytest.raises(DataSourceError):
        CryptoPriceSource(session=FakeSession(FakeResponse({"message": "no"}, 500))).fetch(["BTC/USD"], "2021-01-01", "2021-01-02")


def test_closes_are_cached_and_synced_incrementally(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    first = FakeSession(FakeResponse({"bars": {"BTC/USD": [bar("2021-01-01", 1.0), bar("2021-01-02", 2.0)]}}))
    assert crypto_closes(store, CryptoPriceSource(session=first), ["BTC/USD"], "2021-01-01", "2021-01-02") == \
        {"BTC/USD": {"2021-01-01": 1.0, "2021-01-02": 2.0}}
    second = FakeSession(FakeResponse({"bars": {"BTC/USD": [bar("2021-01-03", 3.0)]}}))
    closes = crypto_closes(store, CryptoPriceSource(session=second), ["BTC/USD"], "2021-01-01", "2021-01-03")
    assert closes["BTC/USD"]["2021-01-03"] == 3.0
    assert second.calls[0]["start"].startswith("2021-01-03")           # only the missing day was fetched
    untouched = FakeSession()
    crypto_closes(store, CryptoPriceSource(session=untouched), ["BTC/USD"], "2021-01-01", "2021-01-03")
    assert untouched.calls == []                                       # fully cached
```

- [ ] **Step 2: Run** → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement** `hedge_fund/data/crypto_prices.py`:

```python
"""Daily crypto bars from Alpaca's market data API — free, no key needed.

BTC, ETH and SOL go back to 2021-01-01. Bars are UTC days. Pairs use
Alpaca's order/data form (`BTC/USD`); positions come back as `BTCUSD`, and
`pair()` maps either to the first. Bars are cached in the MarketStore
`prices` table under the pair name and synced incrementally.
"""

from __future__ import annotations

from datetime import date, timedelta

import requests

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.models import Price
from hedge_fund.data.store import MarketStore

CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"


def pair(symbol: str) -> str:
    """'BTCUSD', 'btc/usd' or 'BTC-USD' → 'BTC/USD'."""
    s = symbol.strip().upper().replace("-", "/")
    if "/" in s and s.endswith("/USD"):
        return s
    if "/" not in s and s.endswith("USD") and len(s) > 3:
        return f"{s[:-3]}/USD"
    raise ValueError(f"not a USD crypto pair: {symbol!r}")


class CryptoPriceSource:
    def __init__(self, *, timeout: float = 30.0, session: requests.Session | None = None) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()

    def fetch(self, symbols: list[str], start: str, end: str) -> dict[str, list[Price]]:
        """Daily bars in [start, end] (UTC days) per pair, oldest first."""
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": f"{start}T00:00:00Z",
                  "end": f"{end}T23:59:59Z", "limit": 10000, "sort": "asc"}
        out: dict[str, list[Price]] = {}
        while True:
            try:
                resp = self._session.get(CRYPTO_BARS_URL, params=dict(params), timeout=self._timeout)
            except requests.RequestException as exc:
                raise DataSourceError(f"Alpaca crypto bars: {exc}") from exc
            if resp.status_code >= 400:
                raise DataSourceError(f"Alpaca crypto bars: HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            for symbol, rows in (data.get("bars") or {}).items():
                out.setdefault(symbol, []).extend(
                    Price(open=r["o"], high=r["h"], low=r["l"], close=r["c"], volume=int(r.get("v") or 0), time=r["t"])
                    for r in rows)
            token = data.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token


def crypto_closes(store: MarketStore, source: CryptoPriceSource, symbols: list[str],
                  start: str, end: str) -> dict[str, dict[str, float]]:
    """Closes per pair per UTC day in [start, end], fetching only days not yet cached."""
    for symbol in symbols:
        synced = store.price_range(symbol)
        if synced and synced[0] <= start and synced[1] >= end:
            continue
        fetch_from = start if not synced or synced[0] > start else (date.fromisoformat(synced[1]) + timedelta(days=1)).isoformat()
        bars = source.fetch([symbol], fetch_from, end).get(symbol, [])
        store.upsert_prices(symbol, bars)
        first = min([start] + ([synced[0]] if synced else []))
        store.set_price_range(symbol, first, max(end, synced[1]) if synced else end)
    return {s: {p.time[:10]: p.close for p in store.prices(s, start, end)} for s in symbols}
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/data/crypto_prices.py hedge_fund/data/test_crypto_prices.py && git commit -m "Fetch and cache free Alpaca crypto daily bars"`

---

### Task 2: Pre-registered trend rules

**Files:** Create `hedge_fund/crypto/__init__.py`, `hedge_fund/crypto/rules.py`; Test `hedge_fund/crypto/test_rules.py`

- [ ] **Step 1: Failing tests**

```python
from hedge_fund.crypto.rules import RULES, ma100, ma_cross, mom12w


def test_ma100():
    assert ma100([1.0] * 99) == 0                       # not enough history: out
    assert ma100([1.0] * 99 + [2.0]) == 1
    assert ma100([2.0] * 99 + [1.0]) == 0


def test_mom12w():
    assert mom12w([1.0] * 84) == 0                      # needs 85 closes (an 84-day return)
    assert mom12w([1.0] * 84 + [1.1]) == 1
    assert mom12w([1.1] + [1.0] * 84) == 0


def test_ma_cross():
    assert ma_cross([1.0] * 199) == 0
    assert ma_cross([1.0] * 150 + [2.0] * 50) == 1      # 50-day avg 2.0 > 200-day avg 1.25
    assert ma_cross([2.0] * 150 + [1.0] * 50) == 0


def test_registry():
    assert set(RULES) == {"ma100", "mom12w", "ma_cross"}
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `hedge_fund/crypto/__init__.py`:

```python
"""Crypto sleeve: free daily data, pre-registered trend rules, backtests."""
```

`hedge_fund/crypto/rules.py`:

```python
"""Pre-registered trend rules (spec §2). Fixed textbook parameters, no tuning.

Each rule takes a coin's closes, oldest first, ending on the decision day,
and returns 1 (hold) or 0 (cash). Too little history means 0.
"""

from __future__ import annotations

from typing import Callable

Rule = Callable[[list[float]], int]


def _sma(closes: list[float], n: int) -> float:
    return sum(closes[-n:]) / n


def ma100(closes: list[float]) -> int:
    """Close above its 100-day simple moving average."""
    return int(len(closes) >= 100 and closes[-1] > _sma(closes, 100))


def mom12w(closes: list[float]) -> int:
    """Positive 84-day (12-week) return."""
    return int(len(closes) >= 85 and closes[-1] > closes[-85])


def ma_cross(closes: list[float]) -> int:
    """50-day average above the 200-day average."""
    return int(len(closes) >= 200 and _sma(closes, 50) > _sma(closes, 200))


RULES: dict[str, Rule] = {"ma100": ma100, "mom12w": mom12w, "ma_cross": ma_cross}
WARMUP_DAYS = 200   # the longest rule's history need; every variant starts after it
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/crypto && git commit -m "Add the pre-registered crypto trend rules"`

---

### Task 3: Backtest and the pass bar

**Files:** Create `hedge_fund/crypto/backtest.py`; Test `hedge_fund/crypto/test_backtest.py`

- [ ] **Step 1: Failing tests**

```python
from datetime import date, timedelta

import pytest

from hedge_fund.crypto.backtest import Metrics, metrics, passes_bar, run_backtest

WEIGHTS = {"A/USD": 0.5, "B/USD": 0.5}


def days(start, n):
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def flat(n=30, start="2024-06-03"):           # 2024-06-03 is a Monday
    return {s: {d: 100.0 for d in days(start, n)} for s in WEIGHTS}


def test_core_on_flat_prices_only_loses_the_first_fee():
    r = run_backtest(flat(), WEIGHTS, None, fee_bps=25, start="2024-06-03", end="2024-07-02", capital=1000.0)
    assert r.nav[0] == pytest.approx(1000.0 - 1000.0 * 0.0025)
    # The first fee leaves cash a little negative, so Mondays trim a few cents: nearly flat after.
    assert r.nav[-1] == pytest.approx(r.nav[0], rel=1e-4)
    assert r.fees == pytest.approx(2.5, rel=0.01)


def test_rule_off_holds_cash_and_is_never_charged():
    r = run_backtest(flat(), WEIGHTS, lambda closes: 0, fee_bps=25, start="2024-06-03", end="2024-07-02", capital=1000.0)
    assert r.nav == [1000.0] * len(r.nav) and r.fees == 0


def test_weekend_signal_executes_on_monday():
    closes = flat(14)
    for s in WEIGHTS:
        closes[s].update({d: 200.0 for d in days("2024-06-08", 9)})   # jumps Saturday 06-08
    seen = []

    def rule(c):
        seen.append(len(c))
        return int(c[-1] > 150)

    r = run_backtest(closes, WEIGHTS, rule, fee_bps=0, start="2024-06-03", end="2024-06-16", capital=1000.0)
    # Out until Mon 06-10 (decision from Sun 06-09's close), then in at 200: flat after.
    assert r.nav[r.dates.index("2024-06-10")] == pytest.approx(1000.0)
    assert r.trades_on("2024-06-08") == 0 and r.trades_on("2024-06-10") > 0


def test_metrics_and_bar():
    m = metrics(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 110.0, 99.0])
    assert m.total_return == pytest.approx(-0.01)
    assert m.max_drawdown == pytest.approx(0.1)

    def m(sharpe, dd):
        return Metrics(total_return=0.0, annualized_return=0.0, sharpe=sharpe, max_drawdown=dd)

    core = {"h1": m(1.0, 0.5), "h2": m(1.0, 0.5), "full": m(1.0, 0.6)}
    good = {"h1": m(1.2, 0.3), "h2": m(1.1, 0.3), "full": m(1.15, 0.4)}    # 0.4 ≤ 0.75 × 0.6
    assert passes_bar(good, core) is True
    assert passes_bar({**good, "h2": m(0.9, 0.3)}, core) is False         # loses one half
    assert passes_bar({**good, "full": m(1.15, 0.5)}, core) is False      # drawdown only 17% smaller
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/crypto/backtest.py`:

```python
"""Daily crypto backtest: a weighted basket, optionally gated by a trend rule.

Prices are UTC daily closes. Trading happens only on weekdays (the live run
does), at that day's close, using the rule's decision from the previous
day's close, so a weekend signal executes on Monday and nothing looks ahead.
The basket rebalances to target on each Monday and whenever a rule decision
changes. Fees are charged per side on traded notional.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Callable

from pydantic import BaseModel

Rule = Callable[[list[float]], int]


class Metrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe: float          # daily, × √365
    max_drawdown: float


class CryptoBacktest(BaseModel):
    dates: list[str]
    nav: list[float]
    fees: float
    traded: dict[str, float]          # date → traded notional

    def trades_on(self, day: str) -> float:
        return self.traded.get(day, 0.0)


def run_backtest(
    closes: dict[str, dict[str, float]], weights: dict[str, float], rule: Rule | None, *,
    fee_bps: float, start: str, end: str, capital: float = 10_000.0,
) -> CryptoBacktest:
    symbols = sorted(weights)
    all_days = sorted(set.intersection(*(set(closes[s]) for s in symbols)))
    days = [d for d in all_days if start <= d <= end]
    history = {s: [closes[s][d] for d in all_days] for s in symbols}
    index = {d: i for i, d in enumerate(all_days)}
    fee = fee_bps / 10_000
    units = dict.fromkeys(symbols, 0.0)
    cash = capital
    held_exposure: dict[str, int] | None = None
    nav: list[float] = []
    traded: dict[str, float] = {}
    total_fees = 0.0
    for day in days:
        i = index[day]
        if date.fromisoformat(day).weekday() < 5:          # weekdays only, like the live run
            exposure = {s: (1 if rule is None else (rule(history[s][:i]) if i > 0 else 0)) for s in symbols}
            if held_exposure is None or exposure != held_exposure or date.fromisoformat(day).weekday() == 0:
                value = cash + sum(units[s] * history[s][i] for s in symbols)
                notional = 0.0
                for s in symbols:
                    target = weights[s] * exposure[s] * value / history[s][i]
                    notional += abs(target - units[s]) * history[s][i]
                    cash -= (target - units[s]) * history[s][i]
                    units[s] = target
                cost = notional * fee
                cash -= cost
                total_fees += cost
                if notional:
                    traded[day] = notional
                held_exposure = exposure
        nav.append(cash + sum(units[s] * history[s][i] for s in symbols))
    return CryptoBacktest(dates=days, nav=nav, fees=round(total_fees, 6), traded=traded)


def metrics(dates: list[str], nav: list[float]) -> Metrics:
    total = nav[-1] / nav[0] - 1
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    annual = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else 0.0
    rets = [b / a - 1 for a, b in zip(nav, nav[1:])]
    mean = sum(rets) / len(rets) if rets else 0.0
    std = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0.0
    peak, dd = nav[0], 0.0
    for v in nav:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    return Metrics(total_return=total, annualized_return=annual,
                   sharpe=(mean / std * math.sqrt(365)) if std > 0 else 0.0, max_drawdown=dd)


def split_metrics(result: CryptoBacktest, split: str) -> dict[str, Metrics]:
    """Full-window metrics plus each half, split at `split` (first half ends on it)."""
    h1 = [i for i, d in enumerate(result.dates) if d <= split]
    h2 = [i for i, d in enumerate(result.dates) if d > split]
    pick = lambda idx: ([result.dates[i] for i in idx], [result.nav[i] for i in idx])
    return {"full": metrics(result.dates, result.nav), "h1": metrics(*pick(h1)), "h2": metrics(*pick(h2))}


def passes_bar(candidate: dict[str, Metrics], core: dict[str, Metrics]) -> bool:
    """Spec §3: Sharpe beats the core in both halves, and full-window drawdown ≥ 25% smaller."""
    return (candidate["h1"].sharpe > core["h1"].sharpe and candidate["h2"].sharpe > core["h2"].sharpe
            and candidate["full"].max_drawdown <= 0.75 * core["full"].max_drawdown)
```

- [ ] **Step 4: Run** → PASS (fix only genuine arithmetic mismatches in test expectations; note any in the commit).
- [ ] **Step 5: Commit** — `git add hedge_fund/crypto && git commit -m "Backtest crypto baskets with weekday execution and fees"`

---

### Task 4: `aihf-crypto backtest`

**Files:** Create `hedge_fund/crypto/cli.py`; Test `hedge_fund/crypto/test_cli.py`; Modify `hedge_fund/paths.py`, `pyproject.toml`

- [ ] **Step 1: Failing test**

```python
from hedge_fund.crypto.cli import CORE_WEIGHTS, SPLIT, build_parser, evaluate
from hedge_fund.crypto.test_backtest import days


def test_core_weights_and_parser():
    assert CORE_WEIGHTS == {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}
    assert build_parser().parse_args(["backtest"]).fee_bps == 25.0


def test_evaluate_runs_core_and_every_rule():
    import math
    closes = {s: {d: 100 * (1 + 0.3 * math.sin(i / 40)) for i, d in enumerate(days("2021-01-01", 1200))}
              for s in CORE_WEIGHTS}
    table = evaluate(closes, fee_bps=25, end=days("2021-01-01", 1200)[-1])
    assert [row["variant"] for row in table] == ["core", "ma100", "mom12w", "ma_cross"]
    assert table[0]["passes"] is None and all(isinstance(r["passes"], bool) for r in table[1:])
    assert table[0]["start"] == "2021-07-20"                       # first day + 200
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `hedge_fund/paths.py` add after `LIVE_SETTINGS_PATH`:

```python
CRYPTO_DIR = USER_DIR / "crypto"             # crypto backtests and the crypto paper lab
```

`hedge_fund/crypto/cli.py`:

```python
"""aihf-crypto — the crypto sleeve's research commands.

    aihf-crypto backtest [--fee-bps 25] [--end YYYY-MM-DD]

Backtests buy-and-hold 60/30/10 BTC/ETH/SOL against each pre-registered trend
rule on free Alpaca daily data since 2021, after fees, and applies the
spec's pre-committed bar. No money and no LLM are involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone

from hedge_fund.crypto.backtest import passes_bar, run_backtest, split_metrics
from hedge_fund.crypto.rules import RULES, WARMUP_DAYS
from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_closes
from hedge_fund.data.store import MarketStore
from hedge_fund.paths import CRYPTO_DIR, MARKET_DB_PATH

CORE_WEIGHTS = {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}
DATA_START = "2021-01-01"
SPLIT = "2023-06-30"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aihf-crypto", description="Crypto sleeve research.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("backtest", help="core vs pre-registered trend rules, after fees")
    p.add_argument("--fee-bps", type=float, default=25.0, help="per side; Alpaca level-1 taker is 25")
    p.add_argument("--end", default=None, help="last UTC day (default: yesterday)")
    return parser


def evaluate(closes: dict[str, dict[str, float]], *, fee_bps: float, end: str) -> list[dict]:
    first = min(min(c) for c in closes.values())
    start = (date.fromisoformat(first) + timedelta(days=WARMUP_DAYS)).isoformat()
    variants = {"core": None, **RULES}
    results = {name: run_backtest(closes, CORE_WEIGHTS, rule, fee_bps=fee_bps, start=start, end=end)
               for name, rule in variants.items()}
    split = {name: split_metrics(r, SPLIT) for name, r in results.items()}
    rows = []
    for name, r in results.items():
        m = split[name]
        rows.append({
            "variant": name, "start": r.dates[0], "end": r.dates[-1],
            "total_return": m["full"].total_return, "annualized": m["full"].annualized_return,
            "sharpe": m["full"].sharpe, "sharpe_h1": m["h1"].sharpe, "sharpe_h2": m["h2"].sharpe,
            "max_drawdown": m["full"].max_drawdown, "fees_pct": r.fees / r.nav[0],
            "passes": None if name == "core" else passes_bar(m, split["core"]),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    closes = crypto_closes(MarketStore(MARKET_DB_PATH), CryptoPriceSource(), list(CORE_WEIGHTS), DATA_START, end)
    table = evaluate(closes, fee_bps=args.fee_bps, end=end)
    print(f"crypto backtest {table[0]['start']} → {table[0]['end']} · 60/30/10 BTC/ETH/SOL · "
          f"{args.fee_bps:g} bps per side · halves split {SPLIT}")
    print(f"{'variant':10} {'total':>9} {'annual':>8} {'sharpe':>7} {'1st half':>9} {'2nd half':>9} {'max dd':>7} {'fees':>6}  passes bar?")
    for r in table:
        verdict = {None: "-", True: "YES", False: "no"}[r["passes"]]
        print(f"{r['variant']:10} {r['total_return']:>+9.0%} {r['annualized']:>+8.1%} {r['sharpe']:>7.2f} "
              f"{r['sharpe_h1']:>9.2f} {r['sharpe_h2']:>9.2f} {r['max_drawdown']:>7.1%} {r['fees_pct']:>6.1%}  {verdict}")
    out = CRYPTO_DIR / "backtests" / f"{end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fee_bps": args.fee_bps, "split": SPLIT, "rows": table}, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`pyproject.toml` `[tool.poetry.scripts]`: add `aihf-crypto = "hedge_fund.crypto.cli:main"`, then `uv pip install --python .venv/bin/python -e .`

- [ ] **Step 4: Run** → PASS; `.venv/bin/aihf-crypto --help` works.
- [ ] **Step 5: Commit** — `git add hedge_fund/crypto hedge_fund/paths.py pyproject.toml && git commit -m "Add aihf-crypto backtest"`

---

### Task 5: Run it and record the verdict

- [ ] **Step 1:** `.venv/bin/aihf-crypto backtest` (free data; no keys, no money, no LLM).
- [ ] **Step 2:** Append "## 11. Phase 1 results (date)" to the spec with the table and the verdict by the pre-committed bar; commit.
- [ ] **Step 3:** If a rule passes → write the Phase 2 plan (overlay + paper lab). If none passes → the crypto half is buy-and-hold only; Phase 2 is the live integration with `crypto_trend_share` fixed at 0.
