# Crypto Phase 2 — Live Crypto Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The live account holds a stock half and a buy-and-hold 60/30/10 BTC/ETH/SOL crypto half (`crypto_share`), with deposits split by target, without the stock logic ever selling, capping or halting on crypto.

**Architecture:** `LiveSettings` gains `crypto_share` and `crypto_core`. `account_book()` scales the existing stock book by (1 − crypto_share) and adds the crypto core × crypto_share. The runner prices crypto via an injected `crypto_marks` function (free Alpaca crypto closes) and keeps crypto out of satellite valuation, returns and the halt check. `AlpacaLiveClient` normalizes crypto symbols and sends crypto orders as `gtc`. The trend sleeve stays at 0 (spec §11).

**Spec:** `docs/superpowers/specs/2026-10-02-crypto-sleeve-design.md` (§6, §11)

**Test command:** `env FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest <path> -q`

---

### Task 1: Live client — crypto symbols and `gtc`

**Files:** Modify `hedge_fund/brokers/alpaca.py`; Test `hedge_fund/brokers/test_alpaca_live.py`

- [ ] **Step 1: Failing tests** — append:

```python
def test_crypto_positions_use_pair_names():
    client, _ = live(FakeResponse(payload=[
        {"symbol": "BTCUSD", "qty": "0.0011", "side": "long", "asset_class": "crypto"},
        {"symbol": "SPY", "qty": "0.2", "side": "long", "asset_class": "us_equity"},
    ]))
    assert client.holdings() == {"BTC/USD": pytest.approx(0.0011), "SPY": pytest.approx(0.2)}


def test_crypto_orders_are_good_til_cancelled():
    order = {"id": "o", "client_order_id": "c", "symbol": "BTC/USD", "side": "buy", "qty": None,
             "notional": "30.00", "filled_qty": "0", "filled_avg_price": None, "status": "accepted"}
    client, session = live(FakeResponse(payload=order), FakeResponse(payload={**order, "symbol": "SPY"}))
    client.buy_notional("BTC/USD", 30.0, "c")
    client.buy_notional("SPY", 30.0, "c")
    assert session.calls[0]["json"]["time_in_force"] == "gtc"
    assert session.calls[1]["json"]["time_in_force"] == "day"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — in `hedge_fund/brokers/alpaca.py` add `from hedge_fund.data.crypto_prices import pair` to the imports; in `AlpacaLiveClient.holdings` replace the assignment with:

```python
            if qty > 1e-9:
                symbol = pair(row["symbol"]) if row.get("asset_class") == "crypto" else row["symbol"]
                held[symbol] = -qty if row.get("side") == "short" else qty
```

and in `_send` replace the body line with:

```python
        # Alpaca rejects "day" for crypto; pairs ("BTC/USD") go good-til-cancelled.
        tif = "gtc" if "/" in body["symbol"] else "day"
        body = {**body, "type": "market", "time_in_force": tif, "client_order_id": client_order_id}
```

- [ ] **Step 4: Run** `hedge_fund/brokers` → PASS. **Step 5: Commit** "Normalize live crypto symbols and send crypto orders good-til-cancelled".

---

### Task 2: Settings and the account book

**Files:** Modify `hedge_fund/live/account/settings.py`, `hedge_fund/live/account/target.py`; Test `hedge_fund/live/account/test_settings.py`, `hedge_fund/live/account/test_target.py`

- [ ] **Step 1: Failing tests** — append to `test_settings.py`:

```python
def test_crypto_defaults_and_validation():
    s = LiveSettings()
    assert s.crypto_share == 0.0
    assert s.crypto_core == {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}
    with pytest.raises(ValueError):
        LiveSettings(crypto_share=0.6)
    with pytest.raises(ValueError, match="sum to 1"):
        LiveSettings(crypto_core={"BTC/USD": 0.5})
    with pytest.raises(ValueError, match="pair"):
        LiveSettings(crypto_core={"BTCUSD": 1.0})
```

append to `test_target.py`:

```python
from hedge_fund.live.account.settings import LiveSettings
from hedge_fund.live.account.target import account_book


def test_account_book_splits_halves():
    book = account_book({"SPY": 0.8, "NVDA": 0.2}, LiveSettings(crypto_share=0.5))
    assert book == {"SPY": pytest.approx(0.4), "NVDA": pytest.approx(0.1), "BTC/USD": pytest.approx(0.3),
                    "ETH/USD": pytest.approx(0.15), "SOL/USD": pytest.approx(0.05)}
    assert sum(book.values()) == pytest.approx(1.0)


def test_no_crypto_share_leaves_the_stock_book_alone():
    assert account_book({"SPY": 1.0}, LiveSettings()) == {"SPY": 1.0}
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — `settings.py`: import `field_validator` from pydantic, and add to `LiveSettings`:

```python
    crypto_share: float = Field(default=0.0, ge=0, le=0.5, description="fraction of the account in the crypto half")
    crypto_core: dict[str, float] = Field(
        default_factory=lambda: {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1},
        description="crypto half's buy-and-hold weights (Alpaca pairs)")

    @field_validator("crypto_core")
    @classmethod
    def _crypto_weights(cls, core: dict[str, float]) -> dict[str, float]:
        if any("/" not in t for t in core):
            raise ValueError("crypto_core keys must be Alpaca pairs like 'BTC/USD'")
        if any(w <= 0 for w in core.values()) or abs(sum(core.values()) - 1) > 1e-6:
            raise ValueError("crypto_core weights must be positive and sum to 1")
        return core
```

`target.py`: add `from hedge_fund.live.account.settings import LiveSettings` and:

```python
def account_book(stock_book: dict[str, float], settings: LiveSettings) -> dict[str, float]:
    """The whole account: the stock book scaled to (1 − crypto_share), plus the crypto core."""
    share = settings.crypto_share
    if share <= 0:
        return stock_book
    book = {t: w * (1 - share) for t, w in stock_book.items()}
    for t, w in settings.crypto_core.items():
        book[t] = book.get(t, 0.0) + w * share
    return book
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** "Add the crypto half to live settings and the account book".

---

### Task 3: Limits exempt the core names

**Files:** Modify `hedge_fund/live/account/orders.py`, `hedge_fund/live/account/test_orders.py`

- [ ] **Step 1: Failing test** — append:

```python
def test_crypto_core_names_are_exempt_from_the_satellite_cap():
    marks = {**MARKS, "BTC/USD": 60_000.0}
    check_orders([PlannedOrder(ticker="BTC/USD", side="buy", dollars=30.0)], {}, 100.0, marks,
                 {"SPY", "BTC/USD"}, shorts_ok=False, max_name=0.1)
```

- [ ] **Step 2: Run** → FAIL (a string `core_ticker` can't hold two names). **Step 3: Implement** — in `check_orders` rename the parameter `core_ticker: str` to `exempt: set[str]` and the condition `t != core_ticker` to `t not in exempt`; update the docstring ("names in `exempt` — the stock core and the crypto core — skip the per-name cap"). In `test_orders.py` replace each positional `"SPY"` argument of `check_orders(...)` with `{"SPY"}`.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** "Exempt every core name from the live per-name cap".

---

### Task 4: Runner — crypto marks, crypto kept out of the satellite

**Files:** Modify `hedge_fund/live/account/ledger.py`, `hedge_fund/live/account/runner.py`, `hedge_fund/live/account/report.py`; Test `hedge_fund/live/account/test_runner.py`

- [ ] **Step 1: Failing tests** — append to `test_runner.py`:

```python
CRYPTO = {"BTC/USD": 60_000.0, "ETH/USD": 3_000.0, "SOL/USD": 150.0}


def crypto_marks(names, day):
    return {n: CRYPTO[n] for n in names}


def test_deposit_is_split_between_the_halves(env):
    env["live"].mark_dry_run_done()
    client = FakeLive(cash=200.0)
    s = LiveSettings(confirm_live=True, crypto_share=0.5, cash_buffer_pct=0.0)
    result = submit_live(s, client, FakeDataClient(CLOSES), env["live"], env["paper"], now=MON,
                         kill_path=env["kill"], crypto_marks=crypto_marks)
    bought = {t: kw["notional"] for t, side, kw in client.sent}
    assert bought == {"SPY": 100.0, "BTC/USD": 60.0, "ETH/USD": 30.0, "SOL/USD": 10.0}
    assert result.target["BTC/USD"] == pytest.approx(0.3)


def test_rebalance_never_sells_crypto_held_at_target(env):
    env["live"].mark_dry_run_done()
    holdings = {"SPY": 0.2, "BTC/USD": 0.001, "ETH/USD": 0.01, "SOL/USD": 0.0666667}   # $100 SPY + $100 crypto 60/30/10
    client = FakeLive(holdings=holdings, cash=0.0)
    s = LiveSettings(confirm_live=True, crypto_share=0.5)
    run_result = submit_live(s, client, FakeDataClient(CLOSES), env["live"], env["paper"], now=MON,
                             kill_path=env["kill"], crypto_marks=crypto_marks)
    assert run_result.status == "nothing_to_do" and client.sent == []


def test_crypto_without_a_price_source_is_an_error(env):
    env["live"].mark_dry_run_done()
    with pytest.raises(ValueError, match="crypto"):
        submit_live(LiveSettings(confirm_live=True, crypto_share=0.5), FakeLive(cash=100.0), FakeDataClient(CLOSES),
                    env["live"], env["paper"], now=MON, kill_path=env["kill"])


def test_reconcile_values_crypto_separately_from_the_satellite(env):
    live = env["live"]
    client = FakeLive(holdings={"SPY": 0.2, "BTC/USD": 0.001}, cash=0.0)
    live.save_holdings("2026-08-20", {"SPY": 0.2, "BTC/USD": 0.001})
    row = reconcile_live(LiveSettings(crypto_share=0.5), client, FakeDataClient(CLOSES), live,
                         session="2026-08-21", crypto_marks=crypto_marks)
    assert (row.core_value, row.crypto_value, row.satellite_value) == (100.0, 60.0, 0.0)
    assert row.satellite_return is None             # crypto is not the satellite
    assert row.equity == 160.0
```

(`CLOSES` must price SPY on 2026-08-20/21 — it already does.)

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement:**

`ledger.py` — add to `LiveNavRow` after `satellite_value`: `crypto_value: float = 0.0`.

`report.py` — add `crypto_value: float` to `LiveReport` and `crypto_value=last.crypto_value,` in `build_live_report`.

`runner.py`:
- imports: `from typing import Callable, Literal`; `from hedge_fund.live.account.target import account_book, latest_paper_weights, satellite_weights, target_book`.
- add above `reconcile_live`:

```python
CryptoMarks = Callable[[list[str], str], dict[str, float]]


def _is_crypto(ticker: str) -> bool:
    return "/" in ticker


def _marks(names, day: str, data_client: DataClient, crypto_marks: CryptoMarks | None) -> dict[str, float]:
    """Stock closes from the data client; crypto pairs from crypto_marks (UTC daily closes)."""
    stocks = sorted(n for n in names if not _is_crypto(n))
    crypto = sorted(n for n in names if _is_crypto(n))
    marks = exact_marks(stocks, day, data_client) if stocks else {}
    if crypto:
        if crypto_marks is None:
            raise ValueError(f"crypto holdings or targets {crypto} need a crypto price source")
        marks.update(crypto_marks(crypto, day))
    return marks
```

- `reconcile_live(..., *, session, crypto_marks: CryptoMarks | None = None)`: replace `closes = exact_marks(sorted(set(holdings) | {core}), session, data_client)` with `closes = _marks(set(holdings) | {core}, session, data_client, crypto_marks)`; compute
  `crypto_value = sum(q * closes[t] for t, q in holdings.items() if _is_crypto(t))` and change `satellite_value` and the satellite-return `sat` dict to exclude crypto (`t != core and not _is_crypto(t)`); add `crypto_value=round(crypto_value, 2)` to the row and include it in equity: `equity = cash + core_value + satellite_value + crypto_value`.
- `submit_live(..., kill_path=KILL_PATH, crypto_marks: CryptoMarks | None = None)`:
  - names: `names = sorted(set(holdings) | {core} | wanted | (set(settings.crypto_core) if settings.crypto_share > 0 else set()))`
  - marks: `marks = _marks(names, mark_day, data_client, crypto_marks)`
  - target: `target = account_book(target_book(satellite, core), settings)`
  - satellite scaling stays relative to the stock half (account_book scales it).
  - limits: `check_orders(..., {core} | set(settings.crypto_core), shorts_ok=..., max_name=...)`.

- [ ] **Step 4: Run** `hedge_fund/live/account` → PASS. **Step 5: Commit** "Hold a crypto half in the live account without touching the satellite rules".

---

### Task 5: CLI wiring and docs

**Files:** Modify `hedge_fund/live/account/cli.py`, `README.md`

- [ ] **Step 1:** In `cli.py` add imports `from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_closes`, `from hedge_fund.data.store import MarketStore`, `MARKET_DB_PATH` from paths, and:

```python
def _crypto_marks(names: list[str], day: str) -> dict[str, float]:
    closes = crypto_closes(MarketStore(MARKET_DB_PATH), CryptoPriceSource(), names, day, day)
    missing = [n for n in names if day not in closes[n]]
    if missing:
        raise ValueError(f"no crypto close on {day} for {', '.join(missing)}")
    return {n: closes[n][day] for n in names}
```

Pass `crypto_marks=_crypto_marks` to `reconcile_live` and `submit_live`; print `crypto ${row.crypto_value:,.2f}` in `_reconcile`; show `crypto half {settings.crypto_share:.0%}` in `_status`.

- [ ] **Step 2:** README "Real-money account" section: add a paragraph — set `crypto_share: 0.5` in `live.yaml` to hold half the account as buy-and-hold 60/30/10 BTC/ETH/SOL; deposits split by target; no crypto trend rule passed its backtest (spec §11), so the crypto half is never traded beyond rebalancing; crypto can fall ~80% (it did in 2021–22).

- [ ] **Step 3: Run** full suite → PASS; `.venv/bin/aihf-live --help`. **Step 4: Commit** "Wire crypto prices into aihf-live and document the crypto half"; push.
