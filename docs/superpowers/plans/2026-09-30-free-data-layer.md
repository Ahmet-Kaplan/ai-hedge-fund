# Free Data Layer Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Replace Financial Datasets with Alpaca prices + SEC EDGAR fundamentals behind the existing `DataClient` protocol, stored in SQLite, with resumable baselines.

**Spec:** `docs/superpowers/specs/2026-09-30-free-data-layer-design.md`

**Architecture:** `MarketStore` (SQLite, no network) ← filled by `AlpacaPriceSource` and `SecSource` (network) ← read by pure `fundamentals.py` functions ← exposed by `FreeDataClient` (a `DataClient`). `open_data_client()` picks free vs Financial Datasets for every entry point.

**Tech:** Python 3.12 stdlib `sqlite3`, `requests`, pydantic v2, pytest. No new dependencies.

**Test command:** `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest hedge_fund -q` (blank key keeps the live Financial Datasets tests skipped).

**Conventions:** tests beside code as `test_*.py`; every network class takes an injectable `session`; every time-dependent function takes `now`/`today`; commit after each task with the Co-Authored-By trailer.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `hedge_fund/paths.py` | modify | `MARKET_DB_PATH` |
| `hedge_fund/data/store.py` | create | SQLite schema + queries |
| `hedge_fund/data/alpaca_prices.py` | create | daily bars → store |
| `hedge_fund/data/sec.py` | create | EDGAR client, ticker/CIK, SIC→sector, → store |
| `hedge_fund/data/fundamentals.py` | create | facts → TTM metrics rows; quarterly EPS; SUE events |
| `hedge_fund/data/free.py` | create | `FreeDataClient` (DataClient) with lazy sync |
| `hedge_fund/data/factory.py` | create | `open_data_client()` |
| `hedge_fund/data/__init__.py` | modify | exports |
| `hedge_fund/backtesting/checkpoint.py` | create | stored baseline variant results |
| `hedge_fund/live/baseline.py`, `hedge_fund/live/cli.py` | modify | checkpoints, `--fresh`, `data-sync`, factory |
| `hedge_fund/run.py`, `hedge_fund/tui/app.py`, `hedge_fund/event_study/__main__.py` | modify | factory instead of `FDClient()` |
| `hedge_fund/tui/keys.py` (or its caller) | modify | FD key required only when `HEDGE_FUND_DATA=fd` |
| `hedge_fund/features/snapshot.py` | modify | `MIN_PERIODS = 4` |
| `.env.example`, `README.md` | modify | `SEC_USER_AGENT`, `HEDGE_FUND_DATA` |

---

### Task 1: `MarketStore`

Schema exactly as spec §2 (`prices`, `price_sync`, `sec_companies`, `sec_facts`, `sec_filings`, `sec_sync`, `backtest_runs`), WAL mode, `MARKET_DB_PATH = USER_DIR / "market.db"`.

API:
- `upsert_prices(ticker, bars: list[Price])`, `prices(ticker, start, end) -> list[Price]` (date-sorted), `price_range(ticker) -> (first, through) | None`, `set_price_range(ticker, first, through)`.
- `upsert_company(ticker, cik, name, sic, sic_description)`, `company(ticker) -> dict | None`.
- `replace_facts(cik, rows)` (delete + insert for that CIK, one transaction), `facts(cik, concepts, filed_lte) -> dict[concept, list[Fact]]`.
- `replace_filings(cik, rows)`, `filings(cik, forms) -> list[dict]`.
- `sync_times(cik) -> (facts_at, submissions_at)`, `mark_synced(cik, facts_at=..., submissions_at=...)`.
- `get_run(run_key)`, `put_run(run_key, label, status, result_json, error)`.

`Fact` = frozen dataclass `(concept, unit, start, end, value, fy, fp, form, filed, accn)`.

Tests (`hedge_fund/data/test_store.py`, `tmp_path` DB): price upsert idempotent + range query; facts filtered by `filed <= date`; `replace_facts` removes stale rows; run round-trip.

### Task 2: `AlpacaPriceSource`

`fetch(tickers, start, end) -> dict[ticker, list[Price]]`: `GET https://data.alpaca.markets/v2/stocks/bars`, params `symbols` (comma-joined, ≤ 100 per request), `timeframe=1Day`, `start`, `end`, `adjustment=all`, `feed=sip`, `limit=10000`, follows `next_page_token`. Keys from `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY`. Bar → `Price(open, high, low, close, volume, time=f"{date}T00:00:00Z")`. HTTP ≥ 400 → `DataSourceError`.

Tests: pagination merges pages; multi-symbol split; error raises; bar mapping.

### Task 3: `SecSource`

- Requires `SEC_USER_AGENT` (constructor raises `ValueError` naming the variable). Header `User-Agent`. Throttle: ≥ 0.125 s between requests (injectable `sleep`/`clock`). Retries 429/5xx with backoff (1, 2, 4 s), then `DataSourceError`.
- `sec_ticker(t)`: `BRK.B → BRK-B` (dots to dashes, upper).
- `ticker_map() -> dict[ticker, cik]` from `company_tickers.json`.
- `sync_company(ticker, store, now)`: submissions (name, sic, sicDescription, recent filings: form, filingDate, accessionNumber, reportDate, items) + companyfacts restricted to the concept list in `fundamentals.CONCEPTS` (us-gaap and dei) → store; `mark_synced`.
- `sic_sector(sic: int) -> str` by SIC division ranges (Agriculture 100–999, Mining 1000–1499, Construction 1500–1799, Manufacturing 2000–3999, Transportation & Utilities 4000–4999, Wholesale 5000–5199, Retail 5200–5999, Finance 6000–6799, Services 7000–8999, Public Administration 9100–9729).

Tests: missing UA raises; ticker normalization; throttle spacing with fake clock; companyfacts → only listed concepts stored with `filed`; submissions → company + filings rows; 429 then 200 retries.

### Task 4: `fundamentals.py` — TTM metrics

Pure functions over `dict[concept, list[Fact]]` (already PIT-filtered by the store query).

- `CONCEPTS`: revenue candidates, GrossProfit, OperatingIncomeLoss, NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities, PaymentsToAcquirePropertyPlantAndEquipment, equity candidates, AssetsCurrent, LiabilitiesCurrent, debt parts (LongTermDebtNoncurrent, LongTermDebtCurrent, LongTermDebt, CommercialPaper, ShortTermBorrowings), WeightedAverageNumberOfDilutedSharesOutstanding, EarningsPerShareDiluted, dei EntityCommonStockSharesOutstanding.
- `latest_by_period(facts) -> dict[(start, end), Fact]`: latest `filed` per period (restatement-aware); `original_by_period` = earliest filed.
- Durations: quarter 80–100 days, FY 350–380 days, YTD otherwise ≤ 290.
- `ttm(facts, end) -> float | None`: FY fact ending at `end` (±3 days) else `YTD(end) + FY(ending start_of_YTD − 1 day ± 7) − YTD(ending end − 1 year ± 7, same length ± 10)`.
- `instant(facts, end)`: instant fact at `end` (±3 days).
- `quarter_ends(facts_by_concept) -> list[(end, filing_date)]`: distinct `end` of 10-Q/10-K duration facts on revenue or net income; `filing_date` = earliest `filed` for that `end`; newest first.
- `metrics_rows(ticker, facts_by_concept, close_on_or_before, limit) -> list[FinancialMetrics]` per spec §4 derivations; per-period concept fallback (first candidate with a value for that period).

Tests (`test_fundamentals.py`, synthetic facts builder): FY row uses FY fact; mid-year row uses YTD formula (cash-flow style facts only YTD); restated value picked by latest filed; ratios hand-computed (market cap, P/E, ROE, margins, D/E, current ratio, growth, EPS, BVPS, FCF/sh); bank-style company with no GrossProfit → `gross_margin is None`; negative NI → `price_to_earnings_ratio is None`; `limit` respected; newest first.

### Task 5: `fundamentals.py` — quarterly EPS and SUE events

- `quarterly_values(facts) -> dict[end, value]` from ORIGINAL filings: 3-month facts; Q4 = FY − 9-month YTD of the same fiscal year.
- `earnings_events(ticker, eps_facts, filings, limit) -> list[EarningsRecord]`: for each quarter end `E` with ≥ 4 prior YoY changes (up to 8), `SUE = (q(E) − q(E−4)) / stdev(prior YoY changes)`; BEAT ≥ 1.0, MISS ≤ −1.0, else MEET; stdev 0 → skip. Date = earliest 8-K with item `2.02` filed 0–60 days after `E`, else earliest 10-Q/10-K filed for `E` (source_type = that form). Newest first, `limit`.

Tests: Q4 derivation; SUE BEAT/MISS/MEET thresholds hand-computed; 8-K matching window; fallback to 10-Q; zero-variance skip; output feeds `PEADModel` and fires on the event date.

### Task 6: `FreeDataClient`

Constructor: `(store, prices: AlpacaPriceSource, sec: SecSource, today=None)`; context manager (`__enter__/__exit__` close store).

- `get_prices(t, start, end)`: end capped at `completed_through()`; if stored range doesn't cover `[start, end]`, fetch the missing head/tail (one call for the gap) and extend range; return stored.
- `prefetch_prices(tickers, start, end)`: one multi-symbol call for tickers whose range is short.
- `_ensure_sec(t, as_of)`: sync if never synced, or (stored older than 20 h and `as_of >= stored fetch date`). Sync failure + stored data → warning and use stored; no stored → raise.
- `get_financial_metrics(t, end_date, period="ttm", limit=10)`: `period != "ttm"` → `ValueError`; facts filed ≤ end_date; prices ensured from 2016-01-01 to end_date for market-cap lookups; `metrics_rows`.
- `get_company_facts(t)` → `CompanyFacts(ticker, name, cik, sector=sic_sector, industry=sic_description, sic_code)`.
- `get_earnings_history(t, limit=12)` → `earnings_events`.
- `get_market_cap(t, end_date)` → latest row's market cap or None. `get_news`/`get_insider_trades` → `[]`; `get_earnings` → None.

Tests with fake sources counting calls: second `get_prices` inside range makes no call; extending end fetches only the tail; SEC not refetched for a past `as_of`; refetched when stale for today; offline with stored data warns and serves; offline without data raises; satisfies `isinstance(client, DataClient)`.

### Task 7: Factory + wiring

`open_data_client()` (contextmanager): `HEDGE_FUND_DATA` = `free` (default) → `FreeDataClient(MarketStore(MARKET_DB_PATH), AlpacaPriceSource(), SecSource())`; `fd` → `CachedDataClient(FDClient())`; other → `ValueError`. Replace every `with FDClient() as raw: … CachedDataClient(raw)` in `live/cli.py`, `run.py`, `tui/app.py`, `event_study/__main__.py`. Where the TUI demands a Financial Datasets key, require it only in `fd` mode.

Tests: factory picks by env; unknown value raises; CLI test that `submit` path calls the factory (monkeypatched).

### Task 8: `aihf-paper data-sync`

Syncs prices (2016-01-01 → completed_through) for universe + benchmark in one prefetch, SEC for each ticker; prints per ticker: price range, fundamentals rows available now, earliest filing, earnings events count. Test: output lines with fake client.

### Task 9: Baseline checkpoints

`run_key` = sha256(spec JSON, start, end, universe, `HEDGE_FUND_LLM_MODEL`). `run_baseline(..., store=None, fresh=False)`: per variant, reuse `done` results unless `fresh`; on exception mark `failed` with the message and continue to the next variant; report includes `failed: dict[name, error]`; if the fund or equal-weight variant failed, verdict flags are None. CLI `--fresh`. Tests: second run doesn't call backtest for finished variants; a failing variant is recorded and others still run; `--fresh` recomputes.

### Task 10: `MIN_PERIODS = 4`

Revert Task-level change from the Financial Datasets workaround (snapshot comment + tests back to 4) — only after Task 11 shows ≥ 4 periods for the universe.

### Task 11: Parity check vs Financial Datasets archive (manual, needs `SEC_USER_AGENT`)

Scratch script: for each `financial_metrics` response in `~/.hedge-fund/backups/financialdatasets-archive-2026-09-30.db`, rebuild rows with `FreeDataClient` at the same `end_date` and compare by `report_period`: filing date exact; margins/ROE/D-E within 2 pp; P/E, market cap within 5%. Print a per-field pass rate and the worst offenders; fix concept mappings until ≥ 90% of comparable values pass, and document known definitional gaps (banks, Berkshire) in `fundamentals.py`.

### Task 12: Docs + default switch

`.env.example`: `SEC_USER_AGENT`, `HEDGE_FUND_DATA`. README: data section (free default, how to use Financial Datasets). Full suite, push, update PR.
