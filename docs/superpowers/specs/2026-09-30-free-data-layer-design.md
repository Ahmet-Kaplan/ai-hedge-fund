# Free Data Layer + Local Market Database — Design

Date: 2026-09-30 · Status: design approved in chat, pending spec review

## Goal

Replace Financial Datasets (paid) with free sources — **Alpaca market data**
(daily prices, already authorized by the paper keys) and **SEC EDGAR** (filed
fundamentals, filings, industry codes; no key) — behind the existing
`DataClient` protocol, stored in a local **SQLite** database so data is fetched
once and never paid for or re-downloaded. Make long runs resumable so a crash
(e.g. LLM credit exhaustion) never loses finished work. The LLM becomes the only
paid dependency.

Verified in this session:

- Alpaca `GET data.alpaca.markets/v2/stocks/bars?timeframe=1Day&feed=sip&adjustment=all`
  with the paper keys returns 2016-01-04 → 2026-09-29 for SPY, AAPL, BRK.B in one
  call; closes match Financial Datasets (SPY 764.20, AAPL 329.40 on 2026-09-29).
- SEC `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` for Apple: 503
  us-gaap concepts, filed 2009-07-22 → 2026-07-31, every fact carrying `filed`
  (point-in-time) and `start`/`end`/`fp`/`form`/`accn`.
- SEC `data.sec.gov/submissions/CIK##########.json`: SIC code + description and
  8-K item lists (item 2.02 = results of operations → earnings announcement date).
- SEC `www.sec.gov/files/company_tickers.json`: ticker → CIK (uses `BRK-B`, not `BRK.B`).

## Decisions (from chat)

| Decision | Choice |
|---|---|
| PEAD surprise | Year-over-year standardized unexpected earnings (SUE), from SEC EPS |
| LLM re-assessment | Unchanged: every agent re-reads every name every rebalance |
| SEC identity | `SEC_USER_AGENT="Name email"` in `.env`, set by the user; client refuses without it |
| Default provider | Free (`HEDGE_FUND_DATA=free`); Financial Datasets stays selectable (`fd`) |

## 1. Components

| Unit | Responsibility |
|---|---|
| `hedge_fund/data/store.py` · `MarketStore` | SQLite at `~/.hedge-fund/market.db`: schema, upserts, point-in-time queries. No network. |
| `hedge_fund/data/alpaca_prices.py` · `AlpacaPriceSource` | Fetch daily bars (multi-symbol, paginated) into the store. |
| `hedge_fund/data/sec.py` · `SecSource` | Rate-limited (≤ 8 req/s) EDGAR client: ticker→CIK map, companyfacts, submissions → store. |
| `hedge_fund/data/fundamentals.py` | Pure functions: SEC facts → point-in-time TTM `FinancialMetrics` rows; quarterly EPS; SUE earnings events. |
| `hedge_fund/data/free.py` · `FreeDataClient` | Implements `DataClient` over the store; syncs lazily (below). |
| `hedge_fund/data/factory.py` · `open_data_client()` | Context manager: `free` → `FreeDataClient`, `fd` → `CachedDataClient(FDClient())`. All entry points use it. |
| `hedge_fund/backtesting/checkpoint.py` | Persist finished backtests and partial progress in the same DB. |

## 2. Database schema (SQLite, WAL mode)

```sql
prices(ticker, date, open, high, low, close, volume, PRIMARY KEY(ticker, date))
price_sync(ticker PRIMARY KEY, first_date, synced_through)        -- inclusive range held
sec_companies(ticker PRIMARY KEY, cik, name, sic, sic_description)
sec_facts(cik, concept, unit, start, end, value, fy, fp, form, filed, accn,
          PRIMARY KEY(cik, concept, unit, start, end, accn))
sec_filings(cik, accn PRIMARY KEY, form, filed, report_date, items)
sec_sync(cik PRIMARY KEY, facts_fetched_at, submissions_fetched_at)
backtest_runs(run_key PRIMARY KEY, label, status, started_at, finished_at, result_json)
```

`run_key` = sha256 of (spec JSON, start, end, universe, LLM model).

## 3. Sync policy (lazy, incremental)

- **Prices:** `get_prices(t, start, end)` first ensures the store holds
  `[min(start, first_date), end]` for `t` (bounded by `completed_through()`),
  fetching only the missing tail/head. One request can cover many tickers:
  `FreeDataClient.prefetch_prices(tickers, start, end)` is called by
  `backtest_fund`/`submit` callers up front so a backtest makes ~1 price request.
- **SEC:** companyfacts + submissions for a CIK are refetched only if the stored
  copy is older than 20 hours **and** the requested `as_of` is on/after the fetch
  date (a backtest as of July never refetches). Ticker→CIK map refreshed weekly.
- All reads after sync come from SQLite. Network failure with a usable stored
  copy → log a warning and use it; with no stored copy → raise (fail loud).

## 4. Fundamentals from SEC facts (`fundamentals.py`)

Point-in-time rule: for `end_date`, only facts with `filed <= end_date`; for each
(concept, start, end) take the **latest filed** value (restatement-aware).

Quarter ends are the `end` dates of 10-Q/10-K duration facts on the revenue or
net-income concept. For each quarter end `E` (newest first, `limit` rows):

- **TTM duration value** (income and cash-flow concepts):
  a ~365-day fact ending at `E` if present (fiscal year end), else
  `YTD(E) + FY(prior) − YTD(E − 1 year)` — standard TTM, correct for cash-flow
  statements that 10-Qs report only year-to-date.
- **Instant value** at `E` (balance sheet).
- `filing_date` = earliest `filed` of the 10-Q/10-K carrying period `E`.

Concepts (first present wins):

| Field | us-gaap concepts |
|---|---|
| revenue | RevenueFromContractWithCustomerExcludingAssessedTax, Revenues, SalesRevenueNet, RevenuesNetOfInterestExpense |
| gross profit | GrossProfit |
| operating income | OperatingIncomeLoss |
| net income | NetIncomeLoss |
| operating cash flow | NetCashProvidedByUsedInOperatingActivities |
| capex | PaymentsToAcquirePropertyPlantAndEquipment |
| equity | StockholdersEquity, StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest |
| current assets / liabilities | AssetsCurrent / LiabilitiesCurrent |
| debt | LongTermDebtNoncurrent + LongTermDebtCurrent + CommercialPaper + ShortTermBorrowings (sum of present) |
| diluted shares | WeightedAverageNumberOfDilutedSharesOutstanding (quarterly) |
| shares outstanding | dei:EntityCommonStockSharesOutstanding (latest filed ≤ filing_date) |

Derived (matching Financial Datasets' definitions, checked against a cached MSFT row):
market_cap = shares outstanding × last close ≤ filing_date · P/E = market_cap / TTM
net income (None if ≤ 0) · ROE = TTM NI / equity · gross/operating/net margin =
TTM ÷ TTM revenue · D/E = debt / equity · current ratio · revenue_growth = TTM
revenue / TTM revenue one year earlier − 1 · EPS = TTM NI / diluted shares ·
BVPS = equity / shares outstanding · FCF/share = TTM (OCF − capex) / shares
outstanding. Any missing input → that field is None (banks have no gross profit;
that is correct, not an error).

`get_company_facts`: name, CIK, `sector` = SIC division (standard SIC ranges,
e.g. 3571 → "Manufacturing"), `industry` = SIC description.

With years of history available, `MIN_PERIODS` returns to **4**.

## 5. Earnings events for PEAD (SUE)

- Quarterly diluted EPS per fiscal quarter: 3-month facts; Q4 = FY − 9-month YTD.
- `SUE(E) = (EPS(E) − EPS(E − 4 quarters)) / stdev` of the previous 8 such
  year-over-year changes (≥ 4 required, else no event).
- `eps_surprise` = `BEAT` if SUE ≥ 1.0, `MISS` if SUE ≤ −1.0, else `MEET`.
- Event date = the 8-K with item 2.02 filed within 0–60 days after `E` (earliest);
  if none, the 10-Q/10-K filing date with `source_type` of that form.
- Returned as `EarningsRecord(source_type="8-K", report_period=E,
  filing_date=…, quarterly=EarningsData(earnings_per_share=…, eps_surprise=…))`,
  so `PEADModel` is unchanged. Known approximation: EPS comes from the 10-Q XBRL,
  which may be filed days after the 8-K; the figures are the ones in the 8-K press
  release, so this is not lookahead in substance. Documented in the module.

## 6. Resumability

- Data and LLM answers are already persisted per call (LLM cache files; the DB for
  data), so a re-run after a crash repeats **no paid work**.
- `backtest_runs`: `run_baseline` stores each finished variant's
  `FundBacktestResult` under its `run_key`; a re-run reuses finished variants and
  only computes the rest. On failure the variant is marked `failed` with the error,
  and `baseline` prints the finished variants' rows plus which ones failed.
- `aihf-paper baseline --fresh` ignores stored results.

## 7. Configuration and entry points

- `.env`: `SEC_USER_AGENT` (required for `free`), `HEDGE_FUND_DATA` (`free` default,
  `fd` optional). `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY` already present.
- `aihf-paper`, `aihf` (CLI), the TUI, and `event_study` open data via
  `open_data_client()`. `FINANCIAL_DATASETS_API_KEY` is no longer required in `free` mode.
- New `aihf-paper data-sync [--tickers …]`: warm the DB (prices from 2016, SEC for
  the universe) and print coverage per ticker — periods available, earliest filing,
  price range.

## 8. Testing

- `MarketStore`: upsert idempotency, PIT fact query (latest filed ≤ date), price ranges.
- `fundamentals`: synthetic companyfacts JSON covering FY vs YTD TTM, Q4 derivation,
  restated value (PIT picks the version filed by the date), missing concepts → None,
  SUE thresholds and 8-K date matching.
- `FreeDataClient`: lazy sync fetches only missing ranges (fake sources count calls);
  offline with stored data works; offline with none raises.
- `SecSource`: refuses without `SEC_USER_AGENT`; rate limiter; ticker normalization
  `BRK.B → BRK-B`.
- Checkpoint: finished variants reused; failure recorded; `--fresh` recomputes.
- **Parity (manual, not committed):** a scratch script compares `FreeDataClient`
  rows with the Financial Datasets responses already cached in
  `~/.hedge-fund/cache/data` (867 files) for the overlapping tickers and periods;
  target: filing dates exact, margins/ROE/D-E within 2 percentage points, P/E and
  market cap within 5%. Outliers investigated before switching the default.

## 9. Out of scope

News and insider trades (unused by current agents); intraday data; replacing the
LLM (it stays paid by choice); reducing LLM calls between filings (declined).

## 10. Later phases (each its own spec → plan; evidence-gated)

Every phase is adopted only if the baseline shows it improves post-cost,
risk-adjusted results. All depend on this data layer.

**Phase 2 — Momentum + risk sizing (free, backtestable over 10 years)**
- `MomentumModel` (`hedge_fund/signals/momentum.py`, long/short, no LLM):
  12-month return skipping the latest month, divided by trailing volatility,
  squashed to [−1, 1]. Registered; library strategy `strategies/momentum.yaml`.
  Used as its own sleeve and as a sixth voice in Fundamental L/S (gives the
  thin short side real short evidence; stops shorting strong uptrends).
- `BlendPolicy.sizing: conviction | inverse_vol` (default `conviction`):
  weight ∝ conviction / 60-day annualized volatility, computed once per
  assessment from stored prices, applied before the per-name cap.
- Price-only sleeves get 10-year backtests from Alpaca data (no LLM cost, no
  memorization concern).

**Phase 3 — Broader universe with an LLM funnel**
- ~100 liquid large caps (`paper_universe_100.list`).
- `FundSpec.llm_shortlist: int | None`: quant models score the whole universe;
  LLM agents assess only the top-N by |quant composite| plus current holdings,
  keeping LLM cost near today's while choosing from 3× more names.
- Shorts filtered by Alpaca `easy_to_borrow`.
- Survivorship caveat: a present-day list flatters long backtests; a
  point-in-time universe (top-N by dollar volume as of each date) is a
  follow-up; until then such backtests are labelled.

**Phase 4 — Insider purchases (optional)**
- SEC Form 4 open-market purchases (transaction code P) by officers/directors,
  clustered over 90 days, as a long-only quant signal. Sales ignored.
