# Live Core + Satellite Account — Design

Date: 2026-10-01 · Status: approved in brainstorming, pending spec review

> Educational project; not investment, tax or financial advice. Real-money
> trading is opt-in, separately keyed, and starts core-only.

## Goal

Let a real Alpaca account start small ($100), take recurring deposits, and
compound over years, with:

- a **core** that builds wealth whether or not the agents have skill (one
  S&P 500 ETF), and
- a **satellite** that copies the paper fund's active bets, sized by an
  **agent share** that the agents earn only through reviewed paper (and later
  live) results.

The paper fund remains the lab; the live account never takes a position the
paper fund did not plan and record first.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Go-live gate | Core live now; satellite only after the paper fund passes its 12-rebalance review |
| Core | One S&P 500 ETF (`core_ticker`, default `SPY` — the fund's benchmark; a cheaper equivalent is the user's choice in config) |
| Agent share | 0% now → 20% after first pass → +10 pts per passed review, cap 50%; −10 pts per failed review, floor 0% |
| Crypto | Out of scope (stocks only) |
| Agent model | Switch to a local Ollama model only if it passes a pre-committed comparison against Opus; else stay on Opus |
| Architecture | Live mirrors the paper fund (one LLM bill; no independent live assessments) |

## 1. Components

| Unit | Purpose |
|---|---|
| `hedge_fund/brokers/alpaca.py` | Factor a shared `_AlpacaRest` base (requests, errors, account, positions, calendar, orders). `AlpacaPaperClient` is unchanged in behavior (paper URL only, `APCA_*` keys). New `AlpacaLiveClient`: base URL `https://api.alpaca.markets` only, keys `ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET_KEY` only (never `APCA_*`), plus `fractional_positions() -> dict[str, float]`, `submit_fractional(ticker, side, qty: float, client_order_id)` (market, `day`), and `cash_flows(after) -> list[CashFlow]` from `/v2/account/activities` (types `CSD`, `CSW`, `DIV`, `JNLC`, `FEE`). |
| `hedge_fund/live/account/settings.py` | `LiveSettings` (pydantic, `extra="forbid"`) loaded from `~/.hedge-fund/live.yaml`: `confirm_live: bool`, `core_ticker: str = "SPY"`, `paper_fund: str = "paper-fund"`, `agent_share: float = 0.0` (0–0.5), `satellite_max_name_pct: float = 0.10`, `short_min_equity: float = 2000`, `shorts_enabled: bool = False` (user sets after enabling margin), `satellite_halt_relative: float = 0.10`, `min_order_usd: float = 1.0`, `stale_plan_days: int = 8`. |
| `hedge_fund/live/account/target.py` | Pure: `target_book(equity, settings, paper_active_weights) -> dict[str, float]` (section 2). |
| `hedge_fund/live/account/ledger.py` | `LiveLedger` at `~/.hedge-fund/live/<account-name>/`: `nav.csv` (`date, equity, net_flow, core_value, satellite_value, benchmark_close`), `plans/`, `fills/`, `flows.csv`, `SATELLITE_HALTED`, `reviews.csv`, `dry_run_done` marker, logs. |
| `hedge_fund/live/account/runner.py` | `reconcile_live(...)`, `submit_live(...)` (sections 3–4). |
| `hedge_fund/live/account/review.py` | `propose_review(paper_ledger, live_ledger, settings) -> ReviewResult`; `apply_review` writes the new `agent_share` to `live.yaml` and appends `reviews.csv` (section 5). |
| `hedge_fund/live/account/report.py` | Deposits, value, time-weighted return, core vs satellite split, satellite vs "same money in core" (section 6). |
| `hedge_fund/live/account/cli.py` | New `aihf-live` entry point: `status`, `reconcile`, `submit [--dry-run]`, `run` (= reconcile then submit), `report`, `review [--apply]`. |
| `hedge_fund/llm/compare.py` (existing) | Used once for the model decision (section 7); no change unless the acceptance thresholds need a flag. |

## 2. Target book

Inputs: account `equity`, `settings`, and the paper fund's latest executed
plan's **active** weights `A` (its `decision.final_weights`, excluding
`equitization`), taken from `~/.hedge-fund/paper/<paper_fund>/plans/` —
only if that plan is ≤ `stale_plan_days` old. If it is stale (the paper fund
missed its rebalance), the satellite **holds**: the target reuses the
satellite weights of the last live plan that mirrored a fresh paper plan (or
`{}` if there is none), so a missed paper run never sells the satellite.

```
share = 0 if SATELLITE_HALTED exists else settings.agent_share
shorts_ok = settings.shorts_enabled and equity >= settings.short_min_equity
A' = A if shorts_ok else {t: w for t, w in A.items() if w > 0}
if share == 0 or gross(A') == 0:  satellite = {}
else: satellite = {t: share * w / gross(A')}            # satellite gross = share
cap each |satellite[t]| at satellite_max_name_pct (excess → core, never redistributed)
core = 1 - sum(|w| for w in satellite.values())
book = {core_ticker: core} + satellite                  # core_ticker never in satellite;
                                                        # if A holds core_ticker, it merges into core
```

Unlevered: gross ≤ 1. Longs are bought with cash only (no margin borrowing).

## 3. Reconcile (daily, before submit)

For the most recent completed session (same rule as the paper reconcile):

1. Pull cash flows since the last reconciled session; append new ones to
   `flows.csv` (idempotent by Alpaca activity id). `net_flow` = deposits −
   withdrawals (dividends and fees are returns, not flows).
2. Value positions (fractional) at that session's closes from the data
   factory (free Alpaca bars); `equity = cash + Σ qty × close`.
3. Split value into `core_value` (core ticker) and `satellite_value` (the rest).
4. Upsert the `nav.csv` row.
5. Satellite halt check (§5.3).

Time-weighted daily return: `r_t = (equity_t − net_flow_t) / equity_{t−1} − 1`
(flows assumed at start of day — they're invested at the next open).

## 4. Submit (daily)

Guards, in order (each returns a status, nothing sent):

1. `KILL` exists → `killed` (shared with paper).
2. `confirm_live` is not `true` → `not_confirmed`.
3. No `dry_run_done` marker and not `--dry-run` → `needs_dry_run`.
4. Not a trading day / after the order cutoff (existing calendar rules) → status.
5. Orders already sent today (client-order-id prefix `live-<date>-`) → `already_submitted`.

Then:

- **Rebalance day** (the first session of the paper fund's period, via the
  existing `is_rebalance_day` — normally Monday, Tuesday after a holiday):
  full realignment to the target book.
- **Other days:** if uninvested cash ≥ `min_order_usd` (new deposit or
  dividends), buy toward target with **cash only** (no sells); else
  `nothing_to_do`.

Sizing: dollar targets = weight × equity; trade = target − current value;
skip trades below `max(min_order_usd, 0.5% of equity)`, except a sell that
closes a position. Sells first, then buys; buys never exceed available cash
(scaled down proportionally if needed). Longs are fractional (qty to 9 dp).
Shorts (only when `shorts_ok`) use whole shares, floor toward zero.

Hard limits checked on the projected book before sending (raise → nothing
sent): no negative cash, no satellite name above `satellite_max_name_pct`,
no short when not `shorts_ok`, gross ≤ 1.0.

The plan (target, orders, inputs including which paper plan it mirrored) is
written to `plans/<date>.json` before any order is sent.

## 5. Agent share

### 5.1 Review (manual command, ~every 12 paper rebalances)

`aihf-live review` computes over the review window — from the previous
review's date (or the paper fund's first rebalance, for the first review) to
the latest reconciled session; the window must contain ≥ 12 executed paper
rebalances:

- **Paper evidence:** paper fund return − SPY return over the window (the
  paper fund equitizes idle capital in SPY, so the difference is the active
  book's contribution after costs).
- **Live evidence (once the satellite is active):** satellite return vs the
  same money held in the core (§6).

Pass = paper evidence > 0 **and** (satellite inactive or live evidence > 0).
Proposal: pass → `min(0.5, share + 0.1)` (first pass from 0 → 0.2); fail →
`max(0, share − 0.1)`. Printed with the numbers; `--apply` writes it to
`live.yaml` and appends `reviews.csv`. Nothing changes the share automatically.

### 5.2 Fewer than 12 rebalances

`review` refuses (prints how many rebalances remain) — the pre-committed
"no verdict before 12" rule.

### 5.3 Satellite halt

If, since the satellite was last activated, the satellite's cumulative
time-weighted return trails the core's by `satellite_halt_relative` (10%) or
more, write `SATELLITE_HALTED`: share is treated as 0 (satellite sold into
the core at the next Monday) until a review applies a new share and clears
the file. The core is never sold because of market drawdowns.

## 6. Report

`aihf-live report`: total deposited, current value, gain (value − net
deposits), time-weighted return since start and YTD, the same for the core
ticker alone (benchmark), core/satellite split, satellite return vs
same-money-in-core, current agent share, halt status, last plan mirrored.

## 7. Agent model decision (one-time, before the satellite matters)

Run `python -m hedge_fund.llm.compare` on ≥ 200 of Opus's saved paper-fund
prompts against local candidates. Pre-committed acceptance: valid ≥ 98%,
agreement ≥ 75%, opposite calls ≤ 5%, value correlation ≥ 0.7, and no single
agent below 60% agreement. If a candidate passes, the paper mandate switches
to it (`HEDGE_FUND_LLM_MODEL`), noted in the paper ledger as a model change so
reviews can see the boundary; otherwise stay on Opus.

## 8. Operations (user-side)

- The user opens and funds the live Alpaca account; transfers are manual.
  The code never moves money.
- Daily: the existing paper command, then `aihf-live run`.
- First live run: `aihf-live submit --dry-run`, reviewed with the user.
- User-side checks, documented in the README section: Alpaca live-account
  availability for UK residents, FX conversion and transfer fees (batching
  deposits may cost less), W-8BEN, and that a US broker account is not an
  ISA (taxable in the UK). Not tax advice.

## 9. Error handling

| Failure | Behavior |
|---|---|
| Live keys missing / `confirm_live` false | refuse before any request |
| Paper plan missing or stale | satellite holds current positions (no buys/sells of satellite names); core still invests new cash |
| Data or Alpaca error during planning | abort before any order; macOS notification (existing `notify`) |
| Order rejected | recorded; others proceed; next run re-plans from actual positions |
| Insufficient cash for buys | buys scaled down proportionally |
| Flow history unavailable | reconcile fails loudly (returns would be wrong without it) |

## 10. Testing

- `target_book`: core-only (share 0), long-only stage drops shorts, shorts
  stage, per-name cap overflow to core, halted satellite, stale plan, core
  ticker present in paper weights.
- Sizing: fractional buys, $1 minimum, dust threshold, cash-only buys on
  non-rebalance days, buys scaled to cash, close-out sells never skipped.
- Guards: kill, not confirmed, needs dry run, already submitted, cutoff.
- Reconcile: flow ingestion idempotent; TWR with deposits; dividends not
  flows; core/satellite split.
- Review: <12 rebalances refused; pass/fail arithmetic; first pass 0 → 0.2;
  cap 0.5 and floor 0; `--apply` writes YAML and log; halt cleared.
- Halt: relative-return trigger.
- `AlpacaLiveClient`: refuses paper URL and `APCA_*` keys; fractional order
  body; activities mapping — all against a fake session, no network.
- `AlpacaPaperClient` behavior unchanged (existing tests pass).

## 11. Out of scope

Crypto; leverage; automatic deposits/withdrawals; ISA wrappers; tax
reports beyond the raw ledger; a live scheduler (the user runs it daily);
satellite strategies other than mirroring the paper fund.

## 12. Agent model decision result (2026-10-01)

200 of Opus 5.5's saved paper-fund prompts (59 bullish, 71 neutral, 70
bearish) replayed through local Ollama models on the user's Mac.

| Model | Valid | Agree | Opposite | Correlation | Lowest agent | s/answer |
|---|---|---|---|---|---|---|
| Bar | ≥ 98% | ≥ 75% | ≤ 5% | ≥ 0.7 | ≥ 60% | — |
| qwen3:8b | 100% | 46% | 14% | 0.30 | Lynch 33% | 62 |
| gpt-oss:20b | 92% | 54% | 8% | 0.53 | Munger 48% | 22 |

Neither passes; the agents stay on Opus 5.5. A local model that says the
opposite of the reference on 8–14% of calls would change the fund's book, so
the paper track record would no longer describe the analyst in use.
Raw results: `~/.hedge-fund/model-compare/`.
