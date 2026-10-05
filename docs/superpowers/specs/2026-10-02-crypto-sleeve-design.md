# Crypto Sleeve (Core + Earned Trend Sleeve, LLM Risk Overlay) — Design

Date: 2026-10-02 · Status: approved in brainstorming, pending spec review

> Educational project; not investment, tax or financial advice. Crypto is far
> more volatile than stocks (BTC fell ~75% in 2022). Real money is opt-in and
> starts as buy-and-hold only.

## Goal

Add a crypto half to the live account on the same principles as the stock
half: a buy-and-hold **core** that builds wealth by itself, and a **trend
sleeve** (a pre-registered quant rule plus an LLM risk overlay) that earns a
share of the crypto half only through paper results that beat the core.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Account structure | One live Alpaca account, two halves: stocks and crypto, ~50/50 (`crypto_share` = 0.5 once the crypto half is funded; default 0); one cash pool, one ledger, one daily command |
| Coins | BTC, ETH, SOL |
| Crypto core | 60% BTC / 30% ETH / 10% SOL, topped up by deposits, rebalanced on rebalance days; never sold because prices fell |
| Trend sleeve | A quant trend rule decides; share of the crypto half earned by review: 0% → 20% → +10 pts per pass, cap 50%; −10 pts per fail; halt if it trails the core by 10% |
| LLM role | Risk overlay only: Opus reads crypto news + market stats daily and may only **reduce** the trend sleeve's exposure, with a reason; switched off unless it beats the rule alone on paper |
| Data | Alpaca's free daily crypto bars (BTC/ETH/SOL from 2021-01-01, covering the 2022 crash). The Massive Market Data plan only covers ~2 years, so it is not used |
| trader.dev | Idea source only: up to 2 candidate rules may be shortlisted from it, then rebuilt and re-tested here. Its leaderboard is not evidence; its exchanges are not used |
| Paper lab | A **separate** Alpaca paper account for crypto, so experiments never touch the stock paper fund's NAV |

## 1. Crypto data

`hedge_fund/data/crypto_prices.py` — `CryptoPriceSource`: daily bars from
`https://data.alpaca.markets/v1beta3/crypto/us/bars` (`timeframe=1Day`,
paginated), cached in the existing `MarketStore` SQLite (new table
`crypto_bars(symbol, date, open, high, low, close, volume)`), incremental
like the stock sync. Bars are UTC days; "the close of day D" is the bar
stamped D 00:00 UTC. Symbols use Alpaca's pair form `BTC/USD`; positions come
back as `BTCUSD`, so one helper normalizes both forms.

## 2. Trend rules (pre-registered)

`hedge_fund/crypto/rules.py`. Each rule maps a coin's daily closes up to
day D (no lookahead) to an exposure in {0, 1} for day D+1. Fixed, textbook
parameters; **no tuning**:

| Id | Rule |
|---|---|
| `ma100` | close > 100-day simple moving average |
| `mom12w` | 84-day return > 0 |
| `ma_cross` | 50-day SMA > 200-day SMA |
| `td1`, `td2` | up to two rules shortlisted on trader.dev, transcribed with their published parameters before any test here (recorded in this spec's results section first) |

No other rule is tested. Adding a rule later is a new spec.

## 3. Crypto backtest (`hedge_fund/crypto/backtest.py`)

Daily, 2021-01-01 → latest completed UTC day.

- **Core:** 60/30/10, rebalanced to weights on the first day of each ISO
  week (the live cadence).
- **Trend sleeve:** same 60/30/10 weights, each coin held only on days its
  rule says 1; otherwise that slice is cash (USD, no yield).
- Execution timing matches live: the live run trades on weekdays only, so a
  rule change seen on day D is executed at the next weekday's close (a
  Friday-night flip waits for Monday).
- Fees: `fee_bps` per side on traded notional (default 25 bps; the real
  Alpaca crypto taker fee for the user's tier must be confirmed and entered).
- Metrics: total and annualized return, Sharpe (daily, ×√365), max drawdown,
  turnover, total fees; for the full window and for two halves
  (2021-01-01 → 2023-06-30, 2023-07-01 → end).

**Pre-committed bar:** a rule passes only if, after fees, its Sharpe beats
the core's in **both** halves **and** its full-window max drawdown is at
least 25% smaller (relative) than the core's. Several rules may pass; the
one with the highest full-window Sharpe is used. If none passes, the crypto
half stays pure buy-and-hold and the sleeve is never activated.

`aihf-crypto backtest` prints the table and writes it to the crypto lab
ledger.

## 4. LLM risk overlay (`hedge_fund/crypto/overlay.py`)

Daily, per coin, only when the chosen rule says 1. Inputs: last 7 days of
Alpaca news headlines for the coin, and stats (7/30-day return, 30-day
volatility, distance from 100-day SMA, rule state). Output (validated JSON):
`{"action": "keep" | "halve" | "exit", "reason": str}` → exposure × 1, 0.5
or 0. It can only reduce. Prompts and answers go through the existing LLM
cache and client (with its hard deadline); an LLM failure means `keep` and
is logged (the overlay never blocks the rule).

The overlay cannot be backtested honestly (the model knows how 2021–2026
news played out), so it is judged **forward only** on paper (§5).

## 5. Crypto paper lab

A separate Alpaca paper account; keys `ALPACA_CRYPTO_PAPER_KEY_ID` /
`ALPACA_CRYPTO_PAPER_SECRET_KEY` (`AlpacaPaperClient` gains explicit-key
construction; it still refuses the live URL).

- `aihf-crypto run` (daily with the other commands): reconcile yesterday, then
  trade the paper account toward **rule × overlay** for the trend sleeve at
  100% of the paper account (the lab measures the sleeve itself).
- The ledger also records, each day, two **shadow** books computed from
  prices only (no orders): the core (60/30/10) and the rule without the
  overlay. So one paper account yields all three comparisons.
- Crypto orders: notional buys, fractional sells, `time_in_force="gtc"`
  (Alpaca rejects `day` for crypto).

## 6. Live integration (extends `aihf-live`)

`LiveSettings` gains:

- `crypto_share: float = 0.0` (0–0.5; the user sets 0.5 when funding crypto)
- `crypto_core: dict[str, float] = {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}`
- `crypto_trend_share: float = 0.0` (0–0.5 of the crypto half, set by review)
- `crypto_rule: str | None = None` (the rule id that passed §3)

Target book: stock targets (existing) × (1 − crypto_share), plus crypto
targets × crypto_share, where crypto targets = core × (1 − crypto_trend_share)
+ sleeve × crypto_trend_share, and sleeve = core weights × rule × overlay
(the live runner reuses the paper lab's latest decisions, the same way the
stock satellite mirrors the paper fund).

Because crypto symbols are part of the target, the stock rebalance no longer
sells them (today's runner would close any holding not in its target).
Crypto orders use `gtc`; stock orders keep `day`. Valuation uses crypto daily
closes from §1. Hard limits gain: crypto half ≤ `crypto_share` + 5 points.

Reviews: `aihf-live review --crypto` uses the paper lab: sleeve (rule ×
overlay) vs shadow core over ≥ 90 days since the last crypto review; pass →
share steps as for stocks. Overlay check in the same review: if rule ×
overlay does not beat the rule-only shadow, the overlay is switched off
(`crypto_overlay: false`).

## 7. Preconditions (user-side, before any real crypto)

- ✅ Confirmed 2026-10-02: Alpaca offers crypto to the user's live account.
- ✅ Fee tier confirmed: level 1 (< $100k 30-day volume) is 0.15% maker /
  0.25% taker. The runner sends market orders (taker), so `fee_bps = 25`.
- UK tax treatment of crypto differs from shares (not tax advice).

## 8. Order of work

1. Crypto data source + backtest + rules → run `aihf-crypto backtest`,
   record results here (no money, no LLM).
2. If a rule passes: overlay + paper lab; the user creates the crypto paper
   account and keys.
3. Live integration with `crypto_share = 0` by default; the user funds and
   sets `crypto_share: 0.5`; the crypto core starts buy-and-hold.
4. First crypto review after ≥ 90 days of paper lab history.

## 9. Testing

Unit tests with synthetic bars: rule outputs and no-lookahead; backtest
arithmetic (fees, weekly rebalance, cash slices); bar evaluation in both
halves; overlay can only reduce and fails safe to `keep`; paper lab shadow
books; live target composition with `crypto_share`; the stock rebalance no
longer closes crypto; `gtc` for crypto orders; symbol normalization. No
network in tests.

## 10. Out of scope

Leverage, derivatives, shorting crypto, staking/yield, stablecoin yield,
coins beyond BTC/ETH/SOL, exchanges other than Alpaca, trader.dev execution.

## 11. Phase 1 results (2026-10-02)

`aihf-crypto backtest`: 2021-07-20 → 2026-10-01 (after the 200-day warm-up),
60/30/10 BTC/ETH/SOL, free Alpaca daily bars, 25 bps per side, weekday
execution, halves split at 2023-06-30.

| Variant | Total | Annual | Sharpe | Sharpe H1 | Sharpe H2 | Max DD | Fees (% of start) | Passes |
|---|---|---|---|---|---|---|---|---|
| Core (buy & hold) | +240% | +26.5% | 0.65 | 0.43 | 0.80 | 79.2% | 3.4% | — |
| ma100 | +291% | +30.0% | 0.85 | 0.42 | 1.13 | 50.1% | 28.8% | no (H1 Sharpe 0.42 < 0.43) |
| mom12w | +124% | +16.8% | 0.62 | 0.05 | 1.00 | 58.3% | 17.5% | no |
| ma_cross | +185% | +22.3% | 0.70 | 0.77 | 0.71 | 45.6% | 5.5% | no (H2 Sharpe 0.71 < 0.80) |

Verdict by the pre-committed bar: **no rule passes**. ma100 misses by 0.01
Sharpe in the first half; per the bar it is not adopted (the bar exists so a
near-miss can't be argued into a pass), and its 29% fee drag shows how much
it whipsaws. The crypto half is buy-and-hold only: Phase 2 is the live
integration with `crypto_trend_share` fixed at 0, and the LLM overlay is not
built (it was defined as an overlay on a passing rule). Up to two trader.dev
candidates may still be pre-registered (§2) and tested the same way.

## 12. Band rebalancing experiment (pre-registered 2026-10-05, before running)

Question: instead of rebalancing the 60/30/10 core every Monday, does
rebalancing only when a coin drifts far from its target — checked every
weekday — do better after fees? It stays fully invested, so the trend rules'
drawdown bar doesn't apply; the bar is efficiency.

- Variants (fixed now, no others): `band10`, `band20` — each weekday, if any
  coin's weight is more than 10% / 20% away from its target *relative* to
  that target (e.g. SOL outside 9–11% / 8–12% of the crypto half), rebalance
  all three to 60/30/10 at that day's close; otherwise do nothing. No
  calendar rebalance. Same data, window, halves and 25 bps fees as §11.
- Bar: Sharpe beats the weekly-rebalanced core in **both** halves **and**
  full-window total return beats the core's, after fees.
- Pass → becomes a live crypto setting (`crypto_rebalance: band`). Fail →
  weekly stays.
