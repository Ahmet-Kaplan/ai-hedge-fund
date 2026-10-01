# Phase A — real-market evidence for four frozen reference strategies

Pre-registration hash `b37d3f1b31a356e7…` (committed before any backtest). Research and simulation only; no live trading. Figures are simulated, net of modelled costs, and not a promise of future returns.

## Summary table (development period unless stated)

| Strategy | CAGR | Sharpe | Max DD | Trades | Costs | DSR | PBO | €200 Final (raw / filtered) | Holdout | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| breakout | -1.6% | -0.58 | 12.4% | 77 | $796 | 0.00 | 0.67 | 40.91 / 200.00 | FAIL | **REJECT** |
| meanrev | 8.8% | 0.54 | 27.1% | 1336 | $7,598 | 0.43 | 0.67 | 0.00 (ruined) / 200.00 | PASS | **INSUFFICIENT_EVIDENCE** |
| tsmom | 10.9% | 0.63 | 30.3% | 536 | $1,130 | 0.51 | 0.67 | 0.00 (ruined) / 230.08 | PASS | **INSUFFICIENT_EVIDENCE** |
| xsmom | 9.5% | 0.57 | 32.4% | 330 | $1,143 | 0.46 | 0.67 | 0.00 (ruined) / 192.98 | FAIL | **REJECT** |

Benchmark SPY total return, development: CAGR 11.8%, Sharpe 0.67, max DD 33.7%. Trades = closed round trips. DSR uses n_trials = 4; PBO is computed once for the family of four strategies.

## Setup

- Universe: runs/buffett-baseline/universe_top10.json — survivorship-aware PIT top-10 by market cap, annual reconstitution 2016-07-01..2025-07-01; membership = latest snapshot on or before the session.
- Development 2016-07-01 → 2022-12-30; embargo 30 days; locked holdout 2023-02-01 → 2026-06-30 (evaluated once per strategy).
- Benchmark: SPY total return (dividends reinvested) — the portfolio is credited dividends. Risk-free rate: 0 (no rate series in the cache); Sharpe is excess over zero.
- Execution: next_open — orders decided on session D's close fill at D+1's open. Costs: commission 1.0 bp, half-spread 2.0 bp, square-root impact coef 0.1, max 5% of ADV.
- Portfolio: long-only, inverse-volatility, gross 100%, max 20% per name; operational breakers off (measure the strategies' own behaviour; operational breakers (drawdown halt, daily loss, vol target) are switched off so they cannot mask or create an edge).
- Strategy configs: code defaults, frozen and hashed: breakout `b13d865ab364ee11` (weekly), meanrev `5775739dba617f29` (weekly), tsmom `f23a4d3fb5b1ee22` (monthly), xsmom `4d7783bca536dc8b` (monthly).

## Data inventory

- 22 instruments (SPY + 21 universe names), 2512 sessions 2016-07-01 → 2026-06-30; Tiingo requests during inventory: 0.
- No missing sessions, closes or volumes during any name's membership; every name is 100% complete.
- Dividends present for all payers; BRK.B, AMZN, TSLA paid none. Splits covered (AAPL, GOOGL, AMZN, NVDA×2, TSLA×2, AVGO, T, GE). GE's 2023/2024 spin-offs are encoded by the vendor as split factors 1.281 and 1.253 (value-preserving restatement, not true splits).
- Delisting coverage: no universe member stopped trading in the window, so delisting handling was not exercised by this universe (survivorship-aware membership, but megacaps did not delist).

## Development results

| Strategy | Total | CAGR | Vol | Sharpe | Sortino | Calmar | Max DD | Hit | Payoff | PF | Expectancy | Turnover/yr | Costs | Slippage | Hold (sessions) | Gross | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breakout | -9.9% | -1.6% | 2.7% | -0.58 | -0.73 | -0.13 | 12.4% | 48.1% | 0.31 | 0.28 | -378 | 4.24 | $796 | $534 | 9 | 4% | 4% |
| meanrev | 72.7% | 8.8% | 18.9% | 0.54 | 0.78 | 0.32 | 27.1% | 55.9% | 0.92 | 1.17 | 32 | 25.13 | $7,598 | $5,105 | 14 | 59% | 59% |
| tsmom | 95.8% | 10.9% | 19.5% | 0.63 | 0.88 | 0.36 | 30.3% | 62.7% | 0.98 | 1.64 | 109 | 3.62 | $1,130 | $758 | 143 | 87% | 87% |
| xsmom | 80.2% | 9.5% | 19.2% | 0.57 | 0.80 | 0.29 | 32.4% | 63.9% | 0.75 | 1.33 | 100 | 3.72 | $1,143 | $766 | 124 | 74% | 74% |

SPY total return over the same window: 104.6%.

### Exposure to the market (post-hoc, descriptive — not used for classification)

| Strategy | Beta vs SPY TR | Correlation | Alpha / yr | Alpha t-stat |
|---|---:|---:|---:|---:|
| breakout | 0.02 | 0.14 | -1.83% | -1.74 |
| meanrev | 0.85 | 0.87 | -0.76% | -0.20 |
| tsmom | 0.91 | 0.90 | 0.50% | 0.15 |
| xsmom | 0.87 | 0.88 | -0.34% | -0.09 |

The three strategies with positive Sharpe are mostly long-only exposure to the same megacaps as the benchmark (beta 0.85–0.91); their alpha is indistinguishable from zero.

### Per year (net return vs SPY total return)

| Year | breakout | meanrev | tsmom | xsmom | SPY TR |
|---|---:|---:|---:|---:|---:|
| 2016 | 0.0% | 5.3% | -1.4% | -1.1% | 7.7% |
| 2017 | 1.1% | 16.2% | 30.1% | 25.3% | 21.7% |
| 2018 | -4.7% | 8.0% | 0.6% | 0.8% | -4.6% |
| 2019 | -2.6% | 18.7% | 19.1% | 19.0% | 31.2% |
| 2020 | -1.6% | 18.0% | 26.5% | 29.8% | 18.4% |
| 2021 | 0.1% | 14.9% | 24.0% | 25.1% | 28.7% |
| 2022 | -2.5% | -18.9% | -18.8% | -25.3% | -18.2% |

### Per regime (SimpleRegime at each decision; annualized mean daily return)

- **breakout**: up/low/risk-on (996d): -0.9%, Sharpe -0.30; down/high/risk-off (226d): 0.3%, Sharpe 0.24; up/high/risk-on (180d): -8.2%, Sharpe -2.64; range/high/risk-on (91d): 0.4%, Sharpe 0.29; down/low/risk-off (53d): 0.8%, Sharpe 2.18
- **meanrev**: up/low/risk-on (996d): 10.4%, Sharpe 1.01; down/high/risk-off (226d): 4.7%, Sharpe 0.13; up/high/risk-on (180d): 8.9%, Sharpe 0.61; range/high/risk-on (91d): -36.4%, Sharpe -1.39; down/low/risk-off (53d): 16.4%, Sharpe 0.59
- **tsmom**: up/low/risk-on (973d): 17.4%, Sharpe 1.24; down/high/risk-off (209d): 17.6%, Sharpe 0.49; up/high/risk-on (143d): -7.2%, Sharpe -0.35; range/high/risk-on (124d): -25.6%, Sharpe -1.07; range/low/risk-on (81d): 45.8%, Sharpe 3.88
- **xsmom**: up/low/risk-on (973d): 16.9%, Sharpe 1.16; down/high/risk-off (209d): 19.1%, Sharpe 0.58; up/high/risk-on (143d): -7.3%, Sharpe -0.40; range/high/risk-on (124d): -30.6%, Sharpe -1.15; range/low/risk-on (81d): 36.3%, Sharpe 2.96

### Largest drawdowns

- **breakout**: -12.4% 2017-06-08→2022-07-18 (recovered not by end); -1.1% 2017-05-16→2017-05-17 (recovered 2017-06-02); -0.8% 2016-07-19→2016-11-30 (recovered 2016-12-13)
- **meanrev**: -27.1% 2020-02-06→2020-03-23 (recovered 2020-07-31); -24.7% 2021-11-08→2022-11-03 (recovered not by end); -15.1% 2018-10-04→2018-12-24 (recovered 2019-03-28)
- **tsmom**: -30.3% 2020-02-19→2020-03-23 (recovered 2020-07-06); -23.0% 2022-01-04→2022-10-12 (recovered not by end); -21.3% 2018-09-20→2018-12-24 (recovered 2019-12-26)
- **xsmom**: -32.4% 2021-11-24→2022-10-12 (recovered not by end); -24.9% 2020-02-19→2020-03-23 (recovered 2020-06-10); -21.6% 2018-09-04→2018-12-24 (recovered 2019-12-26)

### Contribution by instrument (P&L in $, development)

- **breakout**: top AAPL -18, AMZN -1,508, BAC -344, BRK.B -2,690; bottom UNH -178, V -2,003, XOM -1,361
- **meanrev**: top AAPL +404, AMZN +10,060, BAC +4,666, BRK.B +3,258; bottom V +4,780, WFC -23, XOM -13,806
- **tsmom**: top AAPL +29,231, AMZN +25,905, BAC -1,222, BRK.B +10,606; bottom V -1,962, WFC -474, XOM -2,372
- **xsmom**: top AAPL +26,989, AMZN +29,399, BAC -1,486, BRK.B -6,684; bottom UNH -613, V -1,661, XOM -1,609

## Validation (development only)

| Strategy | WF positive years | Purged k-fold Sharpe (min / mean) | CPCV share positive | PSR | DSR | PBO | Bootstrap Sharpe 95% CI (daily) | MC max DD p95 | Gates |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| breakout | 33% | -2.12 / -0.84 | 20% | 0.06 | 0.00 | 0.67 | [-0.0810, 0.0130] | 32.7% | FAIL: bootstrap, deflated_sharpe, oos_sharpe, pbo, walk_forward |
| meanrev | 83% | -0.46 / 0.94 | 93% | 0.91 | 0.43 | 0.67 | [-0.0018, 0.0774] | 36.4% | FAIL: bootstrap, deflated_sharpe, max_drawdown, pbo |
| tsmom | 83% | -0.80 / 0.78 | 87% | 0.94 | 0.51 | 0.67 | [-0.0025, 0.0886] | 10.6% | FAIL: bootstrap, deflated_sharpe, max_drawdown, pbo |
| xsmom | 83% | -1.12 / 0.80 | 80% | 0.93 | 0.46 | 0.67 | [-0.0053, 0.0815] | 21.5% | FAIL: bootstrap, deflated_sharpe, max_drawdown, pbo |

Gates: `configs/validation-gates.yaml` (hash 2fc90f4ad972f482). No gate was changed.

## Locked holdout (2023-02-01 → 2026-06-30, evaluated once)

| Strategy | Total | CAGR | Sharpe | Max DD | Round trips | Costs | SPY TR | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| breakout | 2.0% | 0.6% | 0.15 | 5.3% | 58 | $634 | 90.1% | FAIL |
| meanrev | 88.5% | 20.6% | 1.23 | 22.1% | 669 | $3,684 | 90.1% | PASS |
| tsmom | 80.1% | 19.0% | 1.11 | 20.9% | 270 | $470 | 90.1% | PASS |
| xsmom | 97.5% | 22.2% | 1.05 | 25.4% | 180 | $687 | 90.1% | FAIL |

Criterion (pre-registered): Sharpe ≥ 0.5, max DD ≤ 25%, total return > 0. It does not require beating the benchmark; none of the strategies clearly did.

## Cost stress (development)

| Strategy | Sharpe 1× | 1.5× | 2× | 3× | Total return 1× → 3× | Fragile |
|---|---:|---:|---:|---:|---|---|
| breakout | -0.58 | -0.61 | -0.63 | -0.68 | -9.9% → -11.5% | yes |
| meanrev | 0.54 | 0.52 | 0.50 | 0.46 | 72.7% → 55.7% | no |
| tsmom | 0.63 | 0.63 | 0.62 | 0.62 | 95.8% → 93.4% | no |
| xsmom | 0.57 | 0.57 | 0.57 | 0.56 | 80.2% → 78.1% | no |

## €200 account (development)

| Strategy | Variant | Final | Max DD | Fills | Rejected (exec) | Dropped uneconomic | Avg invested | Cash share | Costs | Costs / avg equity | P(ruin, 1y) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breakout | raw | 40.91 | 79.5% | 146 | 0 | 0 | 5.67 | 95% | 149.37 | 135% | 0.0% |
| breakout | economics_filtered | 200.00 | -0.0% | 0 | 0 | 77 | 0.00 | 100% | 0.00 | 0% | 0.0% |
| meanrev | raw | 0.00 (ruined) | 100.0% | 210 | 16 | 0 | 7.27 | 92% | 211.93 | 2590% | 88.4% |
| meanrev | economics_filtered | 200.00 | -0.0% | 0 | 0 | 1424 | 0.00 | 100% | 0.00 | 0% | 0.0% |
| tsmom | raw | 0.00 (ruined) | 100.0% | 243 | 64 | 0 | 40.23 | 56% | 243.96 | 613% | 80.3% |
| tsmom | economics_filtered | 230.08 | 34.1% | 127 | 12 | 488 | 209.94 | 14% | 130.45 | 53% | 0.0% |
| xsmom | raw | 0.00 (ruined) | 100.0% | 256 | 8 | 0 | 47.86 | 37% | 257.93 | 440% | 100.0% |
| xsmom | economics_filtered | 192.98 | 37.5% | 144 | 0 | 286 | 160.44 | 25% | 148.31 | 69% | 0.0% |

The €1 minimum commission per order dominates: rebalancing a 200-unit account into several names means orders of a few to a few dozen units, where €1 is 2–50% of the order. Unfiltered, every strategy lost the whole account. With the economics filter, orders whose estimated edge did not cover costs were skipped; breakout and meanrev then never traded, tsmom ended at 230 (+15% over 6.5 years) and xsmom at 193. The filter's edge estimate (development round-trip expectancy, bp) includes market beta, so it overstates true edge. Results were simulated at €200 directly, not scaled from $100k.

Simulator finding: the raw small-account runs ended slightly below zero because commissions were still charged on tiny sells when cash was exhausted. The report treats these as ruined (0); the cash floor should be enforced in the ledger before any small-account paper trading.

## Classification (pre-registered rules, unchanged)

- **breakout — REJECT**: development Sharpe -0.58; gates fail (bootstrap, deflated_sharpe, oos_sharpe, pbo, walk_forward); holdout fail; fragile to costs: yes.
- **meanrev — INSUFFICIENT_EVIDENCE**: development Sharpe 0.54; gates fail (bootstrap, deflated_sharpe, max_drawdown, pbo); holdout pass; fragile to costs: no.
- **tsmom — INSUFFICIENT_EVIDENCE**: development Sharpe 0.63; gates fail (bootstrap, deflated_sharpe, max_drawdown, pbo); holdout pass; fragile to costs: no.
- **xsmom — REJECT**: development Sharpe 0.57; gates fail (bootstrap, deflated_sharpe, max_drawdown, pbo); holdout fail; fragile to costs: no.

Ensemble: not run — fewer than two strategies are EVIDENCE_SUPPORTS_FURTHER_RESEARCH.

## Audit

- Financial Datasets clients constructed: 0; Tiingo requests: 0 (offline store); live broker modules loaded: 0.
- Look-ahead violations (fill session not after decision session): 0.
- Pre-registration committed at 2026-10-01T09:20:11Z (commit aa05bc1); holdout evaluated at 09:27:38, 09:27:54, 09:28:05, 09:28:14 UTC, once per strategy.
- Config hashes match the pre-registration: True; no parameter was changed before or after the holdout.
- Experiment registry verified (hash chain): True; distinct trials in family: 4.
- No secrets printed or stored; `main` unchanged.

## Caveats

- Small universe (10 megacaps at a time); results do not generalise to broader or less liquid markets.
- Rebalance cadence per strategy was a pre-registered choice; other cadences were not tried (by design).
- Risk-free rate set to 0; Sharpe ratios are slightly overstated relative to excess-of-cash Sharpe.
- PBO is a family-level statistic over these four strategies, not a per-strategy probability.
- The holdout criterion does not include a benchmark comparison; holdout passes therefore do not imply outperformance.
