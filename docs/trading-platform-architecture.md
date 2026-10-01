# Multi-market systematic trading platform — architecture

Research and paper trading only. Nothing here promises returns; live trading is locked.

## Layers

```
MarketDataProvider ──► MarketPanel ──► AsOfView ──► Strategy ──► Signal
 (Tiingo+EDGAR today)   (systematic/    (no row after      (systematic/
                         panel.py)       its session)        strategies.py)
                                                                   │
           Regime (regime.py) ──► Ensemble (ensemble.py) ◄─────────┘
                                        │  evidence-weighted, NO_TRADE if edge < cost
                                        ▼
                         Portfolio (portfolio.py) ──► RiskEngine (risk.py)
                                                          │ last word, only reduces risk
                                                          ▼
                    DecisionEngine (decision.py) ──► OrderRequest (core/orders.py)
                                                          │
                 ┌────────────────────────────────────────┴───────────────┐
                 ▼                                                        ▼
   SimulatedExecution (execution.py)                         TradingVenue (core/protocols.py)
   backtest: SystematicBacktester (backtest.py)              PaperBroker (brokers/paper.py)
                 │                                           Alpaca/IBKR adapters (paper-only)
                 ▼                                                        │
       FillEvent ──► Ledger (ledger.py) ──► metrics.py      PaperTradingEngine (paper.py)
                                                              reconcile · audit · kill switch
```

| Concept | Where | Notes |
|---|---|---|
| Instrument | `hedge_fund/core/instruments.py` | asset class, quantity step (whole/fractional/lot/contract), multiplier, minimums |
| Order / Fill / lifecycle | `hedge_fund/core/orders.py` | accepted, partial, filled, cancelled, rejected, expired; idempotency keys |
| Protocols | `hedge_fund/core/protocols.py` | MarketDataProvider, Strategy, ExecutionModel, TradingVenue, RiskGate |
| Equities implementation | `hedge_fund/data/` | Tiingo prices + SEC EDGAR; Financial Datasets blocked by `data/policy.py` |
| Buffett/LLM pipeline | `hedge_fund/pipeline`, `backtesting` | unchanged; uses the older `SimBroker` |

Adding FX, futures or crypto means a new `MarketDataProvider` (bars, dividends/splits
or their absence) and `Instrument` definitions; the engine does not change.

## Data flow and point-in-time rules

- The panel is built once from the provider; strategies only ever receive
  `panel.as_of(D)`. Any read after D raises `FutureDataError`.
- Each view restates prices to the split basis in force at D, so a pre-split price
  level does not reveal a later split. Dividends and split factors are dated fields.
- Untradable bars (halts, zero volume, post-delisting vendor placeholders) are masked.
- Universe membership comes from a point-in-time schedule.
- Tests prove a strategy's output, and a backtest's equity up to X, are identical
  when the vendor history is truncated at X.

## Execution

- An order decided on session D fills only at D+1's open or close (`FillTiming`
  is a required argument); filling at or before D raises `LookAheadExecution`.
- Costs: commission (per share, bps, minimum), half-spread, square-root impact on
  order/ADV notional scaled by daily vol; ADV and vol use sessions before the fill.
- Participation cap (share of ADV); zero ADV, untradable or missing prices reject.
- Ledger applies splits and ex-date dividends; shorts pay dividends; every cash flow
  is journaled and reconciled. Benchmarks are reported price-only and total-return;
  a dividend-credited portfolio is compared with the total-return benchmark.

## Risk (`systematic/risk.py`)

Kill switch, drawdown breaker, daily-loss reduce-only, shorting permission, max
position, ADV participation, correlation-cluster cap, volatility target (down only),
gross/net caps, turnover cap. `RiskConfig` is frozen and hashed. Fractional Kelly
(≤ 0.5) is an optional upper bound that disables itself without enough evidence.

## Validation (`hedge_fund/validation`)

Walk-forward, purged k-fold, CPCV; PSR, Deflated Sharpe, PBO; block bootstrap and
Monte Carlo trade reshuffles; a hash-chained experiment registry that counts trials;
a locked holdout evaluated once per candidate that refuses derived candidates of a
failed parent; gates from `configs/validation-gates.yaml` (frozen, hashed).

## AI role

- `research/crossreview.py`: one explicitly named model proposes, another critiques
  (Claude ↔ Kimi). Responses cached. The verdict is the Python gate result.
- `research/pipeline.py`: candidate → backtested → validated → stress-tested →
  holdout → paper → live-candidate. AI may only propose and review; the system
  advances on passing evidence; a human approves live-candidate status. Changing
  risk limits, gates, the holdout, leverage, the kill switch or going live is
  human-only.

## Paper vs live

- `PaperTradingEngine` refuses any venue with `live=True`; idempotent sessions and
  order ids; stale-data and calendar checks; retries; reconciliation; audit log;
  `KILL_SWITCH` file halts and flattens.
- `brokers/adapters.py`: Alpaca (paper REST host only) and IBKR (paper port only)
  are integration-ready; `LIVE_TRADING_ENABLED = False` is a code constant, so going
  live needs a reviewed code change plus credentials plus an env opt-in.

## Small accounts

`configs/small_account.yaml` + `systematic/small_account.py`: fractional sizing,
minimum notional, per-order commission in round-trip cost, an economics filter that
skips trades whose edge cannot pay costs, and a ruin-probability estimate.
