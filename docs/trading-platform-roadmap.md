# Trading platform roadmap

Status as of 2026-10-01, branch `claude/focused-knuth-uj5k9j`.

## DONE

- Buffett LLM baseline 2016-07 → 2026-06 with audit (`runs/buffett-baseline/`).
- Financial Datasets fail-closed policy (`data/policy.py`).
- Project guardrails: `CLAUDE.md`, `.claude/settings.json`.
- Core abstractions: instruments, order lifecycle, protocols (`core/`).
- Market panel with strict as-of views (`systematic/panel.py`).
- Execution costs, next-session fills, dividends/splits, ledger, benchmarks.
- Four reference strategies (tsmom, xsmom, meanrev, breakout).
- Metrics, risk engine, regime layer, ensemble with NO_TRADE, portfolio construction.
- Daily systematic backtester with trade audit trail; shared DecisionEngine.
- Validation framework (splits, DSR, PBO, bootstrap, registry, locked holdout, gates).
- Paper broker, paper trading loop, paper-only Alpaca/IBKR adapters.
- Gated research pipeline (AI proposes; Python validates; human approves).
- Small-account mode.
- Kimi cross-review integration (mocked; key not yet provided).

## CURRENT

- Documentation of the above; final verification run.

## NEXT

1. Run the reference strategies on the real point-in-time universe (Tiingo cache,
   offline) with walk-forward splits and the locked holdout declared up front;
   record every trial in the registry.
2. Estimate `StrategyEvidence` (OOS Sharpe, edge per unit signal) from walk-forward
   results only, then run the ensemble backtest.
3. Broader liquid universe (top 100–500 by point-in-time float), monthly
   reconstitution, average-dollar-volume filter.
4. Event signals from SEC 8-K filings (earnings releases) with acceptance timestamps.
5. FRED risk-free rate and VIX for metrics and the regime layer.
6. Scheduler for the paper loop (daily after close) and a monitoring view.

## LATER

- Additional asset classes via new providers (ETFs first, then FX/futures/crypto).
- Intraday bars for fill simulation; borrow costs if shorting is enabled.
- Headroom A/B evaluation; OmniRoute with explicit named profiles only.
- Dashboard UI (Frontend Design skill if needed).

## BLOCKED

- Kimi critiques: `KIMI_CREDENTIAL_REQUIRED` (`KIMI_API_KEY` or `MOONSHOT_API_KEY`).
- Alpaca paper trading: needs `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (paper account).
- IBKR paper: needs IB Gateway in paper mode and `ib_insync`/`ib_async`.
- Larger universes: Tiingo free tier (50 req/h, 1,000/day) limits first downloads.
- Live trading: intentionally locked; requires a human decision, code change and legal review.
