# ai-hedge-fund — working rules for Claude

Systematic trading research platform. Python 3.11 + Poetry, package `hedge_fund/`.
Educational research only: no profitability claims, no real-money trading.

## Git
- Never modify, merge into, or push to `main`. Work on the assigned `claude/*` branch.
- Never force-push or rewrite pushed history. Checkpoint-commit each finished phase.

## Data
- Prices: Tiingo. Fundamentals: SEC EDGAR point-in-time. Nothing else by default.
- Financial Datasets is forbidden. It is blocked in code (`hedge_fund/data/policy.py`)
  unless `AIHF_ALLOW_FINANCIAL_DATASETS=1`; never set that. Also run commands with
  `env -u FINANCIAL_DATASETS_API_KEY`.
- Strict point-in-time: a decision at session D may use only data published by D.
  Market data for strategies goes through `hedge_fund.systematic` `AsOfView`.
- Fills happen at the next eligible execution event, never the observation session.

## Research discipline
- Never tune parameters on a locked holdout. A holdout failure ends that candidate;
  do not iterate on the same holdout.
- Record every experiment in the registry; count trials for multiple-testing adjustments.
- The quantitative engine (signals → portfolio → risk → execution) is deterministic Python.
- AI output (Claude, Kimi, ...) is a proposal or critique only. Promotion needs the
  Python validation gates to PASS, then paper trading, then explicit human approval.
- AI must never change risk limits, validation thresholds, the locked holdout, or the
  kill switch, and must never enable live trading.

## LLM cost
- Cache every paid LLM response (atomic writes); never re-pay a cached prompt.
- Respect the task's cost cap; estimate before any large batch and stop above it.

## Tests
- Offline suite before and after changes:
  `env -u FINANCIAL_DATASETS_API_KEY poetry run pytest -q -p no:cacheprovider hedge_fund`
- Every new module ships with tests; look-ahead guards must stay tested.

## Secrets
- Never print, log, commit, or echo API keys, tokens or credentials. Report env vars as
  PRESENT/ABSENT only.
