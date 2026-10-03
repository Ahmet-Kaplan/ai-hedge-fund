# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund. The goal of this project is to explore the use of AI to make trading decisions. This project is for **educational** purposes only and is not intended for real trading or investment.

<img width="2400" height="1460" alt="image" src="https://github.com/user-attachments/assets/e3985623-c226-4c1a-a587-e03fb4eca31e" />


Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## How to Install

```bash
pipx install aihf
```

(or `uv tool install aihf`, or `pip install aihf` into an environment of your choice)

Then run it from anywhere:

```bash
aihf
```

### API keys

The app asks for keys the first time it needs them and saves them to `~/.hedge-fund/.env` — nothing to configure up front. It needs:

- A [Financial Datasets](https://financialdatasets.ai) API key, for prices, fundamentals, and earnings.
- One model API key for the investor agents. Supported providers: Anthropic, OpenAI, DeepSeek, Google, xAI, Kimi, TypeSafe (Jev). Or run locally with Ollama — no key; model ids are Ollama tags (`llama3.1`, `qwen2.5`, or `ollama:<tag>` for anything you have pulled).

Keys exported in your shell always win over the saved file.

To point OpenAI-compatible models at a custom host (Groq, a local proxy, ...), set `OPENAI_BASE_URL` or the older `OPENAI_API_BASE` alias. Moonshot/Kimi already uses `MOONSHOT_BASE_URL`. Ollama uses `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`). Bound hung calls with `LLM_REQUEST_TIMEOUT` (seconds; default 60). A missing Ollama daemon fails inside that timeout instead of hanging.

## How to Run

### Interactive app

```bash
aihf
```

With no arguments, this launches the interactive terminal app. Build a fund — pick stocks, strategies, rebalance cadence — or backtest a saved fund and watch its equity curve draw against its benchmark. Funds you build are saved as mandate files in `~/.hedge-fund/mandates/`.

### Non-interactive

Run one live-clock paper cycle from a mandate file (`PaperBroker`, fills at mark, no live venue). `--paper` is the explicit flag; omitting it is the same path. If this mandate has a prior cycle receipt, the run opens that ending book so cash, positions, and NAV carry forward; otherwise it opens at the mandate's capital. A corrupt or incompatible receipt fails the run. The full cycle record prints to stdout as JSON; a short human summary goes to stderr; the receipt is saved next to the mandate:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --paper
```

Run the same mandate with Jev after configuring `TYPESAFE_API_KEY`:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --model jev-1.13.0
```

Backtest the mandate over history at its rebalance cadence (`SimBroker`, not the paper venue):

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest
```

An LLM trained after your backtest window may remember how those companies did, and that memory would score as skill. So a backtest withholds the ticker, industry and calendar dates from the investor agents' prompts; the personas see the fundamentals with periods labelled t-0, t-1, ... instead. This reduces the recall without removing it (distinctive numbers can still give a large company away), and a window after the model's training cutoff is the cleanest read. Live runs are unaffected and name the company.

A mandate is the desk — strategies, staff, risk, capital, cadence — and never names tickers; `--tickers` says what to point it at for this run.

### Scheduler daemon (always-on paper / sim)

The same `run_cycle` on a live clock, polled on an interval, gated by the market calendar and the mandate's rebalance cadence. Each tick is keyed by mandate + session date, so a double-fire is a no-op. Paper is the default venue; `sim` is the other allowed book. There is no live venue on this path.

```bash
# always-on: poll every 60s, fire when the session is due
python -m hedge_fund.daemon ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT

# one evaluation, then exit (no polling sleep) — useful from cron
python -m hedge_fund.daemon ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --once

# optional schedule YAML (CLI flags override)
# interval_seconds: 300
# venue: paper
python -m hedge_fund.daemon ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --config ~/.hedge-fund/daemon.yaml
```

Halt new ticks without killing the process mid-cycle:

```bash
touch ~/.hedge-fund/KILL
# or
export HEDGE_FUND_KILL_SWITCH=1
```

`--once` prints a JSON result on stdout (`status` is `ran`, `skipped`, `halted`, or `not_due`). The always-on loop logs each evaluation to stderr and exits when the kill-switch is on. Idempotency keys live under `~/.hedge-fund/ticks/`.

### Alpaca as the execution venue

The fund's books are broker-agnostic: anything satisfying the `Broker` protocol
(`positions` / `cash` / `place_order`) can execute a cycle. `AlpacaBroker` is a
paper/live venue for that slot. It needs the optional SDK:

```bash
pip install alpaca-py
```

```python
from hedge_fund.brokers import AlpacaBroker, AlpacaSettings
from hedge_fund.pipeline import run_cycle

broker = AlpacaBroker(AlpacaSettings.from_env())   # reads only, by default
record = run_cycle(fund, as_of, broker, data_client, universe)
```

The same book is reachable from the CLI and the daemon:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --broker sim
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --broker alpaca
python -m hedge_fund.daemon ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --venue alpaca
```

With `--broker alpaca` the run reconciles the venue against the newest receipt
before it sizes anything, refuses to trade if the venue is still holding an
order, and journals every submission. Until trading is enabled it stops with a
read-only message rather than failing on the first order mid-cycle.

Order submission is gated four times over, so a half-remembered environment
variable cannot move real money:

| Variable | Default | Effect |
|----------|---------|--------|
| `ALPACA_PAPER` | `true` | paper endpoint; reads always work |
| `ALPACA_TRADING_ENABLED` | `false` | required before *any* order is submitted |
| `ALPACA_LIVE_TRADING_CONFIRMED` | `false` | additionally required when `ALPACA_PAPER=false` |
| `ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET_KEY` | unset | the credential a live run must use |

A live run reads its credential from the `ALPACA_LIVE_*` variables and refuses
the paper key — so `ALPACA_PAPER=false` cannot re-point the key your paper runs
have been using at real money. A key carrying Alpaca's `PK` paper prefix is
refused on a live run even when it arrives through the live variables.

`ALPACA_API_SECRET` is accepted as an alias for `ALPACA_SECRET_KEY`.

Two behaviours differ from the offline brokers on purpose. `place_order` must
fill *completely* or raise — the fund sizes its book from the returned `Fill`,
so Alpaca's asynchronous executions are polled briefly and anything short of a
full fill raises `AlpacaOrderError` carrying the broker order id rather than
inventing a fill. And a fractional position raises instead of being truncated,
because `Position.shares` is an integer share count and rounding would desync
the fund's books from the broker's.

### TradingAgents as an analyst (optional)

[TradingAgents](https://github.com/TauricResearch/TradingAgents) runs a
LangGraph of analysts, a bull/bear debate and a three-way risk debate, and
returns a five-tier rating. `ta_board` turns that rating into a `Signal`, so
the board can sit in a strategy beside the personas and the quant models:

```yaml
strategies:
  - name: research
    models:
      - name: ta_board
        params: {llm_provider: anthropic, deep_think_llm: claude-opus-5-5}
    blend: {mode: long_short}
```

Install it from the repository — **not** from PyPI:

```bash
pip install git+https://github.com/TauricResearch/TradingAgents.git
```

A different project publishes itself on PyPI under the same name, so
`pip install tradingagents` fetches the wrong code. The adapter checks the
rating vocabulary it finds and refuses the impostor with an explanation.

Rating map: `Buy +1.0 · Overweight +0.5 · Hold 0.0 · Underweight -0.5 · Sell -1.0`.
`REVIEW` — the framework's "no readable decision" sentinel — abstains rather
than counting as a Hold, so the record never shows a call nobody made.

Three things to know before using it:

- **It cannot be blinded.** The framework is handed the ticker and fetches its
  own news, social and fundamentals, so a backtest could not hide the company
  from it. `TradingAgentsBoard.supports_blind = False`, and `blind=True` refuses
  to staff it instead of scoring recall as skill. Run it live, or leave it out
  of the mandate you backtest.
- **It bypasses this project's data contract.** Everything else reads a
  point-in-time snapshot built through `DataClient`; this model does its own
  fetching, and its social and news sources reflect *now* even for a historical
  date. `Signal.metadata["data_source"]` records that, so the audit trail shows
  which numbers came from outside the contract.
- **It is expensive.** One call is a multi-agent graph with debate rounds.
  Ratings are cached per (ticker, date, config fingerprint), so a replay is
  free, but a first run costs several LLM calls per ticker.

All of its artefacts (cache, logs, memory) are written under this project's
cache directory rather than a second `~/.tradingagents`.

### Where the data comes from

By default the fund needs **no data subscription**: prices come from Alpaca's
market data and fundamentals from SEC EDGAR.

```bash
# default — free, needs an Alpaca key and an SEC contact
HEDGE_FUND_DATA=free
SEC_USER_AGENT="Your Name your@email.com"
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...

# or take everything from Financial Datasets
HEDGE_FUND_DATA=fd
FINANCIAL_DATASETS_API_KEY=...
```

Prices and filings are cached in one SQLite file, `~/.hedge-fund/market.db`, so
a backtest re-reading the same bar or filing answers from an index instead of
re-fetching.

**Fundamentals are point-in-time.** Every metric row carries the date its
filing was *filed*, and the market cap on that row is priced at the close on or
before that date — so a backtest sees the number the market saw, not the number
the period ended on. Filings that restate an earlier period are kept both ways:
the restated value for "what is true now", the original for "what was public
then".

**The free sources serve prices and SEC fundamentals only.** News, insider
trades and earnings are not among them, and they *say so* — a `DataSourceError`
naming what to set — rather than returning an empty list. That distinction
matters here: `news_sentiment` and `news_analyst` both read news, and an empty
answer is indistinguishable from a company that had no news, so they would
abstain on every ticker and the receipt would attribute the silence to the
company.

If you want those models on a free run, set `FINANCIAL_DATASETS_API_KEY` as
well: the client is then composed, taking prices and fundamentals from the free
sources and news from Financial Datasets.

### What a fill costs

Backtests and paper runs charge the schedule a mandate declares, so a strategy
is measured against what trading it actually costs:

```yaml
commission:
  per_trade: 1.00   # flat charge per order
  per_share: 0.005  # plus this much per share
risk:
  max_position_pct: 0.25
  max_gross_exposure: 1.0
  min_cash_reserve_pct: 0.10   # leave 10% in cash; caps NET exposure at 90%
```

Both default to zero, and a mandate that does not mention them backtests
**bit-identically** to one from before either existed — the cost branch is
skipped rather than applied with zeros.

Commission is booked as a cash expense rather than folded into cost basis, so
"what the position earned" and "what trading it cost" stay separable: every
`Fill` carries its own `commission` and `realized_pnl`, and positions carry the
weighted-average `cost_basis` they were bought at. Closing a position realizes
against that basis; crossing zero in one order closes the old side and opens
the new one at the fill price, rather than pricing a short off shares the book
no longer holds.

The one honest gap: a book seeded from a receipt knows its share counts but not
what they cost. Closing such a position moves cash correctly and reports
`realized_pnl: null` — "nobody can say what this trade made" is a different
statement from "this trade made nothing", and the record keeps them apart.

### Read-only dashboard

A local web view of the same desk snapshot the TUI shows:

```bash
python -m hedge_fund.web                 # http://127.0.0.1:8765
python -m hedge_fund.web --venue alpaca  # watch the Alpaca account
```

Needs the optional web dependencies (`pip install fastapi uvicorn`). It is
deliberately narrow:

- **It cannot trade.** Every route is a GET, and the broker behind the view is
  wrapped so that `place_order` raises. There is no endpoint that submits, and
  no static directory to serve from.
- **Loopback only.** Binding anywhere else is refused unless you pass
  `--allow-remote`, because the page is unencrypted and carries an account
  token. `0.0.0.0` counts as remote — it is every interface, not loopback.
- **Token required.** Taken from `HEDGE_FUND_WEB_TOKEN`, or generated and
  printed at startup. Compared with `secrets.compare_digest`.

`GET /api/desk` returns the snapshot as JSON and `GET /api/stream` is a
server-sent-events feed of it, so a script can watch an account without a
browser.

## Development

This fork lives at [bugman666/ai-hedge-fund](https://github.com/bugman666/ai-hedge-fund). See [CONTRIBUTING.md](CONTRIBUTING.md) for the first-test / first-backtest path.

```bash
git clone https://github.com/bugman666/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
poetry run pytest hedge_fund   # offline: no API keys required
poetry run aihf
```

Live Financial Datasets tests skip unless `FINANCIAL_DATASETS_API_KEY` is set.

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused. This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
