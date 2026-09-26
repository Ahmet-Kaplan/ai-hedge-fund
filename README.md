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

Order submission is gated three times over, so a half-remembered environment
variable cannot move real money:

| Variable | Default | Effect |
|----------|---------|--------|
| `ALPACA_PAPER` | `true` | paper endpoint; reads always work |
| `ALPACA_TRADING_ENABLED` | `false` | required before *any* order is submitted |
| `ALPACA_LIVE_TRADING_CONFIRMED` | `false` | additionally required when `ALPACA_PAPER=false` |

`ALPACA_API_SECRET` is accepted as an alias for `ALPACA_SECRET_KEY`.

Two behaviours differ from the offline brokers on purpose. `place_order` must
fill *completely* or raise — the fund sizes its book from the returned `Fill`,
so Alpaca's asynchronous executions are polled briefly and anything short of a
full fill raises `AlpacaOrderError` carrying the broker order id rather than
inventing a fill. And a fractional position raises instead of being truncated,
because `Position.shares` is an integer share count and rounding would desync
the fund's books from the broker's.

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
