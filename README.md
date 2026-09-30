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
- One model API key for the investor agents. Supported providers: Anthropic, OpenAI, DeepSeek, Google, xAI, Kimi, TypeSafe (Jev).

Keys exported in your shell always win over the saved file.

## How to Run

### Interactive app

```bash
aihf
```

With no arguments, this launches the interactive terminal app. Build a fund — pick stocks, strategies, rebalance cadence — or backtest a saved fund and watch its equity curve draw against its benchmark. Funds you build are saved as mandate files in `~/.hedge-fund/mandates/`.

### Non-interactive

Run one fund cycle from a mandate file. The full cycle record prints to stdout as JSON; a short human summary goes to stderr:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT
```

Run the same mandate with Jev after configuring `TYPESAFE_API_KEY`:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --model jev-1.13.0
```

Backtest the mandate over history at its rebalance cadence:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest
```

An LLM trained after your backtest window may remember how those companies did, and that memory would score as skill. So a backtest withholds the ticker, industry and calendar dates from the investor agents' prompts; the personas see the fundamentals with periods labelled t-0, t-1, ... instead. This reduces the recall without removing it (distinctive numbers can still give a large company away), and a window after the model's training cutoff is the cleanest read. Live runs are unaffected and name the company.

A mandate is the desk — strategies, staff, risk, capital, cadence — and never names tickers; `--tickers` says what to point it at for this run.

## Paper trading on Alpaca

`aihf-paper` runs a fund forward on an Alpaca **paper** account (simulated money;
the client refuses any other endpoint). It trades exactly like the backtester:
on the first session of each rebalance period it assesses with data through the
previous close and submits market-on-close orders, so paper and backtest results
are directly comparable.

Add `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (paper keys) to `~/.hedge-fund/.env`, then:

```bash
aihf-paper status               # account, halts, last NAV
aihf-paper submit --dry-run     # see the plan, send nothing
aihf-paper baseline             # backtest fund, each strategy, SPY, equal-weight
aihf-paper install-schedule     # reconcile 09:00 ET, submit 10:00 ET, weekdays
aihf-paper report               # paper return vs benchmark, slippage, fills
```

Defaults: `hedge_fund/fund/paper.yaml` (all four strategies, long/short, unlevered,
10% per name) over `hedge_fund/fund/paper_universe.list` (32 large caps). The books
live in `~/.hedge-fund/paper/<fund>/`. Safety: `touch ~/.hedge-fund/KILL` stops all
trading; a 15% drawdown from peak halts and flattens the fund until
`aihf-paper resume`; `aihf-paper flatten --yes` does the same by hand.

## Development

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
poetry run aihf
poetry run pytest hedge_fund
```

## How to Contribute

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
