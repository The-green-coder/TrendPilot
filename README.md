# TrendPilot

TrendPilot is a rules-based trend-following backtesting engine. It combines configurable strategies, a rules engine, market-data retrieval, and a Plotly Dash dashboard for visualising results.

## Project layout
- `TrendPilot.py` – CLI entrypoint that merges default, strategy, and CLI configs then runs a backtest.
- `BacktestingEngine.py` – Portfolio simulation, metrics, and CSV output.
- `MarketData.py` – Data retrieval from Yahoo Finance with offline synthetic fallback.
- `Trends_RuleEngine.py` – Moving-average based allocation rules with a registry for discovery.
- `Dashboard.py` – Minimal Dash app to display results.
- `default_config.csv` – Global defaults.
- `Strategy_Centre/USTechTripleTrend` – Example strategy with config and instrument lists.
- `MarketData/` – Cached price history.
- `ResultsData/` – Backtest outputs.

## Usage
Run a strategy with the default configuration:
```bash
python TrendPilot.py --strategy USTechTripleTrend
```

Override parameters at runtime (examples):
```bash
python TrendPilot.py --strategy USTechTripleTrend --backtesting_period 1Y --rebalance_frequency Monthly --rule_name TripleTrend
```

Launch the dashboard after running a backtest:
```bash
python Dashboard.py
```

If network access is unavailable, market data is simulated to keep the workflow operational. Performance outputs now include time-in-market percentages for Risk On/Risk Off, contribution breakdowns by leg and trading costs, and risk metrics that are benchmark-aware (volatility, max drawdown, Sharpe).

### Dependencies and offline mode
The codebase is designed to run even in isolated environments where installing external Python packages is not possible. Lightweight drop-in stubs (`pandas_stub.py` and `numpy_stub.py`) live in the repository so the CLI backtest can execute without fetching wheels from the internet. When full libraries are available they can be installed normally and will be preferred.

### Pulling remote updates after you have local commits
If you have local commits and want to bring in new upstream changes, use a rebase-based pull to keep history tidy:

1. Ensure your working tree is clean (commit or stash any local changes):
   ```bash
   git status
   ```
2. Fetch the latest remote refs:
   ```bash
   git fetch origin
   ```
3. Rebase your local branch onto the updated remote branch (for example, `main`):
   ```bash
   git pull --rebase origin main
   ```
   - If conflicts arise, resolve them file by file, `git add` the fixes, and continue:
     ```bash
     git add <resolved-files>
     git rebase --continue
     ```
4. Once the rebase completes, run your tests/backtest to verify everything still works, then push:
   ```bash
   git push origin main
   ```

This flow preserves your commits on top of the latest remote history and avoids unnecessary merge commits.
