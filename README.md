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
