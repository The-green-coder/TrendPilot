"""
BacktestingEngine.py

Core portfolio backtesting engine for TrendPilot.

Responsibilities:
- Load historical prices (via MarketData.load_market_data)
- Apply allocation rule (Trends_RuleEngine)
- Simulate portfolio with transaction costs & slippage
- Compute performance & risk statistics
- Write CSV outputs for use by the Dashboard
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from MarketData import load_market_data

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """
    Container for key backtest outputs returned by BacktestingEngine.run().
    """

    performance_stats: pd.DataFrame
    risk_stats: pd.DataFrame
    allocation_series: pd.Series
    portfolio_values: pd.Series
    benchmark_values: pd.Series
    allocation_path: str
    performance_path: str
    risk_path: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _parse_backtesting_period(period: str) -> (date, date):
    """
    Parse a backtesting period string like "3M", "6M", "1Y", "5Y" into
    (start_date, end_date). End date is "today".
    """
    period = period.strip().upper()
    today = date.today()

    if period.endswith("M"):
        months = int(period[:-1])
        # naive month subtraction for backtest purposes
        year_delta = months // 12
        month_delta = months % 12
        start_year = today.year - year_delta
        start_month = today.month - month_delta
        while start_month <= 0:
            start_year -= 1
            start_month += 12
        start_day = min(today.day, 28)  # keep it simple
        start = date(start_year, start_month, start_day)
    elif period.endswith("Y"):
        years = int(period[:-1])
        start = date(today.year - years, today.month, today.day)
    else:
        # fallback: 5Y
        LOGGER.warning("Unrecognised backtesting_period '%s', falling back to 5Y", period)
        start = date(today.year - 5, today.month, today.day)

    return start, today


def _map_rebalance_frequency(freq: str) -> str:
    """
    Map human-readable rebalance frequency to pandas offset alias.

    Supported examples:
    - Daily
    - Weekly
    - Monthly
    - Quarterly
    - Yearly
    """
    freq = (freq or "").strip().lower()
    if freq in ("daily", "day", "d"):
        return "D"
    if freq in ("weekly", "week", "w"):
        return "W"
    if freq in ("bi-weekly", "biweekly", "2w"):
        return "2W"
    if freq in ("monthly", "month", "m"):
        return "M"
    if freq in ("bi-monthly", "bimonthly", "2m"):
        return "2M"
    if freq in ("quarterly", "q"):
        return "Q"
    if freq in ("half-yearly", "6m", "semiannual"):
        return "2Q"
    if freq in ("yearly", "year", "y", "annual"):
        return "A"
    LOGGER.warning("Unknown rebalance_frequency '%s', defaulting to Monthly", freq)
    return "M"


def _select_price_column(df: pd.DataFrame, preference: str) -> pd.Series:
    """
    Select a price column from OHLCV DataFrame given a preference string.
    Supported preferences: open, high, low, close, average.
    """
    preference = (preference or "close").strip().lower()

    if preference == "open":
        col = "open"
    elif preference == "high":
        col = "high"
    elif preference == "low":
        col = "low"
    elif preference == "average":
        # average of OHLC if all are available; fallback to close
        required = {"open", "high", "low", "close"}
        if required.issubset(set(df.columns)):
            return df[["open", "high", "low", "close"]].mean(axis=1)
        col = "close"
    else:
        col = "close"

    if col not in df.columns:
        raise KeyError(f"Price column '{col}' not found in DataFrame columns {df.columns}")

    return df[col]


# ---------------------------------------------------------------------------
# Backtesting engine implementation
# ---------------------------------------------------------------------------


class BacktestingEngine:
    """
    Main backtesting engine.

    The API is deliberately flexible: `run` accepts a **config dict via
    keyword arguments**, so TrendPilot.py can evolve without breaking this
    module.
    """

    def __init__(self) -> None:
        pass

    # ----- risk helpers -----------------------------------------------------

    def _sharpe(self, daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Compute annualised Sharpe ratio from daily returns.

        - daily_returns: Series of daily strategy or benchmark returns
        """
        if isinstance(daily_returns, pd.DataFrame):
            if daily_returns.shape[1] == 0:
                return float("nan")
            series = daily_returns.iloc[:, 0]
        else:
            series = daily_returns

        excess = series - risk_free_rate / 252.0
        std = excess.std()

        if std is None or np.isnan(std) or std == 0:
            return float("nan")

        return float(np.sqrt(252.0) * excess.mean() / std)

    def _max_drawdown(self, values: pd.Series) -> float:
        """
        Compute max drawdown (as a negative number, e.g. -0.35 for -35%).
        """
        running_max = values.cummax()
        dd = values / running_max - 1.0
        return float(dd.min())

    def _apply_delay(self, allocation: pd.Series, delay: int) -> pd.Series:
        """
        Apply a delay (in trading days) between signal and actual trade.
        """
        delay = int(delay or 0)
        if delay <= 0:
            return allocation
        # Shift weights forward in time; back-fill initial gap
        return allocation.shift(delay).bfill()

    # ----- core public API --------------------------------------------------

    def run(self, **config) -> BacktestResult:
        """
        Run a backtest with flexible configuration.

        Expected keys in `config` (all passed as keyword arguments):
        - risk_on_symbols: List[str]
        - risk_off_symbols: List[str]
        - benchmark_symbols: List[str]
        - rule: callable(price_series) -> allocation_series
        - rebalance_frequency: str
        - initial_capital: float
        - transaction_cost: float (percent per trade)
        - slippage: float (percent per trade)
        - backtesting_period: str like "5Y", "3M"
        - price_preference_buy: str
        - price_preference_sell: str
        - delay: int (days between signal and trade)
        - results_path: base directory for ResultsData (default "ResultsData")
        - strategy_name or strategy: used for subfolder name
        - risk_free_rate: float, optional (annualised)
        """

        # ---- unpack config with sensible defaults -------------------------
        risk_on_symbols: Sequence[str] = config["risk_on_symbols"]
        risk_off_symbols: Sequence[str] = config["risk_off_symbols"]
        benchmark_symbols: Sequence[str] = config["benchmark_symbols"]
        rule = config["rule"]
        rebalance_frequency: str = config["rebalance_frequency"]
        initial_capital: float = float(config["initial_capital"])
        transaction_cost_pct: float = float(config["transaction_cost"])  # e.g. 0.1 means 0.1%
        slippage_pct: float = float(config["slippage"])  # e.g. 0.05 means 0.05%
        backtesting_period: str = config["backtesting_period"]
        price_preference_buy: str = config["price_preference_buy"]
        price_preference_sell: str = config["price_preference_sell"]
        delay: int = int(config.get("delay", config.get("delay_between_signal_and_trade", 0)))
        results_path: str = config.get("results_path", "ResultsData")
        risk_free_rate: float = float(config.get("risk_free_rate", 0.0))
        strategy_name: str = config.get("strategy_name", config.get("strategy", "Strategy"))

        rebalance_alias = _map_rebalance_frequency(rebalance_frequency)
        start_date, end_date = _parse_backtesting_period(backtesting_period)

        LOGGER.info(
            "Running backtest for strategy '%s' from %s to %s (period=%s, rebalance=%s)",
            strategy_name,
            start_date,
            end_date,
            backtesting_period,
            rebalance_frequency,
        )

        # ---- load market data --------------------------------------------
        all_symbols = sorted(
            set(risk_on_symbols) | set(risk_off_symbols) | set(benchmark_symbols)
        )

        data_path = config.get("data_path", "MarketData")
        data: Dict[str, pd.DataFrame] = load_market_data(
            symbols=all_symbols,
            start=start_date,
            end=end_date,
            data_path=data_path,
            refresh=False,
        )

        # Ensure indices are DateTimeIndex and aligned
        for sym, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        # Common index across all symbols
        common_index = None
        for df in data.values():
            common_index = df.index if common_index is None else common_index.intersection(
                df.index
            )

        if common_index is None or len(common_index) == 0:
            raise RuntimeError("No overlapping dates across instruments for backtest")

        common_index = common_index.sort_values()
        for sym in all_symbols:
            data[sym] = data[sym].reindex(common_index).ffill().bfill()

        # ---- pick specific instruments -----------------------------------
        risk_on = risk_on_symbols[0]
        risk_off = risk_off_symbols[0]
        benchmark = benchmark_symbols[0]

        risk_on_df = data[risk_on]
        risk_off_df = data[risk_off]
        benchmark_df = data[benchmark]

        # ---- build signal price series -----------------------------------
        signal_price_series = _select_price_column(risk_on_df, price_preference_buy).astype(
            float
        )

        # ---- compute allocation from rule (0..1 risk-on) -----------------
        raw_allocation = rule(signal_price_series)
        raw_allocation = raw_allocation.reindex(common_index).astype(float)
        # Clip to [0, 1]
        raw_allocation = raw_allocation.clip(lower=0.0, upper=1.0)
        # Fill any missing values
        raw_allocation = raw_allocation.ffill().bfill().fillna(0.0)

        # ---- apply rebalance frequency -----------------------------------
        # Take allocation only at rebalance dates, then forward-fill.
        rebalance_dates = raw_allocation.resample(rebalance_alias).first().index
        allocation_rebalanced = raw_allocation.copy()
        allocation_rebalanced.loc[:] = np.nan
        allocation_rebalanced.loc[rebalance_dates] = raw_allocation.loc[rebalance_dates]
        allocation_rebalanced = allocation_rebalanced.ffill().bfill()

        # ---- apply delay between signal and trade ------------------------
        allocation = self._apply_delay(allocation_rebalanced, delay)

        # ---- compute returns for instruments -----------------------------
        def _price_series(df: pd.DataFrame) -> pd.Series:
            return _select_price_column(df, price_preference_sell).astype(float)

        risk_on_price = _price_series(risk_on_df)
        risk_off_price = _price_series(risk_off_df)
        benchmark_price = _price_series(benchmark_df)

        risk_on_returns = risk_on_price.pct_change().fillna(0.0)
        risk_off_returns = risk_off_price.pct_change().fillna(0.0)
        benchmark_returns = benchmark_price.pct_change().fillna(0.0)

        # ---- portfolio daily returns (before costs) ----------------------
        portfolio_returns = allocation * risk_on_returns + (1.0 - allocation) * risk_off_returns

        # ---- transaction cost & slippage model ---------------------------
        # Simple model: cost proportional to turnover in allocation
        turnover = allocation.diff().abs().fillna(0.0)
        cost_perc = (transaction_cost_pct + slippage_pct) / 100.0
        trading_costs = turnover * cost_perc

        net_returns = portfolio_returns - trading_costs

        # ---- equity curves -----------------------------------------------
        portfolio_values = (1.0 + net_returns).cumprod() * initial_capital
        benchmark_values = (1.0 + benchmark_returns).cumprod() * initial_capital

        final_portfolio_value = float(portfolio_values.iloc[-1])
        final_benchmark_value = float(benchmark_values.iloc[-1])

        LOGGER.info("Backtest completed. Final portfolio value: %.2f", final_portfolio_value)
        LOGGER.info("Benchmark final value: %.2f", final_benchmark_value)

        # ------------------------------------------------------------------
        # Allocation time statistics
        # ------------------------------------------------------------------
        avg_risk_on_alloc_pct = float(allocation.mean() * 100.0)
        avg_risk_off_alloc_pct = 100.0 - avg_risk_on_alloc_pct

        # % of days with any risk-on allocation and full risk-off allocation
        time_any_risk_on_pct = float((allocation > 0.0).mean() * 100.0)
        time_full_risk_off_pct = float((allocation == 0.0).mean() * 100.0)

        # ------------------------------------------------------------------
        # Performance contribution attribution
        # ------------------------------------------------------------------
        risk_on_leg = allocation * risk_on_returns
        risk_off_leg = (1.0 - allocation) * risk_off_returns

        total_portfolio_return = net_returns.sum()
        if total_portfolio_return != 0:
            risk_on_contrib_pct = float(risk_on_leg.sum() / total_portfolio_return * 100.0)
            risk_off_contrib_pct = float(risk_off_leg.sum() / total_portfolio_return * 100.0)
        else:
            risk_on_contrib_pct = float("nan")
            risk_off_contrib_pct = float("nan")

        # ------------------------------------------------------------------
        # Risk statistics vs benchmark
        # ------------------------------------------------------------------
        daily_portfolio_returns = portfolio_values.pct_change().fillna(0.0)
        daily_benchmark_returns = benchmark_values.pct_change().fillna(0.0)

        sharpe_strategy = self._sharpe(daily_portfolio_returns, risk_free_rate)
        sharpe_benchmark = self._sharpe(daily_benchmark_returns, risk_free_rate)

        max_dd_strategy = self._max_drawdown(portfolio_values)
        max_dd_benchmark = self._max_drawdown(benchmark_values)

        vol_strategy = float(daily_portfolio_returns.std() * np.sqrt(252.0))
        vol_benchmark = float(daily_benchmark_returns.std() * np.sqrt(252.0))

        # ------------------------------------------------------------------
        # Build CSV outputs
        # ------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_root = results_path or "ResultsData"
        strategy_dir = os.path.join(results_root, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)

        performance_stats = pd.DataFrame(
            {
                "final_value": [final_portfolio_value],
                "benchmark_final_value": [final_benchmark_value],
                "total_return_pct": [
                    float((final_portfolio_value / float(portfolio_values.iloc[0]) - 1.0) * 100.0)
                ],
                "benchmark_return_pct": [
                    float(
                        (final_benchmark_value / float(benchmark_values.iloc[0]) - 1.0)
                        * 100.0
                    )
                ],
                "avg_risk_on_alloc_pct": [avg_risk_on_alloc_pct],
                "avg_risk_off_alloc_pct": [avg_risk_off_alloc_pct],
                "time_any_risk_on_pct": [time_any_risk_on_pct],
                "time_full_risk_off_pct": [time_full_risk_off_pct],
                "risk_on_contrib_pct": [risk_on_contrib_pct],
                "risk_off_contrib_pct": [risk_off_contrib_pct],
            }
        )

        risk_stats = pd.DataFrame(
            {
                "sharpe_strategy": [sharpe_strategy],
                "sharpe_benchmark": [sharpe_benchmark],
                "vol_strategy": [vol_strategy],
                "vol_benchmark": [vol_benchmark],
                "max_drawdown_strategy": [max_dd_strategy],
                "max_drawdown_benchmark": [max_dd_benchmark],
            }
        )

        # Allocation & equity path over time
        allocation_df = pd.DataFrame(
            {
                "date": common_index,
                "allocation_risk_on": allocation.values,
                "allocation_risk_off": 1.0 - allocation.values,
                "portfolio_value": portfolio_values.values,
                "benchmark_value": benchmark_values.values,
                "risk_on_price": risk_on_price.values,
                "risk_off_price": risk_off_price.values,
                "benchmark_price": benchmark_price.values,
            }
        )

        perf_path = os.path.join(strategy_dir, f"PerformanceStats_{timestamp}.csv")
        risk_path = os.path.join(strategy_dir, f"RiskStats_{timestamp}.csv")
        alloc_path = os.path.join(strategy_dir, f"Allocation_{timestamp}.csv")

        performance_stats.to_csv(perf_path, index=False)
        risk_stats.to_csv(risk_path, index=False)
        allocation_df.to_csv(alloc_path, index=False)

        # Log summary to console
        LOGGER.info(
            "Average allocation: %.2f%% Risk-On, %.2f%% Risk-Off",
            avg_risk_on_alloc_pct,
            avg_risk_off_alloc_pct,
        )
        LOGGER.info(
            "Return contribution: %.2f%% from Risk-On, %.2f%% from Risk-Off",
            risk_on_contrib_pct,
            risk_off_contrib_pct,
        )
        LOGGER.info(
            "Strategy Sharpe: %.3f vs Benchmark Sharpe: %.3f",
            sharpe_strategy,
            sharpe_benchmark,
        )
        LOGGER.info(
            "Strategy max drawdown: %.2f%% vs Benchmark: %.2f%%",
            max_dd_strategy * 100.0,
            max_dd_benchmark * 100.0,
        )

        return BacktestResult(
            performance_stats=performance_stats,
            risk_stats=risk_stats,
            allocation_series=allocation,
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            allocation_path=alloc_path,
            performance_path=perf_path,
            risk_path=risk_path,
        )
