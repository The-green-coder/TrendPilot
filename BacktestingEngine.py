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
from datetime import date, datetime
from typing import Dict, Sequence

# Prefer real numpy/pandas; fall back to lightweight stubs for offline environments
try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    import numpy_stub as np  # type: ignore

try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    import pandas_stub as pd  # type: ignore

from MarketData import load_market_data

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helper dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    performance_stats: pd.DataFrame
    risk_stats: pd.DataFrame
    allocation_series: pd.Series
    portfolio_values: pd.Series
    benchmark_values: pd.Series
    allocation_path: str
    performance_path: str
    risk_path: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_backtesting_period(period: str) -> (date, date):
    """
    Parse backtesting period like '3M', '6M', '1Y', '5Y' into (start_date, end_date).
    End date is today.
    """
    period = (period or "").strip().upper()
    today = date.today()

    if period.endswith("M"):
        months = int(period[:-1])
        year_delta = months // 12
        month_delta = months % 12
        start_year = today.year - year_delta
        start_month = today.month - month_delta
        while start_month <= 0:
            start_year -= 1
            start_month += 12
        start_day = min(today.day, 28)
        start = date(start_year, start_month, start_day)
    elif period.endswith("Y"):
        years = int(period[:-1])
        start = date(today.year - years, today.month, today.day)
    else:
        LOGGER.warning("Unrecognised backtesting_period '%s', defaulting to 5Y", period)
        start = date(today.year - 5, today.month, today.day)

    return start, today


def _map_rebalance_frequency(freq: str) -> str:
    """
    Map human-readable rebalance frequency to pandas resample alias.
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
    Select OHLC price series according to preference: open, high, low, close, average.
    """
    preference = (preference or "close").strip().lower()

    if preference == "open":
        col = "open"
    elif preference == "high":
        col = "high"
    elif preference == "low":
        col = "low"
    elif preference == "average":
        required = {"open", "high", "low", "close"}
        if required.issubset(df.columns):
            return df[["open", "high", "low", "close"]].mean(axis=1)
        col = "close"
    else:
        col = "close"

    if col not in df.columns:
        raise KeyError(f"Price column '{col}' not found in {df.columns}")

    return df[col]


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class BacktestingEngine:
    """
    Main backtesting engine.
    """

    def __init__(self) -> None:
        pass

    # ---- risk helpers -----------------------------------------------------

    def _sharpe(self, daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio from daily returns.
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
        Max drawdown as a negative number (e.g. -0.35 == -35%).
        """
        running_max = values.cummax()
        dd = values / running_max - 1.0
        return float(dd.min())

    def _apply_delay(self, allocation: pd.Series, delay: int) -> pd.Series:
        """
        Apply a delay in trading days between signal and execution.
        """
        delay = int(delay or 0)
        if delay <= 0:
            return allocation
        return allocation.shift(delay).bfill()

    # ---- main API ---------------------------------------------------------

    def run(self, **config) -> BacktestResult:
        """
        Run a backtest. Expected config keys:

        - risk_on_symbols: List[str]
        - risk_off_symbols: List[str]
        - benchmark_symbols: List[str]
        - rule: callable(price_series) -> allocation_series
        - rebalance_frequency: str
        - initial_capital: float
        - transaction_cost: float (percent per trade)
        - slippage: float (percent per trade)
        - backtesting_period: str
        - price_preference_buy: str
        - price_preference_sell: str
        - record_backtest_details: Yes/No to emit NAV and transaction CSVs
        - delay: int
        - results_path: str
        - strategy_name: str
        - risk_free_rate: float (optional)
        """
        risk_on_symbols: Sequence[str] = config["risk_on_symbols"]
        risk_off_symbols: Sequence[str] = config["risk_off_symbols"]
        benchmark_symbols: Sequence[str] = config["benchmark_symbols"]
        rule = config["rule"]

        rebalance_frequency: str = config["rebalance_frequency"]
        initial_capital: float = float(config["initial_capital"])
        transaction_cost_pct: float = float(config["transaction_cost"])
        slippage_pct: float = float(config["slippage"])
        backtesting_period: str = config["backtesting_period"]
        price_preference_buy: str = config["price_preference_buy"]
        price_preference_sell: str = config["price_preference_sell"]
        record_backtest_details: bool = str(
            config.get("record_backtest_details", "No")
        ).strip().lower() == "yes"
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
        all_symbols = sorted(set(risk_on_symbols) | set(risk_off_symbols) | set(benchmark_symbols))
        data_path = config.get("data_path", "MarketData")

        data: Dict[str, pd.DataFrame] = load_market_data(
            symbols=all_symbols,
            start=start_date,
            end=end_date,
            data_path=data_path,
            refresh=False,
        )

        # normalise indices
        for sym, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        # find common index
        common_index = None
        for df in data.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        if common_index is None or len(common_index) == 0:
            raise RuntimeError("No overlapping dates across instruments for backtest")

        common_index = common_index.sort_values()
        for sym in all_symbols:
            data[sym] = data[sym].reindex(common_index).ffill().bfill()

        # pick primary instruments (single symbol for now)
        risk_on = risk_on_symbols[0]
        risk_off = risk_off_symbols[0]
        benchmark = benchmark_symbols[0]

        risk_on_df = data[risk_on]
        risk_off_df = data[risk_off]
        benchmark_df = data[benchmark]

        # ---- signal prices & rule ----------------------------------------
        # Use robust numeric conversion to avoid issues like 'qqq' strings
        raw_signal_prices = _select_price_column(risk_on_df, price_preference_buy)
        signal_price_series = pd.to_numeric(raw_signal_prices, errors="coerce")
        signal_price_series = signal_price_series.ffill().bfill()

        # ensure rule output is a Series aligned to common_index
        raw_allocation = rule(signal_price_series)
        raw_allocation = pd.Series(raw_allocation, index=signal_price_series.index)
        raw_allocation = raw_allocation.reindex(common_index).astype(float)
        raw_allocation = raw_allocation.clip(0.0, 1.0).ffill().bfill().fillna(0.0)

        # ---- rebalance using resample (no manual .loc) -------------------
        # resample to rebalance frequency (e.g. weekly), then forward-fill
        allocation_resampled = raw_allocation.resample(rebalance_alias).first()
        allocation_rebalanced = allocation_resampled.reindex(common_index, method="ffill").bfill()

        # ---- apply delay --------------------------------------------------
        allocation = self._apply_delay(allocation_rebalanced, delay)

        # ---- compute returns ---------------------------------------------
        def _price_series(df: pd.DataFrame, preference: str) -> pd.Series:
            raw = _select_price_column(df, preference)
            s = pd.to_numeric(raw, errors="coerce")
            return s.ffill().bfill()

        risk_on_price = _price_series(risk_on_df, price_preference_sell)
        risk_off_price = _price_series(risk_off_df, price_preference_sell)
        benchmark_price = _price_series(benchmark_df, price_preference_sell)

        risk_on_buy_price = _price_series(risk_on_df, price_preference_buy)
        risk_off_buy_price = _price_series(risk_off_df, price_preference_buy)

        risk_on_returns = risk_on_price.pct_change().fillna(0.0)
        risk_off_returns = risk_off_price.pct_change().fillna(0.0)
        benchmark_returns = benchmark_price.pct_change().fillna(0.0)

        portfolio_returns = allocation * risk_on_returns + (1.0 - allocation) * risk_off_returns

        # costs
        turnover = allocation.diff().abs().fillna(0.0)
        cost_perc = (transaction_cost_pct + slippage_pct) / 100.0
        trading_costs = turnover * cost_perc
        net_returns = portfolio_returns - trading_costs

        # equity curves
        portfolio_values = (1.0 + net_returns).cumprod() * initial_capital
        benchmark_values = (1.0 + benchmark_returns).cumprod() * initial_capital

        final_portfolio_value = float(portfolio_values.iloc[-1])
        final_benchmark_value = float(benchmark_values.iloc[-1])

        LOGGER.info("Backtest completed. Final portfolio value: %.2f", final_portfolio_value)
        LOGGER.info("Benchmark final value: %.2f", final_benchmark_value)

        # ---- allocation time stats ---------------------------------------
        avg_risk_on_alloc_pct = float(allocation.mean() * 100.0)
        avg_risk_off_alloc_pct = 100.0 - avg_risk_on_alloc_pct
        time_any_risk_on_pct = float((allocation > 0.0).mean() * 100.0)
        time_full_risk_off_pct = float((allocation == 0.0).mean() * 100.0)

        # ---- performance contribution ------------------------------------
        risk_on_leg = allocation * risk_on_returns
        risk_off_leg = (1.0 - allocation) * risk_off_returns

        total_portfolio_return = net_returns.sum()
        if total_portfolio_return != 0:
            risk_on_contrib_pct = float(risk_on_leg.sum() / total_portfolio_return * 100.0)
            risk_off_contrib_pct = float(risk_off_leg.sum() / total_portfolio_return * 100.0)
        else:
            risk_on_contrib_pct = float("nan")
            risk_off_contrib_pct = float("nan")

        # standalone asset returns (for intuition)
        risk_on_total_return_pct = float(
            (risk_on_price.iloc[-1] / risk_on_price.iloc[0] - 1.0) * 100.0
        )
        risk_off_total_return_pct = float(
            (risk_off_price.iloc[-1] / risk_off_price.iloc[0] - 1.0) * 100.0
        )

        # ---- risk metrics -------------------------------------------------
        daily_portfolio_returns = portfolio_values.pct_change().fillna(0.0)
        daily_benchmark_returns = benchmark_values.pct_change().fillna(0.0)

        sharpe_strategy = self._sharpe(daily_portfolio_returns, risk_free_rate)
        sharpe_benchmark = self._sharpe(daily_benchmark_returns, risk_free_rate)

        max_dd_strategy = self._max_drawdown(portfolio_values)
        max_dd_benchmark = self._max_drawdown(benchmark_values)

        vol_strategy = float(daily_portfolio_returns.std() * np.sqrt(252.0))
        vol_benchmark = float(daily_benchmark_returns.std() * np.sqrt(252.0))

        # ---- CSV outputs --------------------------------------------------
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
                "risk_on_total_return_pct": [risk_on_total_return_pct],
                "risk_off_total_return_pct": [risk_off_total_return_pct],
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

        risk_off_alloc = 1.0 - allocation

        allocation_df = pd.DataFrame(
            {
                "date": common_index,
                "allocation_risk_on": allocation.values,
                "allocation_risk_off": risk_off_alloc.values,
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

        if record_backtest_details:
            nav_path = os.path.join(
                strategy_dir, f"{strategy_name}_DailyNAV_{timestamp}.csv"
            )
            nav_df = pd.DataFrame(
                {
                    "Date": common_index,
                    "RiskOnAlloc": allocation.values * 100.0,
                    "RiskOffAlloc": (1.0 - allocation.values) * 100.0,
                    "NAV": (portfolio_values / initial_capital).values,
                    "PortfolioValue": portfolio_values.values,
                }
            )
            nav_df.to_csv(nav_path, index=False)

            txn_records = []
            rebalancing_iteration = 0
            prev_allocation = 0.0
            for i, dt in enumerate(common_index):
                alloc_today = float(allocation.iloc[i])
                prev_value = (
                    initial_capital if i == 0 else float(portfolio_values.iloc[i - 1])
                )
                diff = alloc_today - prev_allocation
                if np.isclose(diff, 0.0):
                    prev_allocation = alloc_today
                    continue

                rebalancing_iteration += 1
                trade_value = abs(diff) * prev_value
                txn_cost_amount = trade_value * transaction_cost_pct / 100.0 * 0.5
                slippage_amount = trade_value * slippage_pct / 100.0 * 0.5
                total_cost = txn_cost_amount + slippage_amount

                if diff > 0:
                    # Increasing exposure to risk-on: sell risk-off, buy risk-on
                    sell_price = float(risk_off_price.iloc[i])
                    sell_qty = trade_value / sell_price if sell_price else 0.0
                    txn_records.append(
                        {
                            "Date": dt,
                            "RebalancingIteration": rebalancing_iteration,
                            "Instrument": risk_off,
                            "Action": "Sell",
                            "Price": sell_price,
                            "Quantity": sell_qty,
                            "TransactionCost": txn_cost_amount,
                            "Slippage": slippage_amount,
                            "TotalCost": total_cost,
                        }
                    )

                    buy_price = float(risk_on_buy_price.iloc[i])
                    buy_qty = trade_value / buy_price if buy_price else 0.0
                    txn_records.append(
                        {
                            "Date": dt,
                            "RebalancingIteration": rebalancing_iteration,
                            "Instrument": risk_on,
                            "Action": "Buy",
                            "Price": buy_price,
                            "Quantity": buy_qty,
                            "TransactionCost": txn_cost_amount,
                            "Slippage": slippage_amount,
                            "TotalCost": total_cost,
                        }
                    )
                else:
                    # Decreasing exposure to risk-on: sell risk-on, buy risk-off
                    sell_price = float(risk_on_price.iloc[i])
                    sell_qty = trade_value / sell_price if sell_price else 0.0
                    txn_records.append(
                        {
                            "Date": dt,
                            "RebalancingIteration": rebalancing_iteration,
                            "Instrument": risk_on,
                            "Action": "Sell",
                            "Price": sell_price,
                            "Quantity": sell_qty,
                            "TransactionCost": txn_cost_amount,
                            "Slippage": slippage_amount,
                            "TotalCost": total_cost,
                        }
                    )

                    buy_price = float(risk_off_buy_price.iloc[i])
                    buy_qty = trade_value / buy_price if buy_price else 0.0
                    txn_records.append(
                        {
                            "Date": dt,
                            "RebalancingIteration": rebalancing_iteration,
                            "Instrument": risk_off,
                            "Action": "Buy",
                            "Price": buy_price,
                            "Quantity": buy_qty,
                            "TransactionCost": txn_cost_amount,
                            "Slippage": slippage_amount,
                            "TotalCost": total_cost,
                        }
                    )

                prev_allocation = alloc_today

            txn_path = os.path.join(
                strategy_dir, f"{strategy_name}_TransactionRecord_{timestamp}.csv"
            )
            txn_df = pd.DataFrame(
                txn_records,
                columns=[
                    "Date",
                    "RebalancingIteration",
                    "Instrument",
                    "Action",
                    "Price",
                    "Quantity",
                    "TransactionCost",
                    "Slippage",
                    "TotalCost",
                ],
            )
            txn_df.to_csv(txn_path, index=False)

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
            "Standalone asset returns: Risk-On %.2f%%, Risk-Off %.2f%%",
            risk_on_total_return_pct,
            risk_off_total_return_pct,
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

        strategy_perf_table = pd.DataFrame(
            {
                "Metric": [
                    "Final Portfolio Value",
                    "Total Return %",
                    "Avg Risk-On Alloc %",
                    "Avg Risk-Off Alloc %",
                    "Time Any Risk-On %",
                    "Time Full Risk-Off %",
                    "Risk-On Contribution %",
                    "Risk-Off Contribution %",
                ],
                "Value": [
                    round(final_portfolio_value, 2),
                    round(performance_stats["total_return_pct"].iloc[0], 2),
                    round(avg_risk_on_alloc_pct, 2),
                    round(avg_risk_off_alloc_pct, 2),
                    round(time_any_risk_on_pct, 2),
                    round(time_full_risk_off_pct, 2),
                    round(risk_on_contrib_pct, 2),
                    round(risk_off_contrib_pct, 2),
                ],
            }
        )

        comparison_table = pd.DataFrame(
            {
                "Metric": ["Final Value", "Total Return %"],
                "Strategy": [
                    round(final_portfolio_value, 2),
                    round(performance_stats["total_return_pct"].iloc[0], 2),
                ],
                "Benchmark": [
                    round(final_benchmark_value, 2),
                    round(performance_stats["benchmark_return_pct"].iloc[0], 2),
                ],
            }
        )

        risk_table = pd.DataFrame(
            {
                "Metric": ["Sharpe", "Volatility", "Max Drawdown %"],
                "Strategy": [
                    round(risk_stats["sharpe_strategy"].iloc[0], 3),
                    round(risk_stats["vol_strategy"].iloc[0], 4),
                    round(risk_stats["max_drawdown_strategy"].iloc[0] * 100.0, 2),
                ],
                "Benchmark": [
                    round(risk_stats["sharpe_benchmark"].iloc[0], 3),
                    round(risk_stats["vol_benchmark"].iloc[0], 4),
                    round(risk_stats["max_drawdown_benchmark"].iloc[0] * 100.0, 2),
                ],
            }
        )

        print("\nStrategy Performance Summary")
        print(strategy_perf_table.to_string(index=False))

        print("\nStrategy vs Benchmark")
        print(comparison_table.to_string(index=False))

        print("\nRisk Statistics")
        print(risk_table.to_string(index=False))

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
