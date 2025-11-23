"""Backtesting logic for TrendPilot.

This module orchestrates strategy execution, including signal
calculation, portfolio simulation, and metric generation.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from MarketData import load_market_data
from Trends_RuleEngine import get_rule


REBALANCE_FREQ = {
    "DAILY": "D",
    "WEEKLY": "W-FRI",
    "MONTHLY": "M",
    "QUARTERLY": "Q",
}


class BacktestResult:
    def __init__(self, performance: pd.DataFrame, risk: pd.DataFrame, allocations: pd.DataFrame, equity_curve: pd.DataFrame):
        self.performance = performance
        self.risk = risk
        self.allocations = allocations
        self.equity_curve = equity_curve


class BacktestingEngine:
    def __init__(self, config: Dict[str, str], strategy_name: str):
        self.config = config
        self.strategy_name = strategy_name

    def _get_price(self, df: pd.DataFrame, preference: str) -> pd.Series:
        preference = preference.lower()
        if preference not in df.columns:
            raise KeyError(f"Price preference '{preference}' not available in data columns {df.columns}")
        return df[preference]

    def _resample_allocations(self, allocation: pd.Series, frequency: str) -> pd.Series:
        freq = REBALANCE_FREQ.get(frequency.upper(), "W-FRI")
        return allocation.resample(freq).last().reindex(allocation.index).ffill()

    def _apply_delay(self, allocation: pd.Series, delay: int) -> pd.Series:
        if delay <= 0:
            return allocation
        return allocation.shift(delay).fillna(method="bfill")

    def _portfolio_returns(self, risk_on_allocation: pd.Series, risk_on_returns: pd.Series, risk_off_returns: pd.Series, transaction_cost: float, slippage: float) -> pd.Series:
        allocation_change = risk_on_allocation.diff().fillna(risk_on_allocation)
        cost = (transaction_cost + slippage) * allocation_change.abs()
        daily_return = risk_on_allocation * risk_on_returns + (1 - risk_on_allocation) * risk_off_returns - cost
        return daily_return

    def _max_drawdown(self, equity: pd.Series) -> float:
        cumulative_max = equity.cummax()
        drawdown = (equity - cumulative_max) / cumulative_max
        return drawdown.min()

    def _annualized_volatility(self, returns: pd.Series) -> float:
        return returns.std() * np.sqrt(252)

    def _cagr(self, equity: pd.Series) -> float:
        total_return = equity.iloc[-1] / equity.iloc[0]
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        return total_return ** (1 / years) - 1 if years > 0 else np.nan

    def _sharpe(self, daily_returns, risk_free_rate: float = 0.0):
        """
        Compute annualised Sharpe ratio.
    
        Handles both:
        - pandas.Series of daily returns
        - pandas.DataFrame (uses the first column by default)
        """
        import pandas as pd
        import numpy as np
    
        # If we get a DataFrame, assume first column is the strategy returns
        if isinstance(daily_returns, pd.DataFrame):
            if daily_returns.shape[1] == 0:
                return np.nan
            series = daily_returns.iloc[:, 0]
        else:
            series = daily_returns
    
        # Excess returns over daily risk-free rate
        excess = series - risk_free_rate / 252.0
    
        std = excess.std()
    
        # Guard against NaN or zero std
        if pd.isna(std) or std == 0:
            return np.nan
    
        return float(np.sqrt(252.0) * excess.mean() / std)

    def run(self, risk_on_symbols: Iterable[str], risk_off_symbols: Iterable[str], benchmark_symbols: Iterable[str], start: datetime, end: datetime, data_path: str, results_path: str) -> BacktestResult:
        data = load_market_data(set(risk_on_symbols) | set(risk_off_symbols) | set(benchmark_symbols), start, end, data_path, refresh=False)

        risk_on_symbol = list(risk_on_symbols)[0]
        risk_off_symbol = list(risk_off_symbols)[0]
        benchmark_symbol = list(benchmark_symbols)[0]

        risk_on = data[risk_on_symbol]
        risk_off = data[risk_off_symbol]
        benchmark = data[benchmark_symbol]

        price_preference_buy = self.config.get("buy_price_preference", "close")
        price_preference_sell = self.config.get("sell_price_preference", "close")
        benchmark_preference = self.config.get("benchmark_price_preference", "close")

        price_series = self._get_price(risk_on, price_preference_buy)
        rule = get_rule(self.config["rule_name"])
        allocation = rule(price_series)

        allocation = self._resample_allocations(allocation, self.config.get("rebalance_frequency", "Weekly"))
        allocation = self._apply_delay(allocation, int(self.config.get("delay_between_signal_and_trade", 0)))
        allocation = allocation.clip(0, 1)

        # Convert selected price series to numeric before computing returns
        risk_on_price = self._get_price(risk_on, price_preference_sell)
        risk_on_price = pd.to_numeric(risk_on_price, errors="coerce")
        risk_on_returns = risk_on_price.pct_change().fillna(0)
        
        risk_off_price = self._get_price(risk_off, price_preference_sell)
        risk_off_price = pd.to_numeric(risk_off_price, errors="coerce")
        risk_off_returns = risk_off_price.pct_change().fillna(0)
        
        benchmark_price = self._get_price(benchmark, price_preference_sell)
        benchmark_price = pd.to_numeric(benchmark_price, errors="coerce")
        benchmark_returns = benchmark_price.pct_change().fillna(0)
        
        
        daily_returns = self._portfolio_returns(allocation, risk_on_returns, risk_off_returns, float(self.config.get("transaction_cost", 0)), float(self.config.get("slippage", 0)))
        equity = (1 + daily_returns).cumprod() * float(self.config.get("initial_capital", 100000))
        benchmark_equity = (1 + benchmark_returns).cumprod() * float(self.config.get("initial_capital", 100000))

        risk_free_rate = float(self.config.get("risk_free_rate", 0.0))

        performance = pd.DataFrame({
            "final_portfolio_value": [equity.iloc[-1]],
            "benchmark_final_value": [benchmark_equity.iloc[-1]],
            "total_return": [equity.iloc[-1] / equity.iloc[0] - 1],
            "benchmark_total_return": [benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1],
            "cagr": [self._cagr(equity)],
        })

        risk_stats = pd.DataFrame({
            "volatility": [self._annualized_volatility(daily_returns)],
            "benchmark_volatility": [self._annualized_volatility(benchmark_returns)],
            "max_drawdown": [self._max_drawdown(equity / equity.iloc[0])],
            "sharpe_ratio": [self._sharpe(daily_returns, risk_free_rate)],
        })

        allocations = pd.DataFrame({
            "risk_on_allocation": allocation,
            "risk_off_allocation": 1 - allocation,
        })

        equity_curve = pd.DataFrame({
            "equity": equity,
            "benchmark": benchmark_equity,
        })

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        strategy_results_path = os.path.join(results_path, self.strategy_name)
        os.makedirs(strategy_results_path, exist_ok=True)

        performance.to_csv(os.path.join(strategy_results_path, f"PerformanceStats_{timestamp}.csv"), index=False)
        risk_stats.to_csv(os.path.join(strategy_results_path, f"RiskStats_{timestamp}.csv"), index=False)
        allocations.reset_index().rename(columns={"index": "date"}).to_csv(os.path.join(strategy_results_path, f"Allocation_{timestamp}.csv"), index=False)
        equity_curve.reset_index().rename(columns={"index": "date"}).to_csv(os.path.join(strategy_results_path, f"Equity_{timestamp}.csv"), index=False)

        return BacktestResult(performance, risk_stats, allocations, equity_curve)
