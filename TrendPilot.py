"""CLI entrypoint for TrendPilot backtesting engine."""
from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, List

from BacktestingEngine import BacktestingEngine
from MarketData import cleanup_data, parse_period
from Trends_RuleEngine import RULE_REGISTRY

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def read_config_csv(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {row["parameter"]: row["value"] for row in reader}


def load_instruments(instrument_path: str, filename: str) -> List[str]:
    path = os.path.join(instrument_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instrument file missing: {path}")
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        symbols = [row["symbol"] for row in reader if row.get("symbol")]
    if not symbols:
        raise ValueError(f"No symbols found in {path}")
    return symbols


def merge_config(default: Dict[str, str], strategy: Dict[str, str], cli_overrides: Dict[str, str]) -> Dict[str, str]:
    merged = {**default, **strategy}
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrendPilot Backtesting Engine")
    parser.add_argument("--strategy", default="USTechTripleTrend", help="Strategy folder name inside Strategy_Centre")
    parser.add_argument("--backtesting_period")
    parser.add_argument("--initial_capital")
    parser.add_argument("--transaction_cost")
    parser.add_argument("--slippage")
    parser.add_argument("--rebalance_frequency")
    parser.add_argument("--buy_price_preference")
    parser.add_argument("--sell_price_preference")
    parser.add_argument("--rule_name")
    parser.add_argument("--delay_between_signal_and_trade")
    parser.add_argument("--eraseDataAfterRun")
    parser.add_argument("--benchmark_price_preference")
    parser.add_argument("--risk_free_rate")
    parser.add_argument("--data_path")
    parser.add_argument("--results_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = read_config_csv("default_config.csv")

    strategy_dir = os.path.join("Strategy_Centre", args.strategy)
    strategy_config_path = os.path.join(strategy_dir, "StrategyConfig.csv")
    strategy_config = read_config_csv(strategy_config_path)

    cli_overrides = {k: v for k, v in vars(args).items() if k != "strategy" and v is not None}
    final_config = merge_config(default_config, strategy_config, cli_overrides)

    instrument_dir = os.path.join(strategy_dir, "Instruments")
    risk_on_symbols = load_instruments(instrument_dir, "RiskOn.csv")
    risk_off_symbols = load_instruments(instrument_dir, "RiskOff.csv")
    benchmark_symbols = load_instruments(instrument_dir, "Benchmark.csv")

    start, end = parse_period(final_config["backtesting_period"])

    LOGGER.info("Running strategy %s using rule %s", args.strategy, final_config["rule_name"])
    LOGGER.info("Available rules: %s", ", ".join(RULE_REGISTRY.keys()))

    engine = BacktestingEngine(final_config, args.strategy)
    result = engine.run(
        risk_on_symbols=risk_on_symbols,
        risk_off_symbols=risk_off_symbols,
        benchmark_symbols=benchmark_symbols,
        start=start,
        end=end,
        data_path=final_config.get("data_path", "MarketData"),
        results_path=final_config.get("results_path", "ResultsData"),
    )

    LOGGER.info("Backtest completed. Final portfolio value: %.2f", result.performance["final_portfolio_value"].iloc[0])
    LOGGER.info("Benchmark final value: %.2f", result.performance["benchmark_final_value"].iloc[0])

    erase_after = final_config.get("eraseDataAfterRun", "No").lower() == "yes"
    if erase_after:
        LOGGER.info("eraseDataAfterRun enabled; cleaning up data and results.")
        cleanup_data(risk_on_symbols + risk_off_symbols + benchmark_symbols, final_config.get("data_path", "MarketData"))
        strategy_results_path = os.path.join(final_config.get("results_path", "ResultsData"), args.strategy)
        if os.path.exists(strategy_results_path):
            for filename in os.listdir(strategy_results_path):
                os.remove(os.path.join(strategy_results_path, filename))


if __name__ == "__main__":
    main()
