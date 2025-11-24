"""
TrendPilot.py

Main CLI entrypoint for TrendPilot backtesting engine.
Loads strategy config, merges with defaults, loads rule,
then executes BacktestingEngine.run().
"""

import argparse
import csv
import logging
import os
from datetime import datetime
import shutil

from BacktestingEngine import BacktestingEngine
from MarketData import cleanup_data
from Trends_RuleEngine import get_rule, list_rules

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helper: Load CSV into dict
# ---------------------------------------------------------------------
def load_csv_as_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        result = {}
        for row in reader:
            if len(row) >= 2:
                key, value = row[0].strip(), row[1].strip()
                result[key] = value
        return result


# ---------------------------------------------------------------------
# Helper: Load instruments (RiskOn.csv, RiskOff.csv, Benchmark.csv)
# ---------------------------------------------------------------------
def load_symbol_list(path: str):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TrendPilot Backtesting Engine")

    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--backtesting_period", help="Override backtesting period, e.g. 5Y")
    parser.add_argument("--initial_capital")
    parser.add_argument("--transaction_cost")
    parser.add_argument("--slippage")
    parser.add_argument("--rebalance_frequency")
    parser.add_argument("--buy_price_preference")
    parser.add_argument("--sell_price_preference")
    parser.add_argument("--rule_name")
    parser.add_argument("--delay_between_signal_and_trade")
    parser.add_argument("--eraseDataAfterRun", default="No")
    parser.add_argument("--record_backtest_details")

    args = parser.parse_args()

    strategy_name = args.strategy

    # -----------------------------------------------------------------
    # Load default config
    # -----------------------------------------------------------------
    default_config_path = "default_config.csv"
    default_config = load_csv_as_dict(default_config_path)

    # -----------------------------------------------------------------
    # Load strategy config from Strategy_Centre/<strategy>/StrategyConfig.csv
    # -----------------------------------------------------------------
    strat_dir = os.path.join("Strategy_Centre", strategy_name)
    strat_cfg_path = os.path.join(strat_dir, "StrategyConfig.csv")

    strat_config = load_csv_as_dict(strat_cfg_path)

    # -----------------------------------------------------------------
    # Merge: CLI args → strategy config → defaults
    # -----------------------------------------------------------------
    final_config = dict(default_config)
    final_config.update(strat_config)

    # APPLY CLI OVERRIDES
    cli_param_map = {
        "backtesting_period": args.backtesting_period,
        "initial_capital": args.initial_capital,
        "transaction_cost": args.transaction_cost,
        "slippage": args.slippage,
        "rebalance_frequency": args.rebalance_frequency,
        "buy_price_preference": args.buy_price_preference,
        "sell_price_preference": args.sell_price_preference,
        "rule_name": args.rule_name,
        "delay_between_signal_and_trade": args.delay_between_signal_and_trade,
        "record_backtest_details": args.record_backtest_details,
        "eraseDataAfterRun": args.eraseDataAfterRun,
    }

    for k, v in cli_param_map.items():
        if v is not None:
            final_config[k] = v

    # -----------------------------------------------------------------
    # Load instrument lists
    # -----------------------------------------------------------------
    riskon_list = load_symbol_list(os.path.join(strat_dir, "Instruments", "RiskOn.csv"))
    riskoff_list = load_symbol_list(os.path.join(strat_dir, "Instruments", "RiskOff.csv"))
    benchmark_list = load_symbol_list(os.path.join(strat_dir, "Instruments", "Benchmark.csv"))

    if not riskon_list or not riskoff_list:
        raise RuntimeError("RiskOn or RiskOff instruments not configured properly")

    # -----------------------------------------------------------------
    # Resolve rule function
    # -----------------------------------------------------------------
    rule_name = final_config.get("rule_name")
    if not rule_name:
        raise RuntimeError("Missing rule_name in strategy config or CLI")

    LOGGER.info(f"Running strategy {strategy_name} using rule {rule_name}")
    LOGGER.info(f"Available rules: {', '.join(list_rules())}")

    rule = get_rule(rule_name)
    if rule is None:
        raise RuntimeError(f"Rule '{rule_name}' not found in Trends_RuleEngine")

    # -----------------------------------------------------------------
    # Create backtesting engine
    # -----------------------------------------------------------------
    engine = BacktestingEngine()  # FIXED — no arguments here

    # -----------------------------------------------------------------
    # Execute backtest
    # -----------------------------------------------------------------
    result = engine.run(
        risk_on_symbols=riskon_list,
        risk_off_symbols=riskoff_list,
        benchmark_symbols=benchmark_list,
        rule=rule,   # <<<<<< FIXED — NOW PASSING RULE CORRECTLY
        rebalance_frequency=final_config["rebalance_frequency"],
        initial_capital=float(final_config["initial_capital"]),
        transaction_cost=float(final_config["transaction_cost"]),
        slippage=float(final_config["slippage"]),
        backtesting_period=final_config["backtesting_period"],
        price_preference_buy=final_config["buy_price_preference"],
        price_preference_sell=final_config["sell_price_preference"],
        delay=int(final_config["delay_between_signal_and_trade"]),
        results_path=final_config.get("results_path", "ResultsData"),
        strategy_name=strategy_name,
        record_backtest_details=final_config.get("record_backtest_details", "No"),
    )

    LOGGER.info("Backtest completed. Results saved in ResultsData/%s", strategy_name)

    erase_after_run = str(final_config.get("eraseDataAfterRun", "No")).strip().lower()
    if erase_after_run == "yes":
        data_path = final_config.get("data_path", "MarketData")
        cleanup_data(riskon_list + riskoff_list + benchmark_list, data_path)

        results_root = final_config.get("results_path", "ResultsData")
        strategy_results_dir = os.path.join(results_root, strategy_name)
        if os.path.isdir(strategy_results_dir):
            shutil.rmtree(strategy_results_dir)
        LOGGER.info("Data erased after run as requested")


if __name__ == "__main__":
    main()
