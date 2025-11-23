"""Minimal Plotly Dash dashboard for TrendPilot results."""
from __future__ import annotations

import importlib.util
import os
from glob import glob
from typing import Tuple

import dash
from dash import dcc, html
_pandas_spec = importlib.util.find_spec("pandas")
if _pandas_spec:
    import pandas as pd  # type: ignore
else:  # pragma: no cover - offline fallback
    import pandas_stub as pd  # type: ignore
import plotly.express as px


def _latest_file(pattern: str) -> str | None:
    files = sorted(glob(pattern))
    return files[-1] if files else None


def load_results(strategy: str, results_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = os.path.join(results_path, strategy)
    performance_file = _latest_file(os.path.join(base, "PerformanceStats_*.csv"))
    equity_file = _latest_file(os.path.join(base, "Equity_*.csv"))
    allocation_file = _latest_file(os.path.join(base, "Allocation_*.csv"))

    if not (performance_file and equity_file and allocation_file):
        raise FileNotFoundError("Result files not found. Run a backtest first.")

    performance = pd.read_csv(performance_file)
    equity = pd.read_csv(equity_file, parse_dates=["date"])
    allocation = pd.read_csv(allocation_file, parse_dates=["date"])
    return performance, equity, allocation


def create_app(strategy: str = "USTechTripleTrend", results_path: str = "ResultsData") -> dash.Dash:
    performance, equity, allocation = load_results(strategy, results_path)

    app = dash.Dash(__name__)

    equity_fig = px.line(equity, x="date", y=["equity", "benchmark"], title="Equity Curve")
    allocation_fig = px.area(allocation, x="date", y=["risk_on_allocation", "risk_off_allocation"], title="Allocation Over Time")

    summary_items = [html.Li(f"{col}: {performance[col].iloc[0]:.2f}") for col in performance.columns]

    app.layout = html.Div(
        children=[
            html.H1(f"TrendPilot Results - {strategy}"),
            html.H3("Performance Summary"),
            html.Ul(summary_items),
            dcc.Graph(figure=equity_fig),
            dcc.Graph(figure=allocation_fig),
        ]
    )
    return app


if __name__ == "__main__":
    dash_app = create_app()
    dash_app.run_server(debug=True)
