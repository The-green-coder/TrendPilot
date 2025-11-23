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


def load_results(strategy: str, results_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = os.path.join(results_path, strategy)
    performance_file = _latest_file(os.path.join(base, "PerformanceStats_*.csv"))
    equity_file = _latest_file(os.path.join(base, "Equity_*.csv"))
    allocation_file = _latest_file(os.path.join(base, "Allocation_*.csv"))
    risk_file = _latest_file(os.path.join(base, "RiskStats_*.csv"))

    if not (performance_file and equity_file and allocation_file and risk_file):
        raise FileNotFoundError("Result files not found. Run a backtest first.")

    performance = pd.read_csv(performance_file)
    equity = pd.read_csv(equity_file, parse_dates=["date"])
    allocation = pd.read_csv(allocation_file, parse_dates=["date"])
    risk = pd.read_csv(risk_file)
    return performance, equity, allocation, risk


def _table_from_df(df: pd.DataFrame) -> html.Table:
    header = [html.Tr([html.Th(col) for col in df.columns])]
    rows = [
        html.Tr([
            html.Td(f"{val:.4f}" if isinstance(val, (int, float)) else val)
            for val in row
        ])
        for row in df.values
    ]
    return html.Table(header + rows, style={"borderCollapse": "collapse", "width": "100%"})


def create_app(strategy: str = "USTechTripleTrend", results_path: str = "ResultsData") -> dash.Dash:
    performance, equity, allocation, risk = load_results(strategy, results_path)

    app = dash.Dash(__name__)

    equity_fig = px.line(equity, x="date", y=["equity", "benchmark"], title="Equity Curve")
    allocation_fig = px.area(allocation, x="date", y=["risk_on_allocation", "risk_off_allocation"], title="Allocation Over Time")

    app.layout = html.Div(
        children=[
            html.H1(f"TrendPilot Results - {strategy}"),
            html.H3("Performance Summary"),
            _table_from_df(performance),
            html.H3("Risk Summary"),
            _table_from_df(risk),
            dcc.Graph(figure=equity_fig),
            dcc.Graph(figure=allocation_fig),
        ]
    )
    return app


if __name__ == "__main__":
    dash_app = create_app()
    run_callable = getattr(dash_app, "run", None) or getattr(dash_app, "run_server", None)
    if not run_callable:
        raise AttributeError("Dash application missing both 'run' and 'run_server' methods")
    run_callable(debug=True)
