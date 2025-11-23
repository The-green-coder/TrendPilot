"""
Dashboard.py

Minimal Plotly Dash dashboard for TrendPilot.

- Reads latest PerformanceStats_*.csv, RiskStats_*.csv, Allocation_*.csv
  for a given strategy.
- Shows key numbers, equity curves, and allocation over time.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import dash
from dash import Dash, dcc, html
import dash_table
import pandas as pd
import plotly.graph_objs as go


RESULTS_ROOT = "ResultsData"
DEFAULT_STRATEGY = "USTechTripleTrend"


def _latest_file(pattern: str) -> Optional[str]:
    """
    Return the most recently modified file matching the glob pattern,
    or None if nothing matches.
    """
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_latest_results(
    strategy: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load the latest performance, risk, and allocation CSVs for a strategy.
    """
    strat_dir = os.path.join(RESULTS_ROOT, strategy)

    perf_file = _latest_file(os.path.join(strat_dir, "PerformanceStats_*.csv"))
    risk_file = _latest_file(os.path.join(strat_dir, "RiskStats_*.csv"))
    alloc_file = _latest_file(os.path.join(strat_dir, "Allocation_*.csv"))

    perf_df = pd.read_csv(perf_file) if perf_file else None
    risk_df = pd.read_csv(risk_file) if risk_file else None
    alloc_df = pd.read_csv(alloc_file, parse_dates=["date"]) if alloc_file else None

    return perf_df, risk_df, alloc_df


# ---------------------------------------------------------------------------
# Load data for default strategy
# ---------------------------------------------------------------------------

strategy = DEFAULT_STRATEGY
perf_df, risk_df, alloc_df = _load_latest_results(strategy)

if perf_df is None or risk_df is None or alloc_df is None:
    raise SystemExit(
        f"No results found for strategy '{strategy}'. "
        f"Run TrendPilot.py for this strategy before starting the dashboard."
    )

perf_row = perf_df.iloc[0].to_dict()
risk_row = risk_df.iloc[0].to_dict()

# Ensure sorted by date
alloc_df = alloc_df.sort_values("date")

# ---------------------------------------------------------------------------
# Build Dash app
# ---------------------------------------------------------------------------

dash_app: Dash = dash.Dash(__name__)
dash_app.title = f"TrendPilot – {strategy}"


def _format_pct(x: float, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "n/a"


def _format_num(x: float, digits: int = 2) -> str:
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "n/a"


summary_items = [
    f"Strategy: {strategy}",
    f"Final portfolio value: {_format_num(perf_row.get('final_value'))}",
    f"Benchmark final value: {_format_num(perf_row.get('benchmark_final_value'))}",
    f"Total return: {_format_pct(perf_row.get('total_return_pct'))}",
    f"Benchmark return: {_format_pct(perf_row.get('benchmark_return_pct'))}",
    f"Average allocation: "
    f"{_format_pct(perf_row.get('avg_risk_on_alloc_pct'))} Risk-On, "
    f"{_format_pct(perf_row.get('avg_risk_off_alloc_pct'))} Risk-Off",
    f"Time any Risk-On: {_format_pct(perf_row.get('time_any_risk_on_pct'))}",
    f"Time fully Risk-Off: {_format_pct(perf_row.get('time_full_risk_off_pct'))}",
    f"Return contribution: "
    f"{_format_pct(perf_row.get('risk_on_contrib_pct'))} from Risk-On, "
    f"{_format_pct(perf_row.get('risk_off_contrib_pct'))} from Risk-Off",
    f"Sharpe (strategy): {_format_num(risk_row.get('sharpe_strategy'), 3)}",
    f"Sharpe (benchmark): {_format_num(risk_row.get('sharpe_benchmark'), 3)}",
    f"Volatility (strategy): {_format_pct(risk_row.get('vol_strategy'), 2)}",
    f"Volatility (benchmark): {_format_pct(risk_row.get('vol_benchmark'), 2)}",
    f"Max drawdown (strategy): {_format_pct(risk_row.get('max_drawdown_strategy') * 100.0, 2)}",
    f"Max drawdown (benchmark): {_format_pct(risk_row.get('max_drawdown_benchmark') * 100.0, 2)}",
]

# Equity curve figure
equity_fig = go.Figure()
equity_fig.add_trace(
    go.Scatter(
        x=alloc_df["date"],
        y=alloc_df["portfolio_value"],
        mode="lines",
        name="Portfolio",
    )
)
equity_fig.add_trace(
    go.Scatter(
        x=alloc_df["date"],
        y=alloc_df["benchmark_value"],
        mode="lines",
        name="Benchmark",
    )
)
equity_fig.update_layout(
    title="Equity Curve",
    xaxis_title="Date",
    yaxis_title="Value",
    hovermode="x unified",
)

# Allocation figure
alloc_fig = go.Figure()
alloc_fig.add_trace(
    go.Scatter(
        x=alloc_df["date"],
        y=alloc_df["allocation_risk_on"],
        mode="lines",
        name="Risk-On allocation",
        stackgroup="one",
    )
)
alloc_fig.add_trace(
    go.Scatter(
        x=alloc_df["date"],
        y=alloc_df["allocation_risk_off"],
        mode="lines",
        name="Risk-Off allocation",
        stackgroup="one",
    )
)
alloc_fig.update_layout(
    title="Allocation over time",
    xaxis_title="Date",
    yaxis_title="Allocation",
    hovermode="x unified",
    yaxis=dict(range=[0, 1]),
)

# Simple tables for performance & risk stats
perf_table = dash_table.DataTable(
    columns=[{"name": c, "id": c} for c in perf_df.columns],
    data=perf_df.to_dict("records"),
    style_table={"overflowX": "auto"},
    style_cell={"textAlign": "left"},
)

risk_table = dash_table.DataTable(
    columns=[{"name": c, "id": c} for c in risk_df.columns],
    data=risk_df.to_dict("records"),
    style_table={"overflowX": "auto"},
    style_cell={"textAlign": "left"},
)

dash_app.layout = html.Div(
    children=[
        html.H1(f"TrendPilot – {strategy}"),
        html.Div(
            [
                html.H2("Summary"),
                html.Ul([html.Li(item) for item in summary_items]),
            ]
        ),
        html.Div(
            [
                html.H2("Equity Curve"),
                dcc.Graph(figure=equity_fig),
            ]
        ),
        html.Div(
            [
                html.H2("Allocation"),
                dcc.Graph(figure=alloc_fig),
            ]
        ),
        html.Div(
            [
                html.H2("Performance Stats"),
                perf_table,
            ]
        ),
        html.Div(
            [
                html.H2("Risk Stats"),
                risk_table,
            ]
        ),
    ]
)


if __name__ == "__main__":
    # Dash 2.x+ uses app.run, not run_server
    dash_app.run(debug=True)
