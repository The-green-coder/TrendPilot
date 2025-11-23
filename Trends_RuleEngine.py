"""Rule engine implementing trend-following allocation logic."""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

def _moving_average_allocations(price_series, windows_and_weights):
    """
    Generic helper to build an allocation series from:
    - price_series: time series of prices
    - windows_and_weights: list of (window, weight) tuples

    It returns a series of allocation to RISK-ON (0.0 .. 1.0).
    Any non-numeric values in price_series are coerced to NaN.
    """

    # Ensure we are working with a 1D numeric Series
    if isinstance(price_series, pd.DataFrame):
        # Use the first column if a DataFrame is passed
        price_series = price_series.iloc[:, 0]

    # Coerce to numeric – strings like "qqq" become NaN
    price_series = pd.to_numeric(price_series, errors="coerce")

    allocations = pd.Series(0.0, index=price_series.index)

    for window, weight in windows_and_weights:
        # Require full window before using the MA
        ma = price_series.rolling(window=window, min_periods=window).mean()

        # 1.0 when price > MA (risk-on), 0.0 otherwise (risk-off)
        signal = (price_series > ma).astype(float)

        # Combine weighted signals
        allocations = allocations.add(weight * signal, fill_value=0.0)

    return allocations


def triple_trend(price_series: pd.Series) -> pd.Series:
    return _moving_average_allocations(price_series, [(50, 0.5), (75, 0.25), (100, 0.25)])


def triple_trend_quicker_response_1(price_series: pd.Series) -> pd.Series:
    return _moving_average_allocations(price_series, [(25, 0.25), (50, 0.5), (100, 0.25)])


def triple_trend_quicker_response_2(price_series: pd.Series) -> pd.Series:
    return _moving_average_allocations(price_series, [(20, 0.5), (50, 0.25), (100, 0.25)])


RULE_REGISTRY: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "TRIPLETREND": triple_trend,
    "TRIPLETREND_QUICKERRESPONSE1": triple_trend_quicker_response_1,
    "TRIPLETREND_QUICKERRESPONSE2": triple_trend_quicker_response_2,
}


def get_rule(rule_name: str) -> Callable[[pd.Series], pd.Series]:
    key = rule_name.upper()
    if key not in RULE_REGISTRY:
        raise KeyError(f"Rule '{rule_name}' is not registered. Available: {', '.join(RULE_REGISTRY)}")
    return RULE_REGISTRY[key]
