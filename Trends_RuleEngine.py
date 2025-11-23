"""Rule engine implementing trend-following allocation logic."""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd


def _moving_average_allocations(price_series: pd.Series, windows_with_weights) -> pd.Series:
    signals = []
    for window, weight in windows_with_weights:
        ma = price_series.rolling(window=window).mean()
        signal = (price_series > ma).astype(float) * weight
        signals.append(signal)
    allocation = sum(signals)
    return allocation.clip(0, 1)


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
