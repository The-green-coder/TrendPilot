"""
Trends_RuleEngine.py

Defines trend-following rules used by TrendPilot.

Each rule must be a function that takes a price series (pd.Series)
and returns an allocation series (0..1 for risk-on %).

We expose:
 - get_rule(rule_name)
 - list_rules()
"""

try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    import pandas_stub as pd  # type: ignore

# -----------------------------------------------------------
# Helper: generic moving average allocation engine
# -----------------------------------------------------------

def _moving_average_allocations(price_series: pd.Series, ma_weights):
    """
    Compute weighted allocation based on price above/below specific MAs.

    ma_weights = list of tuples: [(window, weight), ...]

    Returns a Series of values in [0,1].
    """
    price_series = price_series.astype(float)
    alloc = pd.Series(0.0, index=price_series.index)

    for window, weight in ma_weights:
        ma = price_series.rolling(window=window).mean()
        cond = price_series > ma
        alloc = alloc + weight * cond.astype(float)

    # ensure no NaN at start → backfill
    return alloc.fillna(0.0)


# -----------------------------------------------------------
# Rule 1 – Original TripleTrend
# -----------------------------------------------------------

def triple_trend(price_series: pd.Series) -> pd.Series:
    """
    TripleTrend rule:
    - 50% weight to 50-day MA
    - 25% weight to 75-day MA
    - 25% weight to 100-day MA
    """
    return _moving_average_allocations(
        price_series,
        [(50, 0.50), (75, 0.25), (100, 0.25)]
    )


# -----------------------------------------------------------
# Rule 2 – TripleTrend_QuickerResponse1
# -----------------------------------------------------------

def triple_trend_quicker_response_1(price_series: pd.Series) -> pd.Series:
    """
    A more aggressive triple trend:
    - 25% weight to 25-day MA
    - 50% weight to 50-day MA
    - 25% weight to 100-day MA
    """
    return _moving_average_allocations(
        price_series,
        [(25, 0.25), (50, 0.50), (100, 0.25)]
    )


# -----------------------------------------------------------
# Rule 3 – TripleTrend_QuickerResponse2
# -----------------------------------------------------------

def triple_trend_quicker_response_2(price_series: pd.Series) -> pd.Series:
    """
    Another variation:
    - 50% weight to 20-day MA
    - 25% weight to 50-day MA
    - 25% weight to 100-day MA
    """
    return _moving_average_allocations(
        price_series,
        [(20, 0.50), (50, 0.25), (100, 0.25)]
    )


# -----------------------------------------------------------
# Rule Registry
# -----------------------------------------------------------

_RULES = {
    "TRIPLETREND": triple_trend,
    "TRIPLETREND_QUICKERRESPONSE1": triple_trend_quicker_response_1,
    "TRIPLETREND_QUICKERRESPONSE2": triple_trend_quicker_response_2,
}


def get_rule(name: str):
    """
    Return the rule function by name (case-insensitive).
    """
    if not name:
        return None
    key = name.strip().upper()
    return _RULES.get(key)


def list_rules():
    """
    Return all rule names in the registry.
    """
    return list(_RULES.keys())
