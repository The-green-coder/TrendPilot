"""Utility functions for retrieving and caching market data.

This module fetches historical data from Yahoo Finance when available.
If downloading fails (for example due to lack of network access), it
creates deterministic synthetic data so the rest of the pipeline can run
in offline environments.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, Tuple

_numpy_spec = importlib.util.find_spec("numpy")
if _numpy_spec:
    import numpy as np  # type: ignore
else:  # pragma: no cover - offline fallback
    import numpy_stub as np  # type: ignore

_pandas_spec = importlib.util.find_spec("pandas")
if _pandas_spec:
    import pandas as pd  # type: ignore
else:  # pragma: no cover - offline fallback
    import pandas_stub as pd  # type: ignore

_yfinance_spec = importlib.util.find_spec("yfinance")
if _yfinance_spec:  # pragma: no cover - optional dependency
    import yfinance as yf  # type: ignore
else:  # pragma: no cover - fallback when yfinance is missing
    yf = None  # type: ignore


LOGGER = logging.getLogger(__name__)

PERIOD_LOOKUPS = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
}


def parse_period(period: str) -> Tuple[datetime, datetime]:
    """Convert a shorthand period such as ``5Y`` into start/end datetimes."""
    today = datetime.utcnow()
    period = period.upper().strip()
    if period not in PERIOD_LOOKUPS:
        raise ValueError(f"Unsupported period shorthand: {period}")
    delta_days = PERIOD_LOOKUPS[period]
    start_date = today - timedelta(days=delta_days)
    return start_date, today


def _generate_synthetic_prices(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Create synthetic price data for offline environments."""
    rng = pd.date_range(start=start, end=end, freq="B")
    base = 100 + hash(symbol) % 20
    trend = pd.Series(range(len(rng)), index=rng) * 0.02
    random_state = np.random.RandomState(abs(hash(symbol)) % 2**32)
    noise = pd.Series(random_state.normal(0, 0.5, len(rng)), index=rng)
    prices = base + trend + noise.cumsum()
    df = pd.DataFrame({"date": rng, "open": prices, "high": prices * 1.005, "low": prices * 0.995, "close": prices})
    df.set_index("date", inplace=True)
    return df


def fetch_symbol_data(symbol: str, start: datetime, end: datetime, data_path: str, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(data_path, exist_ok=True)
    filepath = os.path.join(data_path, f"{symbol}.csv")
    if os.path.exists(filepath) and not refresh:
        return pd.read_csv(filepath, parse_dates=["date"], index_col="date")

    if yf is None:
        LOGGER.warning("yfinance unavailable; generating synthetic data for %s", symbol)
        df = _generate_synthetic_prices(symbol, start, end)
    else:
        try:
            df = yf.download(symbol, start=start, end=end)
            if df.empty:
                raise RuntimeError("Empty dataset returned from Yahoo Finance")
            df = df.rename(columns=str.lower)
        except Exception as exc:  # pragma: no cover - network failure path
            LOGGER.warning("Falling back to synthetic data for %s due to error: %s", symbol, exc)
            df = _generate_synthetic_prices(symbol, start, end)

    # Normalize the datetime column so downstream code always sees a "date" column
    # regardless of how yfinance names the index. The ``names`` argument works on
    # modern pandas, but we keep a backwards-compatible fallback for older
    # versions (and for the lightweight stub used offline).
    try:  # pandas >= 1.3
        df.reset_index(names="date", inplace=True)
    except TypeError:  # pragma: no cover - legacy pandas path
        df.reset_index(inplace=True)
    # Handle common datetime column names that may appear after reset_index
    rename_candidates = {
        "index": "date",
        "Date": "date",
        "Datetime": "date",
        "datetime": "date",
    }
    df.rename(columns=rename_candidates, inplace=True)
    if "date" not in df.columns:
        # As a last resort, promote the original index to a column
        df["date"] = df.index
        df.reset_index(drop=True, inplace=True)
    df.to_csv(filepath, index=False)
    df.set_index("date", inplace=True)
    return df


def load_market_data(symbols: Iterable[str], start: datetime, end: datetime, data_path: str, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch data for all symbols, returning a symbol->DataFrame map."""
    return {symbol: fetch_symbol_data(symbol, start, end, data_path, refresh=refresh) for symbol in symbols}


def cleanup_data(symbols: Iterable[str], data_path: str) -> None:
    """Remove cached data files for the given symbols."""
    for symbol in symbols:
        filepath = os.path.join(data_path, f"{symbol}.csv")
        if os.path.exists(filepath):
            os.remove(filepath)
