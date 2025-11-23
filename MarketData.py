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

def _normalize_date_column(df: pd.DataFrame, start: datetime | None = None) -> pd.DataFrame:
    """Ensure a ``date`` column exists (datetime64) regardless of input shape."""

    # Always start from a reset index to avoid surprises with unnamed indexes
    try:
        df = df.reset_index()
    except Exception:  # pragma: no cover - extremely defensive
        df = pd.DataFrame(df)

    # Look for obvious date-like columns first
    candidates = [
        col for col in df.columns if str(col).lower() in {"date", "datetime", "index"}
    ]

    # Also accept auto-generated "Unnamed" columns that commonly hold index values
    candidates.extend([col for col in df.columns if str(col).startswith("Unnamed")])

    if candidates:
        df.rename(columns={candidates[0]: "date"}, inplace=True)

    # Promote the current index to a date column if none matched
    if "date" not in df.columns:
        df["date"] = df.index
        df.reset_index(drop=True, inplace=True)

    # Absolute last resort: simple sequential range
    if "date" not in df.columns:
        df["date"] = pd.RangeIndex(len(df))

    # Guarantee datetime dtype
    if hasattr(pd, "to_datetime"):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:  # pragma: no cover - pandas_stub path
        from datetime import datetime as _dt

        def _coerce(val):
            if isinstance(val, _dt):
                return val
            try:
                return _dt.fromisoformat(str(val))
            except Exception:
                return val

        df["date"] = pd.Series([_coerce(v) for v in df["date"]], getattr(df, "index", None))

    # If the date column is still missing or entirely NA, synthesize reasonable values
    date_values = df["date"] if "date" in df.columns else None
    missing_or_all_na = date_values is None
    if date_values is not None:
        try:
            is_na = getattr(date_values, "isna", None)
            if callable(is_na):
                missing_or_all_na = bool(is_na().all())
        except Exception:
            missing_or_all_na = False

    if "date" not in df.columns or missing_or_all_na:
        if start is not None and len(df) > 0:
            df["date"] = pd.date_range(start=start, periods=len(df), freq="B")
        else:
            df["date"] = pd.RangeIndex(len(df))

    return df


def fetch_symbol_data(symbol: str, start: datetime, end: datetime, data_path: str, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(data_path, exist_ok=True)
    filepath = os.path.join(data_path, f"{symbol}.csv")
    if os.path.exists(filepath) and not refresh:
        df = pd.read_csv(filepath)
        df = _normalize_date_column(df, start)
    else:
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

        df = _normalize_date_column(df, start)
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
