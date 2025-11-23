"""Utility functions for retrieving and caching market data.

This module fetches historical data from Yahoo Finance when available.
If downloading fails (for example due to lack of network access), it
creates deterministic synthetic data so the rest of the pipeline can run
in offline environments.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover - fallback when yfinance is missing
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
    """
    Fetch data for a single symbol between start and end.

    - If a cached CSV exists and refresh is False, load from disk.
    - Otherwise, download from yfinance and save to CSV.
    - Always ensure we end up with a 'date' column as DateTimeIndex.
    """
    os.makedirs(data_path, exist_ok=True)
    filepath = os.path.join(data_path, f"{symbol}.csv")

    # Use cached CSV if present and refresh is False
    if os.path.exists(filepath) and not refresh:
        # Load with 'date' as DateTimeIndex if present
        df = pd.read_csv(filepath)
    else:
        # If yfinance not available, fall back to synthetic data
        if yf is None:
            LOGGER.warning("yfinance unavailable; generating synthetic data for %s", symbol)
            df = _generate_synthetic_prices(symbol, start, end)
        else:
            try:
                df = yf.download(symbol, start=start, end=end)
                if df.empty:
                    raise RuntimeError("Empty dataset returned from Yahoo Finance")

                # Make column names lower-case for consistency (open, high, low, close, adj close, volume)
                df = df.rename(columns=str.lower)

            except Exception as exc:
                # Network failure or other error -> synthetic data
                LOGGER.warning("Falling back to synthetic data for %s due to error: %s", symbol, exc)
                df = _generate_synthetic_prices(symbol, start, end)

        # yfinance gives a DatetimeIndex; after reset_index, we get a 'Date' column.
        df.reset_index(inplace=True)

        # Normalise to a 'date' column
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "date" not in df.columns and "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)

        if "date" not in df.columns:
            raise KeyError(f"No 'date' column found after reset_index for symbol {symbol}")

        df.to_csv(filepath, index=False)

    # Ensure 'date' column exists when loading cached data as well
    if "date" not in df.columns:
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)

    if "date" not in df.columns:
        raise KeyError(f"No 'date' column found in cached CSV for symbol {symbol}")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Standardise column names again in case loaded from CSV
    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )

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
