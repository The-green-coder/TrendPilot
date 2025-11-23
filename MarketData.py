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

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex or tuple columns into simple strings."""

    if hasattr(pd, "MultiIndex") and isinstance(getattr(df, "columns", None), getattr(pd, "MultiIndex", ())):
        flat_columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_columns.append("_".join(str(part) for part in col if part))
            else:
                flat_columns.append(str(col))
        df.columns = flat_columns
    else:
        try:
            df.columns = [str(col) for col in df.columns]
        except Exception:  # pragma: no cover - extremely defensive
            pass
    return df


def _normalize_date_column(df: pd.DataFrame, start: datetime | None = None) -> pd.DataFrame:
    """Ensure a ``date`` column exists (datetime64) regardless of input shape."""

    base_df = df if hasattr(df, "columns") else pd.DataFrame(df)
    df = _flatten_columns(base_df)

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

    # Final guard: if callers still do not see a usable date column after
    # normalization (for example due to unusual pandas/yfinance combinations),
    # construct one deterministically so downstream code never fails when
    # setting the index.
    if "date" not in df.columns:
        df["date"] = pd.date_range(start=start or datetime.utcnow(), periods=len(df), freq="B")
    elif hasattr(df["date"], "isna") and df["date"].isna().all():
        df["date"] = pd.date_range(start=start or datetime.utcnow(), periods=len(df), freq="B")

    # Deduplicate any repeated columns that may arise from odd CSV exports
    # (keeping the first occurrence), ensuring "date" remains present.
    if hasattr(df, "columns") and hasattr(df.columns, "duplicated"):
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def _coerce_numeric_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price-like columns to numeric, dropping fully empty rows.

    Real-world CSVs can accumulate stray header strings (for example after
    conflict resolutions or manual edits). To keep downstream calculations
    robust, coerce every non-date column to numeric when possible.
    """

    if not hasattr(df, "columns"):
        return df

    try:
        numeric_df = df.copy()  # pandas path
    except Exception:
        if hasattr(df, "data"):
            numeric_df = pd.DataFrame(getattr(df, "data"))  # pandas_stub path
        else:
            numeric_df = pd.DataFrame(df)  # final fallback
    for col in numeric_df.columns:
        if str(col).lower() == "date":
            continue
        try:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        except Exception:  # pragma: no cover - extremely defensive
            pass

    # Drop rows where every price column is NaN, retaining the date index
    price_cols = [c for c in numeric_df.columns if str(c).lower() != "date"]
    if price_cols and hasattr(numeric_df, "dropna"):
        numeric_df = numeric_df.dropna(axis=0, how="all", subset=price_cols)

    return numeric_df


def fetch_symbol_data(symbol: str, start: datetime, end: datetime, data_path: str, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(data_path, exist_ok=True)
    filepath = os.path.join(data_path, f"{symbol}.csv")
    if os.path.exists(filepath) and not refresh:
        try:
            df = pd.read_csv(filepath)
            df = _normalize_date_column(df, start)
        except Exception as exc:  # pragma: no cover - handles legacy/stub parsing issues
            LOGGER.warning("Failed to read cached data for %s (%s); regenerating", symbol, exc)
            df = None
    else:
        df = None

    if df is None:
        if yf is None:
            LOGGER.warning("yfinance unavailable; generating synthetic data for %s", symbol)
            df = _generate_synthetic_prices(symbol, start, end)
        else:
            try:
                df = yf.download(symbol, start=start, end=end)
                if df.empty:
                    raise RuntimeError("Empty dataset returned from Yahoo Finance")
                df = _flatten_columns(df)
                df = df.rename(columns=str.lower)
            except Exception as exc:  # pragma: no cover - network failure path
                LOGGER.warning("Falling back to synthetic data for %s due to error: %s", symbol, exc)
                df = _generate_synthetic_prices(symbol, start, end)

        df = _normalize_date_column(df, start)
        df = _coerce_numeric_prices(df)
        # Persist with a guaranteed "date" column to avoid future cache issues
        df.to_csv(filepath, index=False)

    # As a final safety net, rebuild a date column if it somehow vanished after
    # normalization (e.g., user-supplied CSV edits) before indexing.
    if "date" not in df.columns:
        df = _normalize_date_column(df, start)
    df = _coerce_numeric_prices(df)
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
