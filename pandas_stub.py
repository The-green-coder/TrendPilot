"""Minimal pandas-like stub to support TrendPilot in offline environments."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional


def date_range(start: datetime, end: datetime, freq: str = "D") -> List[datetime]:
    days = []
    current = start
    while current <= end:
        if freq == "B":
            if current.weekday() < 5:
                days.append(current)
        else:
            days.append(current)
        current += timedelta(days=1)
    return days


class Series:
    def __init__(self, data: Iterable, index: Optional[List] = None):
        self.data = list(data)
        self.index = list(index) if index is not None else list(range(len(self.data)))
        if len(self.data) != len(self.index):
            raise ValueError("Data and index must be the same length")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other):
        if isinstance(other, Series):
            return Series([a + b for a, b in zip(self.data, other.data)], self.index)
        return Series([a + other for a in self.data], self.index)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Series):
            return Series([a - b for a, b in zip(self.data, other.data)], self.index)
        return Series([a - other for a in self.data], self.index)

    def __rsub__(self, other):
        if isinstance(other, Series):
            return Series([b - a for a, b in zip(self.data, other.data)], self.index)
        return Series([other - a for a in self.data], self.index)

    def __mul__(self, other):
        if isinstance(other, Series):
            return Series([a * b for a, b in zip(self.data, other.data)], self.index)
        return Series([a * other for a in self.data], self.index)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Series):
            return Series([a / b if b != 0 else math.nan for a, b in zip(self.data, other.data)], self.index)
        return Series([a / other for a in self.data], self.index)

    def __gt__(self, other):
        if isinstance(other, Series):
            return Series([1.0 if a > b else 0.0 for a, b in zip(self.data, other.data)], self.index)
        return Series([1.0 if a > other else 0.0 for a in self.data], self.index)

    def astype(self, _type):
        if _type is float:
            return Series([float(x) if x is not None else math.nan for x in self.data], self.index)
        return self

    def rolling(self, window: int):
        series = self

        class Roller:
            def mean(self_nonself):
                rolled = []
                for i in range(len(series.data)):
                    start = max(0, i - window + 1)
                    window_data = series.data[start : i + 1]
                    rolled.append(sum(window_data) / len(window_data))
                return Series(rolled, series.index)

        return Roller()

    def clip(self, lower=None, upper=None):
        clipped = []
        for val in self.data:
            new_val = val
            if lower is not None:
                new_val = max(lower, new_val)
            if upper is not None:
                new_val = min(upper, new_val)
            clipped.append(new_val)
        return Series(clipped, self.index)

    def resample(self, _freq):
        series = self

        class Resampler:
            def last(self_nonself):
                return series

        return Resampler()

    def reindex(self, new_index: List):
        mapping = {idx: val for idx, val in zip(self.index, self.data)}
        values = [mapping.get(idx, None) for idx in new_index]
        return Series(values, new_index)

    def ffill(self):
        filled = []
        last = None
        for val in self.data:
            if val is None:
                filled.append(last)
            else:
                filled.append(val)
                last = val
        return Series(filled, self.index)

    def fillna(self, value=None, method: Optional[str] = None):
        filled = self.data[:]
        if method == "bfill":
            next_val = None
            for i in range(len(filled) - 1, -1, -1):
                if filled[i] is None:
                    filled[i] = next_val
                else:
                    next_val = filled[i]
        elif value is not None:
            filled = [value if v is None else v for v in filled]
        return Series(filled, self.index)

    def pct_change(self):
        changes = [0]
        for i in range(1, len(self.data)):
            prev = self.data[i - 1]
            curr = self.data[i]
            if prev == 0 or prev is None:
                changes.append(0)
            else:
                changes.append((curr - prev) / prev)
        return Series(changes, self.index)

    def abs(self):
        return Series([abs(v) for v in self.data], self.index)

    def diff(self):
        diffs = [self.data[0]]
        for i in range(1, len(self.data)):
            prev = self.data[i - 1]
            curr = self.data[i]
            diffs.append(curr - prev)
        return Series(diffs, self.index)

    def cumsum(self):
        total = 0
        out = []
        for v in self.data:
            total += v
            out.append(total)
        return Series(out, self.index)

    def cumprod(self):
        total = 1
        out = []
        for v in self.data:
            total *= v
            out.append(total)
        return Series(out, self.index)

    def cummax(self):
        max_val = -math.inf
        out = []
        for v in self.data:
            max_val = max(max_val, v)
            out.append(max_val)
        return Series(out, self.index)

    def shift(self, periods=1):
        new_data = [None] * periods + self.data[:-periods]
        return Series(new_data, self.index)

    def std(self):
        if len(self.data) == 0:
            return math.nan
        mean_val = sum(self.data) / len(self.data)
        variance = sum((x - mean_val) ** 2 for x in self.data) / len(self.data)
        return math.sqrt(variance)

    def mean(self):
        if len(self.data) == 0:
            return math.nan
        return sum(self.data) / len(self.data)

    def min(self):
        return min(self.data) if self.data else math.nan

    def max(self):
        return max(self.data) if self.data else math.nan

    @property
    def iloc(self):
        series = self

        class ILoc:
            def __getitem__(self_nonself, item):
                return series.data[item]

        return ILoc()


class DataFrame:
    def __init__(self, data: Dict[str, Iterable]):
        lengths = {len(v) for v in data.values()}
        if len(lengths) > 1:
            raise ValueError("All columns must be the same length")
        self.columns = list(data.keys())
        self.data: Dict[str, Series] = {}
        length = lengths.pop() if lengths else 0
        default_index = list(range(length))
        for key, values in data.items():
            if isinstance(values, Series):
                self.data[key] = values
            else:
                self.data[key] = Series(values, default_index)
        self.index = self.data[self.columns[0]].index if self.columns else []

    def set_index(self, column_name: str, inplace: bool = False):
        if column_name not in self.data:
            raise KeyError(column_name)
        new_index = self.data[column_name].data
        new_data = {k: Series(v.data, new_index) for k, v in self.data.items() if k != column_name}
        result = DataFrame(new_data)
        if inplace:
            self.data = result.data
            self.index = new_index
            self.columns = list(result.columns)
            return None
        return result

    def reset_index(self, inplace: bool = False):
        new_data = {"index": Series(self.index, self.index)}
        for k, v in self.data.items():
            new_data[k] = Series(v.data, self.index)
        df = DataFrame(new_data)
        if inplace:
            self.data = df.data
            self.columns = df.columns
            self.index = df.index
            return None
        return df

    def rename(self, columns: Dict[str, str], inplace: bool = False):
        new_data = {}
        for col, series in self.data.items():
            new_name = columns.get(col, col)
            new_data[new_name] = series
        df = DataFrame(new_data)
        if inplace:
            self.data = df.data
            self.columns = df.columns
            return None
        return df

    def __getitem__(self, key: str) -> Series:
        return self.data[key]

    def __setitem__(self, key: str, value: Series):
        self.data[key] = value
        if key not in self.columns:
            self.columns.append(key)

    def to_csv(self, path: str, index: bool = True, index_label: str = None):
        fieldnames = ([] if not index else [index_label or "index"]) + self.columns
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for i in range(len(self.index)):
                row = []
                if index:
                    row.append(self.index[i])
                for col in self.columns:
                    row.append(self.data[col].data[i])
                writer.writerow(row)

    @classmethod
    def read_csv(cls, path: str, parse_dates: Optional[List[str]] = None, index_col: Optional[str] = None):
        with open(path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        columns: Dict[str, List] = {}
        index = []
        for row in rows:
            for key, value in row.items():
                if key == index_col:
                    if parse_dates and key in parse_dates:
                        index.append(datetime.fromisoformat(value))
                    else:
                        index.append(value)
                else:
                    columns.setdefault(key, []).append(float(value) if value not in (None, "") else 0.0)
        data = {k: Series(v, index if index_col else None) for k, v in columns.items()}
        df = DataFrame(data)
        if index_col:
            df.index = index
            for col_series in df.data.values():
                col_series.index = index
        return df


def read_csv(path: str, parse_dates: Optional[List[str]] = None, index_col: Optional[str] = None):
    return DataFrame.read_csv(path, parse_dates=parse_dates, index_col=index_col)


__all__ = ["Series", "DataFrame", "date_range", "read_csv"]
