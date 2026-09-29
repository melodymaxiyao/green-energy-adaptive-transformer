"""Authoritative aligned-panel interface for the final paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .build_dataset import TickerDataset
from .universe import (
    BOUNDARY_PURGE,
    FINAL_TICKERS,
    FORECAST_HORIZONS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


@dataclass
class PanelDataset:
    """The shared stock-date sample index and its aligned arrays."""

    ticker: np.ndarray
    anchor_date: np.ndarray
    split: np.ndarray
    X_daily: np.ndarray
    X_weekly: np.ndarray
    targets: dict[int, np.ndarray]


@dataclass(frozen=True)
class DateSplit:
    train: tuple[pd.Timestamp, ...]
    validation: tuple[pd.Timestamp, ...]
    test: tuple[pd.Timestamp, ...]


def load_ticker_dataset(ticker: str, processed_dir: Path) -> TickerDataset:
    """Load and validate one per-ticker bundle created by build_dataset.py."""
    ticker_dir = processed_dir / ticker
    required = [
        ticker_dir / "X_daily.npy",
        ticker_dir / "X_weekly.npy",
        *(ticker_dir / f"y_{horizon}d.npy" for horizon in FORECAST_HORIZONS),
        ticker_dir / "meta.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {ticker}: {missing}")

    X_daily = np.load(ticker_dir / "X_daily.npy")
    X_weekly = np.load(ticker_dir / "X_weekly.npy")
    targets = {
        horizon: np.load(ticker_dir / f"y_{horizon}d.npy")
        for horizon in FORECAST_HORIZONS
    }
    metadata = pd.read_csv(ticker_dir / "meta.csv", parse_dates=["anchor_date"])
    lengths = {len(X_daily), len(X_weekly), len(metadata), *(len(x) for x in targets.values())}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths for {ticker}: {sorted(lengths)}")
    return TickerDataset(ticker, X_daily, X_weekly, targets, metadata)


def common_anchor_dates(records: dict[str, TickerDataset]) -> set[pd.Timestamp]:
    """Return the exact intersection of eligible anchor dates for all 17 stocks."""
    if set(records) != set(FINAL_TICKERS):
        raise ValueError("Records must contain exactly the frozen 17-stock universe")
    date_sets = [
        set(pd.to_datetime(records[ticker].metadata["anchor_date"]).dt.normalize())
        for ticker in FINAL_TICKERS
    ]
    return set.intersection(*date_sets)


def chronological_split(
    dates: set[pd.Timestamp] | list[pd.Timestamp],
    purge: int = BOUNDARY_PURGE,
) -> DateSplit:
    """Create the 70/15/15 split and purge Train/Validation boundaries."""
    sorted_dates = tuple(sorted(pd.Timestamp(date).normalize() for date in dates))
    if not sorted_dates:
        raise ValueError("Cannot split an empty anchor-date set")

    train_end = int(len(sorted_dates) * TRAIN_FRACTION)
    validation_end = int(
        len(sorted_dates) * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )
    train = sorted_dates[:train_end]
    validation = sorted_dates[train_end:validation_end]
    test = sorted_dates[validation_end:]

    if purge:
        train = train[:-purge] if len(train) > purge else ()
        validation = validation[:-purge] if len(validation) > purge else ()
    return DateSplit(tuple(train), tuple(validation), tuple(test))


def load_final_panel(processed_dir: Path) -> PanelDataset:
    """Load the balanced, common-date-only, boundary-purged 17-stock panel."""
    records = {
        ticker: load_ticker_dataset(ticker, processed_dir)
        for ticker in FINAL_TICKERS
    }
    split_dates = chronological_split(common_anchor_dates(records))
    date_to_split = {
        **{date: "train" for date in split_dates.train},
        **{date: "val" for date in split_dates.validation},
        **{date: "test" for date in split_dates.test},
    }

    tickers: list[str] = []
    dates: list[pd.Timestamp] = []
    splits: list[str] = []
    daily: list[np.ndarray] = []
    weekly: list[np.ndarray] = []
    targets = {horizon: [] for horizon in FORECAST_HORIZONS}

    for ticker in FINAL_TICKERS:
        record = records[ticker]
        ticker_dates = pd.to_datetime(record.metadata["anchor_date"]).dt.normalize()
        for index, date in enumerate(ticker_dates):
            split = date_to_split.get(pd.Timestamp(date))
            if split is None:
                continue
            tickers.append(ticker)
            dates.append(pd.Timestamp(date))
            splits.append(split)
            daily.append(record.X_daily[index])
            weekly.append(record.X_weekly[index])
            for horizon in FORECAST_HORIZONS:
                targets[horizon].append(record.targets[horizon][index])

    return PanelDataset(
        ticker=np.asarray(tickers),
        anchor_date=np.asarray(dates),
        split=np.asarray(splits),
        X_daily=np.asarray(daily, dtype=np.float32),
        X_weekly=np.asarray(weekly, dtype=np.float32),
        targets={
            horizon: np.asarray(values, dtype=np.float32)
            for horizon, values in targets.items()
        },
    )


def fit_train_only_normalizer(panel: PanelDataset) -> dict:
    """Fit per-ticker, per-resolution, per-feature mean/std on Train only."""
    statistics = {}
    train_mask = panel.split == "train"
    for ticker in np.unique(panel.ticker):
        ticker_train = (panel.ticker == ticker) & train_mask
        for resolution, values in (
            ("daily", panel.X_daily),
            ("weekly", panel.X_weekly),
        ):
            flat = values[ticker_train].reshape(-1, values.shape[-1])
            mean = flat.mean(axis=0)
            std = flat.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            statistics[(ticker, resolution)] = (mean, std)
    return statistics


def apply_normalizer(
    panel: PanelDataset,
    statistics: dict,
    frequency: str,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply fixed Train statistics to one or both input resolutions."""

    def normalize(values: np.ndarray, resolution: str) -> np.ndarray:
        output = np.empty_like(values)
        for ticker in np.unique(panel.ticker):
            mask = panel.ticker == ticker
            mean, std = statistics[(ticker, resolution)]
            output[mask] = (values[mask] - mean) / std
        return output

    if frequency == "daily_only":
        return normalize(panel.X_daily, "daily")
    if frequency == "weekly_only":
        return normalize(panel.X_weekly, "weekly")
    if frequency == "daily_weekly":
        return (
            normalize(panel.X_daily, "daily"),
            normalize(panel.X_weekly, "weekly"),
        )
    raise ValueError(f"Unknown frequency setting: {frequency}")
