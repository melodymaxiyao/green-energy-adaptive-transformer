"""Build per-ticker Daily/Weekly windows and forward-return targets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import (
    DAILY_LOOKBACK,
    FEATURE_COLUMNS,
    FINAL_TICKERS,
    FORECAST_HORIZONS,
    MAX_FORECAST_HORIZON,
    WEEKLY_LOOKBACK,
)


@dataclass
class TickerDataset:
    """Aligned arrays and metadata for one ticker."""

    ticker: str
    X_daily: np.ndarray
    X_weekly: np.ndarray
    targets: dict[int, np.ndarray]
    metadata: pd.DataFrame


def load_daily(ticker: str, raw_dir: Path) -> pd.DataFrame:
    """Load one canonical Daily CSV without filling or interpolating observations."""
    path = raw_dir / f"{ticker}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Daily data for {ticker}: {path}")

    daily = pd.read_csv(path)
    required = ["datetime", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in daily.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    return daily[required].sort_values("datetime").reset_index(drop=True)


def reconstruct_weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate canonical Daily OHLCV into Monday-start calendar weeks."""
    daily = daily.copy()
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    daily["week_start"] = daily["datetime"].apply(
        lambda date: date - pd.Timedelta(days=date.weekday())
    )

    weekly = (
        daily.groupby("week_start", as_index=False)
        .agg(
            week_start=("week_start", "first"),
            available_date=("datetime", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["available_date"] = pd.to_datetime(weekly["available_date"])
    return weekly[["week_start", "available_date", *FEATURE_COLUMNS]]


def completed_weekly_window(
    weekly: pd.DataFrame,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Return the last 26 bars completed by the anchor date, or None."""
    available = weekly[
        weekly["available_date"].notna()
        & (weekly["available_date"] <= pd.Timestamp(anchor_date))
    ]
    if len(available) < WEEKLY_LOOKBACK:
        return None
    return available.sort_values("week_start").tail(WEEKLY_LOOKBACK)


def build_ticker_dataset(ticker: str, daily: pd.DataFrame) -> TickerDataset:
    """Construct the final-paper samples for one ticker."""
    daily = daily.copy()
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    weekly = reconstruct_weekly_from_daily(daily)

    daily_windows: list[np.ndarray] = []
    weekly_windows: list[np.ndarray] = []
    targets = {horizon: [] for horizon in FORECAST_HORIZONS}
    metadata = []

    for index in range(DAILY_LOOKBACK - 1, len(daily) - MAX_FORECAST_HORIZON):
        anchor_date = pd.Timestamp(daily.iloc[index]["datetime"])
        daily_window = daily.iloc[
            index - DAILY_LOOKBACK + 1 : index + 1
        ][list(FEATURE_COLUMNS)]
        if len(daily_window) != DAILY_LOOKBACK:
            continue

        weekly_window = completed_weekly_window(weekly, anchor_date)
        if weekly_window is None:
            continue

        price_at_anchor = float(daily.iloc[index]["close"])
        for horizon in FORECAST_HORIZONS:
            future_price = float(daily.iloc[index + horizon]["close"])
            targets[horizon].append(float(np.log(future_price / price_at_anchor)))

        daily_windows.append(daily_window.to_numpy())
        weekly_windows.append(weekly_window[list(FEATURE_COLUMNS)].to_numpy())
        last_week = weekly_window.iloc[-1]
        metadata.append(
            {
                "ticker": ticker,
                "anchor_date": anchor_date,
                "daily_feature_end": anchor_date,
                "weekly_bar_start": pd.Timestamp(last_week["week_start"]),
                "weekly_available_date": pd.Timestamp(last_week["available_date"]),
                "target_date_5d": daily.iloc[index + 5]["datetime"],
                "target_date_10d": daily.iloc[index + 10]["datetime"],
                "target_date_20d": daily.iloc[index + 20]["datetime"],
            }
        )

    result = TickerDataset(
        ticker=ticker,
        X_daily=np.asarray(daily_windows, dtype=np.float32),
        X_weekly=np.asarray(weekly_windows, dtype=np.float32),
        targets={
            horizon: np.asarray(values, dtype=np.float32)
            for horizon, values in targets.items()
        },
        metadata=pd.DataFrame(metadata),
    )
    arrays = [result.X_daily, result.X_weekly, *result.targets.values()]
    if not arrays or any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Non-finite values found while building {ticker}")
    return result


def save_ticker_dataset(dataset: TickerDataset, output_dir: Path) -> None:
    """Save one ticker bundle in the format consumed by panel.py."""
    ticker_dir = output_dir / dataset.ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    np.save(ticker_dir / "X_daily.npy", dataset.X_daily)
    np.save(ticker_dir / "X_weekly.npy", dataset.X_weekly)
    for horizon, values in dataset.targets.items():
        np.save(ticker_dir / f"y_{horizon}d.npy", values)
    dataset.metadata.to_csv(ticker_dir / "meta.csv", index=False)


def build_all(raw_dir: Path, output_dir: Path) -> None:
    """Build and save per-ticker bundles for the frozen 17-stock universe."""
    for ticker in FINAL_TICKERS:
        dataset = build_ticker_dataset(ticker, load_daily(ticker, raw_dir))
        save_ticker_dataset(dataset, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    build_all(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
