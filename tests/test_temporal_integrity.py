"""Temporal-integrity tests for the clean final-paper data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from green_energy_transformer.data.build_dataset import (
    build_ticker_dataset,
    completed_weekly_window,
    reconstruct_weekly_from_daily,
)
from green_energy_transformer.data.panel import (
    PanelDataset,
    apply_normalizer,
    chronological_split,
    fit_train_only_normalizer,
    load_final_panel,
)
from green_energy_transformer.data.universe import (
    BOUNDARY_PURGE,
    DAILY_LOOKBACK,
    FINAL_TICKERS,
    FORECAST_HORIZONS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    WEEKLY_LOOKBACK,
)


def synthetic_daily(n_rows: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-06", periods=n_rows)
    base = np.arange(n_rows, dtype=np.float64) + 100.0
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": base,
            "high": base + 2.0,
            "low": base - 2.0,
            "close": base + 1.0,
            "volume": np.arange(n_rows, dtype=np.float64) + 1_000.0,
        }
    )


def test_final_dataset_constants() -> None:
    assert FINAL_TICKERS == (
        "AES", "BEP", "BLDP", "BLNK", "CSIQ", "CWEN", "DQ", "ENPH",
        "FCEL", "FSLR", "HASI", "JKS", "NEE", "ORA", "PLUG", "RUN", "SEDG",
    )
    assert DAILY_LOOKBACK == 90
    assert WEEKLY_LOOKBACK == 26
    assert FORECAST_HORIZONS == (5, 10, 20)
    assert TRAIN_FRACTION == 0.70
    assert VALIDATION_FRACTION == 0.15
    assert BOUNDARY_PURGE == 20


def test_weekly_ohlcv_aggregation() -> None:
    daily = synthetic_daily(10)
    weekly = reconstruct_weekly_from_daily(daily)
    first = daily.iloc[:5]
    first_week = weekly.iloc[0]
    assert first_week["open"] == first.iloc[0]["open"]
    assert first_week["high"] == first["high"].max()
    assert first_week["low"] == first["low"].min()
    assert first_week["close"] == first.iloc[-1]["close"]
    assert first_week["volume"] == first["volume"].sum()


def test_unfinished_week_is_not_available() -> None:
    daily = synthetic_daily(160)
    weekly = reconstruct_weekly_from_daily(daily)
    current_week = weekly.iloc[-2]
    anchor = current_week["week_start"] + pd.Timedelta(days=2)
    window = completed_weekly_window(weekly, anchor)
    assert window is not None
    assert window.iloc[-1]["available_date"] < current_week["available_date"]
    assert (window["available_date"] <= anchor).all()


def test_windows_and_targets_use_future_trading_rows() -> None:
    daily = synthetic_daily()
    dataset = build_ticker_dataset("AES", daily)
    assert dataset.X_daily.shape[1:] == (DAILY_LOOKBACK, 5)
    assert dataset.X_weekly.shape[1:] == (WEEKLY_LOOKBACK, 5)

    sample = 10
    anchor = dataset.metadata.iloc[sample]["anchor_date"]
    anchor_index = daily.index[daily["datetime"] == anchor][0]
    for horizon in FORECAST_HORIZONS:
        expected_date = daily.iloc[anchor_index + horizon]["datetime"]
        expected = np.log(
            daily.iloc[anchor_index + horizon]["close"]
            / daily.iloc[anchor_index]["close"]
        )
        assert dataset.metadata.iloc[sample][f"target_date_{horizon}d"] == expected_date
        assert np.isclose(dataset.targets[horizon][sample], expected)


def test_chronological_split_and_boundary_purges() -> None:
    dates = set(pd.bdate_range("2018-01-01", periods=400))
    unpurged = chronological_split(dates, purge=0)
    purged = chronological_split(dates)

    assert (len(unpurged.train), len(unpurged.validation), len(unpurged.test)) == (
        280, 60, 60,
    )
    assert max(purged.train) < min(purged.validation)
    assert max(purged.validation) < min(purged.test)
    assert purged.train == unpurged.train[:-BOUNDARY_PURGE]
    assert purged.validation == unpurged.validation[:-BOUNDARY_PURGE]
    assert purged.test == unpurged.test
    assert (len(purged.train), len(purged.validation), len(purged.test)) == (
        260, 40, 60,
    )


def test_normalization_is_fit_on_train_only() -> None:
    tickers = np.array(["AES"] * 4)
    split = np.array(["train", "train", "val", "test"])
    daily = np.array([1.0, 3.0, 100.0, 1_000.0], dtype=np.float32).reshape(4, 1, 1)
    weekly = np.array([2.0, 4.0, 200.0, 2_000.0], dtype=np.float32).reshape(4, 1, 1)
    panel = PanelDataset(
        ticker=tickers,
        anchor_date=np.array(pd.bdate_range("2020-01-01", periods=4)),
        split=split,
        X_daily=daily,
        X_weekly=weekly,
        targets={
            horizon: np.zeros(4, dtype=np.float32)
            for horizon in FORECAST_HORIZONS
        },
    )
    stats = fit_train_only_normalizer(panel)
    assert np.array_equal(stats[("AES", "daily")][0], np.array([2.0], dtype=np.float32))
    assert np.array_equal(stats[("AES", "weekly")][0], np.array([3.0], dtype=np.float32))
    normalized_daily = apply_normalizer(panel, stats, "daily_only")
    assert np.allclose(normalized_daily[:2, 0, 0], [-1.0, 1.0])


def write_bundle(root: Path, ticker: str, dates: pd.DatetimeIndex) -> None:
    ticker_dir = root / ticker
    ticker_dir.mkdir(parents=True)
    n = len(dates)
    np.save(
        ticker_dir / "X_daily.npy",
        np.ones((n, DAILY_LOOKBACK, 5), dtype=np.float32),
    )
    np.save(
        ticker_dir / "X_weekly.npy",
        np.ones((n, WEEKLY_LOOKBACK, 5), dtype=np.float32),
    )
    for horizon in FORECAST_HORIZONS:
        np.save(ticker_dir / f"y_{horizon}d.npy", np.arange(n, dtype=np.float32))
    pd.DataFrame({"ticker": ticker, "anchor_date": dates}).to_csv(
        ticker_dir / "meta.csv", index=False
    )


def test_final_panel_has_one_shared_stock_date_index(tmp_path: Path) -> None:
    dates = pd.bdate_range("2018-01-01", periods=400)
    excluded_date = dates[-10]
    for position, ticker in enumerate(FINAL_TICKERS):
        ticker_dates = dates.delete(len(dates) - 10) if position == 0 else dates
        write_bundle(tmp_path, ticker, ticker_dates)

    panel = load_final_panel(tmp_path)
    assert len(panel.ticker) == len(panel.X_daily) == len(panel.X_weekly)
    assert all(len(values) == len(panel.ticker) for values in panel.targets.values())

    sample_index = pd.DataFrame(
        {"ticker": panel.ticker, "anchor_date": panel.anchor_date}
    )
    date_sets = [
        set(group["anchor_date"])
        for _, group in sample_index.groupby("ticker")
    ]
    assert set(sample_index["ticker"]) == set(FINAL_TICKERS)
    assert not sample_index.duplicated(["ticker", "anchor_date"]).any()
    assert excluded_date not in set(pd.to_datetime(panel.anchor_date))
    assert len(date_sets) == 17
    assert all(date_set == date_sets[0] for date_set in date_sets[1:])
    assert sample_index.groupby("anchor_date")["ticker"].nunique().eq(17).all()
