"""
Task (Advisor Pivot) - Build Panel Multi-Scale Dataset (Daily + Weekly only)

This is the final Dataset Contract V2 builder for the green-energy panel.

The panel remains anchored on canonical Daily features from yfinance interval="1d"
with auto_adjust=True. Formal Weekly OHLCV is reconstructed deterministically from
that same canonical Daily data using a Monday-start calendar week and actual last
trading date availability. Provider-native Weekly bars are never used for formal
panel samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_universe import GREEN_ENERGY_UNIVERSE

FEATURES = ["open", "high", "low", "close", "volume"]
H_DAILY = 90
H_WEEKLY = 26
TARGET_HORIZONS = [5, 10, 20]
MAX_HORIZON = max(TARGET_HORIZONS)

RAW_DIR = ROOT / "dataset" / "finance" / "panel_raw"
OUT_DIR = ROOT / "dataset" / "multiscale_dataset" / "panel"

MIN_SAMPLES = 200


def _reconstruct_weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct formal weekly bars from canonical Daily OHLCV."""
    daily = daily.copy()
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    daily["week_start"] = daily["datetime"].apply(
        lambda dt: dt - pd.Timedelta(days=dt.weekday())
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
    weekly["datetime"] = weekly["week_start"]
    return weekly[["datetime", "week_start", "available_date", "open", "high", "low", "close", "volume"]]


def build_one(ticker: str) -> dict:
    """Build one ticker's panel sample bundle under the final V2 interim contract."""
    record = {
        "ticker": ticker,
        "n_samples": 0,
        "status": "ok",
        "note": "",
        "dataset_contract_version": 2,
        "canonical_daily_interval": "1d",
        "daily_auto_adjust": True,
        "daily_includes_anchor": True,
        "weekly_source": "daily_aggregation",
        "provider_native_weekly_used": False,
        "weekly_grouping": "monday_start_calendar_week",
        "weekly_availability_rule": "actual_last_trading_day",
        "weekly_lookback": H_WEEKLY,
        "daily_lookback": H_DAILY,
        "split_purge_dates": 20,
        "target_definition": "forward_adjusted_price_log_return",
        "horizons": "5,10,20",
    }

    daily_path = RAW_DIR / f"{ticker}_daily.csv"
    if not daily_path.exists():
        record["status"] = "missing_raw"
        return record

    daily = pd.read_csv(daily_path)
    if "datetime" not in daily.columns:
        record["status"] = "missing_datetime"
        return record

    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    weekly = _reconstruct_weekly_from_daily(daily)

    X_daily, X_weekly = [], []
    y_by_horizon = {h: [] for h in TARGET_HORIZONS}
    meta_records = []

    for i in range(H_DAILY - 1, len(daily) - MAX_HORIZON):
        anchor_dt = pd.Timestamp(daily.iloc[i]["datetime"])
        daily_window = daily.iloc[i - H_DAILY + 1 : i + 1][FEATURES]
        if len(daily_window) != H_DAILY:
            continue

        weekly_subset = weekly[
            (weekly["available_date"].notna()) & (weekly["available_date"] <= anchor_dt)
        ].copy()
        if len(weekly_subset) < H_WEEKLY:
            continue

        weekly_window = weekly_subset.sort_values("week_start").tail(H_WEEKLY)[FEATURES]
        last_week_row = weekly_subset.sort_values("week_start").tail(1).iloc[0]

        p0 = float(daily.iloc[i]["close"])
        for h in TARGET_HORIZONS:
            p1 = float(daily.iloc[i + h]["close"])
            y_by_horizon[h].append(float(np.log(p1 / p0)))

        X_daily.append(daily_window.values)
        X_weekly.append(weekly_window.values)

        meta_records.append(
            {
                "ticker": ticker,
                "anchor_date": anchor_dt,
                "daily_feature_end": anchor_dt,
                "weekly_bar_start": pd.Timestamp(last_week_row["week_start"]),
                "weekly_available_date": pd.Timestamp(last_week_row["available_date"]),
                "target_date_5d": daily.iloc[i + 5]["datetime"],
                "target_date_10d": daily.iloc[i + 10]["datetime"],
                "target_date_20d": daily.iloc[i + 20]["datetime"],
            }
        )

    n = len(X_daily)
    record["n_samples"] = n
    if n < MIN_SAMPLES:
        record["status"] = "too_few_samples"
        record["note"] = f"only {n} samples (< {MIN_SAMPLES})"
        return record

    X_daily = np.asarray(X_daily, dtype=np.float32)
    X_weekly = np.asarray(X_weekly, dtype=np.float32)
    y_5d = np.asarray(y_by_horizon[5], dtype=np.float32)
    y_10d = np.asarray(y_by_horizon[10], dtype=np.float32)
    y_20d = np.asarray(y_by_horizon[20], dtype=np.float32)
    meta = pd.DataFrame(meta_records)

    if np.isnan(X_daily).any() or np.isnan(X_weekly).any() or np.isnan(y_20d).any():
        record["status"] = "nan_found"
        record["note"] = "NaN detected in features or targets, skipped save"
        return record

    ticker_dir = OUT_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    np.save(ticker_dir / "X_daily.npy", X_daily)
    np.save(ticker_dir / "X_weekly.npy", X_weekly)
    np.save(ticker_dir / "y_5d.npy", y_5d)
    np.save(ticker_dir / "y_10d.npy", y_10d)
    np.save(ticker_dir / "y_20d.npy", y_20d)
    np.save(ticker_dir / "y.npy", y_20d)
    meta.to_csv(ticker_dir / "meta.csv", index=False)

    return record


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, ticker in enumerate(GREEN_ENERGY_UNIVERSE, start=1):
        record = build_one(ticker)
        manifest.append(record)
        print(f"[{i}/{len(GREEN_ENERGY_UNIVERSE)}] {ticker}: {record}")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(OUT_DIR / "panel_build_manifest.csv", index=False)

    print("\nBuild Summary")
    print("-" * 60)
    print(manifest_df.to_string(index=False))
    n_ok = (manifest_df["status"] == "ok").sum()
    print(f"\nok={n_ok} total={len(manifest_df)}")


if __name__ == "__main__":
    main()
